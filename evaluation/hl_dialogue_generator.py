"""hl_dialogue_generator.py — Generate health-literacy-conditioned dialogues.

Three generation modes:
  - standard: baseline role-play with profile info
  - literacy_anchored: explicit behavioral guidelines for each HL level
  - structured_state: full patient state as internal reference
"""

import json, time, os, sys
from openai import OpenAI

MODEL = "deepseek-v4-flash"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CLINICIAN_PROMPTS = [
    "What brings you in today?",
    "When did this start?",
    "Can you describe the pain or discomfort? Where exactly do you feel it?",
    "Have you had any fever or chills?",
    "What medications are you currently taking?",
    "Do you have any chronic medical conditions?",
    "Are you allergic to any medications?",
    "Have you tried anything to relieve the symptoms?",
    "Does anything make it better or worse?",
    "How has this been affecting your daily life?",
    "Do you smoke or drink alcohol?",
    "Has anyone in your family had similar problems?",
    "Have you experienced anything like this before?",
    "I've explained quite a bit — does that make sense so far?",
    "On a scale of 1 to 10, how would you rate what you're going through right now — physically, and also emotionally?",
    "Do you have any questions for me?",
    "What do you think might be going on?",
    "Is there anything else you think I should know?",
]

# Extended to 18 prompts to include literacy-sensitive probes at turns 14, 16, 17


def _state_to_snapshot(profile):
    c = profile["clinical_attributes"]
    a = profile["affective_attributes"]
    d = profile["demographics"]
    s = profile["scenario"]
    sec = ", ".join(c["secondary_symptoms"]) or "none"
    chr_c = ", ".join(c["chronic_conditions"])
    meds = ", ".join(c["medications"])
    algs = ", ".join(c["allergies"])
    return (
        f"PATIENT STATE:\n"
        f"Age: {d['age']}, Gender: {d['gender']}, Occupation: {d['occupation']}\n"
        f"Scenario: {s['category']}, Urgency: {s['urgency']}, Setting: {s['setting']}\n"
        f"Chief Complaint: {s['chief_complaint']}\n"
        f"CLINICAL:\n"
        f"  Primary symptom: {c['primary_symptom']}\n"
        f"  Secondary symptoms: {sec}\n"
        f"  Duration: {c['symptom_duration']}\n"
        f"  Severity (1-5): {c['disease_severity']}\n"
        f"  Temperature: {c['temperature_c']}C\n"
        f"  Chronic conditions: {chr_c}\n"
        f"  Medications: {meds}\n"
        f"  Allergies: {algs}\n"
        f"  Smoking: {c['smoking_status']}, Alcohol: {c['alcohol_use']}\n"
        f"AFFECTIVE:\n"
        f"  Anxiety (1-5): {a['anxiety_level']}\n"
        f"  Pain (1-5): {a['pain_level']}\n"
        f"  Mood: {a['mood']}\n"
        f"  Cooperation (1-5): {a['cooperation_willingness']}\n"
        f"  Health literacy: {a['health_literacy']}\n"
        f"  Communication style: {a['communication_style']}"
    )


HL_BEHAVIORAL_GUIDELINES = {
    "low": (
        "HEALTH LITERACY GUIDELINES — Your health literacy is LOW:\n"
        "- You use very simple, everyday language. No medical terms.\n"
        "- You describe symptoms in basic terms: 'it hurts here,' 'I feel bad.'\n"
        "- You rarely ask the doctor questions — you assume they know best.\n"
        "- When the doctor asks if you understand, you usually just say 'yes' or 'okay' even if you're unsure.\n"
        "- You let the doctor make decisions: 'whatever you think, doc.'\n"
        "- Your answers are short. You don't volunteer extra information.\n"
        "- You don't express preferences or opinions about your treatment.\n"
        "- You may seem unsure about dates, durations, or specific details.\n"
        "- You don't use words like 'symptom,' 'diagnosis,' or 'medication.'"
    ),
    "medium": (
        "HEALTH LITERACY GUIDELINES — Your health literacy is MEDIUM:\n"
        "- You can describe your symptoms reasonably well but sometimes imprecise.\n"
        "- You ask a few questions but mainly when the doctor invites them.\n"
        "- You sometimes express preferences but often defer to the doctor.\n"
        "- Your answers are moderate in length — you give detail when asked.\n"
        "- You understand most of what the doctor says but may need clarification on complex terms."
    ),
    "high": (
        "HEALTH LITERACY GUIDELINES — Your health literacy is HIGH:\n"
        "- You describe your symptoms clearly and in some detail.\n"
        "- You sometimes use appropriate medical terms.\n"
        "- You ask questions and want to understand your condition.\n"
        "- You express your preferences and participate in decisions.\n"
        "- Your answers are detailed and structured.\n"
        "- If something is unclear, you ask for clarification.\n"
        "- You can describe your medical history and medications accurately."
    ),
}


def _system_prompt(profile, strategy):
    d = profile["demographics"]
    a = profile["affective_attributes"]
    s = profile["scenario"]
    hl = a["health_literacy"]

    lit_hint = {
        "low": "use simple language and may struggle to explain your symptoms precisely",
        "medium": "can communicate adequately but may sometimes be imprecise",
        "high": "use appropriate medical terms and can describe your condition clearly",
    }

    base = (
        f"You are role-playing a patient in a medical consultation.\n\n"
        f"Your identity:\n"
        f"- You are a {d['age']}-year-old {d['gender']} patient who works as {d['occupation']}.\n"
        f"- You are visiting a {s['setting'].replace('_', ' ')} for a {'routine' if s['urgency'] == 'routine' else 'urgent'} visit.\n"
        f"- Your reason for coming: {s['chief_complaint']}\n\n"
        f"Your communication style is: {a['communication_style']}.\n"
        f"Your health literacy level is: {hl} ({lit_hint.get(hl, '')}).\n"
        f"Your current emotional state: you feel {a['mood']}, your pain level is {a['pain_level']}/5, "
        f"and you are at anxiety level {a['anxiety_level']}/5.\n"
        f"Your willingness to cooperate is {a['cooperation_willingness']}/5.\n\n"
        f"ABSOLUTE RULES:\n"
        f"- Stay in character at all times. You are the patient, never break the fourth wall.\n"
        f"- Answer naturally, like a real person talking to their doctor.\n"
        f"- Never mention your clinical attributes explicitly (don't say 'my pain level is 3/5' — describe how you feel).\n"
        f"- Never list your chronic conditions or medications from a script — only mention them when the doctor asks.\n"
        f"- Never reveal that you are an AI or that you have a 'patient state' or 'profile.'\n"
        f"- If the clinician asks something you don't know the answer to, just say you don't know.\n"
        f"- Keep responses concise (1-4 sentences per turn) but detailed enough to sound real."
    )

    if strategy == "standard":
        return base

    elif strategy == "literacy_anchored":
        return base + "\n\n" + HL_BEHAVIORAL_GUIDELINES.get(hl, "")

    elif strategy == "structured_state":
        snapshot = _state_to_snapshot(profile)
        return (
            base
            + f"\n\nSTRUCTURED PATIENT STATE (internal reference — DO NOT quote directly, use as grounding):\n{snapshot}\n\n"
            "GROUNDING RULES:\n"
            "- The state above defines every detail of your condition. All responses must be consistent with it.\n"
            "- Express clinical attributes naturally: e.g., if severity=4, use phrases like 'really bad', 'can barely function'.\n"
            "- Express affective attributes through tone and word choice."
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def generate_dialogue(profile, strategy="standard", max_turns=18, verbose=False):
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )

    system = _system_prompt(profile, strategy)
    messages = [{"role": "system", "content": system}]
    turns = []

    for turn_i in range(1, max_turns + 1):
        # Clinician turn
        if turn_i <= len(CLINICIAN_PROMPTS):
            clinician_msg = CLINICIAN_PROMPTS[turn_i - 1]
        else:
            clinician_msg = "And how have you been coping with all of this?"

        turns.append({"turn_id": turn_i, "role": "clinician", "content": clinician_msg})
        messages.append({"role": "user", "content": clinician_msg})

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=256,
                timeout=30,
            )
            patient_msg = resp.choices[0].message.content.strip()
        except Exception as e:
            patient_msg = f"[API ERROR: {e}]"
            if verbose:
                print(f"  Turn {turn_i} error: {e}", file=sys.stderr)
            time.sleep(2)

        turns.append({"turn_id": turn_i, "role": "patient", "content": patient_msg})
        messages.append({"role": "assistant", "content": patient_msg})

        if any(phrase in patient_msg.lower() for phrase in ["thank you", "goodbye"]):
            if turn_i >= 6:
                break

        time.sleep(0.3)

    return {
        "profile_id": profile["patient_id"],
        "strategy": strategy,
        "total_turns": len([t for t in turns if t["role"] == "patient"]),
        "turns": turns,
    }


def generate_batch(profiles, strategy, output_dir, verbose=True):
    """Generate dialogues for a list of profiles."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for profile in profiles:
        pid = profile["patient_id"]
        if verbose:
            print(f"  Generating {pid}_{strategy}...", end=" ", flush=True)
        dialogue = generate_dialogue(profile, strategy, verbose=verbose)
        fname = f"{pid}_{strategy}.json"
        path = os.path.join(output_dir, fname)
        with open(path, "w") as f:
            json.dump(dialogue, f, indent=2)
        results.append(dialogue)
        if verbose:
            print(f"done ({dialogue['total_turns']} turns)")
    return results


if __name__ == "__main__":
    # Load profiles
    profiles_path = os.path.join(BASE_DIR, "data", "patient_profiles.json")
    with open(profiles_path) as f:
        data = json.load(f)

    output_dir = os.path.join(BASE_DIR, "data", "dialogues")

    strategy = sys.argv[1] if len(sys.argv) > 1 else "literacy_anchored"

    print(f"Generating {strategy} dialogues for {len(data['profiles'])} profiles...")
    generate_batch(data["profiles"], strategy, output_dir, verbose=True)
    print(f"\nDone! Generated {len(data['profiles'])} dialogues with strategy='{strategy}'")
    print(f"Output: {output_dir}")
