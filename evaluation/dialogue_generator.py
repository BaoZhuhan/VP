"""dialogue_generator.py — Generate clinician–patient dialogues for AMICA benchmark."""

import json, time, os, sys
from openai import OpenAI

MODEL = "deepseek-v4-flash"

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
    "On a scale of 1 to 10, how would you rate what you're going through right now — physically, and also emotionally?",
    "Is there anything else you think I should know?",
]


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


def _system_prompt(profile, strategy):
    d = profile["demographics"]
    a = profile["affective_attributes"]
    s = profile["scenario"]

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
        f"Your health literacy level is: {a['health_literacy']} ({lit_hint.get(a['health_literacy'], '')}).\n"
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

    elif strategy == "consistency":
        return (
            base
            + "\n\nCONSISTENCY RULES:\n"
            "- Throughout this entire conversation, you MUST remain consistent with your underlying patient profile.\n"
            "- Your symptoms, severity, emotional state, and communication style should never change or evolve unless the conversation naturally warrants it.\n"
            "- If you describe a symptom one way early on, do not later describe it differently.\n"
            "- Your mood, anxiety, and cooperation levels should remain stable throughout the conversation.\n"
            "- Do not introduce new symptoms that are not part of your profile.\n"
            "- If the clinician asks about something unrelated to your condition, acknowledge it briefly but stay focused."
        )

    elif strategy == "structured_state":
        snapshot = _state_to_snapshot(profile)
        return (
            base
            + f"\n\nSTRUCTURED PATIENT STATE (internal reference — DO NOT quote directly, use as grounding):\n{snapshot}\n\n"
            "GROUNDING RULES:\n"
            "- The state above defines every detail of your condition. All responses must be consistent with it.\n"
            "- Express clinical attributes naturally: e.g., if severity=4, use phrases like 'really bad', 'can barely function'.\n"
            "- Express affective attributes through tone and word choice: anxious patients speak more hesitantly."
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def generate_dialogue(profile, strategy="standard", max_turns=20, verbose=False):
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

        # Patient turn
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=256,
            )
            patient_msg = resp.choices[0].message.content.strip()
        except Exception as e:
            patient_msg = f"[API ERROR: {e}]"
            if verbose:
                print(f"  Turn {turn_i} error: {e}", file=sys.stderr)

        turns.append({"turn_id": turn_i, "role": "patient", "content": patient_msg})
        messages.append({"role": "assistant", "content": patient_msg})

        # Natural ending check
        if any(phrase in patient_msg.lower() for phrase in ["thank you", "goodbye", "i'll try that"]):
            if turn_i >= 6:
                break

        time.sleep(0.3)

    return {
        "profile_id": profile["patient_id"],
        "strategy": strategy,
        "total_turns": len([t for t in turns if t["role"] == "patient"]),
        "turns": turns,
    }
