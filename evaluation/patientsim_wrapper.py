"""patientsim_wrapper.py — PatientSim-style prompt construction for C4/C5.

Uses PatientSim's persona axes (CEFR/language proficiency, personality,
recall level, dazed level) to construct system prompts, then calls
DeepSeek V4 Flash directly with our fixed clinician questions.

Conditions:
  C4: PatientSim's native persona (CEFR level maps to health literacy)
  C5: C4 + explicit health_literacy injection as additional condition
"""

import json, os, sys, time, re, copy

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── PatientSim persona definitions ─────────────────────────────────

CEFR_LEVELS = {
    "A": (
        "a patient with basic English proficiency who can only use and understand very simple language.\n"
        "Act as a patient with basic English proficiency (CEFR A). You must:\n"
        "1) Speaking: Use only basic, simple words. Respond with short phrases instead of full sentences. "
        "Make frequent grammar mistakes. Do not use any complex words or long phrases.\n"
        "2) Understanding: Understand only simple, everyday words and phrases. Struggle with even slightly "
        "complex words or sentences. Often need repetition or easy explanations to understand.\n"
        "3) Medical Terms: Use and understand only very simple, everyday medical words, with limited medical "
        "knowledge. Cannot use or understand complex medical terms. Need all medical terms to be explained "
        "in very simple, everyday language.\n"
        "IMPORTANT: If a question contains any difficult words, long sentences, or complex grammar, "
        "respond like 'What?' or 'I don't understand'. Keep asking until the question is simple enough for you to answer."
    ),
    "B": (
        "a patient with intermediate English proficiency who can use and understand well in everyday language.\n"
        "Act as a patient with intermediate English proficiency (CEFR B). You must:\n"
        "1) Speaking: Use common vocabulary and form connected, coherent sentences with occasional minor "
        "grammar errors. Discuss familiar topics confidently but struggle with abstract or technical subjects. "
        "Avoid highly specialized or abstract words.\n"
        "2) Understanding: Can understand the main ideas of everyday conversations. Need clarification or "
        "simpler explanations for abstract, technical, or complex information.\n"
        "3) Medical Terms: Use and understand common medical terms related to general health. Cannot use or "
        "understand advanced or specialized medical terms and require these to be explained in simple language.\n"
        "IMPORTANT: If a question contains advanced terms beyond your level, ask for simpler explanation "
        "(e.g., 'I don't get it' or 'What do you mean?'). Keep asking until the question is clear enough for you to answer."
    ),
    "C": (
        "a patient with proficient English proficiency who can use and understand highly complex, detailed "
        "language, including advanced medical terminology.\n"
        "Act as a patient with proficient English proficiency (CEFR C). You must:\n"
        "1) Speaking: Use a full range of vocabulary with fluent, precise language. Can construct well-structured, "
        "complex sentences with diverse and appropriate word choices.\n"
        "2) Understanding: Fully comprehend detailed, complex explanations and abstract concepts.\n"
        "3) Medical Terminology: Use and understand highly specialized medical terms, with expert-level knowledge "
        "of medical topics.\n"
        "IMPORTANT: Reflect your high-level language proficiency mainly through precise vocabulary choices "
        "rather than by making your responses unnecessarily long."
    ),
}

PERSONALITY_TYPES = {
    "plain": (
        "a neutral patient without any distinctive personality traits.\n"
        "1) Provides concise, direct answers focused on the question, without extra details.\n"
        "2) Responds in a neutral tone without any noticeable emotion or personality."
    ),
    "terse": (
        "a patient who speaks very little.\n"
        "1) Gives the shortest possible answers, often one or two words.\n"
        "2) Does not volunteer any extra information.\n"
        "3) Responds with minimal words unless specifically asked for more detail."
    ),
    "detailed": (
        "a patient who provides thorough and comprehensive descriptions.\n"
        "1) Gives detailed answers covering multiple aspects of the question.\n"
        "2) Offers context and background information naturally.\n"
        "3) May elaborate on related topics when relevant."
    ),
    "stoic": (
        "a stoic patient who downplays their symptoms and discomfort.\n"
        "1) Minimizes pain and discomfort, often saying 'it's fine' or 'not too bad'.\n"
        "2) Expresses little emotion even when describing significant problems.\n"
        "3) Shows resilience and reluctance to complain."
    ),
    "emotional": (
        "an emotional patient who expresses their feelings openly.\n"
        "1) Freely expresses worry, fear, or other emotions.\n"
        "2) Describes how symptoms affect them emotionally.\n"
        "3) May become visibly distressed when discussing serious concerns."
    ),
}

RECALL_LEVELS = {
    "low": (
        "have limited medical history recall ability.\n"
        "1) Frequently forget important medical history, such as previous diagnoses, surgeries, or medications.\n"
        "2) May be uncertain about details like dates, names of conditions, or medication dosages.\n"
        "3) Sometimes remember things incorrectly or say 'I don't remember'."
    ),
    "high": (
        "have a clear and detailed ability to recall medical history.\n"
        "1) Accurately remember all health-related information, including past conditions and current medications.\n"
        "2) Can provide specific details about medical history when asked.\n"
        "3) Consistently recall details that match their medical records."
    ),
}

DAZED_LEVELS = {
    "normal": "Act without confusion. Clearly understand the question according to your language proficiency level, and naturally reflect your background and personality in your responses.",
    "moderate": (
        "1) Sometimes provide answers that are slightly off-topic or hesitant.\n"
        "2) Occasionally need questions to be repeated or simplified.\n"
        "3) Become less confused as the conversation progresses."
    ),
}

HL_CEFR_MAP = {"low": "A", "medium": "B", "high": "C"}

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


def build_patientsim_prompt(profile, condition="C4"):
    """Construct PatientSim-style system prompt for a given profile.

    Args:
        profile: Patient profile dict from patient_profiles.json
        condition: 'C4' (native CEFR mapping) or 'C5' (CEFR + HL injection)

    Returns:
        System prompt string
    """
    d = profile["demographics"]
    c = profile["clinical_attributes"]
    a = profile["affective_attributes"]
    s = profile["scenario"]

    hl = a["health_literacy"]
    cefr = HL_CEFR_MAP[hl]
    comm_style = a["communication_style"]

    # Map our communication_style to PatientSim personality
    # If it doesn't exist in our mapping, use "plain"
    personality = PERSONALITY_TYPES.get(comm_style, PERSONALITY_TYPES["plain"])

    # Determine recall level based on HL
    recall = "low" if hl == "low" else "high"

    # Build the prompt sections
    prompt = (
        f"You are role-playing a patient in a medical consultation.\n\n"
        f"Patient Background Information:\n"
        f"    Demographics:\n"
        f"        Age: {d['age']}\n"
        f"        Gender: {d['gender']}\n"
        f"        Occupation: {d['occupation']}\n\n"
        f"    Setting: {s['setting'].replace('_', ' ')} for a {'routine' if s['urgency'] == 'routine' else 'urgent'} visit.\n"
        f"    Chief Complaint: {s['chief_complaint']}\n\n"
        f"    Current symptoms:\n"
        f"        Primary: {c['primary_symptom']}\n"
        f"        Secondary: {', '.join(c['secondary_symptoms'])}\n"
        f"        Duration: {c['symptom_duration']}\n"
        f"        Severity: {c['disease_severity']}/5\n"
        f"    Chronic conditions: {', '.join(c['chronic_conditions'])}\n"
        f"    Medications: {', '.join(c['medications'])}\n\n"
        f"Persona:\n"
        f"    Personality: {personality}\n"
        f"    Language Proficiency (CEFR): {CEFR_LEVELS[cefr]}\n"
        f"    Medical History Recall Ability: {RECALL_LEVELS[recall]}\n"
        f"    Dazedness Level: {DAZED_LEVELS['normal']}\n"
    )

    # Add emotional state context
    prompt += (
        f"\nYour current emotional state: you feel {a['mood']}, your pain level is {a['pain_level']}/5, "
        f"and you are at anxiety level {a['anxiety_level']}/5.\n"
    )

    # Condition-specific additions
    if condition == "C5":
        prompt += (
            f"\nADDITIONAL PATIENT CHARACTERISTIC:\n"
            f"Your health literacy level is: {hl}.\n"
            f"This means you have {'limited' if hl == 'low' else 'moderate' if hl == 'medium' else 'good'} "
            f"ability to understand, process, and act upon health information.\n"
        )

    prompt += (
        f"\nDuring the consultation, follow these guidelines:\n"
        f"1. Fully immerse yourself in the patient role.\n"
        f"2. Ensure responses stay consistent with the patient's profile, current symptoms, and prior conversation.\n"
        f"3. Align responses with the patient's language proficiency.\n"
        f"4. Match the tone and style to the patient's personality.\n"
        f"5. Keep responses realistic and natural.\n"
        f"6. Use everyday language appropriate to your language proficiency level.\n"
        f"7. Respond only with what the patient would say.\n"
        f"8. Never reveal that you are an AI or that you have a profile.\n"
        f"9. Never explicitly state your CEFR level, health literacy, or personality.\n"
        f"10. Keep responses concise (1-4 sentences per turn)."
    )

    return prompt


def generate_patientsim_dialogue(profile, condition="C4", max_turns=18, verbose=False):
    """Generate dialogue using PatientSim-style persona construction.

    Args:
        profile: Patient profile dict
        condition: 'C4' or 'C5'
        max_turns: Number of clinician-patient turn pairs
        verbose: Print progress

    Returns:
        Dialogue dict
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com/v1",
    )

    system_prompt = build_patientsim_prompt(profile, condition)
    messages = [{"role": "system", "content": system_prompt}]
    turns = []

    condition_name = "patientsim_c4" if condition == "C4" else "patientsim_c5"

    for turn_i in range(1, max_turns + 1):
        if turn_i <= len(CLINICIAN_PROMPTS):
            clinician_msg = CLINICIAN_PROMPTS[turn_i - 1]
        else:
            clinician_msg = "And how have you been coping with all of this?"

        turns.append({"turn_id": turn_i, "role": "clinician", "content": clinician_msg})
        messages.append({"role": "user", "content": clinician_msg})

        try:
            resp = client.chat.completions.create(
                model="deepseek-v4-flash",
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
            time.sleep(5)

        turns.append({"turn_id": turn_i, "role": "patient", "content": patient_msg})
        messages.append({"role": "assistant", "content": patient_msg})

        if any(phrase in patient_msg.lower() for phrase in ["thank you", "goodbye"]):
            if turn_i >= 6:
                break

        time.sleep(0.3)

    return {
        "profile_id": profile["patient_id"],
        "strategy": condition_name,
        "total_turns": len([t for t in turns if t["role"] == "patient"]),
        "turns": turns,
    }


def generate_batch(profiles, condition, output_dir, verbose=True):
    """Generate dialogues for a list of profiles using PatientSim style."""
    from openai import OpenAI

    os.makedirs(output_dir, exist_ok=True)
    results = []
    strategy_name = "patientsim_c4" if condition == "C4" else "patientsim_c5"
    for profile in profiles:
        pid = profile["patient_id"]
        fname = f"{pid}_{strategy_name}.json"
        fpath = os.path.join(output_dir, fname)
        if os.path.exists(fpath):
            if verbose:
                print(f"  {fname} already exists, skipping")
            continue

        if verbose:
            print(f"  Generating {fname}...", end=" ", flush=True)
        dialogue = generate_patientsim_dialogue(profile, condition, verbose=verbose)
        with open(fpath, "w") as f:
            json.dump(dialogue, f, indent=2)
        results.append(dialogue)
        if verbose:
            print(f"done ({dialogue['total_turns']} turns)")
    return results


if __name__ == "__main__":
    profiles_path = os.path.join(BASE_DIR, "data", "patient_profiles.json")
    with open(profiles_path) as f:
        data = json.load(f)

    condition = sys.argv[1] if len(sys.argv) > 1 else "C4"
    assert condition in ("C4", "C5"), f"Invalid condition: {condition}"

    output_dir = os.path.join(BASE_DIR, "data", "dialogues")
    print(f"Generating {condition} dialogues for {len(data['profiles'])} profiles...")
    generate_batch(data["profiles"], condition, output_dir, verbose=True)
    print(f"\nDone! Generated {condition} dialogues.")
