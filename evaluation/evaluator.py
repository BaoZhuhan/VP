"""evaluator.py — State faithfulness evaluator for VP (Consistency in Virtual Patient Simulation) benchmark.

Given a dialogue, reconstruct the latent patient state and compare
with ground truth at the attribute level.
"""

import json, os, sys, re, time


CLINICAL_TARGETS = [
    "primary_symptom",
    "secondary_symptoms",
    "symptom_duration",
    "disease_severity",
    "chronic_conditions",
]

AFFECTIVE_TARGETS = [
    "anxiety_level",
    "pain_level",
    "mood",
    "cooperation_willingness",
]

MULTI_SELECT_FIELDS = {"secondary_symptoms", "chronic_conditions"}
ORDINAL_FIELDS = {"disease_severity", "anxiety_level", "pain_level", "cooperation_willingness"}


# ── Normalization helpers ─────────────────────────────────────────

def _normalize_duration(text):
    """Map free-text duration to schema key."""
    text = text.strip().lower()
    text = text.replace("about ", "").replace("around ", "").replace("roughly ", "").replace("almost ", "")
    if "year" in text or "years" in text:
        return "6+_months"
    if "month" in text or "months" in text:
        return "3-6_months"
    if "hour" in text or "hours" in text or "since this morning" in text:
        return "hours"
    if "day" in text or "days" in text:
        # Try to extract number
        for w in text.split():
            if w.isdigit():
                n = int(w)
                if n <= 2: return "1-2_days"
                if n <= 7: return "3-7_days"
        if "yesterday" in text or "last night" in text:
            return "1-2_days"
        return "3-7_days"
    if "week" in text or "weeks" in text:
        for w in text.split():
            if w.isdigit():
                n = int(w)
                if n <= 2: return "1-2_weeks"
                if n <= 4: return "2-4_weeks"
        return "1-2_weeks"  # default
    return text.replace(" ", "_")


def _normalize_mood(text):
    """Map free-text mood to schema key."""
    text = text.strip().lower()
    mood_map = {
        "anxious": ["anxious", "worried", "nervous", "anxious and worried", "anxious but cooperative"],
        "depressed": ["depressed", "sad", "down", "low mood", "depression", "hopeless", "blue"],
        "irritable": ["irritable", "irritated", "frustrated", "cranky"],
        "fearful": ["fearful", "scared", "afraid", "terrified", "frightened"],
        "resigned": ["resigned", "accepting"],
        "neutral": ["neutral", "okay", "fine", "calm"],
        "optimistic": ["optimistic", "hopeful", "positive"],
        "angry": ["angry", "mad", "furious"],
    }
    for key, synonyms in mood_map.items():
        for syn in synonyms:
            if syn in text:
                return key
    return text


def _normalize_symptom(text):
    """Map free-text symptom description to schema key."""
    text = text.strip().lower()
    symptom_map = {
        "cough": ["cough", "coughing"],
        "shortness_of_breath": ["shortness of breath", "breathless", "difficulty breathing", "can't breathe", "hard to breathe"],
        "chest_pain": ["chest pain", "chest tightness", "chest pressure", "chest discomfort"],
        "headache": ["headache", "head pain"],
        "fatigue": ["fatigue", "tired", "exhausted", "weak", "no energy"],
        "fever": ["fever", "feverish", "fever and chills", "high temperature"],
        "abdominal_pain": ["abdominal pain", "stomach pain"],
        "dizziness": ["dizziness", "dizzy", "lightheaded"],
        "back_pain": ["back pain"],
        "nausea": ["nausea", "nauseous"],
        "sore_throat": ["sore throat", "throat pain"],
        "numbness": ["numbness", "numb"],
        "rash": ["rash", "skin rash", "red patches"],
        "palpitations": ["palpitations", "heart racing", "heart pounding", "fluttering"],
        "joint_pain": ["joint pain", "joint aches", "knee pain"],
    }
    for key, synonyms in symptom_map.items():
        for syn in synonyms:
            if syn in text:
                return key
    return text.split(",")[0].strip()


def _normalize_chronic(text):
    """Map free-text chronic condition to schema key."""
    text = text.strip().lower()
    chronic_map = {
        "asthma": ["asthma"],
        "hypertension": ["hypertension", "high blood pressure", "high bp"],
        "diabetes_type2": ["diabetes", "type 2 diabetes", "high blood sugar", "diabetic"],
        "coronary_artery_disease": ["coronary artery", "heart disease", "cad"],
        "hypothyroidism": ["hypothyroidism", "underactive thyroid", "thyroid"],
        "anxiety_disorder": ["anxiety", "anxiety disorder", "generalized anxiety", "panic disorder"],
        "depression": ["depression", "depressed", "major depression", "clinical depression"],
        "GERD": ["gerd", "acid reflux", "reflux"],
        "COPD": ["copd", "emphysema", "chronic bronchitis"],
        "obesity": ["obese", "obesity"],
        "rheumatoid_arthritis": ["rheumatoid arthritis"],
        "migraine": ["migraine", "migraines"],
        "chronic_kidney_disease": ["chronic kidney disease", "ckd", "kidney disease"],
        "anemia": ["anemia", "anaemia"],
    }
    for key, synonyms in chronic_map.items():
        for syn in synonyms:
            if syn in text:
                return key
    return text


# ── Prompt ────────────────────────────────────────────────────────

RECONSTRUCTION_PROMPT = """You are a medical AI evaluator. Your task is to infer the latent patient state from a clinician-patient dialogue.

Read the following dialogue between a clinician (doctor) and a patient. Based ONLY on what the patient says, reconstruct the patient's clinical and affective state by filling in the JSON template below.

IMPORTANT GUIDELINES:
- Infer from what the patient ACTUALLY says — do not guess or invent attributes not mentioned.
- For ordinal 1-5 scales (disease_severity, anxiety_level, pain_level, cooperation_willingness), use the patient's own language to calibrate.
- For multi-select fields (secondary_symptoms, chronic_conditions), list only what is explicitly mentioned or very clearly implied.
- If an attribute is not mentioned at all, use null.
- Do NOT use information that appears only in the clinician's questions without patient confirmation.

Dialogue:
{dialogue_text}

Reconstruct the patient state as a JSON object with this exact structure. Return ONLY valid JSON, no other text:
{{
    "primary_symptom": "string or null",
    "secondary_symptoms": ["list", "of", "strings"],
    "symptom_duration": "string or null",
    "disease_severity": integer 1-5 or null,
    "chronic_conditions": ["list", "of", "strings"],
    "anxiety_level": integer 1-5 or null,
    "pain_level": integer 1-5 or null,
    "mood": "string or null",
    "cooperation_willingness": integer 1-5 or null
}}
"""


def format_dialogue_for_eval(dialogue):
    """Convert dialogue turns to a compact text for the evaluator."""
    lines = []
    for turn in dialogue["turns"]:
        role_label = turn["role"].upper()
        lines.append(f"[{role_label}] {turn['content']}")
    return "\n\n".join(lines)


def reconstruct_state(dialogue, llm_func, verbose=False, max_retries=3):
    """Reconstruct patient state from a dialogue.

    Args:
        dialogue: Dialogue dict
        llm_func: Required. Callable that takes a prompt string and returns {text: ...}
        verbose: Print debug info
        max_retries: Number of API call attempts

    Returns:
        dict of reconstructed attributes, or None
    """
    dialogue_text = format_dialogue_for_eval(dialogue)
    prompt = RECONSTRUCTION_PROMPT.format(dialogue_text=dialogue_text)

    for attempt in range(max_retries):
        try:
            result = llm_func(prompt)
            text = result["text"].strip()
        except Exception as e:
            if verbose:
                print(f"  Attempt {attempt+1} error: {e}", file=sys.stderr)
            time.sleep(2)
            continue

        if not text:
            if verbose:
                print(f"  Attempt {attempt+1}: empty response, retrying...", file=sys.stderr)
            time.sleep(2)
            continue

        # Try parsing full response first
        try:
            state = json.loads(text)
            return state
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        m = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        if verbose:
            print(f"  Attempt {attempt+1}: unparseable, retrying...\n  Resp: {text[:200]}", file=sys.stderr)
        time.sleep(1)

    if verbose:
        print(f"  Failed after {max_retries} attempts", file=sys.stderr)
    return None


def compute_metrics(ground_truth, reconstructed):
    """Compare reconstructed state with ground truth.

    Args:
        ground_truth: The patient profile dict (full)
        reconstructed: Reconstructed state dict (flattened, eval-only attrs)

    Returns:
        dict with per-attribute scores and aggregate metrics
    """
    gt_clinical = ground_truth["clinical_attributes"]
    gt_affective = ground_truth["affective_attributes"]

    results = {}

    # --- Clinical attributes ---
    for attr in CLINICAL_TARGETS:
        gt_val = gt_clinical.get(attr)
        pred_val = reconstructed.get(attr) if reconstructed else None
        results[attr] = _score_attr(attr, gt_val, pred_val)

    # --- Affective attributes ---
    for attr in AFFECTIVE_TARGETS:
        gt_val = gt_affective.get(attr)
        pred_val = reconstructed.get(attr) if reconstructed else None
        results[attr] = _score_attr(attr, gt_val, pred_val)

    # --- Aggregate ---
    clinical_scores = [results[a]["score"] for a in CLINICAL_TARGETS]
    affective_scores = [results[a]["score"] for a in AFFECTIVE_TARGETS]

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    aggregate = {
        "clinical_faithfulness": _mean(clinical_scores),
        "affective_faithfulness": _mean(affective_scores),
        "overall_faithfulness": _mean(clinical_scores + affective_scores),
        "clinical_targets_hit": sum(1 for s in clinical_scores if s == 1.0),
        "clinical_targets_total": len(clinical_scores),
        "affective_targets_hit": sum(1 for s in affective_scores if s == 1.0),
        "affective_targets_total": len(affective_scores),
    }

    return {"per_attribute": results, "aggregate": aggregate}


def _score_attr(attr_name, gt, pred):
    """Score a single attribute. Returns dict with score, gt, pred, method."""
    # Handle null in prediction
    if pred is None:
        return {"score": 0.0, "ground_truth": gt, "predicted": None, "method": "missing"}

    # Multi-select fields (list comparison via Jaccard with normalization)
    if attr_name in MULTI_SELECT_FIELDS:
        gt_set = set(gt) if gt else set()
        pred_raw = set(pred) if pred else set()
        # Normalize predicted values
        norm_fn = _normalize_symptom if attr_name == "secondary_symptoms" else _normalize_chronic
        pred_set = set()
        for item in pred_raw:
            item_str = str(item).strip().lower()
            normed = norm_fn(item_str)
            if normed and normed != item_str.split(",")[0].strip():
                pred_set.add(normed)
            else:
                pred_set.add(item_str)
        if not gt_set and not pred_set:
            return {"score": 1.0, "ground_truth": gt, "predicted": pred, "method": "jaccard"}
        if not gt_set or not pred_set:
            return {"score": 0.0, "ground_truth": gt, "predicted": pred, "method": "jaccard"}
        intersection = gt_set & pred_set
        union = gt_set | pred_set
        score = len(intersection) / len(union)
        # Handle "none" special case
        if "none" in gt_set and "none" not in pred_set and len(gt_set) == 1:
            score = 0.0
        return {"score": round(score, 4), "ground_truth": gt, "predicted": pred, "method": "jaccard"}

    # Ordinal fields (MAE-based scoring)
    if attr_name in ORDINAL_FIELDS:
        try:
            gt_int = int(gt)
            pred_int = int(pred)
        except (ValueError, TypeError):
            return {"score": 0.0, "ground_truth": gt, "predicted": pred, "method": "ordinal_mae"}
        mae = abs(gt_int - pred_int)
        # Normalize: perfect=1.0, off-by-1=0.75, off-by-2=0.5, off-by-3=0.25, off-by-4=0.0
        score = max(0.0, 1.0 - mae * 0.25)
        return {
            "score": round(score, 4),
            "ground_truth": gt,
            "predicted": pred,
            "method": "ordinal_mae",
            "mae": mae,
        }

    # Categorical fields (fuzzy match — allow substring/containment)
    gt_str = str(gt).strip().lower() if gt else ""
    pred_str = str(pred).strip().lower() if pred else ""

    # Duration normalization
    if attr_name == "symptom_duration":
        pred_norm = _normalize_duration(pred_str)
        gt_norm = _normalize_duration(gt_str)
        score = 1.0 if pred_norm == gt_norm else 0.0
        return {"score": score, "ground_truth": gt, "predicted": pred, "method": "duration_norm"}

    # Mood normalization
    if attr_name == "mood":
        pred_norm = _normalize_mood(pred_str)
        score = 1.0 if pred_norm == gt_str else 0.0
        return {"score": score, "ground_truth": gt, "predicted": pred, "method": "mood_norm"}

    # Primary symptom — check if gt is a substring of pred or vice versa
    if attr_name == "primary_symptom":
        if gt_str in pred_str or pred_str in gt_str:
            return {"score": 1.0, "ground_truth": gt, "predicted": pred, "method": "substring"}
        # Also check symptom normalization map
        pred_norm = _normalize_symptom(pred_str)
        if pred_norm == gt_str:
            return {"score": 1.0, "ground_truth": gt, "predicted": pred, "method": "symptom_norm"}
        return {"score": 0.0, "ground_truth": gt, "predicted": pred, "method": "substring"}

    # Generic categorical (exact match)
    score = 1.0 if gt_str == pred_str else 0.0
    return {"score": score, "ground_truth": gt, "predicted": pred, "method": "exact_match"}


def evaluate_dialogue(dialogue, profile, llm_func, verbose=False):
    """Full evaluation pipeline: reconstruct + compare.

    Args:
        dialogue: Dialogue dict
        profile: Patient profile dict
        llm_func: Required. Callable for reconstruction (host.llm or similar)
        verbose: Print debug info
    """
    reconstructed = reconstruct_state(dialogue, llm_func, verbose=verbose)
    if reconstructed is None:
        return {"error": "reconstruction_failed", "aggregate": None, "reconstructed": None}
    metrics = compute_metrics(profile, reconstructed)
    return {
        "profile_id": profile["patient_id"],
        "strategy": dialogue.get("strategy", "unknown"),
        "total_turns": dialogue.get("total_turns", 0),
        "reconstructed": reconstructed,
        "per_attribute": metrics["per_attribute"],
        "aggregate": metrics["aggregate"],
    }


def evaluate_batch(dialogues_dir, profiles_path, llm_func, verbose=False):
    """Evaluate all dialogues in a directory against their profiles.

    Args:
        dialogues_dir: Path to dialogue files directory
        profiles_path: Path to patient_profiles.json
        llm_func: Required. Callable for reconstruction (host.llm or similar)
        verbose: Print progress
    """
    with open(profiles_path) as f:
        profiles_data = json.load(f)
    profiles_map = {p["patient_id"]: p for p in profiles_data["profiles"]}

    results = []
    files = sorted([f for f in os.listdir(dialogues_dir)
                    if f.endswith(".json") and not f.startswith("test")])

    for fname in files:
        with open(f"{dialogues_dir}/{fname}") as f:
            dialogue = json.load(f)
        pid = dialogue["profile_id"]
        profile = profiles_map[pid]

        if verbose:
            print(f"  Evaluating {fname}...", end=" ", flush=True)

        result = evaluate_dialogue(dialogue, profile, llm_func, verbose=False)
        result["filename"] = fname
        results.append(result)

        if verbose:
            agg = result.get("aggregate", {})
            if agg:
                print(f"clinical={agg.get('clinical_faithfulness', '?'):.3f} "
                      f"affective={agg.get('affective_faithfulness', '?'):.3f}")
            else:
                print(f"FAILED")

    # Aggregate across all evaluations
    valid = [r for r in results if r.get("aggregate")]
    if not valid:
        return {"error": "no_valid_evaluations", "results": results}

    agg = {
        "total_evaluated": len(results),
        "total_valid": len(valid),
        "mean_clinical_faithfulness": sum(r["aggregate"]["clinical_faithfulness"] for r in valid) / len(valid),
        "mean_affective_faithfulness": sum(r["aggregate"]["affective_faithfulness"] for r in valid) / len(valid),
        "mean_overall_faithfulness": sum(r["aggregate"]["overall_faithfulness"] for r in valid) / len(valid),
    }

    return {"summary": agg, "results": results}



