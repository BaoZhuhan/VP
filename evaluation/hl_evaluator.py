"""hl_evaluator.py — Health literacy behavioral faithfulness evaluator.

Measures whether LLM-generated patient dialogues exhibit communication
behaviors consistent with the assigned health literacy level, grounded
in clinical communication literature (Aboumatar 2013, Menendez 2017, etc.).
"""

import json, os, re, sys
from collections import Counter

# ── Medical term list (curated from clinical literature) ─────────────

MEDICAL_TERMS = {
    "hypertension", "diabetes", "asthma", "COPD", "GERD", "myocardial",
    "infarction", "coronary", "arrhythmia", "tachycardia", "bradycardia",
    "edema", "hematoma", "aneurysm", "stenosis", "thrombosis", "embolism",
    "ischemia", "necrosis", "biopsy", "prognosis", "diagnosis", "pathology",
    "chronic", "acute", "benign", "malignant", "metastasis", "carcinoma",
    "sarcoma", "lymphoma", "anemia", "leukemia", "antibiotic", "antiviral",
    "antihypertensive", "anticoagulant", "statin", "diuretic", "beta-blocker",
    "ACE inhibitor", "proton pump inhibitor", "NSAID", "corticosteroid",
    "bronchodilator", "antihistamine", "insulin", "metformin", "levothyroxine",
    "symptomatology", "contraindication", "comorbidity", "etiology",
    "idiopathic", "iatrogenic", "palliative", "adjuvant", "prophylactic",
    "auscultation", "palpation", "percussion", "radiograph", "ultrasound",
    "CT scan", "MRI", "echocardiogram", "electrocardiogram", "endoscopy",
    "colonoscopy", "bronchoscopy", "laparoscopy", "arthroscopy",
    "hemoglobin", "hematocrit", "creatinine", "potassium", "sodium",
    "cholesterol", "triglyceride", "glucose", "HbA1c", "sedimentation",
    "inflammation", "infection", "sepsis", "pneumonia", "bronchitis",
    "gastroenteritis", "diverticulitis", "pancreatitis", "cholecystitis",
    "appendicitis", "peritonitis", "cellulitis", "abscess", "ulceration",
    "obstruction", "perforation", "hemorrhage", "ischemic", "neuropathic",
    "arthritic", "osteoporotic", "degenerative", "autoimmune",
    "exacerbation", "remission", "relapse", "progressive", "benign",
    "neoplasm", "nodule", "polyp", "cyst", "lesion", "mass", "tumor",
    "spirometry", "pulse oximetry", "capillary refill", "mucous membrane",
    "scleral icterus", "jugular venous distension", "lymphadenopathy",
    "hepatomegaly", "splenomegaly", "ascites", "cachexia", "dehydration",
}

# ── Clinician comprehension-check probes ─────────────────────────────

COMPREHENSION_PROBE_KEYWORDS = [
    "does that make sense",
    "do you understand",
    "do you have any questions",
    "does that sound",
    "are you following",
    "do you follow",
    "any questions",
    "does that clarify",
]


def load_dialogue(path):
    with open(path) as f:
        return json.load(f)


def load_profile(profiles_path, pid):
    with open(profiles_path) as f:
        data = json.load(f)
    for p in data["profiles"]:
        if p["patient_id"] == pid:
            return p
    return None


def _is_clinician_turn(turn):
    return turn.get("role", "").lower() in ("clinician", "doctor", "physician")


def _is_patient_turn(turn):
    return turn.get("role", "").lower() in ("patient",)


def _count_questions(text):
    """Count interrogative sentences in patient utterance."""
    count = 0
    for sentence in re.split(r'[.!?\n]', text):
        sentence = sentence.strip()
        if not sentence:
            continue
        # Check if it ends with ? or starts with question words
        if sentence.endswith("?"):
            count += 1
        elif re.match(
            r"^(what|why|how|when|where|who|which|is|are|can|could|will|would|do|does|did|should|may|might|have|has|am|was|were)\b",
            sentence,
            re.IGNORECASE,
        ):
            count += 1
    return count


def _count_medical_terms(text):
    """Count occurrences of medical terms in text."""
    text_lower = text.lower()
    count = 0
    for term in MEDICAL_TERMS:
        if term in text_lower:
            count += 1
    return count


def _compute_lexical_diversity(tokens):
    """Type-token ratio."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def _is_comprehension_probe(text):
    """Check if clinician turn is a comprehension check."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in COMPREHENSION_PROBE_KEYWORDS)


def _is_unqualified_affirmation(text):
    """Detect 'pretend understanding' — simple yes/okay without elaboration."""
    text = text.strip().lower()
    if not text:
        return False
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Check for bare affirmations or very short agreements
    bare_affirmations = {"yes", "yeah", "yep", "uh huh", "okay", "ok",
                         "sure", "alright", "right", "mmhmm"}
    if text in bare_affirmations:
        return True
    # Check patterns like "yes, doctor" or "okay, thank you"
    words = text.split()
    if len(words) >= 1 and words[0] in bare_affirmations:
        if len(words) <= 3:
            return True
    return False


def extract_dialogue_turns(dialogue):
    """Separate clinician and patient turns."""
    clinician_turns = []
    patient_turns = []
    for turn in dialogue.get("turns", []):
        if _is_clinician_turn(turn):
            clinician_turns.append(turn)
        elif _is_patient_turn(turn):
            patient_turns.append(turn)
    return clinician_turns, patient_turns


def compute_turn_trajectory(dialogue, window_size=3):
    """Compute per-window trajectory of key metrics for a dialogue.

    Slides a window across patient turns and computes avg_response_length
    and decision_activeness for each window.

    Returns:
        list of dicts: [{window_center, avg_response_length, decision_activeness}, ...]
    """
    _, pat_turns = extract_dialogue_turns(dialogue)
    if len(pat_turns) < window_size:
        return []

    # Pre-compute per-turn metrics
    turn_lengths = [len(t["content"].split()) for t in pat_turns]
    passive_patterns = re.compile(
        r"(whatever you (think|say|recommend)|you're the doctor|you know best"
        r"|i don't know, what do you think|i'll trust you|just tell me what to do)",
        re.IGNORECASE,
    )
    active_patterns = re.compile(
        r"(i'd like to (know|understand|try)|can you explain|tell me more about"
        r"|what are my options|what would happen if|what do you recommend)",
        re.IGNORECASE,
    )
    turn_passive = [1 if passive_patterns.search(t["content"]) else 0 for t in pat_turns]
    turn_active = [1 if active_patterns.search(t["content"]) else 0 for t in pat_turns]

    trajectory = []
    for start in range(len(pat_turns) - window_size + 1):
        end = start + window_size
        window_center = start + window_size // 2
        avg_len = sum(turn_lengths[start:end]) / window_size
        total_pa = sum(turn_passive[start:end]) + sum(turn_active[start:end])
        decision_act = (
            (sum(turn_active[start:end]) - sum(turn_passive[start:end])) / total_pa
            if total_pa > 0 else 0.0
        )
        trajectory.append({
            "window_center": window_center,
            "turn_start": start,
            "turn_end": end - 1,
            "avg_response_length": round(avg_len, 2),
            "decision_activeness": round(decision_act, 4),
        })
    return trajectory


def compute_behavioral_metrics(dialogue):
    """Compute all 5 behavioral dimensions from a dialogue.

    Returns:
        dict with keys: question_count, avg_response_length, lexical_diversity,
                        term_density, preference_expression_rate,
                        unqualified_affirmation_rate, decision_activeness,
                        total_clinician_turns, total_patient_turns
    """
    clin_turns, pat_turns = extract_dialogue_turns(dialogue)
    if not pat_turns:
        return {"error": "no_patient_turns"}

    # 1. Question-asking
    total_questions = 0
    for t in pat_turns:
        total_questions += _count_questions(t["content"])

    # 2. Response complexity
    response_lengths = []
    all_tokens = []
    for t in pat_turns:
        words = t["content"].split()
        response_lengths.append(len(words))
        all_tokens.extend(words)

    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0
    lex_diversity = _compute_lexical_diversity(all_tokens) if all_tokens else 0

    # 3. Term density
    total_med_terms = sum(_count_medical_terms(t["content"]) for t in pat_turns)
    total_patient_words = sum(len(t["content"].split()) for t in pat_turns)
    term_density = total_med_terms / total_patient_words if total_patient_words > 0 else 0

    # 4. Preference expression (patient states what they want/think)
    preference_patterns = re.compile(
        r"(i (think|feel|believe|want|would like|prefer|hope|wish)"
        r"|in my opinion"
        r"|if (i was|it was up to me)"
        r"|what do you recommend|what would you suggest"
        r"|i'd rather|i'd prefer)",
        re.IGNORECASE,
    )
    preference_turns = sum(
        1 for t in pat_turns if preference_patterns.search(t["content"])
    )
    preference_expression_rate = preference_turns / len(pat_turns) if pat_turns else 0

    # 5. Comprehension signaling (pretend understanding)
    probe_turns = [t for t in clin_turns if _is_comprehension_probe(t["content"])]

    # Find patient responses immediately following comprehension probes
    probe_patient_responses = []
    for ct in clin_turns:
        if _is_comprehension_probe(ct["content"]):
            ct_tid = ct.get("turn_id", 0)
            for pt in pat_turns:
                if pt.get("turn_id", 0) > ct_tid:
                    probe_patient_responses.append(pt["content"])
                    break

    unqualified_yes_count = sum(
        1 for resp in probe_patient_responses if _is_unqualified_affirmation(resp)
    )
    unqualified_affirmation_rate = (
        unqualified_yes_count / len(probe_patient_responses)
        if probe_patient_responses
        else None  # No probes in this dialogue
    )

    # 6. Decision-making role (passive → active)
    passive_patterns = re.compile(
        r"(whatever you (think|say|recommend)"
        r"|you're the doctor|you know best"
        r"|i don't know, what do you think"
        r"|i'll trust you|i'll follow your"
        r"|just tell me what to do"
        r"|i'm not sure, you decide)",
        re.IGNORECASE,
    )
    active_patterns = re.compile(
        r"(i'd like to (know|understand|try)"
        r"|can you explain|tell me more about"
        r"|what are my options"
        r"|what would happen if"
        r"|is there anything else"
        r"|i've been thinking about"
        r"|i read that|i heard that"
        r"|what do you recommend"
        r"|what's your opinion)",
        re.IGNORECASE,
    )

    passive_count = sum(1 for t in pat_turns if passive_patterns.search(t["content"]))
    active_count = sum(1 for t in pat_turns if active_patterns.search(t["content"]))

    # Decision activeness score: -1 (fully passive) to +1 (fully active)
    total_pa = passive_count + active_count
    if total_pa > 0:
        decision_activeness = (active_count - passive_count) / total_pa
    else:
        decision_activeness = 0.0

    return {
        "question_count": total_questions,
        "avg_response_length": round(avg_response_length, 2),
        "lexical_diversity": round(lex_diversity, 4),
        "term_density": round(term_density, 4),
        "total_medical_terms": total_med_terms,
        "preference_expression_rate": round(preference_expression_rate, 4),
        "preference_turns": preference_turns,
        "unqualified_affirmation_rate": unqualified_affirmation_rate,
        "unqualified_yes_count": unqualified_yes_count,
        "probe_count": len(probe_patient_responses),
        "decision_activeness": round(decision_activeness, 4),
        "passive_count": passive_count,
        "active_count": active_count,
        "total_patient_turns": len(pat_turns),
        "total_clinician_turns": len(clin_turns),
    }


def score_vs_literature(metrics, hl_level):
    """Score how well behavioral metrics match literature expectations.

    Args:
        metrics: dict from compute_behavioral_metrics()
        hl_level: 'low', 'medium', or 'high'

    Returns:
        dict with per-dimension score (0-1) and composite score
    """
    # Literature-derived expected ranges
    # Sources: Aboumatar 2013, Menendez 2017, PLOS ONE 2022
    expected = {
        "low": {
            "question_count": (4, 5),         # 4.46 per visit, ~5 per visit
            "avg_response_length": (8, 20),   # short responses
            "lexical_diversity": (0.4, 0.6),  # limited vocabulary
            "term_density": (0.0, 0.02),      # near-zero medical jargon
            "preference_expression_rate": (0.0, 0.2),
            "unqualified_affirmation_rate": (0.5, 1.0),  # high pretend understanding
            "decision_activeness": (-1.0, -0.3),  # passive
        },
        "medium": {
            "question_count": (5, 7),
            "avg_response_length": (15, 35),
            "lexical_diversity": (0.55, 0.75),
            "term_density": (0.01, 0.05),
            "preference_expression_rate": (0.15, 0.4),
            "unqualified_affirmation_rate": (0.2, 0.6),
            "decision_activeness": (-0.3, 0.3),
        },
        "high": {
            "question_count": (7, 12),        # 6.82-9 per visit
            "avg_response_length": (30, 60),
            "lexical_diversity": (0.65, 0.85),
            "term_density": (0.03, 0.10),
            "preference_expression_rate": (0.3, 0.7),
            "unqualified_affirmation_rate": (0.0, 0.3),
            "decision_activeness": (0.2, 1.0),
        },
    }

    if hl_level not in expected:
        return {"error": f"unknown hl_level: {hl_level}"}

    ranges = expected[hl_level]
    scores = {}
    dimension_count = 0
    total_score = 0.0

    for dim, (lo, hi) in ranges.items():
        if dim not in metrics or metrics[dim] is None:
            continue
        val = metrics[dim]
        if dim == "term_density":
            val = val  # already a proportion
        elif dim == "avg_response_length":
            pass  # raw word count
        elif dim == "question_count":
            pass  # raw count

        # Score: 1.0 if within range, linear decay outside
        if lo <= val <= hi:
            score = 1.0
        elif val < lo:
            # Distance below lower bound, normalized by range width
            score = max(0.0, 1.0 - (lo - val) / max(lo, 1))
        else:  # val > hi
            score = max(0.0, 1.0 - (val - hi) / max(hi, 1))

        scores[dim] = round(score, 4)
        dimension_count += 1
        total_score += score

    composite = round(total_score / max(dimension_count, 1), 4)
    return {"per_dimension": scores, "composite": composite}


def evaluate_dialogue(dialogue_path, profile, verbose=False):
    """Full evaluation pipeline for one dialogue."""
    dialogue = load_dialogue(dialogue_path)
    metrics = compute_behavioral_metrics(dialogue)
    if "error" in metrics:
        return {"error": metrics["error"], "metrics": metrics}

    hl_level = profile["affective_attributes"]["health_literacy"]
    scoring = score_vs_literature(metrics, hl_level)

    result = {
        "profile_id": dialogue.get("profile_id"),
        "strategy": dialogue.get("strategy", "unknown"),
        "total_turns": dialogue.get("total_turns", len(dialogue.get("turns", []))),
        "assigned_hl": hl_level,
        "metrics": metrics,
        "scoring": scoring,
    }
    return result


def evaluate_batch(dialogues_dir, profiles_path, verbose=False):
    """Evaluate all dialogues in a directory."""
    with open(profiles_path) as f:
        profiles_data = json.load(f)
    profiles_map = {p["patient_id"]: p for p in profiles_data["profiles"]}

    results = []
    files = sorted(
        f for f in os.listdir(dialogues_dir)
        if f.endswith(".json") and not f.startswith("test")
    )

    for fname in files:
        path = os.path.join(dialogues_dir, fname)
        dialogue = load_dialogue(path)
        pid = dialogue.get("profile_id")
        profile = profiles_map.get(pid)
        if not profile:
            if verbose:
                print(f"  Skipping {fname}: no profile for {pid}", file=sys.stderr)
            continue

        if verbose:
            print(f"  Evaluating {fname}...", end=" ", flush=True)

        result = evaluate_dialogue(path, profile, verbose=False)
        result["filename"] = fname
        results.append(result)

        if verbose and "error" not in result:
            c = result["scoring"]["composite"]
            print(f"HL score: {c:.3f}")
        elif verbose:
            print(f"FAILED")

    # Aggregate
    valid = [r for r in results if "error" not in r]
    if not valid:
        return {"error": "no_valid_evaluations", "results": results}

    # Aggregate by strategy and HL level
    from collections import defaultdict
    by_strategy = defaultdict(list)
    by_hl = defaultdict(list)

    for r in valid:
        by_strategy[r["strategy"]].append(r["scoring"]["composite"])
        by_hl[r["assigned_hl"]].append(r["scoring"]["composite"])

    agg = {
        "total_evaluated": len(results),
        "total_valid": len(valid),
        "mean_composite": round(
            sum(r["scoring"]["composite"] for r in valid) / len(valid), 4
        ),
        "by_strategy": {
            s: {"mean": round(sum(v)/len(v), 4), "n": len(v), "scores": v}
            for s, v in sorted(by_strategy.items())
        },
        "by_hl_level": {
            hl: {"mean": round(sum(v)/len(v), 4), "n": len(v), "scores": v}
            for hl, v in sorted(by_hl.items())
        },
    }

    return {"summary": agg, "results": results}


if __name__ == "__main__":
    # Quick test
    import sys
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dialogues_dir = os.path.join(base, "data", "dialogues")
    profiles_path = os.path.join(base, "data", "patient_profiles.json")

    result = evaluate_batch(dialogues_dir, profiles_path, verbose=True)
    summary = result.get("summary", {})
    print("\n=== AGGREGATE SUMMARY ===")
    print(f"Total valid: {summary.get('total_valid', 0)}")
    print(f"Mean composite: {summary.get('mean_composite', 'N/A')}")
    print("\nBy Strategy:")
    for s, v in summary.get("by_strategy", {}).items():
        print(f"  {s}: {v['mean']:.4f} (n={v['n']})")
    print("\nBy HL Level:")
    for hl, v in summary.get("by_hl_level", {}).items():
        print(f"  {hl}: {v['mean']:.4f} (n={v['n']})")
