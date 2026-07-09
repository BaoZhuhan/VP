"""hl_confusion.py — HL-level confusion matrix from behavioral metrics.

Uses k-NN classifier on the 5 behavioral dimensions to predict HL level,
then builds a 3×3 confusion matrix. This is more principled than an LLM
evaluator because it avoids circular evaluation (LLM generating → LLM judging).
"""

import json, os, sys
import statistics
from collections import defaultdict, Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "evaluation"))
from hl_evaluator import compute_behavioral_metrics, score_vs_literature, load_dialogue

DIALOGUES = os.path.join(BASE_DIR, "data", "dialogues")
PROFILES = os.path.join(BASE_DIR, "data", "patient_profiles.json")


def euclidean(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def knn_predict(train_features, train_labels, test_point, k=5):
    """k-NN classification with majority vote."""
    distances = [(euclidean(test_point, tf), tl)
                 for tf, tl in zip(train_features, train_labels)]
    distances.sort(key=lambda x: x[0])
    neighbors = [label for _, label in distances[:k]]
    vote = Counter(neighbors).most_common(1)[0][0]
    return vote


def feature_vector(metrics):
    """Convert behavioral metrics to a fixed-length feature vector."""
    return [
        metrics.get("avg_response_length", 0) / 60,  # normalize to ~0-1
        metrics.get("question_count", 0) / 10,
        metrics.get("preference_expression_rate", 0),
        max(-1, min(1, metrics.get("decision_activeness", 0))),
        metrics.get("lexical_diversity", 0),
        metrics.get("term_density", 0) * 20,
    ]


def main():
    with open(PROFILES) as f:
        profiles_data = json.load(f)
    profiles_map = {p["patient_id"]: p for p in profiles_data["profiles"]}

    files = sorted(f for f in os.listdir(DIALOGUES)
                   if f.endswith(".json") and not f.startswith("test"))

    # Extract features and labels for all dialogues
    data_points = []
    for fname in files:
        path = os.path.join(DIALOGUES, fname)
        dialogue = load_dialogue(path)
        pid = dialogue.get("profile_id")
        profile = profiles_map.get(pid)
        if not profile:
            continue
        hl = profile["affective_attributes"]["health_literacy"]
        metrics = compute_behavioral_metrics(dialogue)
        if "error" in metrics:
            continue
        vec = feature_vector(metrics)
        data_points.append({
            "fname": fname,
            "pid": pid,
            "strategy": dialogue.get("strategy", "unknown"),
            "ground_truth": hl,
            "features": vec,
            "metrics": metrics,
        })

    # Leave-one-profile-out cross-validation
    # Group by patient ID to avoid profile-level confounding
    from collections import defaultdict
    profile_groups = defaultdict(list)
    for dp in data_points:
        profile_groups[dp["pid"]].append(dp)

    pids = list(profile_groups.keys())
    correct = 0
    total = 0
    predictions = []

    for test_pid in pids:
        test_data = profile_groups[test_pid]
        train_data = []
        for train_pid, td in profile_groups.items():
            if train_pid != test_pid:
                train_data.extend(td)

        train_features = [d["features"] for d in train_data]
        train_labels = [d["ground_truth"] for d in train_data]

        for test_dp in test_data:
            pred = knn_predict(train_features, train_labels, test_dp["features"], k=5)
            predictions.append({
                "fname": test_dp["fname"],
                "pid": test_dp["pid"],
                "strategy": test_dp["strategy"],
                "ground_truth": test_dp["ground_truth"],
                "predicted": pred,
            })
            if pred == test_dp["ground_truth"]:
                correct += 1
            total += 1

    # Confusion matrix
    classes = ["low", "medium", "high"]
    matrix = {gt: {p: 0 for p in classes} for gt in classes}
    for p in predictions:
        matrix[p["ground_truth"]][p["predicted"]] += 1

    print("="*60)
    print("CONFUSION MATRIX (k-NN, leave-one-profile-out)")
    print("="*60)
    print(f"{'GT \\ Pred':<12} {'low':<10} {'medium':<10} {'high':<10} {'total':<10}")
    for gt in classes:
        row = matrix[gt]
        row_total = sum(row.values())
        pcts = [f"{row[cl]/row_total*100:.0f}%" if row_total else "-" for cl in classes]
        print(f"{gt:<12} {row['low']:<10} {row['medium']:<10} {row['high']:<10} {row_total:<10}")
        print(f"{'':12} {pcts[0]:<10} {pcts[1]:<10} {pcts[2]:<10}")

    acc = correct / total if total > 0 else 0
    print(f"\nOverall accuracy: {acc:.3f} ({correct}/{total})")

    # By strategy
    print("\n" + "="*60)
    print("ACCURACY BY STRATEGY")
    print("="*60)
    strat_results = defaultdict(list)
    for p in predictions:
        strat_results[p["strategy"]].append(p["predicted"] == p["ground_truth"])

    for strat in sorted(strat_results.keys()):
        cl = strat_results[strat]
        n = len(cl)
        a = sum(cl) / n if n > 0 else 0
        print(f"  {strat:<25} {a:.3f} ({sum(cl)}/{n})")

    # Save
    output_path = os.path.join(BASE_DIR, "data", "results", "hl_confusion_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "matrix": matrix,
            "predictions": predictions,
        }, f, indent=2)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
