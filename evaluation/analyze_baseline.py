"""Analyze HL baseline results in detail."""
import json, os, sys
from collections import defaultdict
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hl_evaluator import (
    compute_behavioral_metrics, score_vs_literature,
    load_dialogue, load_profile,
)

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dialogues_dir = os.path.join(base, "data", "dialogues")
profiles_path = os.path.join(base, "data", "patient_profiles.json")

with open(profiles_path) as f:
    profiles_data = json.load(f)
profiles_map = {p["patient_id"]: p for p in profiles_data["profiles"]}

files = sorted(f for f in os.listdir(dialogues_dir)
               if f.endswith(".json") and not f.startswith("test"))

# Collect detailed results
rows = []
for fname in files:
    path = os.path.join(dialogues_dir, fname)
    dialogue = load_dialogue(path)
    pid = dialogue.get("profile_id")
    profile = profiles_map.get(pid)
    if not profile:
        continue
    hl = profile["affective_attributes"]["health_literacy"]
    style = profile["affective_attributes"]["communication_style"]
    strategy = dialogue.get("strategy", "unknown")
    metrics = compute_behavioral_metrics(dialogue)
    scoring = score_vs_literature(metrics, hl)

    rows.append({
        "fname": fname,
        "pid": pid,
        "hl": hl,
        "style": style,
        "strategy": strategy,
        "composite": scoring.get("composite", 0),
        **{k: metrics.get(k) for k in [
            "question_count", "avg_response_length", "lexical_diversity",
            "term_density", "preference_expression_rate",
            "unqualified_affirmation_rate", "decision_activeness",
            "total_patient_turns",
        ]},
        **{f"dim_{k}": v for k, v in scoring.get("per_dimension", {}).items()},
    })

# ── By HL level × strategy ──
print("="*80)
print("MEAN RAW METRICS BY HL LEVEL (across all strategies)")
print("="*80)

hl_groups = defaultdict(list)
for r in rows:
    hl_groups[r["hl"]].append(r)

metric_keys = [
    "question_count", "avg_response_length", "lexical_diversity",
    "term_density", "preference_expression_rate",
    "unqualified_affirmation_rate", "decision_activeness",
]

header = f"{'Metric':<35} {'Low(8)':<10} {'Med(9)':<10} {'High(10)':<10} {'LvsH p':<10} {'d':<8} {'Low_CI':<16} {'High_CI':<16}"
print(header)
print("-"*105)

def cohens_d(x, y):
    """Cohen's d for two independent groups."""
    from math import sqrt
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1 = statistics.stdev(x)
    s2 = statistics.stdev(y)
    sp = sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if sp == 0:
        return float("nan")
    return (statistics.mean(x) - statistics.mean(y)) / sp


def bootstrap_ci(values, n_iterations=10000):
    """95% CI via bootstrapping (percentile method)."""
    import random
    if len(values) < 2:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n_iterations):
        sample = [random.choice(values) for _ in range(len(values))]
        means.append(statistics.mean(sample))
    means.sort()
    return (means[int(0.025 * n_iterations)], means[int(0.975 * n_iterations)])


def mannwhitney_u(x, y):
    """Simple Mann-Whitney U test (no scipy dependency)."""
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return 1.0
    ranked = sorted([(v, 0) for v in x] + [(v, 1) for v in y])
    rank_sum = sum(i+1 for i, (_, g) in enumerate(ranked) if g == 0)
    u1 = rank_sum - n1*(n1+1)/2
    u2 = n1*n2 - u1
    from math import sqrt, erf
    mu = n1*n2/2
    sigma = sqrt(n1*n2*(n1+n2+1)/12)
    if sigma == 0:
        return 1.0
    z = (min(u1, u2) - mu) / sigma
    # two-tailed p from normal approximation
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return p

for mk in metric_keys:
    low_vals = [r[mk] for r in hl_groups["low"] if r[mk] is not None]
    med_vals = [r[mk] for r in hl_groups["medium"] if r[mk] is not None]
    high_vals = [r[mk] for r in hl_groups["high"] if r[mk] is not None]
    low_m = statistics.mean(low_vals) if low_vals else 0
    med_m = statistics.mean(med_vals) if med_vals else 0
    high_m = statistics.mean(high_vals) if high_vals else 0
    p = mannwhitney_u(low_vals, high_vals)
    d = cohens_d(low_vals, high_vals)
    ci_low = bootstrap_ci(low_vals)
    ci_high = bootstrap_ci(high_vals)
    print(f"{mk:<35} {low_m:<12.3f} {med_m:<12.3f} {high_m:<12.3f} {p:<10.4f} {d:<8.3f} {ci_low[0]:<6.3f}-{ci_low[1]:<6.3f} {ci_high[0]:<6.3f}-{ci_high[1]:<6.3f}")

# ── Composite score breakdown ──
print()
print("="*80)
print("COMPOSITE SCORE BY HL × STRATEGY")
print("="*80)
print(f"{'Strategy':<20} {'Low':<10} {'Medium':<10} {'High':<10} {'Overall':<10}")
print("-"*60)

strat_hl = defaultdict(lambda: defaultdict(list))
for r in rows:
    strat_hl[r["strategy"]][r["hl"]].append(r["composite"])

for strat in ["standard", "consistency", "structured_state", "literacy_anchored"]:
    low_m = statistics.mean(strat_hl[strat].get("low", [0])) if strat_hl[strat].get("low") else 0
    med_m = statistics.mean(strat_hl[strat].get("medium", [0])) if strat_hl[strat].get("medium") else 0
    high_m = statistics.mean(strat_hl[strat].get("high", [0])) if strat_hl[strat].get("high") else 0
    all_v = strat_hl[strat].get("low", []) + strat_hl[strat].get("medium", []) + strat_hl[strat].get("high", [])
    overall = statistics.mean(all_v) if all_v else 0
    print(f"{strat:<20} {low_m:<10.4f} {med_m:<10.4f} {high_m:<10.4f} {overall:<10.4f}")

# ── Per-dimension scoring breakdown ──
print()
print("="*80)
print("PER-DIMENSION SCORE BY HL LEVEL")
print("="*80)
dims = ["question_count", "avg_response_length", "lexical_diversity",
        "term_density", "preference_expression_rate",
        "unqualified_affirmation_rate", "decision_activeness"]
print(f"{'Dimension':<35} {'Low':<10} {'Medium':<10} {'High':<10}")
print("-"*65)
for dim in dims:
    dim_key = f"dim_{dim}"
    low_v = [r[dim_key] for r in hl_groups["low"] if dim_key in r]
    med_v = [r[dim_key] for r in hl_groups["medium"] if dim_key in r]
    high_v = [r[dim_key] for r in hl_groups["high"] if dim_key in r]
    low_m = statistics.mean(low_v) if low_v else 0
    med_m = statistics.mean(med_v) if med_v else 0
    high_m = statistics.mean(high_v) if high_v else 0
    print(f"{dim:<35} {low_m:<10.4f} {med_m:<10.4f} {high_m:<10.4f}")

# ── Individual dialogue list (key outliers) ──
print()
print("="*80)
print("BOTTOM 10 DIALOGUES BY COMPOSITE SCORE")
print("="*80)
sorted_rows = sorted(rows, key=lambda r: r["composite"])
for r in sorted_rows[:10]:
    print(f"  {r['fname']:<40} HL={r['hl']:<8} strategy={r['strategy']:<18} composite={r['composite']:.4f}")

print()
print("TOP 10 DIALOGUES BY COMPOSITE SCORE")
print("="*80)
for r in sorted_rows[-10:]:
    print(f"  {r['fname']:<40} HL={r['hl']:<8} strategy={r['strategy']:<18} composite={r['composite']:.4f}")

# ── RAW METRICS BY STRATEGY × HL LEVEL ──
print()
print("="*80)
print("RAW METRICS BY STRATEGY × HL LEVEL (literacy_anchored vs standard)")
print("="*80)
for mk in metric_keys:
    print(f"\n--- {mk} ---")
    print(f"{'HL':<8} {'Standard':<12} {'LitAnchor':<12} {'StdErr':<12}")
    print("-"*44)
    for hl in ["low", "medium", "high"]:
        std = [r[mk] for r in rows if r["strategy"]=="standard" and r["hl"]==hl and r[mk] is not None]
        la = [r[mk] for r in rows if r["strategy"]=="literacy_anchored" and r["hl"]==hl and r[mk] is not None]
        std_m = statistics.mean(std) if std else 0
        la_m = statistics.mean(la) if la else 0
        p = mannwhitney_u(std, la)
        p_str = f"{p:.4f}"
        print(f"{hl:<8} {std_m:<12.3f} {la_m:<12.3f} {p_str:<12}")
