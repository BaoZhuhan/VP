"""Generate HL baseline figures — per-dimension heatmap, radar, trajectory, interaction."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import json, os, sys, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
from hl_evaluator import (
    compute_behavioral_metrics, compute_turn_trajectory,
    score_vs_literature, load_dialogue
)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 9,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "figure.dpi": 150,
})

BASE = "/Users/zhuhan/Documents/Research/VP/VP/figures"
DIALOGUES = "/Users/zhuhan/Documents/Research/VP/VP/data/dialogues"
PROFILES = "/Users/zhuhan/Documents/Research/VP/VP/data/patient_profiles.json"


def load_all_results():
    """Load profiles and compute per-dimension scores for all dialogues."""
    with open(PROFILES) as f:
        profiles_data = json.load(f)
    profiles_map = {p["patient_id"]: p for p in profiles_data["profiles"]}

    files = sorted(f for f in os.listdir(DIALOGUES)
                   if f.endswith(".json") and not f.startswith("test"))
    results = []
    for fname in files:
        path = os.path.join(DIALOGUES, fname)
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
        trajectory = compute_turn_trajectory(dialogue, window_size=3)

        row = {
            "fname": fname, "pid": pid, "hl": hl, "style": style,
            "strategy": strategy, "metrics": metrics, "scoring": scoring,
            "trajectory": trajectory,
        }
        dims = scoring.get("per_dimension", {})
        for k, v in dims.items():
            row[f"dim_{k}"] = v
        results.append(row)
    return results


def _stratify(results):
    """Group results by strategy name."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r["strategy"]].append(r)
    return groups


# ── A1: Heatmap ──────────────────────────────────────────────────

STRATEGY_LABELS = {
    "standard": "Standard", "consistency": "Consistency",
    "structured_state": "Structured State", "literacy_anchored": "Literacy-Anchored",
    "patientsim_c4": "PatientSim C4", "patientsim_c5": "PatientSim C5",
}
DIM_LABELS = {
    "question_count": "Question\nCount", "avg_response_length": "Response\nLength",
    "preference_expression_rate": "Preference\nRate",
    "unqualified_affirmation_rate": "Prevent\nUnderstanding",
    "decision_activeness": "Decision\nActiveness",
}
DIM_ORDER = ["response_length", "decision_activeness", "preference_expression_rate",
             "question_count", "unqualified_affirmation_rate"]
DIM_KEYS = [f"dim_{d}" for d in DIM_ORDER]


def fig_heatmap(results):
    strat_groups = _stratify(results)
    strategies_ordered = ["standard", "consistency", "structured_state",
                          "patientsim_c4", "patientsim_c5", "literacy_anchored"]

    data = np.zeros((len(strategies_ordered), len(DIM_KEYS)))
    for i, s in enumerate(strategies_ordered):
        for j, dk in enumerate(DIM_KEYS):
            vals = [r[dk] for r in strat_groups.get(s, []) if dk in r]
            data[i, j] = np.mean(vals) if vals else 0.0

    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(DIM_KEYS)))
    ax.set_xticklabels([DIM_LABELS.get(d.replace("dim_", ""), d.replace("dim_",""))
                        for d in DIM_KEYS], fontsize=8)
    ax.set_yticks(range(len(strategies_ordered)))
    ax.set_yticklabels([STRATEGY_LABELS.get(s, s) for s in strategies_ordered],
                       fontsize=8)

    for i in range(len(strategies_ordered)):
        for j in range(len(DIM_KEYS)):
            val = data[i, j]
            color = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Per-Dimension Score")
    ax.set_title("Health Literacy Faithfulness: Per-Dimension Scores", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{BASE}/hl_dimension_heatmap.png", dpi=200, bbox_inches="tight")
    print("  Saved hl_dimension_heatmap.png")
    plt.close(fig)


# ── A1: Radar ─────────────────────────────────────────────────────

def fig_radar(results):
    strat_groups = _stratify(results)
    strategies_ordered = ["standard", "consistency", "structured_state",
                          "patientsim_c4", "patientsim_c5", "literacy_anchored"]
    colors = ["#4C72B0", "#55A868", "#CC974B", "#8172B2", "#C44E52", "#DD8452"]

    labels = ["Response\nLength", "Decision\nActiveness", "Preference\nRate",
              "Question\nCount", "Prev.\nUnderstanding"]
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))

    for idx, strat in enumerate(strategies_ordered):
        vals_list = strat_groups.get(strat, [])
        values = [np.mean([r[DIM_KEYS[j]] for r in vals_list if DIM_KEYS[j] in r])
                  for j in range(num_vars)] if vals_list else [0]*num_vars
        values += values[:1]
        ax.plot(angles, values, color=colors[idx], linewidth=1.2, label=STRATEGY_LABELS.get(strat, strat))
        ax.fill(angles, values, color=colors[idx], alpha=0.05)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=7)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=7)
    ax.set_title("Per-Dimension Profile by Strategy", fontsize=10, pad=20)

    fig.tight_layout()
    fig.savefig(f"{BASE}/hl_radar.png", dpi=200, bbox_inches="tight")
    print("  Saved hl_radar.png")
    plt.close(fig)


# ── A2: HL × Communication Style ─────────────────────────────────

def fig_hl_style_interaction(results):
    """Bar chart: response length and decision activeness by HL × style."""
    strat_groups = _stratify(results)

    # Collapse styles into 2 groups
    def style_group(style):
        if style == "detailed":
            return "Detailed"
        return "Terse/Stoic"

    hl_order = ["low", "medium", "high"]
    hl_labels = ["Low HL", "Medium HL", "High HL"]
    style_groups = ["Detailed", "Terse/Stoic"]

    # Use strategy="standard" only (to control for strategy confound)
    standard = strat_groups.get("standard", [])

    dims = [
        ("dim_avg_response_length", "Response Length", (0, 65), "%.0f"),
        ("dim_decision_activeness", "Decision Activeness", (-0.5, 1.0), "%.2f"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    x = np.arange(len(hl_order))
    width = 0.3

    for col, (dim_key, dim_label, ylim, fmt) in enumerate(dims):
        ax = axes[col]
        for si, sg in enumerate(style_groups):
            offset = (si - 0.5) * width
            vals = []
            for hl in hl_order:
                subset = [r[dim_key] for r in standard
                          if r["hl"] == hl and style_group(r["style"]) == sg
                          and dim_key in r]
                vals.append(np.mean(subset) if subset else 0)
            bars = ax.bar(x + offset, vals, width,
                          label=sg, color=["#4C72B0", "#C44E52"][si],
                          edgecolor="white", linewidth=0.5)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + ylim[1]*0.01,
                        fmt % val, ha="center", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(hl_labels, fontsize=8)
        ax.set_ylabel(dim_label, fontsize=9)
        ax.set_ylim(ylim)
        ax.legend(fontsize=7)

    fig.suptitle("HL × Communication Style (Standard Prompting)", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{BASE}/hl_style_interaction.png", dpi=200, bbox_inches="tight")
    print("  Saved hl_style_interaction.png")
    plt.close(fig)


# ── A3: Trajectory ────────────────────────────────────────────────

def fig_trajectory(results):
    """Line plot: per-turn trajectory of response length, faceted by HL."""
    strat_groups = _stratify(results)
    strategies_plot = ["standard", "literacy_anchored", "patientsim_c4"]
    colors_plot = ["#4C72B0", "#C44E52", "#8172B2"]
    hl_order = ["low", "medium", "high"]
    hl_labels = ["Low HL", "Medium HL", "High HL"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)

    for hi, hl in enumerate(hl_order):
        ax = axes[hi]
        for si, strat in enumerate(strategies_plot):
            # Collect all trajectories for this strategy × HL level
            dialogues = [r for r in strat_groups.get(strat, []) if r["hl"] == hl]
            if not dialogues:
                continue

            # Aggregate trajectory: align by window center
            traj_dict = {}
            for d in dialogues:
                for pt in d.get("trajectory", []):
                    wc = pt["window_center"]
                    if wc not in traj_dict:
                        traj_dict[wc] = []
                    traj_dict[wc].append(pt["avg_response_length"])

            if not traj_dict:
                continue

            windows = sorted(traj_dict.keys())
            means = [np.mean(traj_dict[w]) for w in windows]
            sems = [np.std(traj_dict[w]) / math.sqrt(len(traj_dict[w]))
                    for w in windows]

            ax.errorbar(windows, means, yerr=sems,
                        color=colors_plot[si], label=STRATEGY_LABELS.get(strat, strat),
                        linewidth=1, capsize=2, marker="o", markersize=3)

        ax.set_title(hl_labels[hi], fontsize=9)
        ax.set_xlabel("Turn Window", fontsize=8)
        if hi == 0:
            ax.set_ylabel("Response Length (words)", fontsize=9)
        ax.legend(fontsize=6)

    fig.suptitle("Response Length Trajectory by HL Level", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{BASE}/hl_trajectory.png", dpi=200, bbox_inches="tight")
    print("  Saved hl_trajectory.png")
    plt.close(fig)


# ── Run ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(BASE, exist_ok=True)
    print("Loading data and computing metrics for 162 dialogues...")
    all_results = load_all_results()
    print(f"  Loaded {len(all_results)} dialogues")

    print("Generating figures...")
    fig_heatmap(all_results)
    fig_radar(all_results)
    fig_hl_style_interaction(all_results)
    fig_trajectory(all_results)

    # Also regenerate existing figures
    from importlib import reload
    # Just run the heatmap/radar above — existing figures are still at their old paths
    print("Done! All figures saved to figures/")
