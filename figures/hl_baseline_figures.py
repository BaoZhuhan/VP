"""Generate HL baseline figures for the experimental report."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

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

# ── Data ─────────────────────────────────────────────────────────────

strategies = ["Standard", "Consistency", "Structured\nState", "Literacy\nAnchored"]
hl_levels = ["Low", "Medium", "High"]
n_strat = len(strategies)
n_hl = len(hl_levels)

# Composite scores: [strategy][hl_level]
composite = [
    [0.797, 0.780, 0.777],   # Standard
    [0.792, 0.780, 0.755],   # Consistency
    [0.844, 0.780, 0.772],   # Structured State
    [0.812, 0.783, 0.808],   # Literacy Anchored
]

# Raw metrics by HL level (across all strategies)
raw_labels = [
    "Response Length",
    "Question Count",
    "Preference Rate",
    "Decision Activeness"
]

raw_low  = [22.73, 2.45, 0.059, -0.300]
raw_high = [52.63, 1.00, 0.220,  0.025]

# Strategy × dimension for low HL (literacy_anchored vs standard)
dim_labels = ["Resp.\nLength", "Question\nCount", "Preference\nRate", "Decision\nActiveness"]
dim_standard = [25.10, 2.80, 0.03, -0.20]
dim_lit_anch = [17.70, 1.20, 0.08, -1.00]


# ── Figure 1: Composite Score by Strategy × HL Level ───────────────

def fig1_composite_by_strategy_hl():
    fig, ax = plt.subplots(figsize=(6, 3.5))

    x = np.arange(n_hl)
    width = 0.20
    colors = ["#4C72B0", "#55A868", "#CC974B", "#C44E52"]

    for i, (strat, color) in enumerate(zip(strategies, colors)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, composite[i], width, label=strat,
                      color=color, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, composite[i]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Assigned Health Literacy Level", fontsize=10)
    ax.set_ylabel("Composite Faithfulness Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(hl_levels, fontsize=10)
    ax.set_ylim(0.6, 1.0)
    ax.legend(loc="lower left", fontsize=8, title="Generation Strategy")
    ax.axhline(y=0.785, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.text(2.3, 0.787, "Overall mean = 0.785", fontsize=7, color="gray", ha="right")

    fig.tight_layout()
    fig.savefig(f"{BASE}/hl_baseline_composite.png", dpi=200, bbox_inches="tight")
    print("  Saved hl_baseline_composite.png")
    plt.close(fig)


# ── Figure 2: Raw Metrics — Low vs High HL ─────────────────────────

def fig2_raw_metrics_low_vs_high():
    fig, axes = plt.subplots(1, 4, figsize=(9, 3.2))

    x = np.arange(2)
    colors_comp = ["#4C72B0", "#C44E52"]

    metric_configs = [
        ("Response Length (words)", raw_low[0], raw_high[0], (0, 65),
         lambda y: f"{y:.0f}"),
        ("Question Count (per dialogue)", raw_low[1], raw_high[1], (0, 4),
         lambda y: f"{y:.2f}"),
        ("Preference Expression Rate", raw_low[2], raw_high[2], (0, 0.35),
         lambda y: f"{y:.3f}"),
        ("Decision Activeness", raw_low[3], raw_high[3], (-0.6, 0.2),
         lambda y: f"{y:.2f}"),
    ]

    for idx, (title, low_val, high_val, ylim, fmt) in enumerate(metric_configs):
        ax = axes[idx]
        vals = [low_val, high_val]
        bars = ax.bar(x, vals, width=0.5, color=colors_comp, edgecolor="white",
                      linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + ylim[1]*0.02,
                    fmt(val), ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(["Low HL", "High HL"], fontsize=8)
        ax.set_ylabel(title, fontsize=8)
        ax.set_ylim(ylim)

        # p-value annotation
        p_vals = ["p<0.001", "p=0.388", "p<0.001", "p<0.001"]
        ax.text(0.5, 0.95, p_vals[idx], transform=ax.transAxes,
                fontsize=7, ha="center", va="top",
                color="#C44E52" if p_vals[idx]!="p=0.388" else "gray")

    fig.tight_layout()
    fig.savefig(f"{BASE}/hl_baseline_raw_metrics.png", dpi=200, bbox_inches="tight")
    print("  Saved hl_baseline_raw_metrics.png")
    plt.close(fig)


# ── Figure 3: Strategy Comparison on Low HL ─────────────────────────

def fig3_strategy_comparison_low_hl():
    fig, ax = plt.subplots(figsize=(5.5, 3.2))

    x = np.arange(len(dim_labels))
    width = 0.30

    bars1 = ax.bar(x - width/2, dim_standard, width, label="Standard Prompting",
                   color="#4C72B0", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, dim_lit_anch, width, label="Literacy-Anchored",
                   color="#C44E52", edgecolor="white", linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            val = bar.get_height()
            va = "bottom" if val >= 0 else "top"
            offset = 0.5 if val >= 0 else -0.5
            ax.text(bar.get_x() + bar.get_width()/2, val + offset,
                    f"{val:.2f}", ha="center", va=va, fontsize=7)

    ax.set_ylabel("Raw Metric Value", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=9)
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="-")
    ax.legend(fontsize=8)
    ax.set_title("Low Health Literacy Only", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(f"{BASE}/hl_strategy_comparison.png", dpi=200, bbox_inches="tight")
    print("  Saved hl_strategy_comparison.png")
    plt.close(fig)


# ── Run ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs(BASE, exist_ok=True)
    print("Generating figures...")
    fig1_composite_by_strategy_hl()
    fig2_raw_metrics_low_vs_high()
    fig3_strategy_comparison_low_hl()
    print("Done! All figures saved to figures/")
