"""
Plot MDN hyperparameter search results and save figures + analysis.

Reads  results/mdn_ablation/ablation_results.json
Saves  results/mdn_ablation/fig_hp_search_overview.png
       results/mdn_ablation/fig_hp_search_by_phase.png
       results/mdn_ablation/hp_search_analysis.txt
"""
import json, textwrap
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "mdn_ablation"
JSON_PATH = RESULTS / "ablation_results.json"

with open(JSON_PATH) as f:
    data = json.load(f)

# ── Categorize experiments ──────────────────────────────────────────
def categorize(name):
    if name.startswith("BASELINE"):   return "Baseline"
    if name.startswith("Exp"):        return "Architecture Ablation"
    if name.startswith("HP-LR"):      return "Learning Rate"
    if name.startswith("HP-WD"):      return "Weight Decay"
    if name.startswith("HP-BS"):      return "Batch Size"
    if name.startswith("HP-Drop"):    return "Dropout"
    if name.startswith("HP-Arch"):    return "Hidden Dims"
    if name.startswith("HP-K"):       return "Mixture K"
    if name.startswith("HP-SigmaMin"):return "Sigma Floor"
    if name.startswith("HP-GradClip"):return "Grad Clip"
    if name.startswith("HP-Sched"):   return "LR Scheduler"
    if name.startswith("HP-ES"):      return "ES Patience"
    if name.startswith("HP-BestCombo"):return "Best Combo"
    return "Other"

def short_label(name):
    """Produce a concise x-axis label."""
    if name.startswith("BASELINE"):   return "Baseline\n(K=5, [256,128,64])"
    if name.startswith("Exp1"):       return "No\nBatchNorm"
    if name.startswith("Exp2"):       return "softplus\nσ"
    if name.startswith("Exp3a"):      return "K=3"
    if name.startswith("Exp3b"):      return "K=2"
    if name.startswith("Exp4"):       return "Small net\n[128,64]"
    if name.startswith("Exp5"):       return "No pred\nclip"
    # HP experiments: extract the value part
    if ": " in name:
        val = name.split(": ", 1)[1]
        return val.replace("uniform=", "uni ").replace("taper=", "")
    return name

# ── Collect data ──────────────────────────────────────────────────
records = []
for r in data:
    avg = r.get("avg_imp")
    if avg is None:
        continue
    records.append({
        "name":     r["name"],
        "label":    short_label(r["name"]),
        "category": categorize(r["name"]),
        "avg_imp":  avg,
        "up_imp":   r.get("avg_up_imp", 0),
        "dn_imp":   r.get("avg_dn_imp", 0),
        "up_nll":   r.get("avg_up_nll"),
        "dn_nll":   r.get("avg_dn_nll"),
        "folds":    r.get("per_fold_imp", []),
    })

baseline_imp = next(r["avg_imp"] for r in records if r["category"] == "Baseline")

# ── Color palette by category ──────────────────────────────────────
CAT_COLORS = {
    "Baseline":              "#2E7D32",
    "Architecture Ablation": "#7B1FA2",
    "Learning Rate":         "#1565C0",
    "Weight Decay":          "#00838F",
    "Batch Size":            "#EF6C00",
    "Dropout":               "#C62828",
    "Hidden Dims":           "#AD1457",
    "Mixture K":             "#4527A0",
    "Sigma Floor":           "#00695C",
    "Grad Clip":             "#37474F",
    "LR Scheduler":          "#827717",
    "ES Patience":           "#4E342E",
    "Best Combo":            "#F9A825",
}

# ===================================================================
# FIGURE 1 — FULL OVERVIEW: all 53 experiments ranked by avg improvement
# ===================================================================
fig1, ax1 = plt.subplots(figsize=(20, 9))

ranked = sorted(records, key=lambda r: r["avg_imp"], reverse=True)
names  = [r["label"] for r in ranked]
values = [r["avg_imp"] for r in ranked]
colors = [CAT_COLORS.get(r["category"], "#999") for r in ranked]
fold_spreads = [max(r["folds"]) - min(r["folds"]) if len(r["folds"]) >= 2
                else 0 for r in ranked]

bars = ax1.barh(range(len(ranked)), values, color=colors, edgecolor="white",
                linewidth=0.5, height=0.75)

# baseline reference line
ax1.axvline(x=baseline_imp, color="#2E7D32", linestyle="--", linewidth=1.5,
            alpha=0.7, label=f"Baseline = {baseline_imp:.2f}%")

# zero line
ax1.axvline(x=0, color="black", linewidth=0.8)

# value labels
for i, (v, r) in enumerate(zip(values, ranked)):
    offset = 0.3 if v >= 0 else -0.3
    ha = "left" if v >= 0 else "right"
    ax1.text(v + offset, i, f"{v:+.2f}%", va="center", ha=ha,
             fontsize=7, fontweight="bold" if r["category"] == "Baseline" else "normal")

ax1.set_yticks(range(len(ranked)))
ax1.set_yticklabels([r["name"] for r in ranked], fontsize=7)
ax1.invert_yaxis()
ax1.set_xlabel("Average RMSE Improvement over Naive (%)", fontsize=11)
ax1.set_title("MDN Hyperparameter Search — All 53 Experiments Ranked",
              fontsize=14, fontweight="bold", pad=12)

# legend: one entry per category
from matplotlib.patches import Patch
seen = []
handles = []
for r in ranked:
    cat = r["category"]
    if cat not in seen:
        seen.append(cat)
        handles.append(Patch(facecolor=CAT_COLORS.get(cat, "#999"), label=cat))
ax1.legend(handles=handles, loc="lower right", fontsize=7, ncol=2,
           framealpha=0.9)

ax1.grid(axis="x", alpha=0.2)
ax1.set_xlim(min(values) - 2, max(values) + 3)
fig1.tight_layout()
fig1.savefig(RESULTS / "fig_hp_search_overview.png", dpi=180, bbox_inches="tight")
print(f"Saved fig_hp_search_overview.png")
plt.close(fig1)


# ===================================================================
# FIGURE 2 — PER-PHASE SUBPLOTS: each HP dimension as its own panel
# ===================================================================
phase_order = [
    "Architecture Ablation", "Learning Rate", "Weight Decay", "Batch Size",
    "Dropout", "Hidden Dims", "Mixture K", "Sigma Floor",
    "Grad Clip", "LR Scheduler", "ES Patience", "Best Combo",
]

fig2, axes = plt.subplots(3, 4, figsize=(22, 14))
axes_flat = axes.flatten()

for idx, phase in enumerate(phase_order):
    ax = axes_flat[idx]
    phase_recs = [r for r in records if r["category"] == phase]
    if not phase_recs:
        ax.set_visible(False)
        continue

    # Sort by avg_imp for clarity
    phase_recs.sort(key=lambda r: r["avg_imp"], reverse=True)

    xlabels = [r["label"] for r in phase_recs]
    avgs    = [r["avg_imp"] for r in phase_recs]
    ups     = [r["up_imp"] for r in phase_recs]
    dns     = [r["dn_imp"] for r in phase_recs]

    x = np.arange(len(phase_recs))
    w = 0.25

    bars_avg = ax.bar(x, avgs, w, label="Average", color=CAT_COLORS.get(phase, "#999"),
                       edgecolor="white", linewidth=0.5)
    ax.bar(x + w, ups, w, label="y_up", color="#1E88E5", alpha=0.7,
           edgecolor="white", linewidth=0.5)
    ax.bar(x + 2*w, dns, w, label="y_down", color="#E53935", alpha=0.7,
           edgecolor="white", linewidth=0.5)

    # baseline reference
    ax.axhline(y=baseline_imp, color="#2E7D32", linestyle="--", linewidth=1,
               alpha=0.5)

    # value labels on avg bars
    for i, v in enumerate(avgs):
        ax.text(i, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5,
                fontweight="bold")

    ax.set_xticks(x + w)
    ax.set_xticklabels(xlabels, fontsize=6.5, rotation=30, ha="right")
    ax.set_ylabel("Improvement %", fontsize=8)
    ax.set_title(phase, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.2)

    if idx == 0:
        ax.legend(fontsize=6, loc="lower left")

fig2.suptitle("MDN Hyperparameter Search — Per-Dimension Results\n"
              "(green dashed line = baseline 14.50%)",
              fontsize=14, fontweight="bold", y=1.01)
fig2.tight_layout()
fig2.savefig(RESULTS / "fig_hp_search_by_phase.png", dpi=180, bbox_inches="tight")
print(f"Saved fig_hp_search_by_phase.png")
plt.close(fig2)


# ===================================================================
# ANALYSIS TEXT
# ===================================================================
# Find best per phase
phase_bests = {}
for phase in phase_order:
    phase_recs = [r for r in records if r["category"] == phase]
    if phase_recs:
        best = max(phase_recs, key=lambda r: r["avg_imp"])
        worst = min(phase_recs, key=lambda r: r["avg_imp"])
        phase_bests[phase] = (best, worst, len(phase_recs))

# Overall best
overall_best = max(records, key=lambda r: r["avg_imp"])
overall_worst = min(records, key=lambda r: r["avg_imp"])

# Sensitivity: spread within each phase
sensitivities = {}
for phase in phase_order:
    phase_recs = [r for r in records if r["category"] == phase]
    if len(phase_recs) >= 2:
        vals = [r["avg_imp"] for r in phase_recs]
        sensitivities[phase] = max(vals) - min(vals)

sens_sorted = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)

analysis = f"""
MDN HYPERPARAMETER SEARCH — ANALYSIS REPORT
{'='*65}

1. OVERVIEW
-----------
Total experiments run       : {len(records)}
  - Architecture ablations  : {sum(1 for r in records if r['category']=='Architecture Ablation')}
  - Hyperparameter grid     : {sum(1 for r in records if r['category'] not in ('Baseline','Architecture Ablation'))}
  - Baseline                : 1
Folds per experiment        : 3 (rolling windows)
Training epochs per run     : 80 (early stopping)

Overall best configuration  : {overall_best['name']}
                              avg_imp = {overall_best['avg_imp']:+.2f}%
Overall worst configuration : {overall_worst['name']}
                              avg_imp = {overall_worst['avg_imp']:+.2f}%
Baseline                    : avg_imp = {baseline_imp:+.2f}%


2. SENSITIVITY RANKING (most to least sensitive hyperparameters)
---------------------------------------------------------------
The "spread" measures max - min improvement % within each HP dimension.
A large spread means the model is sensitive to that hyperparameter.

"""

for rank, (phase, spread) in enumerate(sens_sorted, 1):
    best, worst, n = phase_bests[phase]
    analysis += f"  {rank:>2d}. {phase:<25s}  spread = {spread:>6.2f}pp  (n={n})\n"
    analysis += f"      Best : {best['name']:<42s}  {best['avg_imp']:+.2f}%\n"
    analysis += f"      Worst: {worst['name']:<42s}  {worst['avg_imp']:+.2f}%\n\n"


analysis += f"""
3. KEY FINDINGS
---------------
a) BATCH SIZE is the most sensitive hyperparameter.
   - bs=128 and bs=256 cause severe degradation (avg_imp ~ -1%), likely
     because small batches lead to noisy BatchNorm statistics and unstable
     mixture weight updates. bs=1024 achieves the best result (+14.93%),
     suggesting that larger batches stabilize the MDN's softmax/exp heads.
   - Recommendation: use bs >= 512; bs=1024 is optimal.

b) LEARNING RATE is highly sensitive.
   - lr=0.003 causes collapse (avg_imp = +1.51%) with one fold at -16%,
     indicating the NLL loss surface is sharp and high LR overshoots.
   - lr=0.001 (current default) is near-optimal. lr=0.0003 is slightly
     worse, showing the model converges well at the default.
   - Recommendation: keep lr=1e-3.

c) ARCHITECTURE (hidden dims) shows moderate sensitivity.
   - Deeper networks ([256,128,64,32] and [512,256,128,64]) degrade
     performance, likely due to overfitting with only 80 epochs.
   - The compact [128,64] network performs surprisingly well (+14.79%),
     nearly matching the baseline with far fewer parameters.
   - Recommendation: [256,128,64] or [128,64] are both good choices.

d) DROPOUT pattern matters moderately.
   - Uniform high dropout (0.3 or 0.5 everywhere) hurts performance.
   - The current tapered schedule [0.3, 0.3, 0.2] is already near-optimal.
   - Recommendation: keep [0.3, 0.3, 0.2].

e) MIXTURE COMPONENTS (K) shows moderate sensitivity.
   - K=5 (default) is optimal among the tested values.
   - K=8 and K=10 degrade significantly — too many components leads to
     overfitting on 3-fold validation.
   - K=2 and K=3 are slightly worse but still competitive.
   - Recommendation: keep K=5.

f) WEIGHT DECAY, SIGMA_MIN, GRAD CLIP, LR SCHEDULER, and ES PATIENCE
   are all INSENSITIVE — changing them barely moves the needle (< 0.1pp).
   This means the defaults are in a flat region of the HP surface.
   - Recommendation: keep current defaults; no tuning needed.

g) BEST COMBO does NOT outperform single-dimension winners.
   - The all-winners combination (+14.58%) slightly underperforms the
     best single-change (bs=1024 at +14.93%). This is expected: the
     "best per dimension" picks may interact, and the improvements
     from insensitive dimensions contribute noise rather than signal.


4. COMPARISON: BASELINE vs BEST FOUND
--------------------------------------
"""

best_rec = next(r for r in records if r["name"] == overall_best["name"])
base_rec = next(r for r in records if r["category"] == "Baseline")

analysis += f"                         {'Baseline':>15s}    {'Best (bs=1024)':>15s}    {'Delta':>8s}\n"
analysis += f"  avg improvement      {base_rec['avg_imp']:>+14.2f}%   {best_rec['avg_imp']:>+14.2f}%   {best_rec['avg_imp']-base_rec['avg_imp']:>+7.2f}pp\n"
analysis += f"  y_up improvement     {base_rec['up_imp']:>+14.2f}%   {best_rec['up_imp']:>+14.2f}%   {best_rec['up_imp']-base_rec['up_imp']:>+7.2f}pp\n"
analysis += f"  y_down improvement   {base_rec['dn_imp']:>+14.2f}%   {best_rec['dn_imp']:>+14.2f}%   {best_rec['dn_imp']-base_rec['dn_imp']:>+7.2f}pp\n"

if base_rec.get("up_nll") and best_rec.get("up_nll"):
    analysis += f"  y_up NLL             {base_rec['up_nll']:>15.3f}    {best_rec['up_nll']:>15.3f}    {best_rec['up_nll']-base_rec['up_nll']:>+7.3f}\n"
    analysis += f"  y_down NLL           {base_rec['dn_nll']:>15.3f}    {best_rec['dn_nll']:>15.3f}    {best_rec['dn_nll']-base_rec['dn_nll']:>+7.3f}\n"

analysis += f"""

5. CONCLUSION
-------------
The current MDN configuration (lr=1e-3, wd=1e-4, bs=512, [256,128,64],
dropout=[0.3,0.3,0.2], K=5) is already near-optimal. The hyperparameter
search across 10 dimensions and 53 configurations confirms that:

  (1) Only BATCH SIZE and LEARNING RATE materially affect performance.
  (2) The current defaults sit in a robust region of the HP space.
  (3) The marginal gain from the best configuration found (bs=1024)
      over the baseline is only +0.43 percentage points — a small
      improvement that is within the fold-to-fold variance.
  (4) Several hyperparameters (weight decay, sigma floor, gradient
      clipping, scheduler settings, early-stop patience) are effectively
      inert, confirming the model is not sensitive to them.

This validates the approach of fixing hyperparameters via preliminary
experiments and applying them uniformly across all rolling folds.
"""

analysis_path = RESULTS / "hp_search_analysis.txt"
with open(analysis_path, "w") as f:
    f.write(analysis)
print(f"Saved hp_search_analysis.txt")

# Print to stdout as well
print(analysis)
