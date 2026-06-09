#!/usr/bin/env python
"""
Generate A2 (production model) figures for the npj DM paper.
All figures are computed on the A2 model (token+FiLM), NOT v21c/A3.
Outputs PNGs to Phase9_V21/results/paper_figures/.

Figures:
  fig_architecture.png        - two-stage pipeline + Option B vs C schematic
  fig_stage1_ladder.png       - Stage-1 macro-AUC ladder (B/A1/A2/A3/A5), MR+CT
  fig_dca_surgery.png         - decision-curve net benefit (from dca_a2_surgery.json)
  fig_km_combined.png         - KM by predicted-risk tertile (surgery/steroid/biologic)

Calibration / td-AUC / Brier come from extended_evaluation_v19.py (a2_extended/).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SURV = os.environ.get("V21_SURVIVAL_A2_CSV",
                      os.path.join(os.environ.get("V21_SURV_DIR", "./survival"), "survival_a2.csv"))
DCA_JSON = os.environ.get("V21_DCA_A2_SURGERY",
                          os.path.join(os.environ.get("V21_RESULTS_DIR", "./results"),
                                       "dca_a2_surgery.json"))
OUT = os.environ.get("V21_PAPER_FIGURES_DIR",
                     os.path.join(os.environ.get("V21_RESULTS_DIR", "./results"), "paper_figures"))
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans", "savefig.dpi": 200,
                     "savefig.bbox": "tight"})

C_MR, C_CT = "#2E75B6", "#C55A11"


# ------------------------------------------------------------------ ladder
def fig_ladder():
    configs = ["B\n(specialist)", "A1\ntoken", "A2 ★\ntoken+FiLM",
               "A5\nDSBN(MR)", "A3 \"full\"\n+DSBN"]
    mr = [0.8486, 0.8431, 0.8486, 0.8348, 0.8296]
    ct = [0.7599, 0.7764, 0.7722, np.nan, 0.7563]
    x = np.arange(len(configs)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.bar(x - w/2, mr, w, label="MR (n=628)", color=C_MR)
    ct_plot = [v if not np.isnan(v) else 0 for v in ct]
    bars = ax.bar(x + w/2, ct_plot, w, label="CT (n=209)", color=C_CT)
    for i, v in enumerate(ct):
        if np.isnan(v):
            bars[i].set_alpha(0); ax.text(x[i] + w/2, 0.756, "n/a", ha="center",
                                          va="bottom", fontsize=8, color="gray")
    for i, v in enumerate(mr):
        ax.text(x[i] - w/2, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    for i, v in enumerate(ct):
        if not np.isnan(v):
            ax.text(x[i] + w/2, v + 0.001, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(configs, fontsize=9)
    ax.set_ylabel("Test macro-AUC"); ax.set_ylim(0.74, 0.865)
    ax.set_title("Stage-1 finding detection: unified conditioning ladder vs. specialists")
    ax.legend(loc="upper right", frameon=False)
    # annotate the significant DSBN drop on MR (A2 -> A3)
    ax.annotate("", xy=(x[4]-w/2, 0.8296), xytext=(x[2]-w/2, 0.8486),
                arrowprops=dict(arrowstyle="->", color="firebrick", lw=1.5))
    ax.text(3.05, 0.840, "DSBN: −0.019 MR\n95% CI [−0.030,−0.008]",
            color="firebrick", fontsize=8.5, ha="center")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(os.path.join(OUT, "fig_stage1_ladder.png")); plt.close(fig)
    print("ladder ok")


# ------------------------------------------------------------------ DCA
def fig_dca():
    d = json.load(open(DCA_JSON))
    t = np.array(d["thresholds"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(t, d["nb_combined"], color="#2E75B6", lw=2.2, label="Combined (imaging+clinical)")
    ax.plot(t, d["nb_clinical"], color="#7F7F7F", lw=1.8, label="Clinical only")
    ax.plot(t, d["nb_treat_all"], color="#C55A11", lw=1.3, ls="--", label="Treat all")
    ax.plot(t, d["nb_treat_none"], color="black", lw=1.0, ls=":", label="Treat none")
    lo, hi = d["combined_best_threshold_range"]
    ax.axvspan(lo, hi, color="#2E75B6", alpha=0.07)
    ax.set_xlim(0, 0.6); ax.set_ylim(bottom=min(0, np.min(d["nb_combined"])))
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title(f"Decision-curve analysis: surgery @ {d['horizon']:.0f} yr "
                 f"(n={d['n']}, events={d['events']})")
    ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.3)
    fig.savefig(os.path.join(OUT, "fig_dca_surgery.png")); plt.close(fig)
    print("dca ok")


# ------------------------------------------------------------------ KM
def fig_km():
    import pandas as pd
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test
    df = pd.read_csv(SURV)
    pred = [c for c in df.columns if c.startswith("pred_")]
    clin = [c for c in df.columns if c.startswith("clinical_")]
    feats = pred + clin
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df[feats] = df[feats].fillna(df[feats].median(numeric_only=True))
    # drop zero-variance columns; z-score standardize the rest for stable Cox fit
    feats = [c for c in feats if df[c].std() > 1e-8]
    df[feats] = (df[feats] - df[feats].mean()) / df[feats].std()
    outcomes = [("surgery", "Surgery"), ("steroid", "Steroid dependence"),
                ("biologic", "Biologic switch")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    colors = {"LOW": "#2CA02C", "MEDIUM": "#FF7F0E", "HIGH": "#D62728"}
    for ax, (oc, title) in zip(axes, outcomes):
        dur, ev = f"duration_{oc}", f"event_{oc}"
        sub = df[(df[dur] > 0)].copy()
        cph = CoxPHFitter(penalizer=1.0, l1_ratio=0.0)
        cph.fit(sub[feats + [dur, ev]], duration_col=dur, event_col=ev,
                fit_options={"step_size": 0.3})
        risk = cph.predict_partial_hazard(sub[feats]).values
        q1, q2 = np.quantile(risk, [1/3, 2/3])
        grp = np.where(risk <= q1, "LOW", np.where(risk <= q2, "MEDIUM", "HIGH"))
        sub["grp"] = grp
        km = KaplanMeierFitter()
        for g in ["LOW", "MEDIUM", "HIGH"]:
            m = sub["grp"] == g
            km.fit(sub.loc[m, dur], sub.loc[m, ev], label=f"{g} (n={m.sum()})")
            km.plot_survival_function(ax=ax, color=colors[g], ci_show=False, lw=2)
        p = multivariate_logrank_test(sub[dur], sub["grp"], sub[ev]).p_value
        ptxt = f"p = {p:.1e}" if p > 0 else "p < 1e-300"
        ax.set_title(f"{title}\nlog-rank {ptxt}", fontsize=10)
        ax.set_xlabel("Years from MRI"); ax.set_ylim(0, 1.02)
        ax.legend(fontsize=8, frameon=False, loc="lower left")
    axes[0].set_ylabel("Event-free probability")
    fig.suptitle("Risk stratification by predicted-risk tertile (A2 unified model)",
                 fontsize=12, y=1.02)
    fig.savefig(os.path.join(OUT, "fig_km_combined.png")); plt.close(fig)
    print("km ok")


# ------------------------------------------------------------------ architecture
def fig_arch():
    fig, ax = plt.subplots(figsize=(10, 5.6)); ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    def box(x, y, w, h, text, fc, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                                    fc=fc, ec="#333333", lw=1.1))
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, color="#555555", lw=1.2))

    box(0.2, 4.4, 1.7, 1.0, "MR\n(T2 + T1)", "#DEEBF7")
    box(0.2, 0.7, 1.7, 1.0, "CT\n(T2 branch,\nT1 zeroed)", "#FCE4D6")
    box(2.3, 2.3, 2.0, 1.3, "Frozen DINOv3\n+ LoRA\n(shared backbone)", "#E2EFDA", 9)
    box(4.6, 3.6, 1.9, 0.9, "modality token\n+ FiLM  (A2)", "#FFF2CC", 8.5)
    box(4.6, 2.5, 1.9, 0.8, "DSBN  ✖ (harmful)", "#F2DCDB", 8.5)
    box(6.8, 2.4, 1.5, 1.1, "MIL\n10 gated-\nattention heads", "#E2EFDA", 8.5)
    box(8.5, 2.5, 1.3, 0.9, "10 finding\nprobabilities", "#DDEBF7", 8.5)
    # stage 2
    box(6.8, 0.5, 3.0, 1.0, "Stage 2 (MR-only): penalized Cox\n10 probs + 13 clinical → S(t|x)\nsurgery / steroid / biologic",
        "#EDEDED", 8.5)
    arrow(1.9, 4.9, 2.4, 3.4); arrow(1.9, 1.2, 2.4, 2.5)
    arrow(4.3, 3.0, 4.6, 3.2); arrow(6.5, 3.0, 6.8, 3.0)
    arrow(8.3, 3.0, 8.5, 3.0); arrow(9.1, 2.5, 8.6, 1.5)
    ax.text(5.0, 5.4, "Stage 1: unified CT+MR finding detection (Option C = one backbone)",
            fontsize=10, weight="bold")
    ax.text(2.4, 1.95, "warm-start from V20 (symmetric to B & C)", fontsize=7.5,
            style="italic", color="#555")
    fig.savefig(os.path.join(OUT, "fig_architecture.png")); plt.close(fig)
    print("arch ok")


if __name__ == "__main__":
    fig_ladder()
    fig_dca()
    fig_arch()
    fig_km()
    print("ALL FIGURES DONE ->", OUT)
