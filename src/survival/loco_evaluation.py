#!/usr/bin/env python
"""LOCO Cox on survival_v08 (corrected center labels).

Hold out each MR center in turn (Mor Inside / SZMC / Assuta_MR), fit elastic-net
Cox on the rest of MR + all CT, evaluate C-index on the held-out MR center.

CT is excluded from LOCO splits because all CT studies originate from a single
center (Assuta); CT data stays in the training fold every iteration.
"""
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
warnings.filterwarnings("ignore")

LABELS = ["ileum_inflammation","ileum_wall_enhancement","ileum_wall_thickness",
          "ileum_dwi","ileum_stenosis","ileum_pre_stenotic_dil","ileum_comb_sign",
          "ileum_fistula","colon_inflammation","ileum_mesenteric_edema"]
CLINICAL = ["Time_to_index","age_at_diagnosis","CCI_at_diagnosis","disease_activity_5",
            "SES_points","diagnostic_delay","had_surgery_before","had_steroid_dep_before",
            "had_biologic_switch_before","had_EIM_before","had_perianal_before","sex",
            "has_clinical"]
PRED_COLS = [f"pred_{l}" for l in LABELS]
CLIN_COLS = [f"clinical_{c}" for c in CLINICAL]
OUTCOMES = ["surgery","steroid","biologic"]
N_BOOT = 1000


def feat_set(name):
    return {"combined": PRED_COLS + CLIN_COLS,
            "imaging_only": PRED_COLS,
            "clinical_only": CLIN_COLS}[name]


def fit_one(Xtr, ytr, Xte):
    cox = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                 n_alphas=100, max_iter=500_000,
                                 fit_baseline_model=False)
    cox.fit(Xtr, ytr)
    nnz = (np.abs(cox.coef_) > 1e-6).sum(axis=0)
    cands = np.where((nnz >= 5) & (nnz <= 15))[0]
    ix = int(cands[len(cands)//2]) if len(cands) else int(np.argmin(np.abs(nnz - 10)))
    return cox.predict(Xte, alpha=cox.alphas_[ix])


def cindex(events, durations, risk):
    return concordance_index_censored(events.astype(bool), durations, risk)[0]


def bootstrap_ci(events, durations, risk, n_boot=N_BOOT, seed=0):
    rng = np.random.RandomState(seed)
    n = len(events)
    out = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            out.append(cindex(events[idx], durations[idx], risk[idx]))
        except Exception:
            pass
    if not out:
        return None, None
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def evaluate_outcome(df, outcome):
    durations = df[f"duration_{outcome}"].astype(float).to_numpy()
    events = df[f"event_{outcome}"].astype(bool).to_numpy()
    centers = df["center"].to_numpy()
    modality = df["modality"].to_numpy()
    pid = df["patient_id"].astype(str).to_numpy()
    mr_centers = sorted(set(centers[modality == "mr"]))
    print(f"  MR centers found: {mr_centers}")
    result = {fs: {"per_center": {}} for fs in ("combined","imaging_only","clinical_only")}

    for fs in ("combined","imaging_only","clinical_only"):
        cols = feat_set(fs)
        X = df[cols].astype(float).fillna(0.0).to_numpy()
        per_center = {}
        for c in mr_centers:
            te_mask = (centers == c) & (modality == "mr")
            tr_mask = ~te_mask
            if te_mask.sum() < 30 or events[te_mask].sum() < 3:
                continue
            Xtr, Xte = X[tr_mask], X[te_mask]
            mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd<1e-8] = 1.0
            Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
            ytr = Surv.from_arrays(events[tr_mask], durations[tr_mask])
            try:
                risk = fit_one(Xtr, ytr, Xte)
            except Exception as e:
                print(f"    {c}: skipped ({e})"); continue
            cidx = cindex(events[te_mask], durations[te_mask], risk)
            lo, hi = bootstrap_ci(events[te_mask], durations[te_mask], risk)
            per_center[c] = {
                "n_test": int(te_mask.sum()),
                "n_events": int(events[te_mask].sum()),
                "n_train": int(tr_mask.sum()),
                "cindex": float(cidx),
                "ci_low": lo, "ci_high": hi,
            }
            print(f"    {c}: n={te_mask.sum()}, ev={int(events[te_mask].sum())}, "
                  f"c-index={cidx:.4f} (95% CI {lo:.4f}-{hi:.4f})")
        cis = [v["cindex"] for v in per_center.values()]
        summary = {"loco_mean_cindex": float(np.mean(cis)) if cis else None,
                   "loco_std": float(np.std(cis, ddof=1)) if len(cis) > 1 else None,
                   "n_centers": len(cis),
                   "per_center": per_center}
        result[fs] = summary
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survival-csv", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.survival_csv)
    for c in PRED_COLS + CLIN_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    print(f"loaded {len(df)} rows, centers={df['center'].value_counts().to_dict()}")
    out = {}
    for oc in OUTCOMES:
        print(f"\n=== {oc.upper()} ===")
        out[oc] = evaluate_outcome(df, oc)
        for fs in ("combined","imaging_only","clinical_only"):
            r = out[oc][fs]
            if r.get("loco_mean_cindex") is not None:
                print(f"  {fs:14s}  LOCO mean={r['loco_mean_cindex']:.4f}  SD={r['loco_std']:.4f}")
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE -> {args.output}")


if __name__ == "__main__":
    main()
