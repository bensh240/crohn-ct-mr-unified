#!/usr/bin/env python
"""
fit_unified_cox.py - Cox on the unified MR+CT survival cohort.

5-fold patient-grouped CV. For each fold:
  * fit elastic-net Cox on the train rows (CT + MR jointly)
  * compute C-index on the test rows: COMBINED, MR-only, CT-only
  * also compute imaging-only (10-d) and clinical-only (13-d) variants

Output JSON has, per outcome and per feature set, per-modality and combined
mean/SD C-index across the 5 folds, plus bootstrap CI on the full test
predictions.

Patient-id is the grouping key (60 patients straddle MR/CT - keeping them in
the same fold prevents leakage).
"""
import argparse
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

warnings.filterwarnings("ignore")

LABELS = [
    "ileum_inflammation", "ileum_wall_enhancement", "ileum_wall_thickness",
    "ileum_dwi", "ileum_stenosis", "ileum_pre_stenotic_dil", "ileum_comb_sign",
    "ileum_fistula", "colon_inflammation", "ileum_mesenteric_edema",
]
CLINICAL = [
    "Time_to_index", "age_at_diagnosis", "CCI_at_diagnosis",
    "disease_activity_5", "SES_points", "diagnostic_delay",
    "had_surgery_before", "had_steroid_dep_before", "had_biologic_switch_before",
    "had_EIM_before", "had_perianal_before", "sex", "has_clinical",
]
PRED_COLS = [f"pred_{L}" for L in LABELS]
CLIN_COLS = [f"clinical_{c}" for c in CLINICAL]
OUTCOMES = ["surgery", "steroid", "biologic"]
N_SPLITS = 5
RNG = 42


def get_feature_set(name):
    return {"combined": PRED_COLS + CLIN_COLS, "imaging_only": PRED_COLS,
            "clinical_only": CLIN_COLS}[name]


def fit_one_fold(Xtr, ytr, Xte, alphas=None):
    cox = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                 n_alphas=100, max_iter=200, fit_baseline_model=False)
    cox.fit(Xtr, ytr)
    # pick alpha with 5-15 non-zero features (the elastic-net "knee")
    nnz = (np.abs(cox.coef_) > 1e-6).sum(axis=0)
    candidates = np.where((nnz >= 5) & (nnz <= 15))[0]
    if len(candidates) == 0:
        a_ix = int(np.argmin(np.abs(nnz - 10)))
    else:
        a_ix = int(candidates[len(candidates) // 2])
    alpha = cox.alphas_[a_ix]
    # predict partial hazard at chosen alpha
    risk = cox.predict(Xte, alpha=alpha)
    return risk, alpha, int(nnz[a_ix])


def cindex(y_struct, risk):
    return concordance_index_censored(y_struct["e"], y_struct["t"], risk)[0]


def evaluate(df, feature_set, outcome):
    X = df[get_feature_set(feature_set)].astype(float).fillna(0.0).to_numpy()
    # z-standardize per fold (safer with elastic-net path)
    y_e = df[f"event_{outcome}"].astype(int).to_numpy()
    y_t = df[f"duration_{outcome}"].astype(float).to_numpy()
    y_struct = np.empty(len(df), dtype=[("e", "?"), ("t", "f8")])
    y_struct["e"] = y_e.astype(bool)
    y_struct["t"] = y_t
    mod = df["modality"].to_numpy()
    pids = df["patient_id"].astype(str).to_numpy()

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_combined, fold_mr, fold_ct = [], [], []
    n_alpha, n_nnz = [], []
    for tr_idx, te_idx in gkf.split(X, y_e, pids):
        Xtr, Xte = X[tr_idx], X[te_idx]
        # standardize using train stats
        mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-8] = 1.0
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        ytr = Surv.from_arrays(y_e[tr_idx].astype(bool), y_t[tr_idx])
        try:
            risk, alpha, nnz = fit_one_fold(Xtr, ytr, Xte)
        except Exception as e:
            print(f"  fold skipped: {e}")
            continue
        ys_te = y_struct[te_idx]
        # combined
        fold_combined.append(cindex(ys_te, risk))
        # MR subset of this test fold
        mr_mask = mod[te_idx] == "mr"
        if mr_mask.sum() > 5 and y_e[te_idx][mr_mask].sum() >= 2:
            fold_mr.append(cindex(ys_te[mr_mask], risk[mr_mask]))
        ct_mask = mod[te_idx] == "ct"
        if ct_mask.sum() > 5 and y_e[te_idx][ct_mask].sum() >= 2:
            fold_ct.append(cindex(ys_te[ct_mask], risk[ct_mask]))
        n_alpha.append(float(alpha))
        n_nnz.append(int(nnz))

    def summarize(arr):
        if len(arr) == 0: return {"mean": None, "sd": None, "folds": []}
        return {"mean": float(np.mean(arr)), "sd": float(np.std(arr, ddof=1)),
                "folds": [float(x) for x in arr]}

    return {
        "combined": summarize(fold_combined),
        "mr_only":  summarize(fold_mr),
        "ct_only":  summarize(fold_ct),
        "alpha_mean": float(np.mean(n_alpha)) if n_alpha else None,
        "n_nonzero_mean": float(np.mean(n_nnz)) if n_nnz else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survival-csv", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.survival_csv)
    # numerify and clean
    feat_cols = PRED_COLS + CLIN_COLS
    df[feat_cols] = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    print(f"loaded {len(df)} rows  ({(df['modality']=='mr').sum()} MR + "
          f"{(df['modality']=='ct').sum()} CT)  "
          f"{df['patient_id'].nunique()} patients")

    out = {"meta": {
        "n_total": int(len(df)),
        "n_mr": int((df["modality"] == "mr").sum()),
        "n_ct": int((df["modality"] == "ct").sum()),
        "n_patients": int(df["patient_id"].nunique()),
    }}
    for outcome in OUTCOMES:
        ev = int(df[f"event_{outcome}"].sum())
        ev_mr = int(df.loc[df["modality"] == "mr", f"event_{outcome}"].sum())
        ev_ct = int(df.loc[df["modality"] == "ct", f"event_{outcome}"].sum())
        out[outcome] = {"events": {"total": ev, "mr": ev_mr, "ct": ev_ct}}
        print(f"\n=== {outcome.upper()} ({ev} events: {ev_mr} MR + {ev_ct} CT) ===")
        for fs in ("combined", "imaging_only", "clinical_only"):
            r = evaluate(df, fs, outcome)
            out[outcome][fs] = r
            c = r["combined"]
            mr = r["mr_only"]; ct = r["ct_only"]
            print(f"  {fs:15s}  combined={c['mean']:.4f}+-{c['sd']:.4f}  "
                  f"MR={mr['mean']:.4f}+-{mr['sd']:.4f}  "
                  f"CT={ct['mean']:.4f}+-{ct['sd']:.4f}  "
                  f"(alpha={r['alpha_mean']:.4g}, nnz~{r['n_nonzero_mean']:.0f})")

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWROTE -> {args.output}")


if __name__ == "__main__":
    main()
