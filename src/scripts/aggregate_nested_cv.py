#!/usr/bin/env python
"""
After the nested-CV array finishes, run inference on each fold's test split for each
variant (A1/A2/A3) and aggregate macro-AUC.

Output: Phase9_V21/results/nested_cv/summary.json + summary.csv
        per-variant per-fold test macro-AUC + mean +/- SD across 5 folds.
"""
import json, os, glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PREDS_ROOT = os.environ.get("V21_NESTEDCV_PRED_DIR",
                             os.path.join(os.environ.get("V21_PRED_DIR", "./predictions"), "nestedcv"))
CV_DATA = os.environ.get("V21_CV5_DATA_DIR",
                         os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "cv5"))
OUT = os.environ.get("V21_NESTEDCV_RESULTS_DIR",
                     os.path.join(os.environ.get("V21_RESULTS_DIR", "./results"), "nested_cv"))
os.makedirs(OUT, exist_ok=True)

VARIANTS = ["a1", "a2", "a3"]
FINDINGS = [
    "ileum_inflammation", "ileum_wall_enhancement", "ileum_wall_thickness",
    "ileum_dwi", "ileum_stenosis", "ileum_pre_stenotic_dil", "ileum_comb_sign",
    "ileum_fistula", "colon_inflammation", "ileum_mesenteric_edema",
]


def macro_auc(pred_csv, test_csv, modality):
    pred = pd.read_csv(pred_csv)
    test = pd.read_csv(test_csv)
    df = pred.merge(test, on="accession_number", suffixes=("", "_lbl"))
    df = df[df["modality"] == modality]
    aucs = []
    for c in FINDINGS:
        y = df[c].astype(int).values
        p = df[f"pred_{c}"].astype(float).values
        if len(np.unique(y)) < 2:
            continue
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs)) if aucs else float("nan"), len(df)


def main():
    rows = []
    for fold in range(5):
        for v in VARIANTS:
            for mod in ("mr", "ct"):
                pred = f"{PREDS_ROOT}/fold_{fold}/{v}_{mod}_test.csv"
                test = f"{CV_DATA}/fold_{fold}/test.csv"
                if not os.path.isfile(pred):
                    rows.append({"fold": fold, "variant": v, "modality": mod,
                                 "macro_auc": None, "n": None,
                                 "note": "predictions missing"})
                    continue
                auc, n = macro_auc(pred, test, mod)
                rows.append({"fold": fold, "variant": v, "modality": mod,
                             "macro_auc": auc, "n": n})
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/per_fold.csv", index=False)

    summary = []
    for v in VARIANTS:
        for mod in ("mr", "ct"):
            sub = df[(df["variant"] == v) & (df["modality"] == mod) &
                     df["macro_auc"].notna()]
            if len(sub) == 0:
                continue
            arr = sub["macro_auc"].values
            summary.append({"variant": v, "modality": mod,
                            "n_folds": int(len(arr)),
                            "mean": float(np.mean(arr)), "sd": float(np.std(arr, ddof=1)),
                            "fold_values": [float(x) for x in arr]})
    json.dump({"per_fold": rows, "summary": summary},
              open(f"{OUT}/summary.json", "w"), indent=2)
    print("\n=== Nested CV summary ===")
    for s in summary:
        vals = " ".join(f"{x:.4f}" for x in s["fold_values"])
        print(f"  {s['variant'].upper()} {s['modality'].upper()}  "
              f"mean={s['mean']:.4f} sd={s['sd']:.4f}  "
              f"per-fold=[{vals}]")
    print("WROTE ->", OUT)


if __name__ == "__main__":
    main()
