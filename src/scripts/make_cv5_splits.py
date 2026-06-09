#!/usr/bin/env python
"""
Make 5 patient-level CV folds for nested-CV evaluation of the conditioning ladder.

Output layout:
  Phase9_V21/data/cv5/fold_0/{train,val,test}.csv
  ...
  Phase9_V21/data/cv5/fold_4/{train,val,test}.csv

Each fold:
  - test  : the K-th of 5 patient-level partitions of the FULL pooled cohort.
  - val   : a 10% patient-level slice of the remainder, drawn deterministically per fold.
  - train : the rest.

Why: defends the post-hoc-selection critique. We train the conditioning
ladder (A1/A2/A3) on each fold and report mean +/- SD across the 5 outer
test sets. If A2 is stably the winner, the single-split critique dies.

Leakage check: a hard assertion that no patient_id appears in more than
one split per fold. The script will refuse to write if this fails.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

DATA = os.environ.get("V21_DATA_DIR", "./data")
SRC_FILES = ["train.csv", "val.csv", "test.csv"]
OUT_ROOT = os.path.join(DATA, "cv5")
SEED = 42
N_FOLDS = 5
VAL_FRACTION = 0.10


def main():
    parts = [pd.read_csv(os.path.join(DATA, f)) for f in SRC_FILES]
    df = pd.concat(parts, ignore_index=True)
    # patient_id lives in the master table; join on accession_number.
    master = pd.read_csv(os.path.join(DATA, "v21_master_table.csv"),
                         usecols=["accession_number", "patient_id"])
    n_before = len(df)
    df = df.merge(master, on="accession_number", how="left")
    missing = df["patient_id"].isna().sum()
    if missing:
        pct = 100.0 * missing / n_before
        print(f"  warning: dropping {missing}/{n_before} ({pct:.1f}%) studies with no "
              f"patient_id (cannot place them in a patient-level fold safely)")
        df = df.dropna(subset=["patient_id"]).reset_index(drop=True)
    print(f"pooled cohort: {len(df)} studies, {df['patient_id'].nunique()} patients, "
          f"modality counts {df['modality'].value_counts().to_dict()}")

    rng = np.random.default_rng(SEED)
    pids = df["patient_id"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    # GroupKFold ignores y; use a dummy.
    splits = list(gkf.split(df, np.zeros(len(df)), groups=pids))

    os.makedirs(OUT_ROOT, exist_ok=True)
    for k, (train_val_idx, test_idx) in enumerate(splits):
        fold_dir = os.path.join(OUT_ROOT, f"fold_{k}")
        os.makedirs(fold_dir, exist_ok=True)
        # Carve val out of train_val by patient
        tv_pids = np.unique(pids[train_val_idx])
        rng.shuffle(tv_pids)
        n_val = max(1, int(round(VAL_FRACTION * len(tv_pids))))
        val_pids = set(tv_pids[:n_val])
        train_mask = df.index.isin(train_val_idx) & ~df["patient_id"].isin(val_pids)
        val_mask = df.index.isin(train_val_idx) & df["patient_id"].isin(val_pids)
        test_mask = df.index.isin(test_idx)
        train_df = df[train_mask].reset_index(drop=True)
        val_df = df[val_mask].reset_index(drop=True)
        test_df = df[test_mask].reset_index(drop=True)

        # Leakage check
        sets = {"train": set(train_df["patient_id"]),
                "val":   set(val_df["patient_id"]),
                "test":  set(test_df["patient_id"])}
        for a in sets:
            for b in sets:
                if a < b and sets[a] & sets[b]:
                    raise SystemExit(f"LEAKAGE fold {k}: {a} ∩ {b} has "
                                     f"{len(sets[a] & sets[b])} patients")

        # Drop patient_id before writing so the schema matches what train_mil_v21 expects.
        for d in (train_df, val_df, test_df):
            d.drop(columns=["patient_id"], inplace=True)
        train_df.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(fold_dir, "test.csv"), index=False)
        print(f"fold {k}: train={len(train_df)} ({len(sets['train'])}p) "
              f"val={len(val_df)} ({len(sets['val'])}p) "
              f"test={len(test_df)} ({len(sets['test'])}p)  | leakage check OK")
    print("WROTE ->", OUT_ROOT)


if __name__ == "__main__":
    main()
