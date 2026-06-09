"""
make_splits.py - Phase 9 V21 patient-level train/val/test split
================================================================
CRITICAL: we warm-start V21 from V20, which trained on the existing Phase-6
split (multilabel_data_v4). So the V21 split MUST INHERIT that split — any
patient V20 saw in train must never land in V21 val/test, or the warm-started
model evaluates on data it has already seen (leakage).

Logic:
  1. Inherit: every accession already in v4 {train,val,test} keeps its split.
  2. Patient inheritance: a NEW study whose patient_id already has a split
     (from step 1) joins that same split (keeps the 60 CT+MR patients together).
  3. New patients: deterministic patient-level hash -> ~80/10/10.
  4. CT studies WITHOUT patient_id -> TRAIN ONLY (never val/test) — can't be
     leakage-checked, and CT has no outcomes anyway (Stage-1 supervision only).
  5. Assert zero leakage: no patient_id appears in more than one split.

Read-only inputs. Writes only Phase9_V21/data/splits/.

Usage (Argus01):
  python make_splits.py --master Phase9_V21/data/v21_master_table.csv \
      --existing-dir ./data/multilabel_data_v4 \
      --out-dir ./data/data/splits
"""

import os
import csv
import argparse
import hashlib
from collections import Counter, defaultdict

VAL_FRAC, TEST_FRAC = 0.10, 0.10  # train = 1 - val - test


def patient_bucket(patient_id, seed="v21"):
    """Deterministic, salted patient-level hash -> 'train' | 'val' | 'test'."""
    h = hashlib.md5(f"{seed}:{patient_id}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF       # uniform in [0, 1)
    if frac < TEST_FRAC:
        return "test"
    if frac < TEST_FRAC + VAL_FRAC:
        return "val"
    return "train"


def load_existing_split(existing_dir):
    """accession_number -> split, from the v4 CSVs V19/V20 trained on."""
    acc_split = {}
    for split in ("train", "val", "test"):
        path = os.path.join(existing_dir, f"{split}_multilabel.csv")
        with open(path) as f:
            for row in csv.DictReader(f):
                acc_split[row["accession_number"]] = split
    return acc_split


def load_master(master_path):
    with open(master_path) as f:
        return list(csv.DictReader(f))


def assign_splits(master_rows, acc_split):
    # --- step 1+2: build patient -> split from inherited accessions ---
    # v4's split is NOT perfectly patient-level (some patients span splits). Resolve
    # conflicts by PRIORITY train>val>test, so a patient V20 saw in train never lands
    # in V21 val/test (warm-start leakage safety).
    PRIO = {"train": 0, "val": 1, "test": 2}
    patient_split = {}
    conflicts = 0
    for r in master_rows:
        acc, pid = r["accession_number"], (r.get("patient_id") or "").strip()
        if acc in acc_split and pid:
            s = acc_split[acc]
            if pid in patient_split:
                if patient_split[pid] != s:
                    conflicts += 1
                    if PRIO[s] < PRIO[patient_split[pid]]:
                        patient_split[pid] = s   # keep highest-priority (train wins)
            else:
                patient_split[pid] = s

    # --- step 3: new patients get a deterministic bucket (patient-level) ---
    for r in master_rows:
        pid = (r.get("patient_id") or "").strip()
        if pid and pid not in patient_split:
            patient_split[pid] = patient_bucket(pid)

    # --- final per-study assignment ---
    out = {"train": [], "val": [], "test": []}
    no_pid_ct_to_train = 0
    for r in master_rows:
        acc = r["accession_number"]
        pid = (r.get("patient_id") or "").strip()
        if pid and pid in patient_split:     # patient-level split (leakage-free)
            split = patient_split[pid]
        elif acc in acc_split:               # has v4 split but no pid (rare)
            split = acc_split[acc]
        else:                                # no patient_id
            if r["modality"] == "CT":        # step 4: CT w/o pid -> train only
                split = "train"
                no_pid_ct_to_train += 1
            else:
                continue                     # MR w/o pid: skip (unsafe to place)
        out[split].append(r)
    return out, patient_split, conflicts, no_pid_ct_to_train


def assert_no_leakage(splits):
    pid_to_splits = defaultdict(set)
    for split, rows in splits.items():
        for r in rows:
            pid = (r.get("patient_id") or "").strip()
            if pid:
                pid_to_splits[pid].add(split)
    leaked = {pid: s for pid, s in pid_to_splits.items() if len(s) > 1}
    assert not leaked, f"LEAKAGE: {len(leaked)} patients in multiple splits, e.g. {list(leaked.items())[:3]}"
    print(f"  leakage check PASSED ({len(pid_to_splits)} patients, each in exactly one split)")


def write_splits(splits, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fields = ["accession_number", "modality", "patient_id", "center", "study_date", "has_outcome"]
    for split, rows in splits.items():
        with open(os.path.join(out_dir, f"{split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "v21_master_table.csv"))
    ap.add_argument("--existing-dir",
                    default="./data/multilabel_data_v4")
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "splits"))
    args = ap.parse_args()

    acc_split = load_existing_split(args.existing_dir)
    master = load_master(args.master)
    splits, patient_split, conflicts, ct_no_pid = assign_splits(master, acc_split)

    print(f"Inherited {len(acc_split)} accessions from {args.existing_dir}")
    if conflicts:
        print(f"  WARNING: {conflicts} patient/split conflicts in inherited data (Phase-6 not fully patient-level?)")
    print(f"  CT without patient_id forced to train: {ct_no_pid}")
    assert_no_leakage(splits)

    print("\nSplit sizes (studies):")
    for split in ("train", "val", "test"):
        rows = splits[split]
        by_mod = Counter(r["modality"] for r in rows)
        print(f"  {split:5s}: {len(rows):5d}  {dict(by_mod)}")

    write_splits(splits, args.out_dir)
    print(f"\nWrote splits -> {args.out_dir}")


if __name__ == "__main__":
    main()
