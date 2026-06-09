"""
build_v21_dataset.py - assemble V21 train/val/test CSVs (CT + MR)
================================================================
Produces Phase9_V21/data/{train,val,test}.csv in the schema train_mil_v21.py reads:
  accession_number, modality, image_dir, t1_dir, is_sick, <10 labels>, <13 clinical>

Two sources, joined per the patient-level split:
  MR  -> REUSE the existing multilabel_data_v4 CSVs (the exact 6,223 MR scans V19/V20
         trained on -> consistency + valid warm-start). Add modality=mr, t1_dir.
  CT  -> BUILD fresh: labels from hsmp_bert_labels.csv (dedup, binarize>=1),
         clinical/survival-derived from outcomes (DB), image_dir = ct_slices_seg/<acc>.
         Only CT studies that already have 16 PNG slices are included.

Split assignment:
  MR keeps its v4 split (train/val/test file it came from).
  CT uses Phase9_V21/data/splits/{train,val,test}.csv (make_splits.py output;
  patient-level, inherits v4, CT-without-pid -> train).

Naming: join key is accession_number (csv/outcomes) vs AccessionNumber (dicoms).

Usage (Argus01):
  python build_v21_dataset.py            # builds all three CSVs + prints a report
"""

import os
import csv
import sqlite3
import argparse
from collections import defaultdict

DB = os.environ.get("V21_DB_PATH", "./epiirn_v0.0.6.db")   # switch to v0.0.7 if/when labels move into a DB
V4_DIR = "./data/multilabel_data_v4"
T1_DIR = "./data/t1_images"
LABELS_CSV = os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "hsmp_bert_labels.csv")
CT_SLICES = os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "../ct_slices_seg")
SPLITS_DIR = os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "splits")
OUT_DIR = os.environ.get("V21_DATA_DIR", "./data")

# 10 findings, in the canonical order the model expects (= v4 column order).
LABELS = ["ileum_inflammation", "ileum_wall_enhancement", "ileum_wall_thickness",
          "ileum_dwi", "ileum_stenosis", "ileum_pre_stenotic_dil", "ileum_comb_sign",
          "ileum_fistula", "colon_inflammation", "ileum_mesenteric_edema"]
# map canonical -> HSMP-BERT CSV column name (new naming)
CSV_COL = {
    "ileum_inflammation": "Ileum_Inflam_0_1_2_3",
    "ileum_wall_enhancement": "Ileum_wall_enhancement_0_1",
    "ileum_wall_thickness": "Ileum_wall_thickness_0_1",
    "ileum_dwi": "Ileum_dwi_0_1",
    "ileum_stenosis": "Ileum_stenosis_0_1",
    "ileum_pre_stenotic_dil": "Ileum_pre_stenotic_dil_0_1",
    "ileum_comb_sign": "Ileum_combsign_0_1",
    "ileum_fistula": "Ileum_fistula_0_1",
    "colon_inflammation": "Colon_Inflam_0_1_2_3",
    "ileum_mesenteric_edema": "Ileum_mesenteric_edemaORfat_stranding_0_1",
}
CONT_CLIN = ["Time_to_index", "age_at_diagnosis", "CCI_at_diagnosis",
             "disease_activity_5", "SES_points", "diagnostic_delay"]
HIST_CLIN = ["had_surgery_before", "had_steroid_dep_before", "had_biologic_switch_before",
             "had_EIM_before", "had_perianal_before"]
CLINICAL = CONT_CLIN + HIST_CLIN + ["sex", "has_clinical"]
OUT_FIELDS = ["accession_number", "modality", "image_dir", "t1_dir", "is_sick"] + LABELS + CLINICAL


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def load_labels_dedup(path):
    """accession_number -> {label: 0/1}, aggregated by MAX (any positive) across dup rows."""
    agg = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            acc = row["accession_number"]
            if acc not in agg:                   # include ALL accessions, incl. all-negative (healthy)
                agg[acc] = {l: 0 for l in LABELS}
            for l in LABELS:
                if _num(row.get(CSV_COL[l]), 0.0) >= 1:   # binarize >=1 (data already 0/1)
                    agg[acc][l] = 1
    return agg


def load_ct_clinical(db):
    """accession_number -> 13-d clinical dict, from outcomes. has_clinical=1 if present."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = conn.cursor()
    cols = ["accession_number", "sex"] + CONT_CLIN + [
        "Time_to_first_EIM_indication", "Time_to_first_perianal_indication",
        '"time_to_1.0"', '"time_to_2.0"', '"time_to_3.0"']
    rows = cur.execute(f"SELECT {','.join(cols)} FROM outcomes").fetchall()
    conn.close()
    out = {}
    for r in rows:
        d = dict(zip([c.strip('"') for c in cols], r))
        clin = {c: _num(d.get(c)) for c in CONT_CLIN}
        clin["sex"] = 1.0 if str(d.get("sex")).strip().upper().startswith("M") else 0.0
        # history flags: event occurred BEFORE the MRI/CT (time_to_*.0 < 0)
        clin["had_surgery_before"] = 1.0 if _num(d.get("time_to_1.0"), 1) < 0 else 0.0
        clin["had_steroid_dep_before"] = 1.0 if _num(d.get("time_to_2.0"), 1) < 0 else 0.0
        clin["had_biologic_switch_before"] = 1.0 if _num(d.get("time_to_3.0"), 1) < 0 else 0.0
        clin["had_EIM_before"] = 1.0 if _num(d.get("Time_to_first_EIM_indication"), 1) < 0 else 0.0
        clin["had_perianal_before"] = 1.0 if _num(d.get("Time_to_first_perianal_indication"), 1) < 0 else 0.0
        clin["has_clinical"] = 1.0
        out[d["accession_number"]] = clin
    return out


def load_ct_splits(splits_dir):
    """accession_number -> split, for CT rows (from make_splits.py output)."""
    acc_split = {}
    for split in ("train", "val", "test"):
        p = os.path.join(splits_dir, f"{split}.csv")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                if row.get("modality", "").upper() == "CT":
                    acc_split[row["accession_number"]] = split
    return acc_split


def mr_rows_from_v4():
    """Reuse v4 CSVs as the MR arm. Returns {split: [row, ...]}."""
    out = {"train": [], "val": [], "test": []}
    for split in out:
        p = os.path.join(V4_DIR, f"{split}_multilabel.csv")
        with open(p) as f:
            for r in csv.DictReader(f):
                acc = r["accession_number"]
                t1 = os.path.join(T1_DIR, acc)
                row = {"accession_number": acc, "modality": "mr",
                       "image_dir": r["image_dir"],
                       "t1_dir": t1 if os.path.isdir(t1) else "",
                       "is_sick": r["is_sick"]}
                for l in LABELS:
                    row[l] = r[l]
                for c in CLINICAL:
                    row[c] = r.get(c, 0)
                out[split].append(row)
    return out


def ct_rows(labels, clinical, ct_split):
    """Build CT rows for accessions that have slices + labels. Returns {split: [row]}."""
    out = {"train": [], "val": [], "test": []}
    have_slices = set(d for d in os.listdir(CT_SLICES)) if os.path.isdir(CT_SLICES) else set()
    n_no_slices = n_no_label = n_no_split = 0
    for acc in labels:
        if acc not in have_slices:
            n_no_slices += 1
            continue
        d = os.path.join(CT_SLICES, acc)
        if len([f for f in os.listdir(d) if f.endswith(".png")]) < 16:
            n_no_slices += 1
            continue
        split = ct_split.get(acc)
        if split is None:
            n_no_split += 1
            continue
        lab = labels[acc]
        clin = clinical.get(acc, {c: 0.0 for c in CLINICAL})
        row = {"accession_number": acc, "modality": "ct", "image_dir": d, "t1_dir": "",
               "is_sick": int(any(lab[l] for l in LABELS))}
        for l in LABELS:
            row[l] = lab[l]
        for c in CLINICAL:
            row[c] = clin.get(c, 0.0)
        out[split].append(row)
    print(f"  CT skipped: no_slices(yet)={n_no_slices}, no_split={n_no_split}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading labels (dedup)...")
    labels = load_labels_dedup(LABELS_CSV)
    print(f"  labels for {len(labels)} accessions")
    print("Loading CT clinical from outcomes...")
    clinical = load_ct_clinical(DB)
    ct_split = load_ct_splits(SPLITS_DIR)
    print(f"  CT split assignments: {len(ct_split)}")

    mr = mr_rows_from_v4()
    ct = ct_rows(labels, clinical, ct_split)

    for split in ("train", "val", "test"):
        rows = mr[split] + ct[split]
        with open(os.path.join(args.out_dir, f"{split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUT_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        nmr, nct = len(mr[split]), len(ct[split])
        sick = sum(int(_num(r["is_sick"])) for r in rows)
        print(f"  {split:5s}: {len(rows):5d}  (MR={nmr}, CT={nct})  is_sick={sick}")
    print(f"\nWrote train/val/test.csv -> {args.out_dir}")


if __name__ == "__main__":
    main()
