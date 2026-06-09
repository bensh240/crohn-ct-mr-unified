"""
make_master_table.py - Phase 9 V21 cohort assembly (read-only on the DB)
========================================================================
Builds Phase9_V21/data/v21_master_table.csv: one row per CT/MR study, with
patient_id (via reports), center (outcomes.source), study date, outcome flag,
and slice metadata. label_10d is left as a placeholder (NULL) until HSMP-BERT
labels arrive.

Read-only: opens the DB read-only (mode=ro). Never writes the DB.

Usage (on Argus01):
  python make_master_table.py --out Phase9_V21/data/v21_master_table.csv
"""

import os
import argparse
import sqlite3
import csv as _csv
from collections import Counter

DB_PATH = os.environ.get("V21_DB_PATH", "./epiirn_v0.0.6.db")


def connect_ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def build(db_path, out_path):
    conn = connect_ro(db_path)
    cur = conn.cursor()

    # One row per study (distinct accession × modality). dicoms is slice-level,
    # so aggregate slice metadata. patient_id from reports; center from outcomes.
    query = """
        SELECT d.AccessionNumber                AS accession_number,
               d.Modality                       AS modality,
               r.patient_id                     AS patient_id,
               o.source                         AS center,
               MIN(d.StudyDate)                 AS study_date,
               MAX(CASE WHEN o.accession_number IS NOT NULL THEN 1 ELSE 0 END) AS has_outcome,
               AVG(d.SliceThickness)            AS mean_slice_thickness,
               MAX(d.NumberOfSlices)            AS max_slices,
               COUNT(*)                         AS n_series_rows
        FROM dicoms d
        LEFT JOIN reports  r ON d.AccessionNumber = r.accession_number
        LEFT JOIN outcomes o ON d.AccessionNumber = o.accession_number
        WHERE d.Modality IN ('CT', 'MR')
        GROUP BY d.AccessionNumber, d.Modality
    """
    rows = cur.execute(query).fetchall()
    cols = [c[0] for c in cur.description]
    conn.close()

    with open(out_path, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow(cols + ["label_10d"])  # placeholder column for HSMP-BERT labels
        for row in rows:
            w.writerow(list(row) + [""])

    # --- summary ---
    mod_idx, ctr_idx, pid_idx, out_idx = (cols.index(c) for c in
                                          ("modality", "center", "patient_id", "has_outcome"))
    by_mod = Counter(r[mod_idx] for r in rows)
    by_mod_center = Counter((r[mod_idx], r[ctr_idx]) for r in rows)
    with_pid = Counter(r[mod_idx] for r in rows if r[pid_idx] is not None)
    with_outcome = Counter(r[mod_idx] for r in rows if r[out_idx] == 1)
    patients = set(r[pid_idx] for r in rows if r[pid_idx] is not None)

    print(f"Wrote {len(rows)} studies -> {out_path}")
    print(f"\nBy modality:        {dict(by_mod)}")
    print(f"With patient_id:    {dict(with_pid)}  (distinct patients: {len(patients)})")
    print(f"With outcome:       {dict(with_outcome)}")
    print("\nBy modality × center:")
    for (mod, ctr), n in sorted(by_mod_center.items(), key=lambda x: (-x[1])):
        print(f"  {mod:3s} {str(ctr):12s} {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--out", default="./data/data/v21_master_table.csv")
    args = ap.parse_args()
    build(args.db, args.out)


if __name__ == "__main__":
    main()
