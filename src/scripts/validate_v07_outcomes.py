#!/usr/bin/env python
"""
Validation gauntlet for v0.0.7 outcomes before kicking off Stage 2.

Checks (any RED -> stop and fix):
  1. tables present (outcomes, dicoms, reports)
  2. CT-with-outcomes counts match what Angeleene reported (1821)
  3. event counts per outcome (post-imaging time_to_X.0 > 0)
  4. patient_id coverage on outcomes
  5. patient-level leakage check (no patient straddles MR<->CT outcomes)
  6. clinical features needed for Stage 2 are present and not totally null on CT
  7. censoring distribution looks sane (no impossible negative durations)
  8. time-to-event units look sane (years, not days or seconds)
  9. overlap with our trained Phase 9 cohort (the cohort we have A2 features for)

Usage:
  python validate_v07_outcomes.py
"""
import os
import sqlite3
import sys
import pandas as pd

DB = os.environ.get("V21_DB_PATH", "./epiirn_v0.0.7.db")
COHORT_CSVS = [
    os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "train.csv"),
    os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "val.csv"),
    os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "test.csv"),
]
MASTER_TABLE = os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "v21_master_table.csv")

# Raw clinical features in the v0.0.7 outcomes table (the "had_X_before" features
# are derived by make_master_table from time_to_X.0 < 0; not in the DB).
CLINICAL_FEATURES = [
    "Time_to_index", "age_at_diagnosis", "CCI_at_diagnosis", "disease_activity_5",
    "SES_points", "diagnostic_delay", "sex",
]
# Derived flags computed downstream (sanity-check via the signed time-to columns).
DERIVED_FLAGS_FROM = ["time_to_1.0", "time_to_2.0", "time_to_3.0"]


def red(msg):  print(f"  RED  {msg}")
def green(msg): print(f"  OK   {msg}")
def warn(msg):  print(f"  WARN {msg}")


def main():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    fatal = 0

    print("=" * 70)
    print(f"V0.0.7 VALIDATION GAUNTLET")
    print(f"  DB: {DB}")
    print("=" * 70)

    # 1. tables present
    print("\n[1] tables present")
    tabs = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in ("outcomes", "dicoms", "reports"):
        if t in tabs: green(f"table '{t}' present")
        else: red(f"table '{t}' MISSING"); fatal += 1

    # 2. CT with outcomes
    print("\n[2] CT with outcomes")
    n_ct_outc = cur.execute(
        "SELECT COUNT(DISTINCT o.accession_number) FROM outcomes o "
        "INNER JOIN dicoms d ON d.AccessionNumber=o.accession_number WHERE d.Modality=?",
        ("CT",)).fetchone()[0]
    if n_ct_outc >= 1800:
        green(f"CT studies with outcome row = {n_ct_outc} (Angeleene reported 1821)")
    else:
        red(f"CT with outcome row = {n_ct_outc}, expected ~1821"); fatal += 1

    # 3. event counts per outcome
    print("\n[3] events per outcome (post-imaging: time_to_X.0 > 0)")
    counts = {}
    for X, name in [(1, "surgery"), (2, "steroid"), (3, "biologic")]:
        col = f"time_to_{X}.0"
        for mod in ("CT", "MR"):
            ev = cur.execute(
                f'SELECT COUNT(DISTINCT o.accession_number) FROM outcomes o '
                f'INNER JOIN dicoms d ON d.AccessionNumber=o.accession_number '
                f'WHERE d.Modality=? AND o."{col}" > 0', (mod,)).fetchone()[0]
            counts[(mod, name)] = ev
            print(f"  {mod} {name:8s} events = {ev}")
    if counts[("CT", "surgery")] >= 100 and counts[("CT", "biologic")] >= 200:
        green("CT event counts look real (surgery >= 100, biologic >= 200)")
    else:
        warn("CT event counts smaller than expected from earlier query")

    # 4. patient_id coverage on CT-with-outcomes (join through reports OR dicoms.PatientID)
    print("\n[4] patient_id coverage on CT-with-outcomes (via reports.patient_id or dicoms.PatientID)")
    n_via_reports = cur.execute(
        "SELECT COUNT(DISTINCT o.accession_number) FROM outcomes o "
        "INNER JOIN dicoms d ON d.AccessionNumber=o.accession_number "
        "INNER JOIN reports r ON r.accession_number=o.accession_number "
        "WHERE d.Modality='CT' AND r.patient_id IS NOT NULL AND r.patient_id <> ''"
    ).fetchone()[0]
    n_via_dicoms = cur.execute(
        "SELECT COUNT(DISTINCT o.accession_number) FROM outcomes o "
        "INNER JOIN dicoms d ON d.AccessionNumber=o.accession_number "
        "WHERE d.Modality='CT' AND d.PatientID IS NOT NULL AND d.PatientID <> ''"
    ).fetchone()[0]
    total = n_ct_outc
    print(f"  via reports.patient_id : {n_via_reports} / {total}")
    print(f"  via dicoms.PatientID   : {n_via_dicoms} / {total}")
    if max(n_via_reports, n_via_dicoms) >= 0.99 * total:
        green(f"sufficient patient_id coverage to do patient-level grouped CV")
    elif max(n_via_reports, n_via_dicoms) >= 0.90 * total:
        warn(f"only {max(n_via_reports,n_via_dicoms)}/{total} have patient_id; "
             f"~{total-max(n_via_reports,n_via_dicoms)} drop or use accession as fallback group")
    else:
        red(f"patient_id missing for many CT studies"); fatal += 1

    # 5. patient overlap CT/MR (need a single patient-id source; use reports.patient_id)
    print("\n[5] patient overlap MR/CT (for grouped-CV planning)")
    df_overlap = pd.read_sql(
        "SELECT DISTINCT r.patient_id AS pid, d.Modality AS modality FROM outcomes o "
        "INNER JOIN dicoms d ON d.AccessionNumber=o.accession_number "
        "INNER JOIN reports r ON r.accession_number=o.accession_number "
        "WHERE d.Modality IN ('MR','CT') AND r.patient_id IS NOT NULL AND r.patient_id <> ''",
        con)
    n_overlap = df_overlap.groupby("pid")["modality"].nunique().eq(2).sum()
    print(f"  patients with BOTH MR-outcome AND CT-outcome studies: {n_overlap}")
    if n_overlap > 0:
        warn(f"{n_overlap} patients straddle CT/MR -- grouped-CV must group BY patient, not by accession")
    else:
        green("no MR/CT patient overlap")

    # 6. clinical features present and non-null on CT
    print("\n[6] clinical features needed for Stage 2")
    cols = {r[1] for r in cur.execute("PRAGMA table_info(outcomes)")}
    missing = [c for c in CLINICAL_FEATURES if c not in cols]
    if missing:
        red(f"missing clinical columns in outcomes: {missing}"); fatal += 1
    else:
        green(f"all {len(CLINICAL_FEATURES)} clinical features present in outcomes")
        # null-rate per clinical feature on CT
        df = pd.read_sql(
            f"SELECT {','.join(repr(c) for c in CLINICAL_FEATURES)} FROM outcomes o "
            "INNER JOIN dicoms d ON d.AccessionNumber=o.accession_number "
            "WHERE d.Modality='CT'", con)
        df.columns = CLINICAL_FEATURES
        for c in CLINICAL_FEATURES:
            null_pct = 100.0 * df[c].isna().mean()
            tag = "OK  " if null_pct < 10 else "WARN" if null_pct < 50 else "RED "
            print(f"  {tag} {c:32s} null = {null_pct:5.1f}%")
            if null_pct > 50: fatal += 1

    # 7. censoring distribution sanity (durations + events)
    print("\n[7] duration/censoring sanity (CT)")
    for X, name in [(1, "surgery"), (2, "steroid"), (3, "biologic")]:
        col = f"time_to_{X}.0"
        df = pd.read_sql(
            f'SELECT o."{col}" AS t FROM outcomes o '
            "INNER JOIN dicoms d ON d.AccessionNumber=o.accession_number "
            "WHERE d.Modality='CT'", con)
        t = pd.to_numeric(df["t"], errors="coerce")
        n_pos = (t > 0).sum()
        n_zero = (t == 0).sum()
        n_neg = (t < 0).sum()
        n_null = t.isna().sum()
        tmax_pos = t[t > 0].max() if (t > 0).any() else None
        tmin_pos = t[t > 0].min() if (t > 0).any() else None
        print(f"  {name:8s}  events(>0)={n_pos:4d}  zero={n_zero:4d}  "
              f"neg(historical)={n_neg:4d}  null={n_null:5d}  "
              f"range(events)=[{tmin_pos!r}, {tmax_pos!r}]")
        if tmax_pos is not None and tmax_pos > 30:
            red(f"  {name} max duration {tmax_pos} > 30 -- units may be wrong (days? months? not years)"); fatal += 1
        elif tmax_pos is not None and tmax_pos > 0.001:
            green(f"  {name} durations look like years")

    # 8. overlap with our trained Phase 9 cohort
    print("\n[8] overlap with our trained Phase 9 cohort")
    cohort = pd.concat([pd.read_csv(p, usecols=["accession_number", "modality"])
                        for p in COHORT_CSVS])
    ct_trained = set(cohort[cohort["modality"] == "ct"]["accession_number"].astype(str))
    mr_trained = set(cohort[cohort["modality"] == "mr"]["accession_number"].astype(str))
    ct_outc_acc = set(r[0] for r in cur.execute(
        "SELECT DISTINCT o.accession_number FROM outcomes o "
        "INNER JOIN dicoms d ON d.AccessionNumber=o.accession_number WHERE d.Modality='CT'"
    ))
    mr_outc_acc = set(r[0] for r in cur.execute(
        "SELECT DISTINCT o.accession_number FROM outcomes o "
        "INNER JOIN dicoms d ON d.AccessionNumber=o.accession_number WHERE d.Modality='MR'"
    ))
    ct_int = ct_trained & ct_outc_acc
    mr_int = mr_trained & mr_outc_acc
    print(f"  CT  trained {len(ct_trained)},  with-outcomes {len(ct_outc_acc)},  in both = {len(ct_int)}")
    print(f"  MR  trained {len(mr_trained)},  with-outcomes {len(mr_outc_acc)},  in both = {len(mr_int)}")
    print(f"  Stage 2 capacity = {len(ct_int) + len(mr_int)} studies "
          f"(CT {len(ct_int)} + MR {len(mr_int)})")

    print("\n" + "=" * 70)
    if fatal == 0:
        print("RESULT: GREEN. Safe to proceed to Stage 2.")
    else:
        print(f"RESULT: {fatal} RED issue(s). STOP and fix before Stage 2.")
    print("=" * 70)
    return 0 if fatal == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
