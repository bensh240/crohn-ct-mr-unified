#!/usr/bin/env python
"""
build_survival_v07.py - Stage 2 unified CT+MR survival CSV from v0.0.7.

For every study (CT or MR) with A2 predictions AND a v0.0.7 outcome row:
  accession_number, patient_id, modality, center,
  pred_<10 labels>,
  clinical_<13 features>  (7 raw + 5 "had_X_before" derived from time_to_X.0 < 0
                           + sex),
  duration_<surgery|steroid|biologic>, event_<surgery|steroid|biologic>
  has_clinical = 1 if any clinical feature is non-null

Convention (Phase 7 / v21):
  event=1 and duration=time_to_X.0  if time_to_X.0 > 0   (event AFTER imaging)
  event=0 and duration=MAX_FU (10 yr)  otherwise (NULL or <= 0 = censored / pre-imaging)

  "had_X_before" = 1 if time_to_X.0 < 0 (event happened before imaging)
"""
import os
import argparse
import csv
import sqlite3
import sys

DB_DEFAULT = os.environ.get("V21_DB_PATH", "./epiirn_v0.0.7.db")
MAX_FU = 10.0

LABELS = ["ileum_inflammation", "ileum_wall_enhancement", "ileum_wall_thickness",
          "ileum_dwi", "ileum_stenosis", "ileum_pre_stenotic_dil", "ileum_comb_sign",
          "ileum_fistula", "colon_inflammation", "ileum_mesenteric_edema"]
CONT_CLIN = ["Time_to_index", "age_at_diagnosis", "CCI_at_diagnosis",
             "disease_activity_5", "SES_points", "diagnostic_delay"]
HIST_CLIN = ["had_surgery_before", "had_steroid_dep_before",
             "had_biologic_switch_before", "had_EIM_before", "had_perianal_before"]
CLINICAL_OUT = CONT_CLIN + HIST_CLIN + ["sex", "has_clinical"]
OUTCOMES = [("surgery", "1"), ("steroid", "2"), ("biologic", "3")]


def num(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_preds(paths):
    """Read one or more prediction CSVs; keep only rows whose modality matches the
    expected modality of the file. Returns {acc: {label: prob, modality: ...}}."""
    preds = {}
    for path, expect_mod in paths:
        with open(path) as f:
            for r in csv.DictReader(f):
                mod = r.get("modality", expect_mod).lower()
                if expect_mod and mod != expect_mod:
                    continue
                acc = r["accession_number"]
                preds[acc] = {"modality": mod}
                for L in LABELS:
                    preds[acc][f"pred_{L}"] = num(r.get(f"pred_{L}"))
    return preds


def load_outcomes(db):
    """Return {acc: row_dict} from outcomes joined with reports.patient_id and
    dicoms.Modality/InstitutionName."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    cur = con.cursor()

    selects = [
        ("o.accession_number", "accession_number"),
        ("COALESCE(r.patient_id, d.PatientID)", "patient_id"),
        ("o.source", "source"),
        ("d.InstitutionName", "center"),
        ("o.sex", "sex"),
    ]
    for c in CONT_CLIN:
        selects.append((f"o.{c}", c))
    for c, code in OUTCOMES:
        selects.append((f'o."time_to_{code}.0"', f"time_to_{code}.0"))

    expr = ", ".join(f'{s} AS "{n}"' for s, n in selects)
    # one center per accession (dicoms can have many series rows per study)
    sql = (
        f"SELECT {expr} FROM outcomes o "
        "LEFT JOIN reports r ON r.accession_number = o.accession_number "
        "LEFT JOIN (SELECT AccessionNumber, MIN(InstitutionName) AS InstitutionName, "
        "                 MIN(PatientID) AS PatientID, MIN(Modality) AS Modality "
        "          FROM dicoms GROUP BY AccessionNumber) d "
        "       ON d.AccessionNumber = o.accession_number"
    )
    rows = {}
    for r in cur.execute(sql).fetchall():
        d = {n: v for (_, n), v in zip(selects, r)}
        rows[d["accession_number"]] = d
    return rows


def derive(time_val):
    v = num(time_val)
    if v is not None and v > 0:
        return v, 1, 0
    had_before = 1 if (v is not None and v < 0) else 0
    return MAX_FU, 0, had_before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-mr", required=True, help="A2 predictions on MR (a2_mr_all.csv)")
    ap.add_argument("--preds-ct", required=True, help="A2 predictions on CT (a2_ct_all.csv)")
    ap.add_argument("--db", default=DB_DEFAULT)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    preds = load_preds([(args.preds_mr, "mr"), (args.preds_ct, "ct")])
    outcomes = load_outcomes(args.db)

    rows = []
    skipped_no_outcome = 0
    skipped_no_clin = 0
    for acc, p in preds.items():
        if acc not in outcomes:
            skipped_no_outcome += 1
            continue
        o = outcomes[acc]

        out = {"accession_number": acc, "patient_id": o.get("patient_id"),
               "modality": p["modality"],
               "center": (o.get("center") or "Unknown").strip()}
        for L in LABELS:
            out[f"pred_{L}"] = p.get(f"pred_{L}")

        clinical_vals = {}
        for c in CONT_CLIN:
            clinical_vals[c] = num(o.get(c))
        # derived "had_X_before" + duration/event
        had = {}
        durs = {}
        evs = {}
        for outc, code in OUTCOMES:
            dur, ev, hb = derive(o.get(f"time_to_{code}.0"))
            had[outc] = hb
            durs[outc] = dur
            evs[outc] = ev
        clinical_vals["had_surgery_before"] = had["surgery"]
        clinical_vals["had_steroid_dep_before"] = had["steroid"]
        clinical_vals["had_biologic_switch_before"] = had["biologic"]
        clinical_vals["had_EIM_before"] = 1 if num(o.get("Time_to_first_EIM_indication"
                                                         if "Time_to_first_EIM_indication" in o else "")) else 0
        clinical_vals["had_perianal_before"] = 1 if num(o.get("Time_to_first_perianal_indication"
                                                              if "Time_to_first_perianal_indication" in o else "")) else 0
        sex_raw = o.get("sex")
        clinical_vals["sex"] = 1 if (str(sex_raw or "").lower() in ("m", "male", "1")) else 0
        # has_clinical: any of the continuous features is non-null
        clinical_vals["has_clinical"] = int(any(clinical_vals[c] is not None for c in CONT_CLIN))

        for c in CLINICAL_OUT:
            out[f"clinical_{c}"] = clinical_vals.get(c, 0) or 0
        for outc in ("surgery", "steroid", "biologic"):
            out[f"duration_{outc}"] = durs[outc]
            out[f"event_{outc}"] = evs[outc]
        rows.append(out)

    # filter rows missing patient_id (can't do grouped CV without it)
    rows_with_pid = [r for r in rows if r.get("patient_id") not in (None, "", "None")]
    n_drop_pid = len(rows) - len(rows_with_pid)

    cols = (["accession_number", "patient_id", "modality", "center"]
            + [f"pred_{L}" for L in LABELS]
            + [f"clinical_{c}" for c in CLINICAL_OUT]
            + [f"duration_{o}" for o, _ in OUTCOMES]
            + [f"event_{o}" for o, _ in OUTCOMES])

    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows_with_pid:
            w.writerow({c: r.get(c, "") for c in cols})

    # quick report
    from collections import Counter
    mods = Counter(r["modality"] for r in rows_with_pid)
    print(f"loaded predictions: {len(preds)}")
    print(f"loaded outcomes:    {len(outcomes)}")
    print(f"matched + had patient_id -> wrote: {len(rows_with_pid)}")
    print(f"  by modality: {dict(mods)}")
    print(f"  skipped (no outcome row): {skipped_no_outcome}")
    print(f"  dropped (no patient_id):  {n_drop_pid}")
    for outc in ("surgery", "steroid", "biologic"):
        for mod in ("mr", "ct"):
            n_ev = sum(1 for r in rows_with_pid if r["modality"] == mod and r[f"event_{outc}"] == 1)
            n_tot = mods[mod]
            print(f"  {mod.upper():2s} {outc:8s} events {n_ev:4d} / {n_tot}")
    print(f"WROTE -> {args.output}")


if __name__ == "__main__":
    main()
