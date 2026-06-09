#!/usr/bin/env python
"""
Generate the three remaining tables for the A2 paper:

  Table A : per-finding AUC (and sens/spec @ Youden) on MR and CT test sets
  Table B : leave-one-modality-out -- A2(CT+MR) vs A2(MR-only) on the MR test set,
            macro-AUC and per-finding, with paired bootstrap 95% CI
  Table C : surgery elastic-net Cox hazard ratios on the A2 cohort

Outputs:
  Phase9_V21/results/paper_tables/per_finding_a2.json
                                /leave_one_modality_out_a2.json
                                /hazards_a2_surgery.json
"""
import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

DATA_DIR = os.environ.get("V21_DATA_DIR", "./data")
PRED_DIR = os.environ.get("V21_PRED_DIR", "./predictions")
SURV_DIR = os.environ.get("V21_SURV_DIR", "./survival")
OUT = os.environ.get("V21_PAPER_TABLES_DIR",
                     os.path.join(os.environ.get("V21_RESULTS_DIR", "./results"), "paper_tables"))
os.makedirs(OUT, exist_ok=True)

FINDINGS = [
    ("ileum_inflammation",     "Ileal inflammation"),
    ("ileum_wall_enhancement", "Ileal wall enhancement"),
    ("ileum_wall_thickness",   "Ileal wall thickening"),
    ("ileum_dwi",              "Restricted diffusion"),
    ("ileum_stenosis",         "Ileal stenosis"),
    ("ileum_pre_stenotic_dil", "Pre-stenotic dilatation"),
    ("ileum_comb_sign",        "Comb sign"),
    ("ileum_fistula",          "Fistula"),
    ("colon_inflammation",     "Colonic inflammation"),
    ("ileum_mesenteric_edema", "Mesenteric edema"),
]


def youden_sens_spec(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j))
    return float(tpr[k]), float(1.0 - fpr[k])


def merge(pred_csv, label_csv):
    """Merge a predictions CSV with the labels test.csv on accession_number."""
    p = pd.read_csv(pred_csv)
    l = pd.read_csv(label_csv)
    return p.merge(l, on="accession_number", suffixes=("", "_lbl"))


# ----------------------------------------------------------------- Table A
def per_finding_table():
    test_csv = os.path.join(DATA_DIR, "test.csv")
    mr = merge(os.path.join(PRED_DIR, "a2_mr_test.csv"), test_csv)
    mr = mr[mr["modality"] == "mr"]
    ct = merge(os.path.join(PRED_DIR, "a2_ct_test.csv"), test_csv)
    ct = ct[ct["modality"] == "ct"]
    rows = []
    aucs_mr, aucs_ct = [], []
    for col, name in FINDINGS:
        rec = {"finding": name, "col": col}
        for tag, df in (("mr", mr), ("ct", ct)):
            y = df[col].astype(int).values
            p = df[f"pred_{col}"].astype(float).values
            if len(np.unique(y)) < 2:
                rec[f"auc_{tag}"] = None
                continue
            auc = float(roc_auc_score(y, p))
            sens, spec = youden_sens_spec(y, p)
            rec[f"auc_{tag}"] = auc
            rec[f"sens_{tag}"] = sens
            rec[f"spec_{tag}"] = spec
            rec[f"prev_{tag}"] = float(y.mean())
            (aucs_mr if tag == "mr" else aucs_ct).append(auc)
        rows.append(rec)
    macro_mr = float(np.mean(aucs_mr)) if aucs_mr else None
    macro_ct = float(np.mean(aucs_ct)) if aucs_ct else None
    out = {"per_finding": rows, "macro_auc_mr": macro_mr, "macro_auc_ct": macro_ct,
           "n_mr": int(len(mr)), "n_ct": int(len(ct))}
    json.dump(out, open(f"{OUT}/per_finding_a2.json", "w"), indent=2)
    print(f"[A] MR n={len(mr)} macro={macro_mr:.4f}  CT n={len(ct)} macro={macro_ct:.4f}")
    return out


# ----------------------------------------------------------------- Table B
def leave_one_modality_out():
    test_csv = os.path.join(DATA_DIR, "test.csv")
    joint = merge(os.path.join(PRED_DIR, "a2_mr_test.csv"), test_csv)
    joint = joint[joint["modality"] == "mr"].set_index("accession_number")
    mron = merge(os.path.join(PRED_DIR, "a2mronly_mr_test.csv"), test_csv)
    mron = mron[mron["modality"] == "mr"].set_index("accession_number")
    common = joint.index.intersection(mron.index)
    joint, mron = joint.loc[common].copy(), mron.loc[common].copy()
    n = len(common)
    per = []
    j_aucs, m_aucs = [], []
    for col, name in FINDINGS:
        y = joint[col].astype(int).values
        if len(np.unique(y)) < 2:
            per.append({"finding": name, "auc_joint": None, "auc_mronly": None,
                        "delta": None}); continue
        pj = joint[f"pred_{col}"].astype(float).values
        pm = mron[f"pred_{col}"].astype(float).values
        aj, am = float(roc_auc_score(y, pj)), float(roc_auc_score(y, pm))
        j_aucs.append(aj); m_aucs.append(am)
        per.append({"finding": name, "auc_joint": aj, "auc_mronly": am,
                    "delta": aj - am})
    macro_j = float(np.mean(j_aucs)); macro_m = float(np.mean(m_aucs))
    delta = macro_j - macro_m

    # Paired bootstrap on macro-AUC (resample accessions)
    rng = np.random.default_rng(42)
    B = 1000
    diffs = np.empty(B)
    accs = list(common)
    y_mat = {c: joint[c].astype(int).values for c, _ in FINDINGS}
    pj_mat = {c: joint[f"pred_{c}"].astype(float).values for c, _ in FINDINGS}
    pm_mat = {c: mron[f"pred_{c}"].astype(float).values for c, _ in FINDINGS}
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        ja, ma = [], []
        for col, _ in FINDINGS:
            y = y_mat[col][idx]
            if len(np.unique(y)) < 2: continue
            ja.append(roc_auc_score(y, pj_mat[col][idx]))
            ma.append(roc_auc_score(y, pm_mat[col][idx]))
        if ja and ma:
            diffs[b] = np.mean(ja) - np.mean(ma)
        else:
            diffs[b] = np.nan
    diffs = diffs[~np.isnan(diffs)]
    ci_lo, ci_hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
    out = {"n_mr_test": int(n), "per_finding": per,
           "macro_auc_joint": macro_j, "macro_auc_mronly": macro_m,
           "delta_joint_minus_mronly": delta, "ci95": [ci_lo, ci_hi],
           "p_joint_better_pct": float(100.0 * (diffs > 0).mean()),
           "n_bootstrap": int(B)}
    json.dump(out, open(f"{OUT}/leave_one_modality_out_a2.json", "w"), indent=2)
    print(f"[B] joint {macro_j:.4f} vs MR-only {macro_m:.4f}  Delta={delta:+.4f}  "
          f"CI95[{ci_lo:+.4f},{ci_hi:+.4f}]  p(joint>MRonly)={out['p_joint_better_pct']:.0f}%")
    return out


# ----------------------------------------------------------------- Table C
def hazard_ratios():
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv
    df = pd.read_csv(os.path.join(SURV_DIR, "survival_a2.csv"))
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    clin_cols = [c for c in df.columns if c.startswith("clinical_")
                 and c != "clinical_has_clinical"]  # has_clinical kept too
    feats = pred_cols + [c for c in df.columns if c.startswith("clinical_")]
    df[feats] = df[feats].apply(pd.to_numeric, errors="coerce")
    df[feats] = df[feats].fillna(df[feats].median(numeric_only=True))
    feats = [c for c in feats if df[c].std() > 1e-8]
    Xs = (df[feats] - df[feats].mean()) / df[feats].std()
    sub = df[df["duration_surgery"] > 0].copy()
    X = Xs.loc[sub.index]
    y = Surv.from_arrays(event=sub["event_surgery"].astype(bool).values,
                         time=sub["duration_surgery"].values)
    # path with 100 alphas, l1_ratio=0.5; pick alpha with 8-15 non-zero features
    cox = CoxnetSurvivalAnalysis(l1_ratio=0.5, n_alphas=100, alpha_min_ratio=0.01)
    cox.fit(X, y)
    chosen, picked_nnz = None, None
    for ai, a in enumerate(cox.alphas_):
        coefs = cox.coef_[:, ai]
        nnz = int((np.abs(coefs) > 1e-6).sum())
        if 8 <= nnz <= 15:
            chosen, picked_nnz = ai, nnz
            break
    if chosen is None:
        # fall back: pick alpha closest to 10 non-zero
        nnz_arr = (np.abs(cox.coef_) > 1e-6).sum(axis=0)
        chosen = int(np.argmin(np.abs(nnz_arr - 10)))
        picked_nnz = int(nnz_arr[chosen])
    coefs = cox.coef_[:, chosen]
    nonzero = [(feats[i], float(coefs[i])) for i in range(len(feats))
               if abs(coefs[i]) > 1e-6]
    nonzero.sort(key=lambda kv: abs(kv[1]), reverse=True)

    def src(name):
        return "MRI (MIL)" if name.startswith("pred_") else "Clinical"

    def pretty(name):
        # strip prefix; convert underscores to spaces
        s = name.split("_", 1)[1] if "_" in name else name
        s = s.replace("ileum_", "").replace("_", " ")
        return s
    table = [{"feature": pretty(n), "source": src(n), "coef": c,
              "hr": float(np.exp(c)),
              "direction": "Risk up" if c > 0 else "Risk down"}
             for n, c in nonzero]
    out = {"alpha": float(cox.alphas_[chosen]), "n_nonzero": picked_nnz,
           "n_total_features": int(len(feats)),
           "n_events": int(sub["event_surgery"].sum()),
           "n_samples": int(len(sub)),
           "table": table}
    json.dump(out, open(f"{OUT}/hazards_a2_surgery.json", "w"), indent=2)
    print(f"[C] alpha={cox.alphas_[chosen]:.5g}  non-zero={picked_nnz}/{len(feats)}")
    for r in table[:10]:
        print(f"    HR={r['hr']:.3f}  {r['source']:12s}  {r['feature']}")
    return out


if __name__ == "__main__":
    print("=== A: per-finding A2 AUC ===");          per_finding_table()
    print("=== B: leave-one-modality-out ===");      leave_one_modality_out()
    print("=== C: surgery hazard ratios ===");       hazard_ratios()
    print("DONE ->", OUT)
