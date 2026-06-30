"""
decision_curve.py - Decision Curve Analysis (net benefit) for the survival models
=================================================================================
Censoring-aware survival DCA (Vickers, with Kaplan-Meier within the treat-positive
group). For a horizon t and a grid of threshold probabilities p_t, compares:
  - model (combined)         : treat patients whose predicted t-year risk >= p_t
  - model (clinical_only)    : same, clinical features only
  - treat-all / treat-none   : reference strategies

Net benefit (censoring-adjusted):
  NB(p_t) = f * (1 - S_t_pos) - f * S_t_pos * p_t/(1-p_t)
  where f = fraction with predicted risk >= p_t, S_t_pos = KM survival at t in that group.

CPU only (sksurv). Reuses the same Cox setup as fit_cox_on_predictions.py.

Usage:
  python decision_curve.py --survival-csv survival_v21c.csv --outcome surgery --horizon 5.0 \
     --output results/dca_v21c_surgery.json
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator

MIL_LABELS = ["ileum_inflammation", "ileum_wall_enhancement", "ileum_wall_thickness",
              "ileum_dwi", "ileum_stenosis", "ileum_pre_stenotic_dil", "ileum_comb_sign",
              "ileum_fistula", "colon_inflammation", "ileum_mesenteric_edema"]
CLIN = ["Time_to_index", "age_at_diagnosis", "CCI_at_diagnosis", "disease_activity_5",
        "SES_points", "diagnostic_delay", "had_surgery_before", "had_steroid_dep_before",
        "had_biologic_switch_before", "had_EIM_before", "had_perianal_before", "sex", "has_clinical"]


def km_surv_at(t, T, E, horizon):
    """KM survival S(horizon) for a subgroup (arrays T,E)."""
    if len(T) == 0:
        return 1.0
    x, y = kaplan_meier_estimator(E.astype(bool), T)
    s = y[x <= horizon]
    return float(s[-1]) if len(s) else 1.0


def risk_at_horizon(X, T, E, horizon):
    """Fit Coxnet, return predicted risk = 1 - S(horizon) per patient (in-sample)."""
    Xs = StandardScaler().fit_transform(X)
    m = CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01, fit_baseline_model=True, max_iter=500000)
    m.fit(Xs, Surv.from_arrays(event=E.astype(bool), time=T))
    surv_fns = m.predict_survival_function(Xs)
    risk = np.array([1.0 - fn(horizon) for fn in surv_fns])
    return risk


def net_benefit(risk, T, E, horizon, thresholds):
    n = len(risk)
    nb = []
    for pt in thresholds:
        pos = risk >= pt
        f = pos.mean()
        if pos.sum() == 0:
            nb.append(0.0); continue
        s_pos = km_surv_at(None, T[pos], E[pos], horizon)   # KM survival at horizon in treat-positive
        event_rate = 1 - s_pos
        nb.append(f * event_rate - f * s_pos * (pt / (1 - pt)))
    return np.array(nb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survival-csv", required=True)
    ap.add_argument("--outcome", default="surgery")
    ap.add_argument("--horizon", type=float, default=5.0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.survival_csv)
    dur, evt = f"duration_{args.outcome}", f"event_{args.outcome}"
    pred_cols = [f"pred_{l}" for l in MIL_LABELS]
    clin_cols = [f"clinical_{c}" for c in CLIN]
    df = df.dropna(subset=pred_cols + clin_cols + [dur, evt]).reset_index(drop=True)
    T = df[dur].values.astype(float)
    E = df[evt].values.astype(int)
    thr = np.linspace(0.01, 0.60, 60)

    out = {"outcome": args.outcome, "horizon": args.horizon, "thresholds": thr.tolist(), "n": len(df),
           "events": int(E.sum())}
    # model risks
    risk_comb = risk_at_horizon(df[pred_cols + clin_cols].values.astype(float), T, E, args.horizon)
    risk_clin = risk_at_horizon(df[clin_cols].values.astype(float), T, E, args.horizon)
    out["nb_combined"] = net_benefit(risk_comb, T, E, args.horizon, thr).tolist()
    out["nb_clinical"] = net_benefit(risk_clin, T, E, args.horizon, thr).tolist()
    # treat-all / none
    s_all = km_surv_at(None, T, E, args.horizon)
    out["nb_treat_all"] = [( (1 - s_all) - s_all * (pt/(1-pt)) ) for pt in thr]
    out["nb_treat_none"] = [0.0] * len(thr)

    # summary: at which thresholds does combined beat clinical & treat-all?
    nbc, nbk, nba = np.array(out["nb_combined"]), np.array(out["nb_clinical"]), np.array(out["nb_treat_all"])
    win = (nbc >= nbk) & (nbc >= nba)
    out["combined_best_threshold_range"] = [float(thr[win].min()), float(thr[win].max())] if win.any() else None
    out["mean_nb_gain_vs_clinical"] = float(np.mean(nbc - nbk))
    out["mean_nb_gain_vs_treat_all"] = float(np.mean(nbc - nba))

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"DCA {args.outcome} @ {args.horizon}y (n={len(df)}, events={int(E.sum())})")
    print(f"  combined best over treat-all+clinical in threshold range: {out['combined_best_threshold_range']}")
    print(f"  mean net-benefit gain vs clinical:  {out['mean_nb_gain_vs_clinical']:+.4f}")
    print(f"  mean net-benefit gain vs treat-all: {out['mean_nb_gain_vs_treat_all']:+.4f}")
    print(f"  -> {args.output}")


if __name__ == "__main__":
    main()
