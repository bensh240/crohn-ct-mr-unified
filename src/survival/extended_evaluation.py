"""
Phase 8.5 — Extended Evaluation for V19 (npj DM gap closure: calibration + time-dep metrics)
============================================================================================

Computes the venue-grade survival metrics that the headline V19 (DINOv2 + LoRA + Cox)
results are missing, for each of three outcomes (surgery / steroid dependence / biologic
switch) under three feature modes (combined / imaging_only / clinical_only):

  1. C-index (re-computed for sanity, with patient-level bootstrap 95 percent CI)
  2. Time-dependent cumulative/dynamic AUC at 1, 2, 3, 5, 7 years
  3. Integrated Brier Score (IBS) over 0.5y - 7y, with bootstrap 95 percent CI
  4. Brier score curve sampled at the same time points (per-horizon plot)
  5. Calibration plot at 5 years: 10 deciles by predicted 5-yr survival probability,
     mean predicted vs observed Kaplan-Meier in each decile.

Cross-validation uses GroupKFold by patient_id (no patient-level leakage), mirroring
fit_cox_on_predictions.py exactly. Cox model: CoxnetSurvivalAnalysis(l1_ratio=0.5,
alpha_min_ratio=0.01, fit_baseline_model=True, max_iter=500_000) with StandardScaler
fit on train fold only.

Designed to slot into the same toolchain as fit_cox_on_predictions.py and to run
later for V21-C with no code changes (just a different --survival-csv).

Run on argus01 CPU only; no GPU / bf16 / SLURM dependencies.

Usage:
  python -u extended_evaluation_v19.py \
      --survival-csv ./data/survival.csv \
      --output-dir   ./results/extended_evaluation \
      --n-bootstrap 1000 \
      --n-splits 5
"""

from __future__ import annotations

import os
import json
import argparse
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv
    from sksurv.metrics import (
        concordance_index_censored,
        cumulative_dynamic_auc,
        integrated_brier_score,
        brier_score,
    )
except ImportError as e:
    raise SystemExit(f"sksurv required: pip install scikit-survival  ({e})")

try:
    from lifelines import KaplanMeierFitter
except ImportError as e:
    raise SystemExit(f"lifelines required: pip install lifelines  ({e})")

# Reuse the exact feature definition + helpers from fit_cox_on_predictions.py
# so the two scripts cannot drift. If the import path is unavailable (e.g. when
# running from a different cwd), fall back to local duplicates.
try:
    from fit_cox_on_predictions import (
        MIL_LABELS,
        CLINICAL_COLS,
        load_df,
        build_feature_matrix,
    )
except Exception:  # pragma: no cover - fallback duplicate
    MIL_LABELS = [
        "ileum_inflammation", "ileum_wall_enhancement", "ileum_wall_thickness",
        "ileum_dwi", "ileum_stenosis", "ileum_pre_stenotic_dil",
        "ileum_comb_sign", "ileum_fistula", "colon_inflammation",
        "ileum_mesenteric_edema",
    ]
    CLINICAL_COLS = [
        "Time_to_index", "age_at_diagnosis", "CCI_at_diagnosis",
        "disease_activity_5", "SES_points", "diagnostic_delay",
        "had_surgery_before", "had_steroid_dep_before", "had_biologic_switch_before",
        "had_EIM_before", "had_perianal_before", "sex", "has_clinical",
    ]

    def load_df(path):
        df = pd.read_csv(path)
        for col in [f"pred_{l}" for l in MIL_LABELS] + [f"clinical_{c}" for c in CLINICAL_COLS]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def build_feature_matrix(df, mode="combined", include_ext=False):
        pred_cols = [f"pred_{l}" for l in MIL_LABELS]
        clin_cols = [f"clinical_{c}" for c in CLINICAL_COLS]
        if mode == "imaging_only":
            X, names = df[pred_cols].values, pred_cols
        elif mode == "clinical_only":
            X, names = df[clin_cols].values, clin_cols
        elif mode == "combined":
            X, names = df[pred_cols + clin_cols].values, pred_cols + clin_cols
        else:
            raise ValueError(mode)
        return X.astype(np.float32), names


warnings.filterwarnings("ignore")

OUTCOMES = ["surgery", "steroid", "biologic"]
MODES = ["combined", "imaging_only", "clinical_only"]
TIME_POINTS_YEARS = [1.0, 2.0, 3.0, 5.0, 7.0]
BRIER_GRID_YEARS = np.linspace(0.5, 7.0, 27)  # 0.25y spacing, 27 points
CAL_HORIZON_YEAR = 5.0


# --------------------------------------------------------------------------------------
# Core CV: returns out-of-fold predictions + survival functions per patient
# --------------------------------------------------------------------------------------

def _filter_times(times_year: np.ndarray, T_train: np.ndarray, T_eval: np.ndarray) -> np.ndarray:
    """Keep only horizons strictly inside the overlap of train and eval observed time ranges.
    sksurv requires test times be < max train time for cumulative_dynamic_auc / brier_score."""
    t_min = max(float(T_train.min()), float(T_eval.min())) + 1e-3
    t_max = min(float(T_train.max()), float(T_eval.max())) - 1e-3
    return times_year[(times_year > t_min) & (times_year < t_max)]


def cv_predict(
    df: pd.DataFrame,
    outcome: str,
    mode: str,
    n_splits: int,
) -> Dict:
    """Run GroupKFold CV. For each held-out fold, store risk scores and survival
    function evaluated on a global time grid, plus the truth (T, E, patient_id).

    Returns:
        {
          "T": np.ndarray, "E": np.ndarray, "groups": np.ndarray,
          "risk": np.ndarray,
          "surv_at_grid": np.ndarray (N x len(grid)),  # S(t | x) on BRIER_GRID_YEARS
          "fold_cindex": list[float],
          "y_train_per_fold": list[structured array],  # for cumulative_dynamic_auc
          "fold_idx": np.ndarray (N,),  # which fold each row belongs to
        }
    """
    dur_col = f"duration_{outcome}"
    evt_col = f"event_{outcome}"

    pred_cols = [f"pred_{l}" for l in MIL_LABELS]
    clin_cols = [f"clinical_{c}" for c in CLINICAL_COLS]
    required = pred_cols + clin_cols + [dur_col, evt_col, "patient_id"]
    dfx = df.dropna(subset=required).reset_index(drop=True)
    if dfx.empty:
        return None

    X, _ = build_feature_matrix(dfx, mode=mode, include_ext=False)
    T = dfx[dur_col].values.astype(np.float64)
    E = dfx[evt_col].values.astype(np.int32)
    groups = dfx["patient_id"].values

    n = len(dfx)
    risk_oof = np.full(n, np.nan, dtype=np.float64)
    surv_oof = np.full((n, len(BRIER_GRID_YEARS)), np.nan, dtype=np.float64)
    fold_assign = np.full(n, -1, dtype=np.int32)
    fold_cindex: List[float] = []
    y_train_per_fold: List[np.ndarray] = []

    kf = GroupKFold(n_splits=n_splits)
    for fold_idx, (tr, va) in enumerate(kf.split(X, E, groups)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr])
        X_va = scaler.transform(X[va])
        y_tr = Surv.from_arrays(event=E[tr].astype(bool), time=T[tr])
        try:
            m = CoxnetSurvivalAnalysis(
                l1_ratio=0.5, alpha_min_ratio=0.01,
                fit_baseline_model=True, max_iter=500_000,
            )
            m.fit(X_tr, y_tr)
            risks = m.predict(X_va)
            risk_oof[va] = risks
            fold_assign[va] = fold_idx

            # Survival functions evaluated on global grid
            sfns = m.predict_survival_function(X_va)
            for i, s in zip(va, sfns):
                # s is a StepFunction; query at each time
                surv_oof[i, :] = np.array([s(t) for t in BRIER_GRID_YEARS])

            c = concordance_index_censored(E[va].astype(bool), T[va], risks)[0]
            fold_cindex.append(float(c))
            y_train_per_fold.append(y_tr)
        except Exception as ex:
            print(f"    [{outcome}/{mode}] fold {fold_idx+1} failed: {type(ex).__name__}: {ex}")
            fold_cindex.append(float("nan"))
            y_train_per_fold.append(None)

    return {
        "T": T, "E": E, "groups": groups,
        "risk": risk_oof,
        "surv_at_grid": surv_oof,
        "fold_cindex": fold_cindex,
        "y_train_per_fold": y_train_per_fold,
        "fold_idx": fold_assign,
        "n_samples": int(n),
        "n_events": int(E.sum()),
    }


# --------------------------------------------------------------------------------------
# Metric computation on OOF predictions
# --------------------------------------------------------------------------------------

def compute_td_auc(cv: Dict) -> Dict[str, float]:
    """Time-dependent AUC at TIME_POINTS_YEARS using OOF risks.

    sksurv's cumulative_dynamic_auc expects a single y_train per call; we use the
    union of all fold training survival data as a proxy (concatenating them gives
    a near-equivalent IPCW estimator and avoids ambiguity). Keeps only horizons
    in the valid time range.
    """
    valid = [yt for yt in cv["y_train_per_fold"] if yt is not None]
    if not valid:
        return {f"{int(t)}y" if t == int(t) else f"{t}y": float("nan") for t in TIME_POINTS_YEARS}
    y_train_all = np.concatenate(valid)
    y_eval = Surv.from_arrays(event=cv["E"].astype(bool), time=cv["T"])
    risk = cv["risk"]
    times_arr = np.array(TIME_POINTS_YEARS, dtype=float)

    # Restrict to horizons in valid range for sksurv
    T_train_all = y_train_all["time"]
    times_valid = _filter_times(times_arr, T_train_all, cv["T"])
    out = {}
    if len(times_valid) == 0:
        for t in TIME_POINTS_YEARS:
            key = f"{int(t)}y" if t == int(t) else f"{t}y"
            out[key] = float("nan")
        return out
    try:
        aucs, _mean = cumulative_dynamic_auc(y_train_all, y_eval, risk, times_valid)
    except Exception as ex:
        print(f"    cumulative_dynamic_auc failed: {ex}")
        aucs = np.full_like(times_valid, np.nan, dtype=float)
    auc_map = dict(zip(times_valid.tolist(), aucs.tolist()))
    for t in TIME_POINTS_YEARS:
        key = f"{int(t)}y" if t == int(t) else f"{t}y"
        out[key] = float(auc_map.get(t, float("nan")))
    return out


def compute_brier(cv: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
    """Returns (IBS over 0.5-7y, time grid used, brier values)."""
    valid = [yt for yt in cv["y_train_per_fold"] if yt is not None]
    if not valid:
        return float("nan"), BRIER_GRID_YEARS, np.full_like(BRIER_GRID_YEARS, np.nan, dtype=float)
    y_train_all = np.concatenate(valid)
    y_eval = Surv.from_arrays(event=cv["E"].astype(bool), time=cv["T"])

    T_train_all = y_train_all["time"]
    times_valid = _filter_times(BRIER_GRID_YEARS, T_train_all, cv["T"])
    if len(times_valid) < 2:
        return float("nan"), BRIER_GRID_YEARS, np.full_like(BRIER_GRID_YEARS, np.nan, dtype=float)

    # Subset surv predictions at valid times
    idx = np.array([np.argmin(np.abs(BRIER_GRID_YEARS - t)) for t in times_valid])
    surv = cv["surv_at_grid"][:, idx]

    try:
        ibs = float(integrated_brier_score(y_train_all, y_eval, surv, times_valid))
    except Exception as ex:
        print(f"    integrated_brier_score failed: {ex}")
        ibs = float("nan")

    try:
        _, bvals = brier_score(y_train_all, y_eval, surv, times_valid)
    except Exception as ex:
        print(f"    brier_score failed: {ex}")
        bvals = np.full_like(times_valid, np.nan, dtype=float)

    return ibs, times_valid, np.asarray(bvals, dtype=float)


# --------------------------------------------------------------------------------------
# Bootstrap (patient-level)
# --------------------------------------------------------------------------------------

def patient_level_bootstrap(
    cv: Dict,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Dict:
    """Bootstrap C-index, td-AUC at each horizon, and IBS. Resample at the
    patient_id level (with replacement) and recompute metrics on the OOF
    predictions. Returns dict with mean / 2.5pct / 97.5pct."""
    T = cv["T"]; E = cv["E"]; groups = cv["groups"]; risk = cv["risk"]
    surv_grid = cv["surv_at_grid"]

    valid = [yt for yt in cv["y_train_per_fold"] if yt is not None]
    y_train_all = np.concatenate(valid) if valid else None

    unique_pids = np.unique(groups)
    pid_to_rows = {p: np.where(groups == p)[0] for p in unique_pids}

    c_samples: List[float] = []
    auc_samples: Dict[float, List[float]] = {t: [] for t in TIME_POINTS_YEARS}
    ibs_samples: List[float] = []

    times_arr = np.array(TIME_POINTS_YEARS, dtype=float)

    for b in range(n_bootstrap):
        sampled_pids = rng.choice(unique_pids, size=len(unique_pids), replace=True)
        rows = np.concatenate([pid_to_rows[p] for p in sampled_pids])
        Tb, Eb, riskb = T[rows], E[rows], risk[rows]
        survb = surv_grid[rows]

        # Skip degenerate samples
        if Eb.sum() == 0 or np.isnan(riskb).all():
            continue

        # C-index
        try:
            cb = concordance_index_censored(Eb.astype(bool), Tb, riskb)[0]
            c_samples.append(float(cb))
        except Exception:
            pass

        if y_train_all is None:
            continue

        # td-AUC
        y_eval_b = Surv.from_arrays(event=Eb.astype(bool), time=Tb)
        T_train_all = y_train_all["time"]
        times_valid = _filter_times(times_arr, T_train_all, Tb)
        if len(times_valid) > 0:
            try:
                aucs_b, _ = cumulative_dynamic_auc(y_train_all, y_eval_b, riskb, times_valid)
                for t, a in zip(times_valid, aucs_b):
                    auc_samples[float(t)].append(float(a))
            except Exception:
                pass

        # IBS
        bg_valid = _filter_times(BRIER_GRID_YEARS, T_train_all, Tb)
        if len(bg_valid) >= 2:
            idx = np.array([np.argmin(np.abs(BRIER_GRID_YEARS - t)) for t in bg_valid])
            try:
                ibs_b = integrated_brier_score(y_train_all, y_eval_b, survb[:, idx], bg_valid)
                ibs_samples.append(float(ibs_b))
            except Exception:
                pass

    def _summ(arr: List[float]) -> Dict[str, float]:
        if len(arr) == 0:
            return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
        a = np.asarray(arr, dtype=float)
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
        return {
            "mean": float(np.mean(a)),
            "ci_low": float(np.percentile(a, 2.5)),
            "ci_high": float(np.percentile(a, 97.5)),
            "n": int(len(a)),
        }

    auc_summary = {}
    for t in TIME_POINTS_YEARS:
        key = f"{int(t)}y" if t == int(t) else f"{t}y"
        auc_summary[key] = _summ(auc_samples[t])

    return {
        "c_index_boot": _summ(c_samples),
        "td_auc_boot": auc_summary,
        "ibs_boot": _summ(ibs_samples),
    }


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------

def plot_calibration_5yr(cv: Dict, outcome: str, out_path: str):
    """Decile calibration at 5 years using OOF predicted survival vs KM-observed."""
    if CAL_HORIZON_YEAR not in BRIER_GRID_YEARS.tolist():
        # Find nearest
        idx = int(np.argmin(np.abs(BRIER_GRID_YEARS - CAL_HORIZON_YEAR)))
    else:
        idx = BRIER_GRID_YEARS.tolist().index(CAL_HORIZON_YEAR)
    surv5 = cv["surv_at_grid"][:, idx]
    T = cv["T"]; E = cv["E"]
    mask = ~np.isnan(surv5)
    if mask.sum() < 50:
        print(f"    [calibration] too few valid rows ({mask.sum()}), skipping")
        return
    surv5 = surv5[mask]; Tm = T[mask]; Em = E[mask]

    # Decile bins by predicted survival probability
    deciles = pd.qcut(surv5, q=10, duplicates="drop")
    df_cal = pd.DataFrame({"surv5": surv5, "T": Tm, "E": Em, "bin": deciles})

    pred_means = []
    obs_means = []
    obs_lows = []
    obs_highs = []
    bins_kept = []
    kmf = KaplanMeierFitter()
    for b, g in df_cal.groupby("bin", observed=True):
        if len(g) < 5:
            continue
        kmf.fit(g["T"].values, event_observed=g["E"].values)
        # KM survival at 5y
        try:
            obs = float(kmf.survival_function_at_times(CAL_HORIZON_YEAR).values[0])
            ci = kmf.confidence_interval_survival_function_
            # Find row at or just before CAL_HORIZON_YEAR
            ci_idx = ci.index.searchsorted(CAL_HORIZON_YEAR, side="right") - 1
            ci_idx = max(0, min(ci_idx, len(ci) - 1))
            lo = float(ci.iloc[ci_idx, 0])
            hi = float(ci.iloc[ci_idx, 1])
        except Exception:
            obs, lo, hi = float("nan"), float("nan"), float("nan")
        pred_means.append(float(g["surv5"].mean()))
        obs_means.append(obs)
        obs_lows.append(lo); obs_highs.append(hi)
        bins_kept.append(str(b))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="Ideal")
    yerr_low = np.array(obs_means) - np.array(obs_lows)
    yerr_high = np.array(obs_highs) - np.array(obs_means)
    ax.errorbar(
        pred_means, obs_means,
        yerr=[yerr_low, yerr_high],
        fmt="o", capsize=3, color="C0",
        label=f"Deciles (n={len(bins_kept)})",
    )
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"Mean predicted {int(CAL_HORIZON_YEAR)}-yr survival probability")
    ax.set_ylabel(f"Observed {int(CAL_HORIZON_YEAR)}-yr survival (KM)")
    ax.set_title(f"Calibration at {int(CAL_HORIZON_YEAR)}y - {outcome} (combined)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_td_auc(td_auc: Dict[str, Dict[str, float]], outcome: str, out_path: str):
    """Plot td-AUC mean +/- 95 CI across horizons for all 3 modes."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    horizons = TIME_POINTS_YEARS
    keys = [f"{int(t)}y" if t == int(t) else f"{t}y" for t in horizons]
    for mode, color in zip(MODES, ["C0", "C1", "C2"]):
        if mode not in td_auc:
            continue
        means = [td_auc[mode][k]["mean"] for k in keys]
        lows = [td_auc[mode][k]["ci_low"] for k in keys]
        highs = [td_auc[mode][k]["ci_high"] for k in keys]
        means = np.array(means, dtype=float)
        lows = np.array(lows, dtype=float)
        highs = np.array(highs, dtype=float)
        ax.plot(horizons, means, "-o", color=color, label=mode)
        ax.fill_between(horizons, lows, highs, color=color, alpha=0.15)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Time horizon (years)")
    ax.set_ylabel("Time-dependent AUC")
    ax.set_title(f"Time-dependent AUC - {outcome}")
    ax.set_ylim(0.3, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_brier_curve(brier_curves: Dict[str, Dict], outcome: str, out_path: str):
    """Plot Brier(t) for each mode."""
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for mode, color in zip(MODES, ["C0", "C1", "C2"]):
        if mode not in brier_curves:
            continue
        times = brier_curves[mode]["times"]
        vals = brier_curves[mode]["values"]
        ax.plot(times, vals, "-", color=color, label=mode)
    # Reference: Brier of 0.25 = uninformative (always predict 0.5)
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1, label="Uninformative (0.25)")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Brier score")
    ax.set_title(f"Brier score over time - {outcome}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------

def evaluate_outcome_mode(df: pd.DataFrame, outcome: str, mode: str,
                          n_splits: int, n_bootstrap: int, rng: np.random.Generator) -> Dict:
    cv = cv_predict(df, outcome, mode, n_splits=n_splits)
    if cv is None:
        return None

    fold_cs = cv["fold_cindex"]
    valid_cs = [c for c in fold_cs if not np.isnan(c)]
    point_c = float(np.mean(valid_cs)) if valid_cs else float("nan")

    td_auc = compute_td_auc(cv)
    ibs, brier_times, brier_vals = compute_brier(cv)
    boots = patient_level_bootstrap(cv, n_bootstrap=n_bootstrap, rng=rng)

    # Compose unified record
    record = {
        "n_samples": cv["n_samples"],
        "n_events": cv["n_events"],
        "n_splits": n_splits,
        "c_index": {
            "point_mean": point_c,
            "fold_values": fold_cs,
            "mean": boots["c_index_boot"]["mean"],
            "ci_low": boots["c_index_boot"]["ci_low"],
            "ci_high": boots["c_index_boot"]["ci_high"],
        },
        "td_auc": {
            k: {
                "point": td_auc[k],
                "mean": boots["td_auc_boot"][k]["mean"],
                "ci_low": boots["td_auc_boot"][k]["ci_low"],
                "ci_high": boots["td_auc_boot"][k]["ci_high"],
            }
            for k in [f"{int(t)}y" if t == int(t) else f"{t}y" for t in TIME_POINTS_YEARS]
        },
        "ibs": {
            "point": float(ibs),
            "mean": boots["ibs_boot"]["mean"],
            "ci_low": boots["ibs_boot"]["ci_low"],
            "ci_high": boots["ibs_boot"]["ci_high"],
        },
        "brier_curve": {
            "times": [float(t) for t in brier_times],
            "values": [float(v) for v in brier_vals],
        },
    }
    return record, cv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survival-csv", required=True,
                    help="Path to survival_filled_v19_dinov2.csv (or compatible).")
    ap.add_argument("--output-dir", required=True,
                    help="Directory to write extended_metrics.json and PNGs.")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    df = load_df(args.survival_csv)
    print(f"Loaded {len(df)} patient rows from {args.survival_csv}")
    n_with_preds = df["pred_ileum_inflammation"].notna().sum() if "pred_ileum_inflammation" in df.columns else 0
    print(f"Rows with MIL predictions filled: {n_with_preds}/{len(df)}")
    if n_with_preds == 0:
        raise SystemExit("No rows have pred_* filled - cannot fit Cox.")

    results: Dict[str, Dict] = {}
    cvs: Dict[Tuple[str, str], Dict] = {}

    for outcome in OUTCOMES:
        print(f"\n=== {outcome} ===")
        results[outcome] = {}
        for mode in MODES:
            print(f"  [{mode}] running CV + metrics...")
            out = evaluate_outcome_mode(
                df, outcome, mode,
                n_splits=args.n_splits,
                n_bootstrap=args.n_bootstrap,
                rng=rng,
            )
            if out is None:
                print(f"    no valid rows - skipped")
                continue
            record, cv = out
            results[outcome][mode] = record
            cvs[(outcome, mode)] = cv
            ci = record["c_index"]
            print(
                f"    C-index = {ci['point_mean']:.4f}  "
                f"boot mean={ci['mean']:.4f} [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]  "
                f"IBS={record['ibs']['mean']:.4f} [{record['ibs']['ci_low']:.4f}, {record['ibs']['ci_high']:.4f}]"
            )

        # Plots: one per outcome, with all modes overlaid except calibration (combined only)
        td_auc_for_plot = {m: results[outcome][m]["td_auc"] for m in MODES if m in results[outcome]}
        brier_for_plot = {m: results[outcome][m]["brier_curve"] for m in MODES if m in results[outcome]}
        if td_auc_for_plot:
            plot_td_auc(td_auc_for_plot, outcome,
                        os.path.join(args.output_dir, f"td_auc_curve_{outcome}.png"))
        if brier_for_plot:
            plot_brier_curve(brier_for_plot, outcome,
                             os.path.join(args.output_dir, f"brier_curve_{outcome}.png"))
        if (outcome, "combined") in cvs:
            plot_calibration_5yr(
                cvs[(outcome, "combined")], outcome,
                os.path.join(args.output_dir, f"calibration_5yr_{outcome}.png"),
            )

    out_json = os.path.join(args.output_dir, "extended_metrics.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    print(f"\nSaved {out_json}")
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
