# Stage-2 survival scripts

Time-to-event (Cox) modeling on the per-patient survival table. Patient data is
NOT distributed with this repository; supply your own `survival_*.csv` in the
schema documented in `docs/data_format.md`.

- **`build_survival_v11.py`** assembles the per-patient survival table:
  `survival_v09` (imaging predictions + 13 clinical features + the outcome
  `duration_*` / `event_*` columns) LEFT-joined with the lab and medication
  features on `accession_number`. Output: `survival_v11.csv` (full MR+CT cohort,
  5,765 rows x 210 columns). All rows carry complete `duration_*` / `event_*`
  values for the three outcomes; lab/med columns are `NaN` where a patient has
  no matching record.

- **`fit_unified_cox.py`** fits an elastic-net Cox model with 5-fold
  patient-grouped cross-validation on the unified MR+CT cohort and reports the
  combined / MR-only / CT-only C-index per outcome (surgery, steroid, biologic).
  The model uses **23 features only**: 10 imaging predictions (`pred_*`) plus 13
  clinical features (`clinical_*`). Identifiers (`accession_number`,
  `patient_id`, `modality`, `center`), the outcome targets themselves, and the
  `lab_*` / `med_*` columns are NOT fed to this model. Usage:

  ```bash
  python fit_unified_cox.py --survival-csv survival_v11.csv --output cox.json
  ```

The `lab_*` and `med_*` columns present in `survival_v11.csv` support the
separate labs/medications and causal-inference analyses, not the Cox model above.
