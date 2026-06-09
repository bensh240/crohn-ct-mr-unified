# Data format

This document describes the file and DB layouts the public-facing pipeline
expects. Patient data is NOT distributed with the repository; you must supply
your own data in this schema.

## 1. Cohort CSVs (`train.csv`, `val.csv`, `test.csv`)

Patient-level splits of the imaging cohort. All three files share the same
schema. **Patient-level leakage is required to be zero** (no `patient_id`
appears in more than one split).

| column | type | example | notes |
|---|---|---|---|
| `accession_number` | str | `A123456789` | unique study identifier |
| `modality` | str | `mr` or `ct` | lowercase |
| `image_dir` | path | `./data/images/A123456789/` | directory holding 16 PNG slices |
| `t1_dir` | path \| empty | `./data/t1/A123456789/` | MR only; empty for CT |
| `is_sick` | 0/1 | `1` | bag-level binary disease label |
| `ileum_inflammation` | 0/1 | `1` | finding label |
| `ileum_wall_enhancement` | 0/1 | `1` | |
| `ileum_wall_thickness` | 0/1 | `1` | |
| `ileum_dwi` | 0/1 | `0` | MR only; can be `0` on CT |
| `ileum_stenosis` | 0/1 | `1` | |
| `ileum_pre_stenotic_dil` | 0/1 | `0` | |
| `ileum_comb_sign` | 0/1 | `0` | |
| `ileum_fistula` | 0/1 | `0` | |
| `colon_inflammation` | 0/1 | `0` | |
| `ileum_mesenteric_edema` | 0/1 | `0` | |
| `Time_to_index` | float | `2.3` | years from diagnosis to imaging |
| `age_at_diagnosis` | float | `27` | years |
| `CCI_at_diagnosis` | int | `0` | Charlson Comorbidity Index |
| `disease_activity_5` | int | `3` | 1-5 activity cluster |
| `SES_points` | int | `7` | socioeconomic score 0-17 |
| `diagnostic_delay` | float | `0.6` | years from symptom onset to dx |
| `had_surgery_before` | 0/1 | `0` | derived from `time_to_1.0 < 0` |
| `had_steroid_dep_before` | 0/1 | `0` | derived from `time_to_2.0 < 0` |
| `had_biologic_switch_before` | 0/1 | `0` | derived from `time_to_3.0 < 0` |
| `had_EIM_before` | 0/1 | `0` | extra-intestinal manifestation flag |
| `had_perianal_before` | 0/1 | `1` | |
| `sex` | 0/1 | `0` | 1 = male |
| `has_clinical` | 0/1 | `1` | any clinical feature non-null |

28 columns total. `train.csv` and `val.csv` are used by
`src/training/train_mil_v21.py`; `test.csv` is used by
`src/inference/run_inference_v21.py` and downstream scripts.

## 2. Image directory layout

Each study folder under `image_dir` must contain **exactly 16 PNG files**,
named `slice00.png` ... `slice15.png` (or any sort-stable scheme - we use
`sorted(os.listdir(...))`). All slices must be the same size; the loader
resizes to `224 x 224`.

For MR with a T1 sequence, `t1_dir` follows the same layout. When `t1_dir`
is empty, the model uses the T1 zero-mask path (`has_t1 = 0`).

```
images/
  A123456789/
    slice00.png
    slice01.png
    ...
    slice15.png
  A234567890/
    ...
```

## 3. Master table (`v21_master_table.csv`)

Used by `src/scripts/make_cv5_splits.py` (and by some `data_prep/` scripts).

| column | type | notes |
|---|---|---|
| `accession_number` | str | join key |
| `modality` | str | `MR` / `CT` (capital - we lower-case downstream) |
| `patient_id` | str | required for patient-level CV |
| `center` | str | `Mor Inside` / `SZMC` / `Assuta` / ... (for LOCO) |
| `study_date` | date | optional |
| ... | | other audit columns are allowed |

## 4. Outcomes database (`epiirn_v0.0.7.db`, SQLite)

Used by `src/survival/build_survival_v07.py` and
`src/scripts/validate_v07_outcomes.py`.

The DB must contain three tables.

### Table `outcomes`

| column | type | notes |
|---|---|---|
| `accession_number` | TEXT | join key |
| `sex` | TEXT | 'M' / 'F' / 0 / 1 (we coerce to 0/1) |
| `Time_to_index` | REAL | years from diagnosis to imaging |
| `age_at_diagnosis` | REAL | |
| `CCI_at_diagnosis` | REAL | |
| `disease_activity_5` | REAL | 1-5 |
| `SES_points` | REAL | 0-17 |
| `diagnostic_delay` | REAL | |
| `Time_to_first_EIM_indication` | REAL | nullable |
| `Time_to_first_perianal_indication` | REAL | nullable |
| `time_to_1.0` | REAL | signed years to surgery from imaging (positive=after, negative=before, null=never) |
| `time_to_2.0` | REAL | signed years to steroid dependence |
| `time_to_3.0` | REAL | signed years to biologic switch |

`time_to_1.0` is the **signed time from the imaging study to the event**:

- `> 0` -> event happened AFTER imaging -> Cox event indicator = 1, duration = `time_to_1.0`
- `< 0` -> event happened BEFORE imaging -> `had_surgery_before = 1`, study censored
- `NULL` or `0` -> no event in follow-up -> censored at `MAX_FU = 10` years

Identical convention for `time_to_2.0` (steroid) and `time_to_3.0` (biologic).

### Table `reports`

| column | type | notes |
|---|---|---|
| `accession_number` | TEXT | join key |
| `patient_id` | TEXT | used by `build_survival_v07.py` for grouped CV |
| `report` | TEXT | free-text radiology report (input to label-extraction model) |

### Table `dicoms` (study-level metadata)

| column | type | notes |
|---|---|---|
| `AccessionNumber` | TEXT | join key (CamelCase here) |
| `PatientID` | TEXT | fallback patient_id when `reports.patient_id` is null |
| `Modality` | TEXT | `MR` / `CT` |
| `InstitutionName` | TEXT | used for LOCO (some rows can be null) |

Multiple `dicoms` rows per accession are allowed; the build script uses
`MIN(...) GROUP BY AccessionNumber` to deduplicate.

## 5. Survival CSV produced by `build_survival_v07.py`

The output of `build_survival_v07.py` (and the input to
`fit_unified_cox.py`, `extended_evaluation.py`, `loco_evaluation.py`,
`decision_curve.py`) has this exact schema:

| column | notes |
|---|---|
| `accession_number` | |
| `patient_id` | grouping key for `GroupKFold` |
| `modality` | `mr` / `ct` |
| `center` | for LOCO |
| `pred_<10 findings>` | 10 columns, each in `[0, 1]` |
| `clinical_<13 features>` | 13 columns: 6 continuous + 5 "had_X_before" + sex + has_clinical |
| `duration_surgery`, `event_surgery` | Cox-ready (years, 0/1) |
| `duration_steroid`, `event_steroid` | |
| `duration_biologic`, `event_biologic` | |

37 columns total.

## 6. Pretrained checkpoint format

The trained A2 checkpoint (released separately on HuggingFace Hub / Zenodo)
contains:

- `best_model.pt` - PyTorch state dict for `V21UnifiedMIL` with
  `conditioning="film"`. Compatible with the `--ckpt` argument of
  `src/inference/run_inference_v21.py`.
- `clinical_scaler.joblib` - a fitted `sklearn.preprocessing.StandardScaler`
  for the 6 continuous clinical features (so test-time clinical features
  are standardized identically to training). Pass via `--scaler`.

Without the scaler, inference falls back to fitting a scaler on the
inference CSV, which is fine for sanity-checking but does not match the
trained model's expected feature distribution. **Always pass `--scaler`
for reproducible numbers.**
