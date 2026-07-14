# A Unified CT and MR Enterography Foundation-Model Pipeline for Crohn's Disease

Code accompanying the paper:

> **A Unified CT and MR Enterography Foundation-Model Pipeline for Crohn's
> Disease: Lightweight Modality Conditioning Matches Dual Specialists, While
> Domain-Specific Normalization Induces Negative Transfer.**
> Ben Shaya, Moti Freiman. MSc thesis, Reichman University &times;
> Shaare Zedek Medical Center. Manuscript submitted to *npj Digital Medicine*.

## What this code does

A two-stage pipeline for Crohn's disease enterography.

**Stage 1 (finding detection).** A frozen DINOv3-Base backbone with LoRA
adapters and dual-branch multi-instance learning (MIL) detects 10
radiological findings (fistula, stenosis, wall thickening, etc.) from
either CT or MR enterography. One unified backbone serves both modalities
via a learned modality token and FiLM conditioning. On a held-out Crohn's-only
test set it reaches macro-AUC 0.780 (MR) / 0.744 (CT). A single unified model
matches a dedicated MR specialist (0.780 vs 0.793) and outperforms a dedicated
CT specialist (0.744 vs 0.673), since the data-limited CT side gains from joint
cross-modality learning.

**Stage 2 (time-to-event prediction).** A penalized Cox model on the unified
CT+MR cohort predicts time to surgery, steroid dependence, and biologic
therapy switch. Surgery C-index 0.792 (95% CI 0.776-0.808), imaging adding
+0.103 over clinical features alone; IBS 0.074, LOCO 0.754 +/- 0.053 (MR centers).
CT contributes a meaningful prognostic signal on its own (surgery C-index 0.711).

All numbers are on the clean Crohn's-disease cohort: ulcerative-colitis studies
were excluded (their outcomes are not comparable to CD), which is why an earlier
all-inflammatory-bowel-disease evaluation reported a higher but UC-inflated
detection macro-AUC of 0.849. Reported metrics use patient-grouped 5-fold
cross-validation with paired bootstrap 95% confidence intervals.

**Key methodological finding.** Domain-specific normalization (DSBN) causes
statistically significant negative transfer on the dominant MR modality
(paired-fold mean macro-AUC delta of +0.0075 in favor of removing DSBN,
95% CI [+0.003, +0.012] across 5 folds). A modality token plus FiLM is the
sweet spot; heavier conditioning should be omitted.

## Layout

```
.
├── README.md
├── LICENSE                          MIT
├── CITATION.cff                     citation metadata
├── requirements.txt                 pip dependencies
├── environment.yml                  conda environment ("crohn_vlm")
├── src/
│   ├── config.py                    env-var driven path config
│   ├── training/
│   │   └── train_mil_v21.py         Stage-1 training (DINOv3 + LoRA + MIL)
│   ├── inference/
│   │   └── run_inference_v21.py     batch inference from a checkpoint
│   ├── survival/
│   │   ├── build_survival_v07.py    build unified survival CSV (MR + CT)
│   │   ├── fit_unified_cox.py       patient-grouped 5-fold Cox CV
│   │   ├── extended_evaluation.py   IBS, td-AUC, calibration
│   │   ├── loco_evaluation.py       leave-one-center-out
│   │   └── decision_curve.py        DCA
│   └── scripts/
│       ├── make_cv5_splits.py       5 patient-level CV folds
│       ├── aggregate_nested_cv.py   summarize nested-CV test mAUC
│       ├── validate_v07_outcomes.py pre-flight DB sanity gauntlet
│       ├── paper_tables_a2.py       reproduce the per-finding and HR tables
│       ├── make_paper_figures_a2.py reproduce the paper figures
│       ├── modules/
│       │   └── conditioning.py      ModalityToken / FiLM / DSBN modules
│       └── data_prep/               cohort-specific data prep (reference only)
├── slurm_examples/                  example SLURM submit scripts
├── paper/
│   ├── main.tex
│   └── Phase9_Paper_CT_MR.pdf
└── docs/
    └── data_format.md               CSV / DB schema we expect
```

## Install

```bash
git clone https://github.com/bensh240/crohn-ct-mr-unified.git
cd crohn-ct-mr-unified

# conda (recommended)
conda env create -f environment.yml
conda activate crohn_vlm

# or pip
pip install -r requirements.txt
```

You will also need:
- A HuggingFace account and access to the gated `facebook/dinov3-vitb16-pretrain-lvd1689m`
  model. Run `huggingface-cli login` once, or `export HF_TOKEN=...`.

## Configure paths

All paths are environment-variable driven. Set whichever you need:

```bash
export V21_DATA_DIR=/path/to/v21_csvs         # train.csv / val.csv / test.csv
export V21_CKPT_DIR=/path/to/checkpoints      # where to write & read checkpoints
export V21_PRED_DIR=/path/to/predictions      # inference outputs
export V21_SURV_DIR=/path/to/survival         # survival CSVs
export V21_RESULTS_DIR=/path/to/results       # JSON metrics, figures
export V21_DB_PATH=/path/to/epiirn_v0.0.7.db  # SQLite outcomes DB
export V21_WARMSTART_CKPT=/path/to/V20.pt     # optional warm-start ckpt
```

Defaults fall back to `./data`, `./checkpoints`, etc. under the repo root.
See `src/config.py`.

## Pretrained checkpoints

The trained **A2** production checkpoint (`v21c_a2_film/best_model.pt`,
~720 MB) and the 15 nested-CV checkpoints are too large for GitHub. They
will be released on:

- Hugging Face Hub: `bensh240/crohn-ct-mr-a2` (planned)
- Zenodo: DOI to follow

upon paper acceptance. The repository URL will be added at proof stage.

## Pipeline at a glance

Given a trained checkpoint and a v0.0.7-format SQLite outcomes DB, here is
the end-to-end:

```bash
# 0. sanity check the DB
python src/scripts/validate_v07_outcomes.py

# 1. run A2 inference on all MR studies
python src/inference/run_inference_v21.py \
  --ckpt $V21_CKPT_DIR/v21c_a2_film/best_model.pt \
  --data-csv $V21_DATA_DIR/train.csv \
  --modality both \
  --scaler $V21_CKPT_DIR/v21c_a2_film/clinical_scaler.joblib \
  --output $V21_PRED_DIR/a2_mr_all.csv
# (and similarly for CT into a2_ct_all.csv)

# 2. build the unified survival CSV
python src/survival/build_survival_v07.py \
  --preds-mr $V21_PRED_DIR/a2_mr_all.csv \
  --preds-ct $V21_PRED_DIR/a2_ct_all.csv \
  --db       $V21_DB_PATH \
  --output   $V21_SURV_DIR/survival_v07.csv

# 3. fit unified Cox (patient-grouped 5-fold CV)
python src/survival/fit_unified_cox.py \
  --survival-csv $V21_SURV_DIR/survival_v07.csv \
  --output       $V21_RESULTS_DIR/cox_unified_v07.json

# 4. calibration + td-AUC + IBS
python src/survival/extended_evaluation.py \
  --survival-csv $V21_SURV_DIR/survival_v07.csv \
  --output-dir   $V21_RESULTS_DIR/v07_extended \
  --n-bootstrap 1000 --n-splits 5

# 5. multi-center LOCO (MR only; CT has no center labels in v0.0.7)
python src/survival/loco_evaluation.py \
  --survival-csv $V21_SURV_DIR/survival_v07.csv \
  --output-dir   $V21_RESULTS_DIR/v07_loco

# 6. decision-curve analysis
python src/survival/decision_curve.py \
  --survival-csv $V21_SURV_DIR/survival_v07.csv \
  --outcome surgery --horizon 5.0 \
  --output $V21_RESULTS_DIR/dca_unified_v07_surgery.json
```

To reproduce the **nested CV** experiment that defends the choice of A2:

```bash
python src/scripts/make_cv5_splits.py        # 5 patient-level folds
# train all 15 (fold, variant) runs via slurm_examples/02_nested_cv_train.sh
# infer:                                       slurm_examples/03_nested_cv_inference.sh
python src/scripts/aggregate_nested_cv.py    # summarize
```

## Reproducibility notes

- Stage 1 training was performed on NVIDIA A6000 (48 GB) GPUs.
- A single A2 training takes 2-9 hours depending on early-stop epoch
  (we use `PATIENCE = 5`).
- Stage 1 inference takes minutes on a single GPU for ~1,800 scans.
- Stage 2 (Cox fit + evaluation) runs in minutes on CPU.

## Data

Patient data is NOT included in this repository for privacy reasons. The
v0.0.7 outcomes DB and the linked DICOM cohort are governed by individual
hospital data-use agreements. The CSV and DB schemas this code expects are
documented in [`docs/data_format.md`](docs/data_format.md).

The 10 radiological-finding labels in our cohort are derived from
free-text radiology reports by a clinical information-extraction model
(HSMP-BERT). See the paper for the labeling pipeline.

## License

[MIT](LICENSE).

## Citation

If you use this code, please cite the accompanying paper. A BibTeX entry
will be added here at acceptance. See [CITATION.cff](CITATION.cff) for
metadata.

## Contact

Ben Shaya (bensh240@gmail.com)
Moti Freiman (moti.freiman@technion.ac.il)
