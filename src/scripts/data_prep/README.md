# Data preparation (reference only)

These scripts were used to build the V21 training cohort from our specific
clinical database (`epiirn_v0.0.6` / `epiirn_v0.0.7`) and DICOM layout
(per-center folders defined in `db/config.json`). They are included here for
transparency and reproducibility of the published cohort construction, but
they are NOT portable as-is: the table schemas, accession-number conventions,
and per-center DICOM roots are specific to our consortium.

If you have a different cohort, treat these scripts as a worked example.
The portable, public-facing pipeline starts at `src/inference/run_inference_v21.py`
(given a trained checkpoint) and `src/survival/*` (given a survival CSV in
the schema documented in `docs/data_format.md`).

| Script | What it does |
|---|---|
| `preprocess_ct.py` | Converts CT DICOM series to 16 evenly-sampled PNG axial slices. |
| `preprocess_ct_seg.py` | Same, but with TotalSegmentator bowel-localization crop first. |
| `make_master_table.py` | Joins DICOMs + reports + outcomes into a master accession-level table. |
| `make_splits.py` | Patient-level train/val/test split with leakage check. |
| `build_v21_dataset.py` | Produces the final `train.csv` / `val.csv` / `test.csv` consumed by `train_mil_v21.py`. |
