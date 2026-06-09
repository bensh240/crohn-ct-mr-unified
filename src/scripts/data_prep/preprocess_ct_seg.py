"""
preprocess_ct_seg.py - CT enterography -> bowel-localized 16 PNG slices (Phase 9 V21)
=====================================================================================
"Best for results" CT preprocessing, reusing the EXACT prior-knowledge approach from
Phase4_VLM_Extraction/training/run_segmentation.py (the MR crop pipeline):

  TotalSegmentator(fast) -> small_bowel + colon masks -> combine -> bounding box
  (+pad voxels) -> crop the volume to the bowel region -> THEN sample 16 axial slices.

This fixes the CT-specific problem that naive even-sampling grabs chest slices on
thoraco-abdominal CT: after cropping to small_bowel+colon, all 16 slices are bowel.
TotalSegmentator is CT-native, so masks are reliable on CT.

Falls back to the full volume if no bowel is detected (same as run_segmentation.py).

Pipeline per CT accession:
  pick axial series (config.json per-center root) -> load (SimpleITK) -> save tmp NIfTI
  -> TotalSegmentator -> small_bowel|colon mask -> bbox(+pad) -> crop -> HU window
  -> 16 linspace slices -> 224 center-crop -> sliceNN.png

Reuses helpers from preprocess_ct.py (series selection, config, windowing, PNG save).
GPU strongly recommended (TotalSegmentator fast ~10-30s/vol on GPU; minutes on CPU).
READ-ONLY on the DICOM store; writes only under --out-dir.

Usage (Argus03/04 GPU, conda crohn_vlm):
  python preprocess_ct_seg.py --sample 3 --sanity          # validate
  python preprocess_ct_seg.py --chunk-idx 0 --chunk-size 500   # SLURM array chunk
"""

import os
import sys
import argparse
import json
import shutil
import sqlite3
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preprocess_ct as base  # reuse pick_series, load_volume, window_normalize, etc.

try:
    import SimpleITK as sitk
except ImportError:
    raise SystemExit("SimpleITK required (env crohn_vlm)")
from PIL import Image

PAD = 10            # voxels of margin around the bowel bbox (matches run_segmentation.py)
NUM_SLICES = base.NUM_SLICES
CROP = base.CROP


def segment_and_crop(vol_img, tmp_dir, tag):
    """vol_img: SimpleITK image (real geometry). Returns a cropped HU numpy array (z,y,x).

    Runs TotalSegmentator(fast), unions small_bowel + colon, crops to bbox(+PAD).
    Falls back to the full array if no bowel is detected or segmentation fails.
    """
    from totalsegmentator.python_api import totalsegmentator

    arr = sitk.GetArrayFromImage(vol_img).astype(np.float32)   # (z,y,x) HU
    tmp_in = os.path.join(tmp_dir, f"seg_in_{tag}.nii.gz")
    tmp_out = os.path.join(tmp_dir, f"seg_out_{tag}")
    try:
        sitk.WriteImage(vol_img, tmp_in)
        totalsegmentator(tmp_in, tmp_out, fast=True, quiet=True,
                         roi_subset=["small_bowel", "colon"])
        mask = np.zeros(arr.shape, dtype=bool)
        for organ in ("small_bowel", "colon"):
            p = os.path.join(tmp_out, f"{organ}.nii.gz")
            if os.path.exists(p):
                m = sitk.GetArrayFromImage(sitk.ReadImage(p)) > 0   # (z,y,x), aligned
                if m.shape == arr.shape:
                    mask |= m
        coords = np.where(mask)
        if coords[0].size == 0:
            return arr, "no_bowel_full_volume"
        z1, z2 = max(0, coords[0].min() - PAD), min(arr.shape[0], coords[0].max() + PAD)
        y1, y2 = max(0, coords[1].min() - PAD), min(arr.shape[1], coords[1].max() + PAD)
        x1, x2 = max(0, coords[2].min() - PAD), min(arr.shape[2], coords[2].max() + PAD)
        return arr[z1:z2, y1:y2, x1:x2], f"cropped {arr.shape}->{(z2-z1, y2-y1, x2-x1)}"
    finally:
        if os.path.exists(tmp_in):
            os.remove(tmp_in)
        if os.path.isdir(tmp_out):
            shutil.rmtree(tmp_out, ignore_errors=True)


def process_accession(conn, accession, folders, out_dir, tmp_dir):
    # skip-existing: already has all 16 slices -> don't redo (cheap requeue/rerun)
    acc_done = os.path.join(out_dir, accession)
    if os.path.isdir(acc_done) and len([f for f in os.listdir(acc_done) if f.endswith(".png")]) >= NUM_SLICES:
        return ["__exists__"], "already done (skip)"
    relpath, source = base.pick_series(conn, accession)
    if relpath is None:
        return None, "no usable axial series"
    bdir = folders.get(str(source).lower().strip())
    if bdir is None:
        return None, f"no folder root for source={source!r}"
    series_dir = os.path.join(str(bdir), relpath)
    if not os.path.isdir(series_dir):
        return None, f"series dir missing: {series_dir}"

    _, vol_img = base.load_volume(series_dir)            # SimpleITK image w/ real geometry
    crop, msg = segment_and_crop(vol_img, tmp_dir, f"{accession}")
    vol = base.window_normalize(crop)                    # HU window -> [0,1] on the CROP
    slices = base.sample_slices(vol, NUM_SLICES)
    acc_out = os.path.join(out_dir, accession)
    os.makedirs(acc_out, exist_ok=True)
    paths = []
    for i, sl in enumerate(slices):
        p = os.path.join(acc_out, f"slice{i:02d}.png")
        base.center_crop_resize(sl).save(p)
        paths.append(p)
    return paths, msg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=base.DB_PATH)
    ap.add_argument("--config", default=base.CONFIG_PATH)
    ap.add_argument("--out-dir", default="./data/ct_slices_seg")
    ap.add_argument("--tmp-dir", default="./data/tmp_seg")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--chunk-idx", type=int, default=-1, help="SLURM array chunk index")
    ap.add_argument("--chunk-size", type=int, default=500)
    args = ap.parse_args()

    with open(args.config) as f:
        folders = json.load(f)["folders"]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    accs = [r[0] for r in conn.execute(
        "SELECT DISTINCT AccessionNumber FROM dicoms WHERE Modality='CT' AND relpath IS NOT NULL").fetchall()]
    if args.sample:
        accs = accs[:args.sample]
    elif args.chunk_idx >= 0:
        s = args.chunk_idx * args.chunk_size
        accs = accs[s:s + args.chunk_size]
        print(f"chunk {args.chunk_idx}: studies [{s}:{s+args.chunk_size}] -> {len(accs)}")

    ok = skipped = 0
    for acc in accs:
        try:
            paths, msg = process_accession(conn, acc, folders, args.out_dir, args.tmp_dir)
            if paths:
                ok += 1
                print(f"  OK {acc}: {msg}")
                if args.sanity and ok == 1:
                    sheet = Image.new("RGB", (CROP * 4, CROP * 4))
                    for i, p in enumerate(paths):
                        sheet.paste(Image.open(p), ((i % 4) * CROP, (i // 4) * CROP))
                    sheet.save(os.path.join(args.out_dir, f"_sanity_seg_{acc}.png"))
                    print(f"  sanity sheet -> {args.out_dir}/_sanity_seg_{acc}.png")
            else:
                skipped += 1
                print(f"  SKIP {acc}: {msg}")
        except Exception as e:
            skipped += 1
            print(f"  FAIL {acc}: {type(e).__name__}: {e}")
    conn.close()
    print(f"\nDone. ok={ok} skipped={skipped} / {len(accs)}")


if __name__ == "__main__":
    main()
