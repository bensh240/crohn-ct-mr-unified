"""
preprocess_ct.py - CT enterography -> 16 PNG slices (Phase 9 V21)
================================================================
Produces, per CT accession, a directory of 16 PNG slices in the SAME format
the V21 dataset (and the V19/V20 MR pipeline) consumes, so CT plugs straight
into training without a new data path.

Recipe (plan §1c):
  HU window [-150, 250] -> clip -> min-max [0,1] -> (in-plane resample) ->
  sample 16 axial slices -> 224x224 center crop -> save sliceNN.png (RGB).

Series selection (CT studies have scouts/localizers/failed series):
  From dicoms rows for an accession, pick the axial volume with the largest
  z-extent whose slice thickness is sane (0.5-6.0 mm). Drops z<MIN_Z, junk
  thickness (0, >10 e.g. 15/560/600), and 1-slice scouts.

v1 caveat: "16 slices around bowel" -> we sample 16 EVENLY-SPACED axial slices
through the abdominal volume (no bowel localization yet). TODO: use Angeleene's
T2 segmentor / an abdominal crop to center on bowel. Flagged in PHASE9_STATE.

READ-ONLY on the DICOM store. Writes only under --out-dir.

Usage (Argus01, conda crohn_vlm):
  # sanity test on a few studies (also dumps a contact sheet)
  python preprocess_ct.py --dicom-root <ROOT> --sample 8 --sanity
  # full cohort
  python preprocess_ct.py --dicom-root <ROOT> --out-dir ./data/ct_slices
"""

import os
import argparse
import sqlite3
import numpy as np

try:
    import SimpleITK as sitk
except ImportError:
    raise SystemExit("SimpleITK required: conda install -c simpleitk simpleitk  (env crohn_vlm)")
from PIL import Image

DB_PATH = os.environ.get("V21_DB_PATH", "./epiirn_v0.0.6.db")   # switch to v0.0.7 once labels land
# relpath in `dicoms` is relative to a PER-CENTER base dir (Angeleene's CTE.py):
#   full_path = config['folders'][source.lower()] + '/' + relpath   (roots under /argusdata3)
CONFIG_PATH = "./db/config.json"
NUM_SLICES = 16
HU_MIN, HU_MAX = -150.0, 250.0
CROP = 224
MIN_Z = 40                 # reject scouts / thin stacks
THK_LO, THK_HI = 0.5, 6.0  # sane CT slice thickness (mm)


def connect_ro(p):
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def pick_series(conn, accession):
    """Return (relpath, source) of the best axial series for this CT accession, or (None, None).

    Prefers ImageType LIKE '%ORIGINAL%' (Angeleene's CTE.py filter for real,
    non-derived acquisitions) and the largest sane-thickness axial volume.
    """
    rows = conn.execute(
        "SELECT relpath, SliceThickness, NumberOfSlices, sitk_size, sitk_size_2, ImageType, source "
        "FROM dicoms WHERE AccessionNumber=? AND Modality='CT' AND relpath IS NOT NULL",
        (accession,)).fetchall()
    # Prefer ORIGINAL acquisitions; if none, fall back to all rows.
    original = [r for r in rows if r[5] and "ORIGINAL" in str(r[5]).upper()]
    candidates = original if original else rows
    best, best_src, best_z = None, None, -1
    for relpath, thk, nsl, sitk_size, z2, _imgtype, source in candidates:
        # z-extent: prefer sitk_size_2, fall back to NumberOfSlices
        try:
            z = int(z2) if z2 not in (None, "") else int(nsl or 0)
        except (TypeError, ValueError):
            z = 0
        if z < MIN_Z:
            continue
        if thk is not None and not (THK_LO <= float(thk) <= THK_HI):
            continue
        if z > best_z:
            best, best_src, best_z = relpath, source, z
    return best, best_src


def load_volume(series_dir):
    """Load a DICOM series directory into a 3D HU array (z, y, x)."""
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(series_dir)
    if not files:
        raise FileNotFoundError(f"no DICOM files under {series_dir}")
    reader.SetFileNames(files)
    img = reader.Execute()                       # rescale slope/intercept applied by reader
    return sitk.GetArrayFromImage(img).astype(np.float32), img


def window_normalize(vol):
    """HU window -> [0,1]."""
    vol = np.clip(vol, HU_MIN, HU_MAX)
    return (vol - HU_MIN) / (HU_MAX - HU_MIN)


def sample_slices(vol, k=NUM_SLICES):
    """k evenly-spaced axial slices through the central 80% of the stack."""
    z = vol.shape[0]
    lo, hi = int(0.10 * z), int(0.90 * z)
    idx = np.linspace(lo, max(lo, hi - 1), k).round().astype(int)
    return vol[idx]


def center_crop_resize(slice2d, size=CROP):
    img = Image.fromarray((slice2d * 255).astype(np.uint8))
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size))
    return img.convert("RGB")


def process_accession(conn, accession, folders, out_dir):
    relpath, source = pick_series(conn, accession)
    if relpath is None:
        return None, "no usable axial series"
    base = folders.get(str(source).lower().strip())
    if base is None:
        return None, f"no folder root for source={source!r}"
    series_dir = os.path.join(str(base), relpath)
    if not os.path.isdir(series_dir):
        return None, f"series dir missing: {series_dir}"
    vol, _ = load_volume(series_dir)
    vol = window_normalize(vol)
    slices = sample_slices(vol)
    acc_out = os.path.join(out_dir, accession)
    os.makedirs(acc_out, exist_ok=True)
    paths = []
    for i, sl in enumerate(slices):
        p = os.path.join(acc_out, f"slice{i:02d}.png")
        center_crop_resize(sl).save(p)
        paths.append(p)
    return paths, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DB_PATH)
    ap.add_argument("--config", default=CONFIG_PATH, help="EPIIRN config.json with per-center 'folders'")
    ap.add_argument("--out-dir", default=os.path.join(os.environ.get("V21_DATA_DIR", "./data"), "../ct_slices"))
    ap.add_argument("--sample", type=int, default=0, help="process only N studies (sanity)")
    ap.add_argument("--sanity", action="store_true", help="also write a contact sheet PNG")
    args = ap.parse_args()

    import json
    with open(args.config) as f:
        folders = json.load(f)["folders"]
    print(f"per-center DICOM roots: {folders}")

    conn = connect_ro(args.db)
    accs = [r[0] for r in conn.execute(
        "SELECT DISTINCT AccessionNumber FROM dicoms WHERE Modality='CT' AND relpath IS NOT NULL").fetchall()]
    if args.sample:
        accs = accs[:args.sample]
    os.makedirs(args.out_dir, exist_ok=True)

    ok = skipped = 0
    thk_report = []
    for acc in accs:
        try:
            paths, msg = process_accession(conn, acc, folders, args.out_dir)
            if paths:
                ok += 1
                if args.sanity and ok == 1:
                    sheet = Image.new("RGB", (CROP * 4, CROP * 4))
                    for i, p in enumerate(paths):
                        sheet.paste(Image.open(p), ((i % 4) * CROP, (i // 4) * CROP))
                    sheet.save(os.path.join(args.out_dir, f"_sanity_{acc}.png"))
                    print(f"  sanity sheet -> {args.out_dir}/_sanity_{acc}.png")
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
