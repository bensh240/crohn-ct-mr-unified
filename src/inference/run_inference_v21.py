"""
run_inference_v21.py - V21 checkpoint -> per-accession 10 finding probabilities
================================================================================
Runs a trained V21 model over a v21-format CSV and writes pred_<label> per scan.
For Stage 2 (Cox) we run on MR scans (the cohort with outcomes).

Output CSV columns: accession_number, modality, pred_<10 labels>.

Usage (compute node, e.g. argus03/04):
  python inference/run_inference_v21.py \
      --ckpt ./data/checkpoints/v21c_unified/best_model.pt \
      --data-csv ./data/data/train.csv \
      --modality mr \
      --output ./data/predictions/v21c_mr_preds.csv
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "training"))
import train_mil_v21 as t  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-csv", required=True, help="a v21-format CSV (train/val/test) to run on")
    ap.add_argument("--modality", choices=["both", "mr", "ct"], default="mr")
    ap.add_argument("--output", required=True)
    ap.add_argument("--scaler", default=None, help="clinical_scaler.joblib from the run (else fit on this csv)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    conditioning = ckpt.get("conditioning", "dsbn")
    print(f"ckpt conditioning={conditioning} | device={device}")

    model = t.V21UnifiedMIL(num_labels=t.NUM_LABELS, hidden_dim=t.HIDDEN_DIM, dropout=0.0,
                            modality_dropout=0.0, clinical_dim=t.CLINICAL_DIM,
                            conditioning=conditioning).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_t = t._build_transforms()
    scaler = None
    if args.scaler and os.path.exists(args.scaler):
        import joblib
        scaler = joblib.load(args.scaler)
    mod_filter = None if args.modality == "both" else args.modality
    ds = t.CrohnV21Dataset(args.data_csv, transform=val_t, scaler=scaler,
                           is_train=(scaler is None), modality_filter=mod_filter)
    loader = DataLoader(ds, batch_size=t.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    print(f"running inference on {len(ds)} scans (modality={args.modality})")

    rows = []
    with torch.no_grad():
        for t2, t1, has_t1, lab, is_sick, accs, mod_ids, clin in loader:
            t2, t1, has_t1 = t2.to(device), t1.to(device), has_t1.to(device)
            mod_ids, clin = mod_ids.to(device), clin.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=t.USE_BF16):
                logits = model(t2, t1, has_t1, mod_ids, clinical=clin)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            for i, acc in enumerate(accs):
                row = {"accession_number": acc, "modality": args.modality}
                for j, l in enumerate(t.LABEL_NAMES):
                    row[f"pred_{l}"] = float(probs[i, j])
                rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fields = ["accession_number", "modality"] + [f"pred_{l}" for l in t.LABEL_NAMES]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
