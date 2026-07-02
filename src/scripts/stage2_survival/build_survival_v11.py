#!/usr/bin/env python
"""Assemble survival_v11 = survival_v09 + CORRECTED med/lab features (accession-linked).
Same schema as v10, but with the SZMC linkage bug fixed."""
import pandas as pd

V09 = "/synology-data/users/bensh240/Phase9_V21/survival/survival_v09.csv"
MED = "/synology-data/users/bensh240/Phase9_V21/data/med_features_per_scan_fixed.csv"
LAB = "/synology-data/users/bensh240/Phase9_V21/data/lab_features_per_scan_fixed.csv"
OUT = "/synology-data/users/bensh240/Phase9_V21/survival/survival_v11.csv"

v09 = pd.read_csv(V09); v09["accession_number"] = v09["accession_number"].astype(str)
med = pd.read_csv(MED); med["accession_number"] = med["accession_number"].astype(str)
lab = pd.read_csv(LAB); lab["accession_number"] = lab["accession_number"].astype(str)
print(f"v09 {v09.shape} | med {med.shape} | lab {lab.shape}")

df = v09.merge(med.drop(columns=["patient_id"], errors="ignore"), on="accession_number", how="left")
df = df.merge(lab.drop(columns=["patient_id"], errors="ignore"), on="accession_number", how="left")
df.to_csv(OUT, index=False)
print(f"WROTE {OUT}  shape={df.shape}")

# sanity vs the old (buggy) v10
print("\n=== v11 sanity: med_advanced_ever_before rate by center ===")
print(df.groupby("center")["med_advanced_ever_before"].mean().round(3).to_string())
allc = ["med_advanced_ever_before", "med_steroid_rx_ever_before", "med_asa_ever_before", "med_imm_ever_before"]
df["all4"] = (df[allc].fillna(0) > 0).all(axis=1)
print("all-4-groups rate by center (was ~1.0 for SZMC in v10):")
print(df.groupby("center")["all4"].mean().round(3).to_string())
