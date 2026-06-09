"""
train_mil_v21.py - V21: Unified CT+MR model with modality conditioning
======================================================================
Phase 9. ONE script covers the whole experimental matrix via flags:

  --arch v21c   Unified shared backbone, trained on CT+MR jointly  (Option C)
  --arch v21b   Specialist: train on a single modality, no conditioning (Option B control)

  --conditioning {none,token,film,dsbn}   ablations A1/A2/A3 (dsbn = full V21-C)
  --modality {both,mr,ct}                  A5 (mr-only), single-modality specialist
  --t2-only                                A8 (contrast-free unified)
  --warm-start PATH                        init from V20 (or V19) checkpoint  <-- default ON

ARCHITECTURE (kept identical to V20 so we can warm-start every weight):
  Two DINOv3-B branches (frozen + LoRA) -> project 768->512 -> concat 1024
  -> [DSBN modality LayerNorm] -> 10x gated attention -> + clinical -> 10 sigmoids
  Modality token + FiLM condition the 768-d per-slice features (post-backbone).

  MR  -> T2 branch + T1 branch         (dual, as in V19/V20)
  CT  -> T2 branch only, T1 zeroed     (reuses the existing has_t1 mechanism)

This is joint training (shared backbone sees both modalities) -> the unification
claim holds. B is the control: same arch+init, but one modality, conditioning off.

Requires: transformers>=4.56, peft>=0.10  (DINOv3 is a gated HF repo)
"""

import os
import sys
import types

# --- triton.ops stub (same as V19/V20: peft->bnb->triton.ops breaks on triton>=3) ---
try:
    import triton  # noqa: F401
    try:
        import triton.ops  # noqa: F401
    except Exception:
        _ops = types.ModuleType("triton.ops")
        _perf = types.ModuleType("triton.ops.matmul_perf_model")
        _perf.early_config_prune = lambda *a, **k: None
        _perf.estimate_matmul_time = lambda *a, **k: None
        _ops.matmul_perf_model = _perf
        sys.modules["triton.ops"] = _ops
        sys.modules["triton.ops.matmul_perf_model"] = _perf
except Exception:
    pass
os.environ.setdefault("BITSANDBYTES_NOWELCOME", "1")

import csv
import json
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torch.amp import autocast
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
from datetime import datetime

from transformers import AutoModel
from peft import LoraConfig, get_peft_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modules.conditioning import ModalityToken, FiLM, ModalityLayerNorm, MODALITY_IDS

# ======================== CONFIG ========================
# V21 dataset CSVs (built by Phase9_V21/scripts/make_master_table.py + label fill).
# Each row: accession_number, modality{mr,ct}, image_dir (16 PNG slices), t1_dir
# (MR only, may be ''), is_sick, the 10 LABEL_NAMES, and the CLINICAL_FEATURES.
V21_DATA_DIR = os.environ.get("V21_DATA_DIR", "./data")
OUTPUT_ROOT = os.environ.get("V21_CKPT_DIR", "./checkpoints")
# Warm-start source: the trained MRI DINOv3 model (Phase 8 V20).
V20_CKPT = os.environ.get("V21_WARMSTART_CKPT", "")

BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"
FEATURE_DIM = 768
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.1
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]  # DINOv3 attention naming

NUM_SLICES = 16
HIDDEN_DIM = 512
NUM_EPOCHS = 50
BATCH_SIZE = 2
ACCUM_STEPS = 4
LR_LORA = 5e-5
LR_HEAD = 1e-4
LR_COND = 1e-4              # modality token / FiLM / DSBN
WARMUP_EPOCHS = 5
WEIGHT_DECAY = 1e-2
DROPOUT = 0.5
MODALITY_DROPOUT = 0.2      # MR T1 dropout (unchanged from V19); CT has no T1 anyway
PATIENCE = 5   # was 12; empirically all V21 trainings reach their best by epoch 7, then plateau; 5 is plenty.
SEED = 42
USE_BF16 = False            # fp32 (bf16 unstable for V18/V19/V20)
GRAD_CLIP = 0.5
EMA_DECAY = 0.999
EMA_START_EPOCH = 3
DECORR_WEIGHT = 0.1
DECORR_PAIRS = [(1, 2)]

TIER_1_LABELS = ["ileum_inflammation", "ileum_wall_enhancement", "ileum_wall_thickness",
                 "ileum_dwi", "ileum_stenosis", "ileum_pre_stenotic_dil", "ileum_comb_sign"]
TIER_2_LABELS = ["ileum_fistula", "colon_inflammation", "ileum_mesenteric_edema"]
LABEL_NAMES = TIER_1_LABELS + TIER_2_LABELS
NUM_LABELS = len(LABEL_NAMES)
GAMMA_NEG_PER_LABEL = [4] * len(TIER_1_LABELS) + [6] * len(TIER_2_LABELS)

CONTINUOUS_CLINICAL = ["Time_to_index", "age_at_diagnosis", "CCI_at_diagnosis",
                       "disease_activity_5", "SES_points", "diagnostic_delay"]
HISTORY_BINARY = ["had_surgery_before", "had_steroid_dep_before",
                  "had_biologic_switch_before", "had_EIM_before", "had_perianal_before"]
OTHER_BINARY = ["sex", "has_clinical"]
CLINICAL_FEATURES = CONTINUOUS_CLINICAL + HISTORY_BINARY + OTHER_BINARY
CLINICAL_DIM = len(CLINICAL_FEATURES)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# =================== EMA / LOSS / SCHED (identical to V19) ===================

class EMAModel:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self, model):
        for n, p in model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


class TierAwareASL(nn.Module):
    def __init__(self, gamma_neg_per_label, gamma_pos=1, clip=0.05, pos_weights=None):
        super().__init__()
        self.gamma_neg = torch.tensor(gamma_neg_per_label, dtype=torch.float32)
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.pos_weights = pos_weights

    def forward(self, logits, targets):
        gamma_neg = self.gamma_neg.to(logits.device)
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)
        los_pos = targets * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=1e-8))
        pt_pos = xs_pos * targets
        pt_neg = xs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + gamma_neg.unsqueeze(0) * (1 - targets)
        w = torch.pow(1 - pt_pos - pt_neg, gamma)
        loss = -(los_pos + los_neg) * w
        if self.pos_weights is not None:
            weight = targets * self.pos_weights.unsqueeze(0) + (1 - targets) * 1.0
            loss = loss * weight
        return loss.mean()


def decorrelation_loss(logits, pairs):
    if not pairs:
        return torch.tensor(0.0, device=logits.device)
    probs = torch.sigmoid(logits.float())
    penalty = torch.tensor(0.0, device=logits.device)
    for i, j in pairs:
        pi, pj = probs[:, i], probs[:, j]
        pic, pjc = pi - pi.mean(), pj - pj.mean()
        cov = (pic * pjc).mean()
        si = pic.pow(2).mean().sqrt().clamp(min=1e-6)
        sj = pjc.pow(2).mean().sqrt().clamp(min=1e-6)
        penalty = penalty + (cov / (si * sj)).abs()
    return penalty / len(pairs)


class CosineWithWarmup:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            scale = (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg, base in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base * scale)


# =================== DATASET ===================

class CrohnV21Dataset(Dataset):
    """Unified CT+MR dataset. CSV columns: accession_number, modality, image_dir,
    t1_dir, is_sick, <10 labels>, <clinical features>.

    MR: image_dir = T2 slices, t1_dir = T1 slices (may be missing -> has_t1=0).
    CT: image_dir = CT slices, t1_dir = '' -> fed through the T2 branch, T1 zeroed.
    """

    def __init__(self, csv_path, transform=None, scaler=None, is_train=False,
                 modality_filter=None, t2_only=False):
        self.samples = []
        self.transform = transform
        self.t2_only = t2_only
        raw_cont, raw_bin = [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                modality = row["modality"].strip().lower()
                if modality_filter and modality != modality_filter:
                    continue
                mod_id = MODALITY_IDS[modality]
                t2_dir = row["image_dir"]
                acc = row["accession_number"]
                is_sick = int(row["is_sick"])
                labels = [int(float(row[n])) for n in LABEL_NAMES]

                t2_files = sorted(os.path.join(t2_dir, fn) for fn in os.listdir(t2_dir)
                                  if fn.endswith(".png"))[:NUM_SLICES]
                if not t2_files:
                    continue
                while len(t2_files) < NUM_SLICES:
                    t2_files.append(t2_files[-1])

                # T1 only for MR (CT has none); zeroed by has_t1 otherwise.
                t1_files, has_t1 = [], False
                t1_dir = row.get("t1_dir", "").strip()
                if modality == "mr" and not t2_only and t1_dir and os.path.isdir(t1_dir):
                    t1_files = sorted(os.path.join(t1_dir, fn) for fn in os.listdir(t1_dir)
                                      if fn.endswith(".png"))[:NUM_SLICES]
                    if len(t1_files) >= NUM_SLICES:
                        has_t1 = True
                    else:
                        t1_files = []

                raw_cont.append([float(row[c]) for c in CONTINUOUS_CLINICAL])
                raw_bin.append([float(row[c]) for c in HISTORY_BINARY + OTHER_BINARY])
                self.samples.append([t2_files, t1_files, labels, is_sick, acc, has_t1, mod_id, None])

        raw_cont = np.array(raw_cont, dtype=np.float64)
        raw_bin = np.array(raw_bin, dtype=np.float64)
        if is_train:
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(raw_cont)
        else:
            assert scaler is not None
            self.scaler = scaler
            scaled = self.scaler.transform(raw_cont)
        for i in range(len(self.samples)):
            self.samples[i][7] = list(scaled[i]) + list(raw_bin[i])

    def __len__(self):
        return len(self.samples)

    def modality_ids(self):
        return np.array([s[6] for s in self.samples])

    def label_matrix(self):
        return np.array([s[2] for s in self.samples])

    def __getitem__(self, idx):
        t2_paths, t1_paths, labels, is_sick, acc, has_t1, mod_id, clinical = self.samples[idx]
        t2_imgs, t1_imgs = [], []
        for i in range(NUM_SLICES):
            t2 = Image.open(t2_paths[i]).convert("RGB")
            if self.transform:
                t2 = self.transform(t2)
            t2_imgs.append(t2)
            if has_t1 and t1_paths:
                t1 = Image.open(t1_paths[i]).convert("RGB")
                if self.transform:
                    t1 = self.transform(t1)
                t1_imgs.append(t1)
            else:
                t1_imgs.append(torch.zeros_like(t2))
        return (torch.stack(t2_imgs), torch.stack(t1_imgs),
                torch.tensor(1.0 if has_t1 else 0.0),
                torch.tensor(labels, dtype=torch.float32),
                is_sick, acc, torch.tensor(mod_id, dtype=torch.long),
                torch.tensor(clinical, dtype=torch.float32))


# =================== MODEL ===================

class DINOv3Branch(nn.Module):
    def __init__(self):
        super().__init__()
        base = AutoModel.from_pretrained(BACKBONE, attn_implementation="eager")
        for p in base.parameters():
            p.requires_grad = False
        cfg = LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                         target_modules=LORA_TARGETS, bias="none")
        self.model = get_peft_model(base, cfg)
        self.feature_dim = FEATURE_DIM

    def forward(self, x):
        return self.model(pixel_values=x, output_hidden_states=False).last_hidden_state[:, 0, :]


class IndependentGatedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        a = self.attention_w(self.attention_V(h) * self.attention_U(h)).squeeze(-1)
        return F.softmax(a.float(), dim=1).to(h.dtype)


class V21UnifiedMIL(nn.Module):
    """Dual DINOv3 branches + modality conditioning. Submodule names match V20
    (t2_backbone/t1_backbone/t2_transform/t1_transform/attention_heads/classifiers)
    so a V20 checkpoint warm-starts every non-conditioning weight."""

    def __init__(self, num_labels=10, hidden_dim=512, dropout=0.5, modality_dropout=0.2,
                 clinical_dim=0, conditioning="dsbn"):
        super().__init__()
        self.num_labels = num_labels
        self.modality_dropout = modality_dropout
        self.clinical_dim = clinical_dim
        self.conditioning = conditioning
        self.use_token = conditioning in ("token", "film", "dsbn")
        self.use_film = conditioning in ("film", "dsbn")
        self.use_dsbn = conditioning == "dsbn"

        self.t2_backbone = DINOv3Branch()
        self.t1_backbone = DINOv3Branch()
        feat_dim = self.t2_backbone.feature_dim  # 768

        # --- modality conditioning (new params; warm-start loads them as missing) ---
        if self.use_token:
            self.mod_token = ModalityToken(feat_dim)
        if self.use_film:
            self.film = FiLM(feat_dim, feat_dim)        # shared FiLM, applied per branch

        self.t2_transform = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                          nn.ReLU(), nn.Dropout(dropout))
        self.t1_transform = nn.Sequential(nn.Linear(feat_dim, hidden_dim), nn.LayerNorm(hidden_dim),
                                          nn.ReLU(), nn.Dropout(dropout))

        fused_dim = hidden_dim * 2
        if self.use_dsbn:
            self.dsbn = ModalityLayerNorm(fused_dim)    # DSBN inside MIL pooling

        attn_hidden = fused_dim // 4
        self.attention_heads = nn.ModuleList(
            [IndependentGatedAttention(fused_dim, attn_hidden) for _ in range(num_labels)])
        clf_in = fused_dim + clinical_dim
        self.classifiers = nn.ModuleList([
            nn.Sequential(nn.Linear(clf_in, hidden_dim), nn.LayerNorm(hidden_dim),
                          nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
            for _ in range(num_labels)])

    def _condition(self, feats, token):
        if self.use_film:
            return self.film(feats, token)
        return feats + token.unsqueeze(1)  # token-only

    def forward(self, t2_images, t1_images, has_t1_mask, modality_ids, clinical=None):
        B, S, C, H, W = t2_images.shape
        t2_feats = self.t2_backbone(t2_images.view(B * S, C, H, W)).view(B, S, -1)
        t1_feats = self.t1_backbone(t1_images.view(B * S, C, H, W)).view(B, S, -1)

        if self.use_token:
            tok = self.mod_token(modality_ids)          # (B, 768)
            t2_feats = self._condition(t2_feats, tok)
            t1_feats = self._condition(t1_feats, tok)

        t2_h = self.t2_transform(t2_feats)
        t1_h = self.t1_transform(t1_feats)

        if self.training and self.modality_dropout > 0:
            drop = torch.bernoulli(torch.full((B,), 1.0 - self.modality_dropout, device=t2_h.device))
            t1_active = drop * has_t1_mask
            t1_h = t1_h * t1_active.view(B, 1, 1) * (1.0 / (1.0 - self.modality_dropout))
        else:
            t1_h = t1_h * has_t1_mask.view(B, 1, 1)

        fused = torch.cat([t2_h, t1_h], dim=-1)
        if self.use_dsbn:
            fused = self.dsbn(fused, modality_ids)

        logits_list = []
        for i in range(self.num_labels):
            a_i = self.attention_heads[i](fused)
            bag = torch.bmm(a_i.unsqueeze(1), fused).squeeze(1)
            if self.clinical_dim > 0 and clinical is not None:
                bag = torch.cat([bag, clinical], dim=-1)
            logits_list.append(self.classifiers[i](bag))
        return torch.cat(logits_list, dim=1)


def warm_start(model, ckpt_path):
    """Load V20 (DINOv2/v3) weights; conditioning params stay at init (strict=False)."""
    if not os.path.exists(ckpt_path):
        print(f"[warm-start] checkpoint not found: {ckpt_path} -> training from DINOv3 pretrained only")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    loaded = len(sd) - len(unexpected)
    print(f"[warm-start] from {ckpt_path}")
    print(f"  loaded {loaded}/{len(sd)} tensors | missing {len(missing)} (conditioning/new) | unexpected {len(unexpected)}")


# =================== TRANSFORMS ===================

def _build_transforms():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_t = transforms.Compose([
        transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10), transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), transforms.ToTensor(), norm])
    val_t = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
    return train_t, val_t


def build_sampler(train_ds):
    """Oversample the minority modality (CT ~1/3 of MR) — plan §2 WeightedRandomSampler."""
    mids = train_ds.modality_ids()
    counts = np.bincount(mids, minlength=len(MODALITY_IDS)).astype(np.float64)
    counts[counts == 0] = 1.0
    w = (1.0 / counts)[mids]
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(mids), replacement=True)


# =================== TRAIN / EVAL ===================

def train(args, device):
    set_seed(SEED)
    out_dir = os.path.join(OUTPUT_ROOT, args.run_name)
    os.makedirs(out_dir, exist_ok=True)
    train_t, val_t = _build_transforms()
    mod_filter = None if args.modality == "both" else args.modality
    data_dir = args.data_dir if args.data_dir else V21_DATA_DIR
    print(f"  data_dir = {data_dir}")

    train_ds = CrohnV21Dataset(os.path.join(data_dir, "train.csv"), transform=train_t,
                               is_train=True, modality_filter=mod_filter, t2_only=args.t2_only)
    val_ds = CrohnV21Dataset(os.path.join(data_dir, "val.csv"), transform=val_t,
                             scaler=train_ds.scaler, modality_filter=mod_filter, t2_only=args.t2_only)
    print(f"Train {len(train_ds)} | Val {len(val_ds)} | modality={args.modality} | "
          f"arch={args.arch} | conditioning={args.conditioning}")
    print(f"  train modality counts: {np.bincount(train_ds.modality_ids(), minlength=2).tolist()} [mr, ct]")
    joblib.dump(train_ds.scaler, os.path.join(out_dir, "clinical_scaler.joblib"))

    pos = train_ds.label_matrix().sum(axis=0)
    pos_weights = torch.tensor((len(train_ds) - pos) / np.maximum(pos, 1), dtype=torch.float32).to(device)

    # Joint CT+MR training -> weighted sampler. Single-modality -> plain shuffle.
    sampler = build_sampler(train_ds) if (args.arch == "v21c" and args.modality == "both") else None
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, shuffle=(sampler is None),
                              num_workers=4, pin_memory=True, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
                            pin_memory=True, persistent_workers=True)

    conditioning = "none" if args.arch == "v21b" else args.conditioning
    model = V21UnifiedMIL(num_labels=NUM_LABELS, hidden_dim=HIDDEN_DIM, dropout=DROPOUT,
                          modality_dropout=(0.0 if args.t2_only else MODALITY_DROPOUT),
                          clinical_dim=CLINICAL_DIM, conditioning=conditioning).to(device)
    if args.warm_start:
        warm_start(model, args.warm_start)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total {total:,} | trainable {trainable:,} ({100*trainable/total:.2f}%)")

    lora_p, cond_p, head_p = [], [], []
    cond_prefixes = ("mod_token", "film", "dsbn")
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_" in n:
            lora_p.append(p)
        elif n.startswith(cond_prefixes):
            cond_p.append(p)
        else:
            head_p.append(p)
    groups = [{"params": lora_p, "lr": LR_LORA}, {"params": head_p, "lr": LR_HEAD}]
    if cond_p:
        groups.append({"params": cond_p, "lr": LR_COND})
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)
    scheduler = CosineWithWarmup(optimizer, WARMUP_EPOCHS, NUM_EPOCHS)
    criterion = TierAwareASL(GAMMA_NEG_PER_LABEL, gamma_pos=1, clip=0.05, pos_weights=pos_weights)
    ema = EMAModel(model, decay=EMA_DECAY)

    best, patience, history = 0.0, 0, []
    for epoch in range(NUM_EPOCHS):
        t0 = datetime.now()
        scheduler.step(epoch)
        model.train()
        nan_count = 0
        optimizer.zero_grad()
        for step, (t2, t1, has_t1, labels, _, _, mod_ids, clin) in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")):
            t2, t1, has_t1 = t2.to(device), t1.to(device), has_t1.to(device)
            labels, mod_ids, clin = labels.to(device), mod_ids.to(device), clin.to(device)
            with autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
                logits = model(t2, t1, has_t1, mod_ids, clinical=clin)
            logits = logits.float().clamp(-15.0, 15.0)
            loss = (criterion(logits, labels) + DECORR_WEIGHT * decorrelation_loss(logits, DECORR_PAIRS)) / ACCUM_STEPS
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                optimizer.zero_grad()
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            if (step + 1) % ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                if epoch >= EMA_START_EPOCH:
                    ema.update(model)
        if nan_count > len(train_loader) * 0.1:
            print("[FATAL] >10% NaN steps. Stopping.")
            break

        use_ema = epoch >= EMA_START_EPOCH
        if use_ema:
            ema.apply_shadow(model)
        v_macro, v_aucs = evaluate_loader(model, val_loader, device)
        if use_ema:
            ema.restore(model)
        print(f"Epoch {epoch+1:3d} | Val mAUC={v_macro:.4f}{' [EMA]' if use_ema else ''} | "
              f"{(datetime.now()-t0).total_seconds():.0f}s")
        history.append({"epoch": epoch + 1, "val_macro": v_macro,
                        "val_per_label": dict(zip(LABEL_NAMES, v_aucs))})
        if v_macro > best:
            best, patience = v_macro, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch + 1,
                        "val_macro_auc": v_macro, "args": vars(args), "backbone": BACKBONE,
                        "conditioning": conditioning, "version": "v21"},
                       os.path.join(out_dir, "best_model.pt"))
            print(f"  -> New best mAUC={best:.4f}")
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    with open(os.path.join(out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Best val mAUC: {best:.4f}")


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    preds, labels = [], []
    for t2, t1, has_t1, lab, _, _, mod_ids, clin in loader:
        t2, t1, has_t1 = t2.to(device), t1.to(device), has_t1.to(device)
        mod_ids, clin = mod_ids.to(device), clin.to(device)
        with autocast("cuda", dtype=torch.bfloat16, enabled=USE_BF16):
            logits = model(t2, t1, has_t1, mod_ids, clinical=clin)
        preds.append(torch.sigmoid(logits).float().cpu().numpy())
        labels.append(lab.numpy())
    preds = np.nan_to_num(np.concatenate(preds), nan=0.5)
    labels = np.concatenate(labels)
    aucs = [roc_auc_score(labels[:, i], preds[:, i]) if len(np.unique(labels[:, i])) > 1 else 0.5
            for i in range(NUM_LABELS)]
    return float(np.mean(aucs)), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["v21c", "v21b"], default="v21c")
    ap.add_argument("--conditioning", choices=["none", "token", "film", "dsbn"], default="dsbn")
    ap.add_argument("--modality", choices=["both", "mr", "ct"], default="both")
    ap.add_argument("--t2-only", action="store_true")
    ap.add_argument("--warm-start", type=str, default=V20_CKPT,
                    help="checkpoint to init from; '' to disable")
    ap.add_argument("--run-name", type=str, required=True)
    ap.add_argument("--phase", choices=["all", "train"], default="all")
    ap.add_argument("--data-dir", type=str, default="",
                    help="override V21_DATA_DIR (used for nested-CV fold subdirs)")
    args = ap.parse_args()
    if args.arch == "v21b" and args.modality == "both":
        raise SystemExit("v21b is a single-modality specialist: pass --modality mr or --modality ct")

    print(f"Start {datetime.now()} | run={args.run_name}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(args, device)
    print(f"End {datetime.now()}")


if __name__ == "__main__":
    main()
