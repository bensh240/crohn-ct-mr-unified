"""
conditioning.py - Modality-conditioning building blocks for V21-C
=================================================================
Three components, all keyed on a modality id in {0 = MR, 1 = CT}:

  1. ModalityToken  - nn.Embedding(2, dim) learned modality embedding.
  2. FiLM           - MLP(token) -> (gamma, beta); applies h*(1+gamma)+beta
                      to the per-slice backbone features (post-backbone, pre-MIL).
  3. DSBN-style LN  - one LayerNorm per modality, selected per-sample.

Trainable parameter budget (dim=768): token ~1.5k, FiLM ~1.2M (2-layer MLP),
DSBN ~3k. Backbone stays frozen; LoRA + these + the MIL head are all that train.

Each module degrades to a no-op when disabled, so the SAME model class covers:
  - V21-C full        (token + film + dsbn)
  - A2 (token + film) (dsbn off)
  - A1 (token only)   (film + dsbn off)
  - V21-B / A6 naive  (all off -> plain unified or single-modality backbone)

MODALITY_IDS is the single source of truth for the id<->name mapping.
"""

import torch
import torch.nn as nn

MODALITY_IDS = {"mr": 0, "ct": 1}
NUM_MODALITIES = len(MODALITY_IDS)


class ModalityToken(nn.Module):
    """Learned per-modality embedding. Returns (B, dim) given modality ids (B,)."""

    def __init__(self, dim, num_modalities=NUM_MODALITIES):
        super().__init__()
        self.embedding = nn.Embedding(num_modalities, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, modality_ids):
        return self.embedding(modality_ids)  # (B, dim)


class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on the modality token.

    token (B, dim) -> 2-layer MLP -> (gamma, beta) each (B, dim).
    Applied per-slice: h' = h * (1 + gamma) + beta, broadcast over the slice axis.
    gamma/beta init ~0 so the module starts as identity (stable warm-start).
    """

    def __init__(self, token_dim, feature_dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.net = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * feature_dim),
        )
        # Start as identity: last layer -> 0, so gamma=beta=0 initially.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.feature_dim = feature_dim

    def forward(self, features, token):
        """features: (B, S, D) per-slice; token: (B, token_dim)."""
        gamma, beta = self.net(token).chunk(2, dim=-1)      # each (B, D)
        gamma = gamma.unsqueeze(1)                          # (B, 1, D)
        beta = beta.unsqueeze(1)
        return features * (1.0 + gamma) + beta


class ModalityLayerNorm(nn.Module):
    """DSBN-style: a separate LayerNorm per modality, selected per sample.

    Avoids the BatchNorm cross-sample stats problem (our batch is tiny, fp32,
    and mixes modalities via a weighted sampler) by using LayerNorm instead.
    """

    def __init__(self, dim, num_modalities=NUM_MODALITIES):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(num_modalities)])

    def forward(self, x, modality_ids):
        """x: (B, ..., dim); modality_ids: (B,). Applies per-sample LN by modality."""
        out = torch.empty_like(x)
        for m, norm in enumerate(self.norms):
            mask = (modality_ids == m)
            if mask.any():
                out[mask] = norm(x[mask])
        return out


class ModalityConditioning(nn.Module):
    """Bundles token + FiLM + DSBN with flags, so one model class covers every ablation.

    level:
      'none'  -> no conditioning            (V21-B specialist, A6 naive-unified)
      'token' -> token added to features    (A1)
      'film'  -> token + FiLM                (A2)
      'dsbn'  -> token + FiLM + ModalityLN   (A3 = full V21-C)
    """

    LEVELS = ("none", "token", "film", "dsbn")

    def __init__(self, feature_dim, level="dsbn", token_dim=None):
        super().__init__()
        assert level in self.LEVELS, f"level must be one of {self.LEVELS}"
        self.level = level
        token_dim = token_dim or feature_dim
        self.use_token = level in ("token", "film", "dsbn")
        self.use_film = level in ("film", "dsbn")
        self.use_dsbn = level == "dsbn"

        if self.use_token:
            self.token = ModalityToken(token_dim)
        if self.use_film:
            self.film = FiLM(token_dim, feature_dim)
        if self.use_dsbn:
            self.dsbn = ModalityLayerNorm(feature_dim)

    def forward(self, features, modality_ids):
        """features: (B, S, D) per-slice backbone output. Returns conditioned (B, S, D)."""
        if not self.use_token:
            return features
        tok = self.token(modality_ids)                      # (B, token_dim)
        if self.use_film:
            features = self.film(features, tok)
        else:
            # token-only: add the modality embedding to every slice
            features = features + tok.unsqueeze(1)
        if self.use_dsbn:
            features = self.dsbn(features, modality_ids)
        return features
