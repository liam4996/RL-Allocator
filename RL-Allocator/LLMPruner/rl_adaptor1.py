# LLMPruner/rl_adaptor.py
# Minimal RL adaptor for layer-wise sparsity assignment in AdaptPruner
# Location: AdaptPruner/LLMPruner/rl_adaptor.py

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========= 1) Per-layer features =========
@dataclass
class LayerFeat:
    # If any field is unavailable, use 0.0 as a placeholder; it won't block execution
    width: float = 0.0               # d_model or FFN width
    n_heads: float = 0.0
    d_model: float = 0.0
    mlp_ratio: float = 0.0
    imp_mag: float = 0.0             # magnitude proxy (mean/norm of |W|)
    imp_first: float = 0.0           # first-order proxy (|w * grad|)
    hess_trace: float = 0.0          # Hutchinson-approx. trace (can be 0)
    latency_ms: float = 0.0          # per-layer latency under a representative config (can be 0)
    mem_mb: float = 0.0              # peak memory contribution (can be 0)
    flops_m: float = 0.0             # estimated FLOPs (can be 0)
    prev_sparsity: float = 0.0       # pruned ratio from the previous stage

    def to_vec(self) -> List[float]:
        return [
            self.width, self.n_heads, self.d_model, self.mlp_ratio,
            self.imp_mag, self.imp_first, self.hess_trace,
            self.latency_ms, self.mem_mb, self.flops_m, self.prev_sparsity
        ]

# ========= 2) Policy network =========
class PolicyMLP(nn.Module):
    """Input: per-layer feature vectors; Output: logits over sparsity bins"""
    def __init__(self, in_dim: int, n_bins: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, n_bins)
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # [L, B]

# ========= 3) Budget projection =========
def project_to_budget(sparsities: torch.Tensor,
                      target_avg: Optional[float] = None,
                      hard_cap: float = 0.95) -> torch.Tensor:
    """Project per-layer sparsities toward a target average and apply an upper cap"""
    s = sparsities.clamp(0.0, hard_cap)
    if target_avg is None:
        return s
    cur = float(s.mean().item())
    if abs(cur - target_avg) < 1e-6:
        return s
    scale = (target_avg / max(cur, 1e-8))
    s = (s * scale).clamp(0.0, hard_cap)
    # fine-tune the mean
    s = s + (target_avg - s.mean())
    return s.clamp(0.0, hard_cap)

# ========= 4) Reward aggregation =========
def default_reward(metrics: Dict[str, float],
                   weights: Dict[str, float]) -> float:
    """
    metrics: {'delta_ppl', 'latency_gain', 'mem_gain', 'flops_gain', 'metric_var'}
    weights: {'alpha','b_latency','b_mem','b_flops','var_penalty'}
    R = -alpha*ΔPPL + b1*lat + b2*mem + b3*flops - var_penalty*var
    """
    alpha = weights.get('alpha', 1.0)
    b1 = weights.get('b_latency', 1.0)
    b2 = weights.get('b_mem', 0.0)
    b3 = weights.get('b_flops', 0.0)
    vp = weights.get('var_penalty', 0.0)
    return (-alpha * metrics.get('delta_ppl', 0.0)
            + b1 * metrics.get('latency_gain', 0.0)
            + b2 * metrics.get('mem_gain', 0.0)
            + b3 * metrics.get('flops_gain', 0.0)
            - vp * metrics.get('metric_var', 0.0))

# ========= 5) Adaptor =========
class RLAdaptor:
    """
    Usage:
      adaptor = RLAdaptor(n_bins=6, bins=[0,0.1,0.2,0.3,0.4,0.5])
      spars = adaptor.assign(layer_feats, target_avg_sparsity=0.30)
      # Optional learning:
      adaptor.learn(layer_feats, apply_fn, eval_fn, target_avg_sparsity=0.30, steps=80)
    """
    def __init__(self,
                 n_bins: int = 6,
                 bins: Optional[List[float]] = None,
                 reward_weights: Optional[Dict[str, float]] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.bins = bins if bins is not None else [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        assert len(self.bins) == n_bins
        self.n_bins = n_bins
        self.device = device
        self.policy: Optional[PolicyMLP] = None
        self.reward_weights = reward_weights or {'alpha': 1.0, 'b_latency': 1.0}
        self._loaded_state = None  # for lazy load

    # ---- Utility: feature normalization ----
    @staticmethod
    def _normalize(feats: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        mean, std = feats.mean(0, keepdim=True), feats.std(0, keepdim=True)
        std = torch.where(std < eps, torch.ones_like(std), std)
        return (feats - mean) / std

    def _maybe_build(self, in_dim: int):
        if self.policy is None:
            self.policy = PolicyMLP(in_dim=in_dim, n_bins=self.n_bins).to(self.device)
            if self._loaded_state is not None:
                self.policy.load_state_dict(self._loaded_state)
                self._loaded_state = None

    # ---- Inference: assign per-layer sparsity ----
    @torch.no_grad()
    def assign(self,
               layer_feats: List[LayerFeat],
               target_avg_sparsity: Optional[float] = None,
               temperature: float = 1.0,
               greedy: bool = True) -> List[float]:
        """
        Return per-layer sparsities (0–1) and project to meet target_avg_sparsity
        """
        x = torch.tensor([lf.to_vec() for lf in layer_feats], dtype=torch.float32, device=self.device)
        x = self._normalize(x)
        self._maybe_build(x.shape[1])
        logits = self.policy(x) / max(temperature, 1e-6)  # [L, B]
        probs = F.softmax(logits, dim=-1)
        if greedy:
            idx = probs.argmax(dim=-1)  # [L]
        else:
            m = torch.distributions.Categorical(probs=probs)
            idx = m.sample()
        s = torch.tensor([self.bins[i.item()] for i in idx], device=self.device)
        s = project_to_budget(s, target_avg=target_avg_sparsity)
        return s.clamp(0.0, 0.95).tolist()

    # ---- Training: single-step REINFORCE (minimal runnable version) ----
    def learn(self,
              layer_feats: List[LayerFeat],
              apply_fn: Callable[[List[float]], None],
              eval_fn: Callable[[], Dict[str, float]],
              target_avg_sparsity: Optional[float] = None,
              steps: int = 100,
              lr: float = 1e-3,
              entropy_bonus: float = 1e-3,
              grad_clip: float = 1.0) -> None:
        """
        Per step: propose sparsities -> apply once (prune + optional light recovery) -> evaluate -> update policy with scalar reward
        - apply_fn: receives the per-layer sparsity list; performs pruning/recovery internally (must be resettable or work on a copy)
        - eval_fn:  returns {'delta_ppl':..., 'latency_gain':..., 'mem_gain':..., 'flops_gain':..., 'metric_var':...}
        """
        x = torch.tensor([lf.to_vec() for lf in layer_feats], dtype=torch.float32, device=self.device)
        x = self._normalize(x)
        self._maybe_build(x.shape[1])

        optim = torch.optim.Adam(self.policy.parameters(), lr=lr)
        for t in range(steps):
            logits = self.policy(x)              # [L, B]
            logp = F.log_softmax(logits, dim=-1)
            probs = logp.exp()
            m = torch.distributions.Categorical(probs=probs)
            idx = m.sample()                     # [L]
            log_prob = m.log_prob(idx).sum()     # sum over layers

            s = torch.tensor([self.bins[i.item()] for i in idx], device=self.device)
            s = project_to_budget(s, target_avg=target_avg_sparsity)

            # Apply once and evaluate
            apply_fn(s.detach().tolist())
            metrics = eval_fn()
            R = default_reward(metrics, self.reward_weights)

            # REINFORCE loss (with entropy regularization)
            entropy = -(probs * logp).sum(dim=-1).mean()
            loss = -(R * log_prob) - entropy_bonus * entropy

            optim.zero_grad()
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clip)
            optim.step()

            if (t + 1) % max(1, steps // 10) == 0:
                print(f"[RL-Adaptor] step {t+1}/{steps} reward={R:.4f} avg_s={float(s.mean().item()):.3f}")

    # ---- Save/Load ----
    def save(self, path: str) -> None:
        assert self.policy is not None, "call assign()/learn() once to init the policy."
        torch.save({'state_dict': self.policy.state_dict(), 'bins': self.bins, 'n_bins': self.n_bins}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.bins = ckpt.get('bins', self.bins)
        self.n_bins = ckpt.get('n_bins', len(self.bins))
        self._loaded_state = ckpt['state_dict']
        self.policy = None  # Initialize on the next assign()/learn() based on in_dim
