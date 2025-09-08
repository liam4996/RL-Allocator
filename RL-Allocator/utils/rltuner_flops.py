# rltuner_flops.py
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- FLOPs ring weights & minimum granularity estimation ----------------

def estimate_flops_weight_per_layer(model, context_len=4096):
    """
    Estimate a FLOPs weight for each decoder layer (larger means more “expensive”).
    Approximation: Attention linear (4*d^2) + Attention interaction (~4*L*d) + MLP (2*d*d_ff)
    Returns: a float list of length [L]
    """
    layers = list(model.model.layers)
    L = len(layers)
    w = []
    for i, layer in enumerate(layers):
        d = layer.self_attn.num_heads * layer.self_attn.head_dim  # == hidden_size
        d_kv = layer.self_attn.num_key_value_heads * layer.self_attn.head_dim
        # Linear projections (q, k, v, o)
        attn_linear = 4.0 * d * d
        # Interaction part (QK^T and PV), roughly at hidden_size scale
        attn_interact = 4.0 * context_len * d
        # Two matmuls for the MLP
        d_ff = layer.mlp.gate_proj.out_features
        mlp_cost = 2.0 * d * d_ff
        w.append(attn_linear + attn_interact + mlp_cost)
    # Normalize to [0,1] to avoid overly large numbers
    m = max(w) if w else 1.0
    return [x / m for x in w]

def estimate_min_block_per_layer(model, group_size=64):
    """
    Estimate each layer's minimum movement granularity (expressed as *sparsity percentage*):
      Attention: whole head => 1 / num_heads
      MLP: group pruning => group_size / d_ff
    Return the per-layer minimum so the RL layer-level move can be finer; the actual
    dispatch will be aligned to submodule granularity by the lower layer.
    """
    layers = list(model.model.layers)
    mins = []
    for layer in layers:
        # Attention head-level granularity
        num_heads = max(1, int(layer.self_attn.num_heads))
        attn_block = 1.0 / float(num_heads)
        # MLP group granularity
        d_ff = int(layer.mlp.gate_proj.out_features)
        mlp_block = float(group_size) / float(d_ff) if d_ff > 0 else 1.0
        mins.append(min(attn_block, mlp_block))
    return mins

# ----------------- Simple two-head policy network (REINFORCE) -----------------

class FlopsEnv:
    def __init__(self, sparsity, flops_weight, min_block, lower, upper):
        """
        sparsity: [L] initial per-layer sparsity (0~1)
        flops_weight: [L] per-layer FLOPs weights (normalized to 0~1 is fine)
        min_block: [L] per-layer minimum movement step (ratio, e.g., 0.01 = 1%)
        lower, upper: [L] per-layer sparsity bounds
        """
        self.s = torch.tensor(sparsity, dtype=torch.float32)
        self.w = torch.tensor(flops_weight, dtype=torch.float32)
        self.step_unit = torch.tensor(min_block, dtype=torch.float32)
        self.lower, self.upper = torch.tensor(lower), torch.tensor(upper)
        self.target_sum = float(self.s.sum())
        self.L = len(sparsity)

    def flops(self, s):
        # Linear approximation: FLOPs ≈ Σ w_l * s_l
        return float((self.w * s).sum())

    def step(self, i, j):
        """Take one block from layer i and give it to layer j; return (reward, success_flag)."""
        if i == j:
            return 0.0, False
        delta = min(float(self.step_unit[i]), float(self.step_unit[j]))
        if (self.s[i] - delta < self.lower[i]) or (self.s[j] + delta > self.upper[j]):
            return 0.0, False
        prev = self.flops(self.s)
        s2 = self.s.clone()
        s2[i] -= delta
        s2[j] += delta
        # Keep the total sum exact (tweak for numerical error)
        diff = float(s2.sum() - self.target_sum)
        if abs(diff) > 1e-9:
            # Absorb the error into j (or i), as long as we don't violate bounds
            k = j if diff > 0 else i
            s2[k] -= diff
            if s2[k] < self.lower[k] - 1e-9 or s2[k] > self.upper[k] + 1e-9:
                return 0.0, False
        cur = self.flops(s2)
        self.s = s2
        return prev - cur, True  # FLOPs decrease is a positive reward

class TwoHeadPolicy(nn.Module):
    """Two classification heads: pick donor layer i and receiver layer j."""
    def __init__(self, in_dim, L, hidden=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.head_give = nn.Linear(hidden, L)
        self.head_take = nn.Linear(hidden, L)

    def forward(self, feats):
        # feats: [L, in_dim]  one row of features per layer
        x = self.backbone(feats)              # [L, hidden]
        g_logits = self.head_give(x).mean(0)  # [L]
        t_logits = self.head_take(x).mean(0)  # [L]
        return g_logits, t_logits

@torch.no_grad()
def _normalize_weights_(w):
    m = w.max().item() if isinstance(w, torch.Tensor) else max(w)
    if m <= 0: return w
    return w / m

def rl_tune_flops(
    s_init, w, I, depth, min_block, lower, upper,
    steps=400, moves_per_step=8, lr=1e-2, device="cuda",
    eval_hook=None, gate_cfg=None
):
    """
    s_init, w, I, depth, min_block, lower, upper: all are lists/tensors of length [L]
    eval_hook(s)->metric: optional external gating metric (e.g., PPL); gate_cfg controls threshold & frequency
    """
    L = len(s_init)
    env = FlopsEnv(s_init, w, min_block, lower, upper)
    feats = torch.stack([
        torch.tensor(I, dtype=torch.float32),
        torch.tensor(w, dtype=torch.float32),
        torch.tensor(depth, dtype=torch.float32),
        env.s.clone()
    ], dim=1)  # [L, 4]

    policy = TwoHeadPolicy(in_dim=feats.size(1), L=L).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline = 0.0

    if gate_cfg is None:
        gate_cfg = {"metric":"ppl", "max_increase":0.5, "validate_every":0, "rollback":True}

    # Segmental reward buffers (for “external gate: no-reward admission”)
    segment_logps, segment_rewards = [], []
    last_safe_state = env.s.clone()

    for _ in range(steps):
        logps, rewards = [], []
        for mv in range(moves_per_step):
            g_logits, t_logits = policy(feats.to(device))
            dist_g = torch.distributions.Categorical(logits=g_logits)
            dist_t = torch.distributions.Categorical(logits=t_logits)
            gi = int(dist_g.sample().item())
            tj = int(dist_t.sample().item())

            r, ok = env.step(gi, tj)
            if not ok:
                # Invalid action: give a tiny gradient signal to avoid collapse
                continue

            # Record RL trajectory
            lp = (dist_g.log_prob(torch.tensor(gi, device=feats.device)) +
                  dist_t.log_prob(torch.tensor(tj, device=feats.device)))
            logps.append(lp)
            rewards.append(r)
            segment_logps.append(lp)
            segment_rewards.append(r)

            # Update the s component in the state features
            feats[gi, 3] = env.s[gi]
            feats[tj, 3] = env.s[tj]

            # --- External gating: periodic validation (no-reward / optional rollback) ---
            if eval_hook and gate_cfg.get("validate_every", 0) > 0:
                if (len(segment_rewards) % gate_cfg["validate_every"]) == 0:
                    metric = float(eval_hook(env.s.tolist()))
                    bad = False
                    if gate_cfg.get("metric", "ppl") == "ppl":
                        # Assume eval_hook returns current ppl; as an example, > max_increase is treated as “bad”.
                        bad = (metric > gate_cfg.get("max_increase", 0.5))
                    if bad:
                        # 1) No admission to reward: zero out this segment's rewards
                        segment_rewards = [0.0] * len(segment_rewards)
                        # 2) Roll back
                        if gate_cfg.get("rollback", True):
                            env.s = last_safe_state.clone()
                            # Sync s in the state features as well
                            feats[:, 3] = env.s.clone()
                    else:
                        last_safe_state = env.s.clone()
                        # Let the segment naturally remain in logps/rewards (simpler implementation)

        if not logps:
            continue
        R = sum(rewards)
        baseline = 0.9 * baseline + 0.1 * R
        loss = -torch.stack(logps).sum() * (R - baseline)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

    return env.s.tolist()
