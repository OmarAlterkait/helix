"""Per-cell flow-matching head (MAR-style "Diffusion Loss", flow variant).

Replaces the MSE/Gaussian-NLL value head: instead of predicting the conditional
MEAN of the charge coeffs (which blurs fluctuations -> var(pred)/var(true) < 1),
it models the full conditional p(values | cell-features) by velocity regression
along straight noise->data paths. Sampling re-injects the fluctuations the mean
throws away. Conditioned on the transformer's per-cell feature z (and, via z,
band/sigma which are already encoded upstream).

Why flow matching (not DDPM): plain velocity loss (no noise schedule), low-var
gradients, and 1-4 ODE steps to sample (vs ~100) -> cheap at 32k cells/event.
"""
import math
import torch
import torch.nn as nn


class FlowBlock(nn.Module):
    """Residual MLP block with AdaLN conditioning on (cell-feature + flow-time)."""
    def __init__(self, h, mult=4):
        super().__init__()
        self.n = nn.LayerNorm(h, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(h, mult * h), nn.SiLU(), nn.Linear(mult * h, h))
        self.ada = nn.Linear(h, 3 * h)
        nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)   # AdaLN-Zero: identity at init

    def forward(self, x, c):
        s, b, g = self.ada(c).chunk(3, -1)
        return x + g * self.mlp(self.n(x) * (1 + s) + b)


class FlowHead(nn.Module):
    """v_theta(y_t, t, z): per-cell velocity field over the n_slot value vector."""
    def __init__(self, d, n_slot, h=512, blocks=3):
        super().__init__()
        self.n_slot = n_slot
        self.in_proj = nn.Linear(n_slot, h)
        self.cond = nn.Linear(d, h)
        self.t_embed = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h))
        self.blocks = nn.ModuleList(FlowBlock(h) for _ in range(blocks))
        self.norm = nn.LayerNorm(h)
        self.out = nn.Linear(h, n_slot)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, y_t, t, z):
        # y_t (N, n_slot), t (N, 1) in [0,1], z (N, d)
        c = self.cond(z) + self.t_embed(t)
        x = self.in_proj(y_t)
        for blk in self.blocks:
            x = blk(x, c)
        return self.out(self.norm(x))                  # velocity (N, n_slot)


def flow_loss(head, z, y1, slot_mask, gen=None):
    """Conditional flow-matching (rectified/OT path). Loss on slot_mask only.
      y1 (N, n_slot) clean targets ; slot_mask (N, n_slot) bool where a real target lives."""
    N = y1.shape[0]
    y0 = torch.randn_like(y1) if gen is None else torch.randn(y1.shape, generator=gen, device=y1.device)
    t = (torch.rand(N, 1, device=y1.device) if gen is None
         else torch.rand(N, 1, generator=gen, device=y1.device))
    y_t = (1 - t) * y0 + t * y1
    v_star = y1 - y0
    v = head(y_t, t, z)
    se = (v - v_star) ** 2
    return se[slot_mask].mean()


@torch.no_grad()
def flow_sample(head, z, n_slot, steps=4, k=1, gen=None):
    """Integrate dy/dt = v_theta from N(0,I) to t=1 (Euler). Returns (k, N, n_slot) samples."""
    N = z.shape[0]
    outs = []
    for _ in range(k):
        y = (torch.randn(N, n_slot, device=z.device) if gen is None
             else torch.randn(N, n_slot, generator=gen, device=z.device))
        for i in range(steps):
            t = torch.full((N, 1), i / steps, device=z.device)
            y = y + (1.0 / steps) * head(y, t, z)
        outs.append(y)
    return torch.stack(outs, 0)
