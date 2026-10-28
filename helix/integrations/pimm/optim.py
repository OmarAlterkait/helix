"""``FlatAdamW``: AdamW for a model whose weights are bf16 (``bf16_params``).

The fp32 master weights and both moments live here, one flat buffer per param
group; each step is

  cat(grads) per group (bf16)  ->  one global norm  ->  ONE Triton kernel per
  group: clip scale, AdamW, write fp32 master / m / v and a bf16 copy  ->
  copy the bf16 copy into the parameters.

What it replaces, measured at d1536 / 18.5k tokens (fused/RESULTS2.md): autocast
re-casting every fp32 weight to bf16 every forward, the fp32 casts of every
gradient, clip_grad_norm_'s separate multiply, and torch's fused AdamW -- about
52 bytes per parameter per step down to ~38.

Per-group ``lr`` / ``weight_decay`` are honoured, so muP's groups
(``FMModel.param_groups``) and any LambdaLR schedule work unchanged. With
identical gradients it matches ``torch.optim.AdamW`` + ``clip_grad_norm_`` to
~1e-7 relative (tests/test_flat_adamw.py).

Checkpoints: the state is exposed per parameter (``master``, ``exp_avg``,
``exp_avg_sq``, ``step``) as views into the flat buffers, so
``torch.distributed.checkpoint.state_dict`` -- what pimm saves with -- sees an
ordinary optimizer. ``load_state_dict`` is overridden because torch's casts
every state tensor to its parameter's dtype, which would round the fp32 master
and moments to bf16 on every resume.

Clipping is the optimizer's (``clip``, set by FMTrainer from ``cfg.clip_grad``):
the trainer must not also run clip_grad_norm_, and runs no GradScaler (bf16
needs none).
"""

from __future__ import annotations

import torch

from pimm.utils.optimizer import OPTIMIZERS

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _adamw_kernel(G, M32, Mo, Vo, P16, N, scale_ptr, lr, b1, b2, eps, wd, bc1, bc2,
                      BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        msk = i < N
        sc = tl.load(scale_ptr)
        g = tl.load(G + i, mask=msk, other=0.).to(tl.float32) * sc
        p = tl.load(M32 + i, mask=msk, other=0.)
        m = tl.load(Mo + i, mask=msk, other=0.) * b1 + (1 - b1) * g
        v = tl.load(Vo + i, mask=msk, other=0.) * b2 + (1 - b2) * g * g
        p = p * (1 - lr * wd) - lr * (m / bc1) / (tl.sqrt(v / bc2) + eps)
        tl.store(M32 + i, p, mask=msk)
        tl.store(Mo + i, m, mask=msk)
        tl.store(Vo + i, v, mask=msk)
        tl.store(P16 + i, p.to(tl.bfloat16), mask=msk)
except ImportError:                                   # CPU-only installs
    triton = None

_BLOCK = 4096


@OPTIMIZERS.register_module()
class FlatAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, fused=None, foreach=None):
        # fused/foreach: accepted so an AdamW config switches by `type` alone
        super().__init__(params, dict(lr=lr, betas=tuple(betas), eps=eps,
                                      weight_decay=weight_decay))
        self.clip = None
        self._t = 0
        self._flat = []
        for g in self.param_groups:
            ps = list(g["params"])
            bad = [tuple(p.shape) for p in ps if p.dtype != torch.bfloat16]
            if bad:
                raise TypeError(f"FlatAdamW holds the fp32 master itself and needs bf16 "
                                f"parameters (model.bf16_params=True); got non-bf16 shapes {bad[:3]}")
            master = torch.cat([p.detach().float().reshape(-1) for p in ps])
            m, v = torch.zeros_like(master), torch.zeros_like(master)
            out = torch.empty_like(master, dtype=torch.bfloat16)
            views, off = [], 0
            for p in ps:
                n = p.numel()
                sl = lambda t: t[off:off + n].view_as(p)
                self.state[p] = {"step": torch.zeros((), dtype=torch.float32),
                                 "master": sl(master), "exp_avg": sl(m), "exp_avg_sq": sl(v)}
                views.append(sl(out))
                off += n
            self._flat.append(dict(params=ps, master=master, m=m, v=v, out=out, views=views))
        self.master_of = {p: self.state[p]["master"] for fl in self._flat for p in fl["params"]}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not any(p.grad is not None for fl in self._flat for p in fl["params"]):
            return loss                                  # every gradient dropped (FiniteGuard)
        self._t += 1
        grads = [torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                            for p in fl["params"]]) for fl in self._flat]
        dev = grads[0].device
        if self.clip:
            sq = torch.stack([torch.linalg.vector_norm(g, dtype=torch.float32) for g in grads]).square().sum()
            scale = torch.clamp(float(self.clip) / (sq.sqrt() + 1e-6), max=1.0).reshape(1)
        else:
            scale = torch.ones(1, device=dev)
        for fl, g, grp in zip(self._flat, grads, self.param_groups):
            b1, b2 = grp["betas"]
            lr, wd = float(grp["lr"]), float(grp["weight_decay"])
            n = g.numel()
            if g.is_cuda and triton is not None:
                _adamw_kernel[(triton.cdiv(n, _BLOCK),)](
                    g, fl["master"], fl["m"], fl["v"], fl["out"], n, scale, lr, b1, b2,
                    grp["eps"], wd, 1 - b1 ** self._t, 1 - b2 ** self._t, BLOCK=_BLOCK)
            else:                                          # reference arithmetic (CPU tests)
                gf = g.float() * scale
                fl["m"].mul_(b1).add_(gf, alpha=1 - b1)
                fl["v"].mul_(b2).addcmul_(gf, gf, value=1 - b2)
                fl["master"].mul_(1 - lr * wd).sub_(
                    lr * (fl["m"] / (1 - b1 ** self._t)) /
                    ((fl["v"] / (1 - b2 ** self._t)).sqrt() + grp["eps"]))
                fl["out"].copy_(fl["master"])
            torch._foreach_copy_(fl["params"], fl["views"])
        return loss

    def state_dict(self):
        for st in self.state.values():
            st["step"].fill_(float(self._t))
        return super().state_dict()

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """Copy saved state INTO the flat buffers, keeping it fp32, then rewrite the
        bf16 weights from the restored master."""
        saved_groups, state = state_dict["param_groups"], state_dict["state"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(f"checkpoint has {len(saved_groups)} param groups, "
                             f"this optimizer {len(self.param_groups)}")
        t = None
        for sg, grp in zip(saved_groups, self.param_groups):
            if len(sg["params"]) != len(grp["params"]):
                raise ValueError("param group sizes differ from the checkpoint's")
            for k, v in sg.items():
                if k != "params":
                    grp[k] = v
            for pid, p in zip(sg["params"], grp["params"]):
                st = state.get(pid)
                if not st:
                    continue
                mine = self.state[p]
                for k in ("master", "exp_avg", "exp_avg_sq"):
                    if k in st:
                        mine[k].copy_(st[k].to(mine[k].device))
                if "master" not in st:                   # torch AdamW state: master = weights
                    mine["master"].copy_(p.float())
                t = int(float(st.get("step", 0)))
        if t is not None:
            self._t = t
        for fl in self._flat:
            fl["out"].copy_(fl["master"])
            torch._foreach_copy_(fl["params"], fl["views"])
