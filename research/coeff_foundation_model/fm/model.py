"""FM architecture (FULL_ARCHITECTURE.md), modular. One event = one token set;
full attention (cross-plane). Components map 1:1 to the design:

  embed(values,occ) -> + ResponseFiLM(band,plane,wire) -> + learned(band,plane)
  -> axial RoPE(physical_time, wire) inside attention -> K ViT blocks
  -> asymmetric decoder -> two heads (occupancy logit + value) per slot.

Objective: masked cross-plane coefficient AE (mask tokens, reconstruct CLEAN
coeffs of masked tokens). L2 value head default; --nll = Gaussian NLL (A/B).
Attention = plain SDPA over the event's tokens (batch=1); flash-varlen packing
is a later optimization. Framework (data/scaling) is deliberately thin — to be
replaced by pimm-data.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---- axial RoPE on (physical_time, wire) -----------------------------------
def rope_angles(pos, dim, lam_min=2.0, lam_max=10000.0):
    """Axial RoPE angles with geometric wavelengths in [lam_min, lam_max]
    (n = dim//4 frequencies). Set lam_min ~ 2x the finest position spacing and
    lam_max ~ the full coordinate span, PER AXIS. The old base=10000 spread
    wavelengths over [6.3, 47000] on both axes -> wasted ~55% of the WIRE dims
    (span 1904, spacing 16 -> useful band [32,1904]) and ~28% of the TIME dims
    (span 4336)."""
    n = dim // 4
    k = torch.arange(n, device=pos.device).float() / max(n - 1, 1)
    inv = (2 * math.pi) / (lam_min * (lam_max / lam_min) ** k)
    return pos[:, None].float() * inv[None, :]                 # (T, half/2)


def apply_rope(x, ang_t, ang_w):
    """x: (T, H, hd). First half of hd rotated by time, second half by wire."""
    h2 = x.shape[-1] // 2

    def rot(v, ang):
        c = torch.cos(ang)[:, None, :].repeat_interleave(2, -1)
        s = torch.sin(ang)[:, None, :].repeat_interleave(2, -1)
        v2 = torch.stack([-v[..., 1::2], v[..., 0::2]], -1).reshape_as(v)
        return v * c + v2 * s
    xt = rot(x[..., :h2], ang_t)
    xw = rot(x[..., h2:], ang_w) if ang_w is not None else x[..., h2:]
    return torch.cat([xt, xw], -1)


# ---- response conditioning -------------------------------------------------
class ResponseFiLM(nn.Module):
    """gamma,beta from [band_emb, plane_emb, MLP(wire features)]. Conditions the
    token on the response chain (scale, plane-type bipolar/unipolar, wire geom)."""

    def __init__(self, d, n_band, n_plane, n_wirefeat, de=32, use=("band", "plane", "wire")):
        super().__init__()
        self.use = set(use)
        self.band = nn.Embedding(n_band, de)
        self.plane = nn.Embedding(n_plane, de)
        self.wire = nn.Sequential(nn.Linear(n_wirefeat, de), nn.GELU())
        cin = de * len(self.use)
        self.mlp = nn.Sequential(nn.Linear(cin, 2 * d))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)  # start = identity

    def forward(self, band, plane, wirefeat):
        parts = []
        if "band" in self.use:  parts.append(self.band(band))
        if "plane" in self.use: parts.append(self.plane(plane))
        if "wire" in self.use:  parts.append(self.wire(wirefeat))
        g, b = self.mlp(torch.cat(parts, -1)).chunk(2, -1)
        return 1 + g, b


# ---- transformer block (pre-LN, SDPA full attention with RoPE) -------------
class Block(nn.Module):
    def __init__(self, d, heads, ffn_mult=4, adaln=False, attn_scale=None):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.attn_scale = attn_scale          # muP: 1/head_dim (else None => SDPA default 1/sqrt(hd))
        self.adaln = adaln
        self.n1 = nn.LayerNorm(d, elementwise_affine=not adaln)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=not adaln)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))
        if adaln:                              # AdaLN-Zero: per-token scale/shift/gate, zero-init = identity
            self.ada = nn.Linear(d, 6 * d)
            nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, x, ang_t, ang_w, c=None):
        T, d = x.shape
        if self.adaln:
            sa, ba, ga, sm, bm, gm = self.ada(c).chunk(6, -1)    # each (T, d)
            h = self.n1(x) * (1 + sa) + ba
        else:
            h = self.n1(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        q = apply_rope(q.view(T, self.h, self.hd), ang_t, ang_w)
        k = apply_rope(k.view(T, self.h, self.hd), ang_t, ang_w)
        v = v.view(T, self.h, self.hd)
        o = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                           v.transpose(0, 1)[None], scale=self.attn_scale)[0].transpose(0, 1)
        ao = self.proj(o.reshape(T, d))
        x = x + (ga * ao if self.adaln else ao)
        if self.adaln:
            return x + gm * self.mlp(self.n2(x) * (1 + sm) + bm)
        return x + self.mlp(self.n2(x))


# ---- CrossMAE decoder block (cheaper decoder) ------------------------------
class CrossBlock(nn.Module):
    """CrossMAE decoder: masked-token queries cross-attend the VISIBLE encoded
    tokens (no mask-mask self-attention) -> O(N_mask * N_vis), far cheaper than the
    full-self-attention decoder over all N. Axial RoPE on BOTH query (masked) and
    key (visible) positions, so it keeps the relative-position addressing.
    NOTE: queries see the FULL visible set (not a latent summary) — unlike the
    Perceiver bottleneck that capped masked prediction at ~33%."""

    def __init__(self, d, heads, ffn_mult=4, attn_scale=None):
        super().__init__()
        self.h, self.hd = heads, d // heads
        self.attn_scale = attn_scale          # muP: 1/head_dim (else None => SDPA default)
        self.nq = nn.LayerNorm(d); self.nk = nn.LayerNorm(d)
        self.q = nn.Linear(d, d); self.kv = nn.Linear(d, 2 * d); self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, q, kv, qa_t, qa_w, ka_t, ka_w):
        Tq, Tk = q.shape[0], kv.shape[0]
        qh = apply_rope(self.q(self.nq(q)).view(Tq, self.h, self.hd), qa_t, qa_w)
        k, v = self.kv(self.nk(kv)).chunk(2, -1)
        kh = apply_rope(k.view(Tk, self.h, self.hd), ka_t, ka_w)
        vh = v.view(Tk, self.h, self.hd)
        o = F.scaled_dot_product_attention(qh.transpose(0, 1)[None], kh.transpose(0, 1)[None],
                                           vh.transpose(0, 1)[None], scale=self.attn_scale)[0].transpose(0, 1)
        q = q + self.proj(o.reshape(Tq, self.h * self.hd))
        return q + self.mlp(self.n2(q))


# ---- the model -------------------------------------------------------------
class FMModel(nn.Module):
    def __init__(self, n_slot, n_band, n_plane, n_wirefeat=1, d=128, blocks=4,
                 dec_blocks=2, heads=4, film=("band", "plane", "wire"), nll=False, ffn_mult=4,
                 lam_t=(8.0, 4336.0), lam_w=(32.0, 2048.0), cond="film", dec_mode="self",
                 mup=False, d_base=128, wire_rope=True):
        super().__init__()
        self.d, self.n_slot, self.nll, self.cond, self.heads = d, n_slot, nll, cond, heads
        self.lam_t, self.lam_w = lam_t, lam_w     # per-axis RoPE wavelength band (time / wire)
        # wire_rope=False: RoPE on TIME ONLY. wire_pos is a PER-PLANE projection axis (non-metric
        # across planes), so relative wire rotation injects nonsense phase onto cross-plane
        # attention pairs (the triangulation pairs). Wire identity still enters via FiLM. (audit bug 2)
        self.wire_rope = wire_rope
        self.dec_mode = dec_mode                  # "self" = full-attn decoder over all N; "cross" = CrossMAE (cheaper)
        # --- muP (Yang & Hu, Tensor Programs V, arXiv:2203.03466) ---
        # m = d/d_base is the width multiplier. Under muP the optimal Adam LR is
        # width-invariant: hidden init var and Adam LR are divided by m; readout
        # forward is divided by m; input/embedding untouched.
        #
        # ATTENTION SCALE: muP replaces 1/sqrt(d_head) with 1/d_head ("8/d_head" in
        # the mup pkg, backward-compatible at d_head=64) ONLY when d_head GROWS with
        # width. HERE d_head is FIXED (=64; heads=d/64), so q.k over a fixed d_head is
        # already O(1) under the standard 1/sqrt(d_head) — that scaling is width-correct
        # and we keep it (forcing 1/d_head would just shrink logits ~8x, verified by the
        # coord check). So the attention-scale change is a NO-OP for this fixed-head_dim
        # model; we leave attn_scale=None (SDPA default 1/sqrt(head_dim)) in both paths.
        self.mup, self.d_base = mup, d_base
        self.m = d / d_base if mup else 1.0
        self.readout_mult = 1.0 / self.m         # output multiplier (1/m under muP, 1 else)
        attn_scale = None                        # None => SDPA default 1/sqrt(head_dim); width-correct for fixed d_head
        adaln = (cond == "adaln")
        self.embed = nn.Linear(2 * n_slot, d)
        self.film = ResponseFiLM(d, n_band, n_plane, n_wirefeat, use=film) if (film and not adaln) else None
        self.band_emb = nn.Embedding(n_band, d)              # learned identity (non-metric)
        self.plane_emb = nn.Embedding(n_plane, d)
        self.cond_wire = nn.Sequential(nn.Linear(n_wirefeat, d), nn.SiLU()) if adaln else None
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.enc = nn.ModuleList(Block(d, heads, ffn_mult, adaln, attn_scale) for _ in range(blocks))
        if dec_mode == "cross":
            self.dec = nn.ModuleList(CrossBlock(d, heads, ffn_mult, attn_scale) for _ in range(dec_blocks))
        else:
            self.dec = nn.ModuleList(Block(d, heads, ffn_mult, adaln, attn_scale) for _ in range(dec_blocks))
        self.dec_norm = nn.LayerNorm(d)
        self.occ_head = nn.Linear(d, n_slot)
        self.val_head = nn.Linear(d, n_slot * (2 if nll else 1))  # mu (+ log-var if nll)
        if mup:
            self._mup_init()

    # ---- muP machinery (Tensor Programs V, arXiv:2203.03466; mup pkg; EleutherAI guide) ----
    # Category map for THIS model. INPUT = fixed-fan_in maps into width-d space
    # (LR const, init unchanged). HIDDEN = fan_in,fan_out both O(d) (init var /m, LR /m).
    # OUTPUT = fixed-fan_out readouts (init unchanged, forward mult 1/m, LR const).
    # LayerNorm gains/biases and all biases: no width treatment (LR const).
    def _mup_categories(self):
        """-> dict name->('input'|'hidden'|'output') for WEIGHT params (2-D).
        Biases / LayerNorm / 1-D params are always 'input' (LR const, no scaling)."""
        hidden, output = set(), set()
        for i, blk in enumerate(self.enc):
            for nm in ("qkv", "proj", "mlp.0", "mlp.2", "ada", "q", "kv"):
                hidden.add(f"enc.{i}.{nm}.weight")
        for i, blk in enumerate(self.dec):
            for nm in ("qkv", "proj", "mlp.0", "mlp.2", "ada", "q", "kv"):
                hidden.add(f"dec.{i}.{nm}.weight")
        output.update({"occ_head.weight", "val_head.weight"})
        cats = {}
        for n, p in self.named_parameters():
            if n in hidden:   cats[n] = "hidden"
            elif n in output: cats[n] = "output"
            else:             cats[n] = "input"     # embed/emb/mask_tok/film/cond_wire/all biases/LN
        return cats

    def _mup_init(self):
        """Rescale hidden WEIGHT init so var = base_var / m (std /= sqrt(m)).
        Input/output init left at the framework default (= base param at m=1 => the
        mup=False path is reproduced exactly when d==d_base). Query/readout are NOT
        zero-init'd here: the codebase already zero-inits FiLM/AdaLN, and a plain
        (non-zero) readout is a valid muP choice as long as the 1/m multiplier is applied."""
        import math as _m
        cats = self._mup_categories()
        with torch.no_grad():
            for n, p in self.named_parameters():
                if cats.get(n) == "hidden" and p.dim() == 2:
                    p.mul_(1.0 / _m.sqrt(self.m))       # var -> var / m

    def param_groups(self, base_lr, weight_decay=None):
        """AdamW param groups with muP per-category LR multipliers (call from train.py).
        HIDDEN: lr = base_lr / m. INPUT & OUTPUT (+ biases, LayerNorm): lr = base_lr.
        With mup=False, m==1 so every group gets base_lr (identical to a flat AdamW).

        weight_decay: if given, DECOUPLE decay from the muP LR scaling. AdamW applies
        decoupled decay as lr*wd*p, so a hidden LR of base_lr/m would otherwise shrink
        effective decay by 1/m across width (a width-scaling confound -- up to ~6x weaker
        at d768). We set the hidden group's wd = weight_decay*m so effective decay
        (lr*wd = base_lr*weight_decay) is width-invariant. weight_decay=None preserves the
        LEGACY path (per-group wd unset -> optimizer's constructor default applies, LR-coupled);
        used by callers that pass weight_decay to the AdamW constructor instead."""
        cats = self._mup_categories()
        # NO-DECAY group (nanoGPT/timm standard): decay only 2-D weight matrices; exclude
        # all 1-D params (biases, LayerNorm) and tokens from weight decay. Only splits when
        # weight_decay is passed; the legacy (weight_decay=None) path stays 2 groups.
        buckets = {"hidden": [], "decay": [], "nodecay": []}
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if cats.get(n) == "hidden":
                buckets["hidden"].append(p)                                     # muP-scaled 2-D weights
            elif p.dim() >= 2:
                buckets["decay"].append(p)                                      # other 2-D weights (heads, embed, film)
            else:
                buckets["nodecay"].append(p)                                    # biases, LayerNorm, 1-D tokens
        groups = []
        if buckets["nodecay"]:
            g = {"params": buckets["nodecay"], "lr": base_lr}
            if weight_decay is not None:
                g["weight_decay"] = 0.0                                         # standard: no decay on 1-D params
            groups.append(g)
        if buckets["decay"]:
            g = {"params": buckets["decay"], "lr": base_lr}
            if weight_decay is not None:
                g["weight_decay"] = weight_decay
            groups.append(g)
        if buckets["hidden"]:
            g = {"params": buckets["hidden"], "lr": base_lr / self.m}           # hidden /= m
            if weight_decay is not None:
                g["weight_decay"] = weight_decay * self.m                       # decouple: lr*wd = base_lr*weight_decay
            groups.append(g)
        return groups

    def _cond(self, B):                                          # AdaLN conditioning vector (per token)
        return self.band_emb(B["band_id"]) + self.plane_emb(B["plane_id"]) + self.cond_wire(B["wirefeat"])

    def encode(self, B):
        """Per-token ENCODER representation with ALL tokens visible (no masking).
        This is the frozen feature the deconvolution probe reads."""
        band, plane = B["band_id"], B["plane_id"]
        at = rope_angles(B["t_phys"], self.d // self.heads, *self.lam_t)
        aw = rope_angles(B["wire_pos"], self.d // self.heads, *self.lam_w) if self.wire_rope else None
        x = self.embed(torch.cat([B["inp"], B["occ"]], -1))
        c = self._cond(B) if self.cond == "adaln" else None
        if c is None:
            if self.film is not None:
                g, b = self.film(band, plane, B["wirefeat"]); x = g * x + b
            x = x + self.band_emb(band) + self.plane_emb(plane)
        for blk in self.enc:
            x = blk(x, at, aw, c)
        return x                                                  # (N, d)

    def encode_layers(self, B, layers):
        """Per-token features after each requested encoder block (1-based). {k: (N,d)}.
        For probing WHICH layer carries the best representation (MAE: often mid-encoder)."""
        band, plane = B["band_id"], B["plane_id"]
        at = rope_angles(B["t_phys"], self.d // self.heads, *self.lam_t)
        aw = rope_angles(B["wire_pos"], self.d // self.heads, *self.lam_w) if self.wire_rope else None
        x = self.embed(torch.cat([B["inp"], B["occ"]], -1))
        c = self._cond(B) if self.cond == "adaln" else None
        if c is None:
            if self.film is not None:
                g, b = self.film(band, plane, B["wirefeat"]); x = g * x + b
            x = x + self.band_emb(band) + self.plane_emb(plane)
        out = {}
        for i, blk in enumerate(self.enc, 1):
            x = blk(x, at, aw, c)
            if i in layers:
                out[i] = x
        return out

    def forward_feat(self, B, tok_mask, return_ctx=False):
        # --- conditioning is known for ALL tokens (band/plane/wire/pos are inputs,
        #     not predicted), so it can be applied to visible AND masked positions ---
        N = B["inp"].shape[0]
        band, plane = B["band_id"], B["plane_id"]
        at = rope_angles(B["t_phys"], self.d // self.heads, *self.lam_t)
        aw = rope_angles(B["wire_pos"], self.d // self.heads, *self.lam_w) if self.wire_rope else None
        adaln = (self.cond == "adaln")
        c = self._cond(B) if adaln else None                     # AdaLN per-token conditioning
        if self.film is not None:
            g, b = self.film(band, plane, B["wirefeat"])
        cond = None if adaln else (self.band_emb(band) + self.plane_emb(plane))  # additive (FiLM path)

        # --- ENCODER-DROP (true MAE): encoder runs on VISIBLE tokens only ---
        vis = ~tok_mask
        vis_idx = vis.nonzero(as_tuple=True)[0]
        xv = self.embed(torch.cat([B["inp"][vis], B["occ"][vis]], -1))
        if self.film is not None:
            xv = g[vis] * xv + b[vis]
        if cond is not None:
            xv = xv + cond[vis]
        atv, awv = at[vis], (aw[vis] if aw is not None else None)
        cv = c[vis] if adaln else None
        for blk in self.enc:
            xv = blk(xv, atv, awv, cv)                            # attention over N_vis only

        # --- CrossMAE decoder: only MASKED positions decoded, x-attending visible xv ---
        if self.dec_mode == "cross":
            mask_idx = tok_mask.nonzero(as_tuple=True)[0]
            qm = self.mask_tok.expand(mask_idx.numel(), self.d)
            if self.film is not None:
                qm = g[tok_mask] * qm + b[tok_mask]
            if cond is not None:
                qm = qm + cond[tok_mask]                          # mask queries carry pos/response
            qm = qm.to(xv.dtype)
            atm, awm = at[tok_mask], (aw[tok_mask] if aw is not None else None)
            for blk in self.dec:
                qm = blk(qm, xv, atm, awm, atv, awv)              # masked x-attend visible (full set, RoPE both sides)
            x = torch.zeros(N, self.d, dtype=xv.dtype, device=xv.device)
            x = x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm)   # visible=encoder feats, masked=decoded
            return (self.dec_norm(x), xv) if return_ctx else self.dec_norm(x)

        # --- (default) full-self-attention decoder over ALL N tokens ---
        xm = self.mask_tok.expand(N, self.d)
        if self.film is not None:
            xm = g * xm + b
        if cond is not None:
            xm = xm + cond                                        # mask tokens carry pos/response
        x = xm.to(xv.dtype).index_copy(0, vis_idx, xv)           # overwrite visible w/ encoder out
        for blk in self.dec:
            x = blk(x, at, aw, c)                                 # decoder over full N
        return (self.dec_norm(x), xv) if return_ctx else self.dec_norm(x)   # (N, d) per-cell features

    def forward(self, B, tok_mask):
        x = self.forward_feat(B, tok_mask)
        occ = self.occ_head(x) * self.readout_mult      # muP output multiplier (1/m; 1 if mup=False)
        val = self.val_head(x) * self.readout_mult
        if self.nll:
            val = val.view(-1, self.n_slot, 2)
            return occ, val[..., 0], val[..., 1]            # logit, mu, logvar
        return occ, val, None


def losses(occ, mu, logvar, B, tok_mask, vis_w=0.0, noisy=False):
    """Dual-target loss. CLEAN coeff is the target at every active slot:
      - MASKED active slots  -> INFERENCE (predict clean from cross-context only)
      - VISIBLE active slots -> DENOISING (predict clean from own noisy value+context)
    Because our input is noisy and target is clean, the visible term is NOT a trivial
    copy (as in pixel MAE) — it's the denoising objective we actually want. vis_w
    weights it (0 = masked-only, the original behavior). Occupancy BCE stays masked-only
    (visible occupancy is observed, so supervising it would just leak)."""
    mvalid = tok_mask[:, None] & B["valid"]
    bce = F.binary_cross_entropy_with_logits(occ[mvalid], B["occ"][mvalid]) if mvalid.any() else occ.sum() * 0

    cell_masked = tok_mask[B["cell"]]                       # per-active-row: is its cell masked?

    def _value(rowsel):
        if not rowsel.any():
            return mu.sum() * 0
        pred = mu[B["cell"], B["slot"]][rowsel]
        # noisy=True -> self-supervised: predict the NOISY input coeff (no clean truth needed)
        tgt = (B["inp"][B["cell"], B["slot"]] if noisy else B["target"])[rowsel]
        if logvar is not None:
            lv = logvar[B["cell"], B["slot"]][rowsel].clamp(-8, 8)
            return 0.5 * (((pred - tgt) ** 2) * torch.exp(-lv) + lv).mean()   # Gaussian NLL
        return F.mse_loss(pred, tgt)

    val = _value(cell_masked)                               # masked: inference
    if vis_w > 0:
        val = val + vis_w * _value(~cell_masked)            # visible: denoising
    return bce, val


def losses_fused(occ, mu, logvar, B, tok_mask, vis_w=0.0, noisy=False):
    """Same loss as losses() but computed DENSELY over the (n_cells, n_slot) grid with masks
    instead of advanced-indexing gathers (mu[cell,slot], occ[mvalid]). Removes the scatter
    (indexing_backward) and collapses to elementwise ops + masked reductions that torch.compile
    fuses into ~1 kernel. Needs the dense B["tgt"] (threaded through data._to_fm)."""
    valid, occ_t = B["valid"], B["occ"]                     # dense (n_cells, n_slot)
    tgt = B["inp"] if noisy else B["tgt"]                   # dense target
    mrow = tok_mask[:, None]                                # (n_cells, 1) masked cells
    m_occ = mrow & valid                                    # masked & valid slots (occupancy BCE support)
    bce_e = F.binary_cross_entropy_with_logits(occ, occ_t, reduction="none")
    bce = (bce_e * m_occ).sum() / m_occ.sum().clamp(min=1)
    if logvar is not None:
        v_e = 0.5 * (((mu - tgt) ** 2) * torch.exp(-logvar.clamp(-8, 8)) + logvar.clamp(-8, 8))
    else:
        v_e = (mu - tgt) ** 2
    act = occ_t.bool() & valid & mrow                       # masked & ACTIVE & valid slots
    val = (v_e * act).sum() / act.sum().clamp(min=1)
    if vis_w > 0:
        av = occ_t.bool() & valid & ~mrow
        val = val + vis_w * (v_e * av).sum() / av.sum().clamp(min=1)
    return bce, val
