"""The coefficient foundation model.

A masked autoencoder over patch tokens of sparse wavelet coefficients: the
encoder runs on the visible tokens only, in grouped attention along four token
orders (:mod:`helix.model.blocks`); a CrossMAE decoder's masked queries
cross-attend the encoded visible tokens in drift-time groups; a categorical head
predicts each slot's occupancy and its value's bin. ``forward`` implements the
pimm Trainer contract (``model(batch) -> dict`` with a ``loss`` key).
"""
import torch
import torch.nn as nn
from helix.model.blocks import (Block, CrossBlock, ResponseFiLM, _blocks, _ckpt, _cu_chunks,
                                _cu_even, _inverse, _permute, rope_angles, rope_tables)
from helix.model.loss import cat_head_sparse


# Training-policy options: how to mask, how a step executes. They carry no
# parameters, so they are NOT arguments of __init__: the constructor signature is
# the architecture, which keeps a checkpoint's architecture metadata and its
# parameter tree in exact correspondence. build_fm() sets them as attributes.
#
# plane_frac mixes the two masking modes PER STEP: whole-plane masking with that
# probability, mask_mode otherwise. It is NOT the same as mask_mode="plane",
# which would mask planes every step.
#
# act_ckpt recomputes each block's activations in the backward pass instead of
# storing them: ~1.3x the step for a fraction of the activation memory, which is
# what lets d > 1024 fit a 40 GB card.
#
# compile_blocks runs the block functions through torch.compile (dynamic
# shapes): ~1.2x steady-state for ~5 minutes' compilation per job, so the
# training config turns it on and everything else (tests, probes, smoke runs)
# leaves it off.
#
# fused_qk runs QK-norm + RoPE as one Triton kernel each way
# (helix.model.kernels); the same arithmetic to ~1e-5, so it is an
# implementation choice, not an architecture one. Needs qk_norm and head dim 64.
_TRAIN_OPTS = dict(mask_mode="random", mask_ratio=0.75, n_planes=1,
                   plane_frac=0.0, compile_blocks=False, act_ckpt=False,
                   fused_qk=False)


# ---- MAE token masking ----------------------------------------------------------------------
# Model-owned: the mask is part of the objective, and the one-argument ``model(batch)`` contract
# leaves nowhere to pass one in.

#: Views (U, V, Y) per TPC volume. `plane_id` is the global plane gid, so
#: ``gid // VIEWS_PER_VOLUME`` is the volume and ``gid % VIEWS_PER_VOLUME`` the
#: view — the same decomposition tokenize and probe/designs already assume.
VIEWS_PER_VOLUME = 3


def make_mask(B, mode, ratio, n_planes, gen=None):
    """mode: random | plane.

    * ``random`` — ``ratio`` of tokens, drawn independently.
    * ``plane``  — ``n_planes`` whole planes PER VOLUME; every volume punctured
      (``ratio`` is unused).
    """
    n = B["n_cells"]; dev = B["plane_id"].device
    rnd = (lambda *s: torch.rand(*s, generator=gen, device=dev)) if (gen is not None and gen.device.type == dev.type) \
        else (lambda *s: torch.rand(*s, device=dev))
    if mode == "random":
        return rnd(n) < ratio
    if mode != "plane":
        raise ValueError(f"mask mode must be 'random' or 'plane', got {mode!r}")
    # Cross-plane: hide whole planes. Puncturing EVERY volume, measured on a real
    # R1 event (gids 0..5 = 2 volumes x 3 views):
    #
    #   n=1   32.9% of cells   every volume down to 2/3 views
    #   n=2   66.1% of cells   every volume down to 1/3 views
    #
    # Difficulty is set by the views left in a punctured volume. Scored on a
    # cooldown checkpoint, 388-event probe split:
    #
    #   random, 0.75   masked   var_expl 0.703
    #   plane,  n=1    0.333    var_expl 0.506
    #   plane,  n=2    0.665    var_expl 0.015   <- one view left: collapse
    #
    # Hiding a third of the cells as whole planes is far harder than hiding 75%
    # at random: cross-plane is the hard axis, not the masked fraction.
    #
    # randperm takes the generator, so a fixed `gen` reproduces the mask (the
    # evaluator relies on that).
    gid = B["plane_id"]
    gids = torch.unique(gid)
    shuffle = lambda g: g[torch.randperm(len(g), generator=gen, device=dev)]
    # volume = gid // 3, view = gid % 3: the convention across the codebase
    # (`toff[plane_gid % 3]` in tokenize's tick transform, `vol, pl = plane // 3,
    # plane % 3` in probe/designs).
    vol = torch.div(gids, VIEWS_PER_VOLUME, rounding_mode="floor")
    pick = torch.cat([shuffle(gids[vol == v])[:n_planes]
                      for v in torch.unique(vol)])
    return torch.isin(gid, pick)


class FMModel(nn.Module):
    """``rope_split=True`` rotates by time AND wire on the plane-order layers but
    by TIME ONLY on the drift-order ones (whose groups mix planes, and wire_pos
    is a per-plane axis); False rotates by both everywhere. ``wire_rope=False``
    rotates by time only, everywhere. ``gp`` / ``gd`` are the group sizes of the
    plane-order and the drift-order layers (and the decoder)."""

    def __init__(self, n_slot, n_band, n_plane, n_bins, d=128, blocks=4, dec_blocks=2,
                 heads=4, ffn_mult=4, lam_t=(8.0, 4336.0), lam_w=(32.0, 2048.0),
                 mup=False, d_base=128, wire_rope=True, qk_norm=False,
                 rope_split=False, gp=1024, gd=2048):
        super().__init__()
        if int(n_bins) <= 0:
            raise ValueError(f"n_bins={n_bins}: the value head is categorical and needs >= 1 bin")
        self.d, self.n_slot, self.heads, self.n_bins = d, n_slot, heads, int(n_bins)
        self.lam_t, self.lam_w = lam_t, lam_w     # per-axis RoPE wavelength band (time / wire)
        # wire_rope=False: RoPE on TIME ONLY. wire_pos is a PER-PLANE projection axis (non-metric
        # across planes), so relative wire rotation injects nonsense phase onto cross-plane
        # attention pairs (the triangulation pairs). Wire identity still enters via FiLM.
        self.wire_rope = wire_rope
        self.rope_split, self.gp, self.gd = rope_split, gp, gd
        # PERSISTENT. These are training-set statistics that inference is
        # WRONG without — the BatchNorm running_mean/running_var case, not
        # the causal-mask case. `persistent=False` is for tensors __init__
        # can regenerate from its own arguments (rotary inv_freq, attention
        # masks); bin edges cannot be regenerated from anything, and a
        # checkpoint must be usable away from its experiment directory.
        #
        # Allocated here so the shape exists before any load; set_bins fills
        # it, and a state_dict load overwrites it.
        # NaN, not zeros: "allocated but never set" must stay distinguishable
        # from real edges. All-zero edges are not valid (they must increase),
        # but zeros would still silently bucketise everything into one bin,
        # trading a loud failure for a wrong number. NaN cannot be mistaken
        # for data and `forward` checks it.
        self.register_buffer("bin_edges",
                             torch.full((n_band, n_bins + 1), float("nan")))
        # PERSISTENT, like the edges. `bin_cent_asinh` is the var_expl
        # estimator (token space); `bin_cent_ratio` is E[coeff/sigma | bin],
        # the charge read-back. `set_bins` DERIVES either one that is not
        # supplied, so neither is ever NaN.
        for _nm in ("bin_cent_asinh", "bin_cent_ratio"):
            self.register_buffer(_nm, torch.full((n_band, n_bins), float("nan")))
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
        # coord check). So attention keeps SDPA's default 1/sqrt(head_dim).
        self.mup, self.d_base = mup, d_base
        self.m = d / d_base if mup else 1.0
        self.readout_mult = 1.0 / self.m         # output multiplier (1 if mup=False)
        self.embed = nn.Linear(2 * n_slot, d)
        self.film = ResponseFiLM(d, n_band, n_plane)
        self.band_emb = nn.Embedding(n_band, d)              # learned identity (non-metric)
        self.plane_emb = nn.Embedding(n_plane, d)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.enc = nn.ModuleList(Block(d, heads, ffn_mult, qk_norm) for _ in range(blocks))
        self.dec = nn.ModuleList(CrossBlock(d, heads, ffn_mult, qk_norm) for _ in range(dec_blocks))
        self.dec_norm = nn.LayerNorm(d)
        self.occ_head = nn.Linear(d, n_slot)
        self.val_head = nn.Linear(d, n_slot * n_bins)        # K logits per slot
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
        hidden = set()
        for stack, blocks in (("enc", self.enc), ("dec", self.dec)):
            for i in range(len(blocks)):
                for nm in ("qkv", "proj", "mlp.0", "mlp.2", "q", "kv"):
                    hidden.add(f"{stack}.{i}.{nm}.weight")
        output = {"occ_head.weight", "val_head.weight"}
        cats = {}
        for n, p in self.named_parameters():
            if n in hidden:       cats[n] = "hidden"
            elif n in output:     cats[n] = "output"
            else:                 cats[n] = "input"     # embed/emb/mask_tok/film/all biases/LN
        return cats

    def _mup_init(self):
        """Rescale hidden WEIGHT init so var = base_var / m (std /= sqrt(m)).
        Input/output init left at the framework default (= base param at m=1 => the
        mup=False path is reproduced exactly when d==d_base). Query/readout are NOT
        zero-init'd here: FiLM is already zero-init'd, and a plain (non-zero)
        readout is a valid muP choice as long as the 1/m multiplier is applied."""
        import math as _m
        cats = self._mup_categories()
        with torch.no_grad():
            for n, p in self.named_parameters():
                if cats.get(n) == "hidden" and p.dim() == 2:
                    p.mul_(1.0 / _m.sqrt(self.m))       # var -> var / m

    def param_groups(self, base_lr, weight_decay=None):
        """AdamW param groups with muP per-category LR multipliers.
        HIDDEN: lr = base_lr / m. INPUT & OUTPUT (+ biases, LayerNorm): lr = base_lr.
        With mup=False, m==1 so every group gets base_lr (identical to a flat AdamW).

        weight_decay: if given, DECOUPLE decay from the muP LR scaling. AdamW applies
        decoupled decay as lr*wd*p, so a hidden LR of base_lr/m would otherwise shrink
        effective decay by 1/m across width (a width-scaling confound -- up to ~6x weaker
        at d768). We set the hidden group's wd = weight_decay*m so effective decay
        (lr*wd = base_lr*weight_decay) is width-invariant. weight_decay=None leaves
        per-group wd unset (the optimizer's constructor value applies, LR-coupled)."""
        cats = self._mup_categories()
        # NO-DECAY group (nanoGPT/timm standard): decay only 2-D weight matrices; exclude
        # all 1-D params (biases, LayerNorm) and tokens from weight decay. The split
        # takes effect only when weight_decay is passed.
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

    # ---- encoder / decoder -------------------------------------------------

    def _fused(self):
        """``fused_qk`` (a train option) after checking it can apply: the kernel
        is QK-norm + RoPE at head dim 64."""
        if not self.fused_qk:
            return False
        hd = self.d // self.heads
        if not (self.enc[0].qn is not None and hd == 64):
            raise ValueError(f"fused_qk needs qk_norm=True and head dim 64 "
                             f"(got qk_norm={self.enc[0].qn is not None}, hd={hd})")
        return True

    def _layouts(self, plane, t, wire):
        """The encoder's distinct (order, group size, wire RoPE on) layouts;
        block i uses ``layouts[i % 4]``."""
        bp = plane.double()
        o_pt = torch.argsort(bp * 1e9 + t.double())
        o_pw = torch.argsort(bp * 1e13 + wire.double() * 1e6 + t.double())
        o_t = torch.argsort(t.double()); o_ts = torch.roll(o_t, self.gd // 2)
        dw = not self.rope_split                                   # drift-layer wire RoPE: off if split
        cell = [(o_pt, self.gp, True), (o_t, self.gd, dw), (o_pw, self.gp, True), (o_ts, self.gd, dw)]
        return cell[:len(self.enc)]

    def _emb(self, B, idx=None):
        """Embedded tokens (rows ``idx`` of B, None = all), FiLM-conditioned on
        (band, plane, wire) plus the additive band and plane identities."""
        sel = slice(None) if idx is None else idx
        band, plane = B["band_id"][sel], B["plane_id"][sel]
        x = self.embed(torch.cat([B["inp"][sel], B["occ"][sel]], -1))
        g, b = self.film(band, plane, B["wirefeat"][sel])
        return g * x + b + self.band_emb(band) + self.plane_emb(plane)

    def _angles(self, B):
        """RoPE angles (time, wire) per token; wire is None when ``wire_rope`` is off."""
        hd = self.d // self.heads
        return (rope_angles(B["t_phys"], hd, *self.lam_t),
                rope_angles(B["wire_pos"], hd, *self.lam_w) if self.wire_rope else None)

    def _encode(self, B, idx, at, aw, layers=()):
        """The encoder over rows ``idx`` of B (None = all) -> (x, {layer: x}),
        both in natural order. ``at``/``aw`` are already those rows'. The residual
        stream is fp32 (the weights' dtype), whatever autocast made of the embedding."""
        sel = slice(None) if idx is None else idx
        x = self._emb(B, idx).to(self.embed.weight.dtype)
        T, dev = x.shape[0], x.device
        lay = []
        for o, g, uw in self._layouts(B["plane_id"][sel], B["t_phys"][sel], B["wire_pos"][sel]):
            cos, sin = rope_tables(at[o], aw[o] if (uw and aw is not None) else None, x.dtype)
            lay.append((o, _inverse(o), _cu_chunks(T, g, dev), min(g, T), cos, sin))
        if not lay:
            return x, {}
        # layout j's rows, gathered from layout j-1's stream
        step = [lay[j - 1][1][lay[j][0]] for j in range(len(lay))]
        self_blk, fused = _ckpt(_blocks(self)[0], self), self._fused()
        xp, out = _permute(x, lay[0][0]), {}
        for i, blk in enumerate(self.enc):
            j = i % len(lay)
            if i:
                xp = _permute(xp, step[j])
            _, inv, cu, mx, cos, sin = lay[j]
            xp = self_blk(blk, xp, cos, sin, cu, mx, fused)
            if i + 1 in layers:
                out[i + 1] = _permute(xp, inv)
        return _permute(xp, inv), out

    def _decode(self, qm, xv, qpos, kpos):
        """The grouped-cross decoder with even time groups. ``qpos`` / ``kpos``
        are (t_phys, ang_t, ang_w) of the queries / keys.

        ``rope_split`` does not reach the decoder: it keeps the wire rotation,
        since it needs the wire address to know which wire it reconstructs
        (dropping it froze reconstruction, var_expl ~2%)."""
        (qt, qat, qaw), (kt, kat, kaw) = qpos, kpos
        Tq, Tk, dev = qm.shape[0], xv.shape[0], xv.device
        oq = torch.argsort(qt.double()); okv = torch.argsort(kt.double())
        nb = max(1, min((max(Tq, Tk) + self.gd - 1) // self.gd, Tq, Tk))
        cuq, cuk = _cu_even(Tq, nb, dev), _cu_even(Tk, nb, dev)
        mq, mk = (Tq + nb - 1) // nb, (Tk + nb - 1) // nb
        qcos, qsin = rope_tables(qat[oq], None if qaw is None else qaw[oq], qm.dtype)
        kcos, ksin = rope_tables(kat[okv], None if kaw is None else kaw[okv], xv.dtype)
        cross_blk, fused = _ckpt(_blocks(self)[1], self), self._fused()
        qp, kvp = _permute(qm, oq), _permute(xv, okv)
        for blk in self.dec:
            qp = cross_blk(blk, qp, kvp, qcos, qsin, kcos, ksin, cuq, cuk, mq, mk, fused)
        return _permute(qp, _inverse(oq))

    def encode(self, B):
        """Per-token encoder representation with ALL tokens visible (no masking).
        This is the frozen feature the deconvolution probe reads."""
        at, aw = self._angles(B)
        return self._encode(B, None, at, aw)[0]

    def encode_layers(self, B, layers):
        """Per-token features after each requested encoder block (1-based). {k: (N,d)}.
        For probing WHICH layer carries the best representation (MAE: often mid-encoder)."""
        at, aw = self._angles(B)
        return self._encode(B, None, at, aw, layers)[1]

    def forward_feat(self, B, tok_mask, masked_only=False):
        """(N, d) decoded features: visible rows are the encoder's output, masked
        rows the decoder's. ``masked_only`` -> (decoded rows, their indices), for
        an objective that reads nothing else."""
        at, aw = self._angles(B)
        # ONE host sync (the visible count) instead of one per boolean-mask index
        # below: a stable argsort of the mask lists visible rows then masked rows,
        # each ascending -- the same rows, in the same order, as nonzero().
        n_vis = int((~tok_mask).sum())
        order = torch.argsort(tok_mask.to(torch.uint8), stable=True)
        vis_idx, mask_idx = order[:n_vis], order[n_vis:]
        rows = lambda a, i: None if a is None else a[i]
        atv, awv = at[vis_idx], rows(aw, vis_idx)
        xv, _ = self._encode(B, vis_idx, atv, awv)

        # the mask queries carry the (known) position and response conditioning
        band, plane = B["band_id"][mask_idx], B["plane_id"][mask_idx]
        g, b = self.film(band, plane, B["wirefeat"][mask_idx])
        qm = g * self.mask_tok.expand(mask_idx.numel(), self.d) + b
        qm = (qm + self.band_emb(band) + self.plane_emb(plane)).to(xv.dtype)
        qm = self._decode(qm, xv, (B["t_phys"][mask_idx], at[mask_idx], rows(aw, mask_idx)),
                          (B["t_phys"][vis_idx], atv, awv))
        if masked_only:
            return self.dec_norm(qm), mask_idx
        x = torch.zeros(B["inp"].shape[0], self.d, dtype=xv.dtype, device=xv.device)
        return self.dec_norm(x.index_copy(0, vis_idx, xv).index_copy(0, mask_idx, qm))

    def raw_heads(self, B, tok_mask):
        """Head outputs over the whole grid: (occupancy logits (n_cells, n_slot),
        value logits (n_cells, n_slot, n_bins)). (Not ``heads``: ``self.heads``
        is the attention-head count.)"""
        x = self.forward_feat(B, tok_mask)
        occ = self.occ_head(x) * self.readout_mult      # muP output multiplier (1/m; 1 if mup=False)
        val = self.val_head(x) * self.readout_mult
        return occ, val.view(-1, self.n_slot, self.n_bins)

    # ---- objective -----------------------------------------------------------

    def set_bins(self, edges, *, cent_asinh=None, cent_ratio=None):
        """Register the categorical head's bin edges and centroid tables.

        Required before ``forward``: the objective needs the edges. They are
        training-set statistics the model cannot invent, so they are PERSISTENT
        buffers and travel inside the state_dict.

        Centroids are DERIVED from the edges when not supplied, so no centroid
        buffer is ever NaN and no consumer has to ask whether they are present.
        A measured table (from ``helix.data.bins``) is strictly better, but its
        absence cannot disable anything.

        The centroid arguments are KEYWORD-ONLY, so a table passed positionally
        cannot silently bind to the wrong slot.
        """
        self._bins_checked = False
        from helix.model.tokenize import bin_centroids_asinh, bin_centroids_ratio

        edges = torch.as_tensor(edges, dtype=self.bin_edges.dtype)
        assert edges.shape[1] == self.n_bins + 1, \
            f"edges {tuple(edges.shape)} inconsistent with n_bins={self.n_bins}"
        assert edges.shape[0] == self.bin_edges.shape[0], \
            f"edges has {edges.shape[0]} bands, model has {self.bin_edges.shape[0]}"
        self.bin_edges.copy_(edges.to(self.bin_edges.device))

        e_np = edges.detach().cpu().numpy()
        for nm, given, derive in (("bin_cent_asinh", cent_asinh, bin_centroids_asinh),
                                  ("bin_cent_ratio", cent_ratio, bin_centroids_ratio)):
            v = torch.as_tensor(given if given is not None else derive(e_np),
                                dtype=self.bin_edges.dtype)
            assert v.shape == (self.bin_edges.shape[0], self.n_bins), \
                f"{nm} {tuple(v.shape)} != {(self.bin_edges.shape[0], self.n_bins)}"
            getattr(self, nm).copy_(v.to(self.bin_edges.device))
        return self

    def make_mask(self, B, ratio=None, mode=None, n_planes=None, gen=None):
        """Draw a token mask for this batch under the model's configured policy.

        ``plane_frac`` mixes the two modes PER STEP: with that probability the
        step masks whole planes (``"plane"``), otherwise it uses ``mask_mode``.
        That mix is the point — a run that only ever masks randomly never has to
        reconstruct a plane it cannot see, so nothing forces cross-plane
        triangulation, and a run that only ever masks planes never learns the
        within-plane task.

        An explicit ``mode=`` overrides the policy and skips the draw, so callers
        that want one specific mode (evaluation, tests) are unaffected.
        """
        m = self.mask_mode if mode is None else mode
        if mode is None and self.plane_frac > 0:
            # torch, not numpy's global RNG: the draw stays on the batch's device
            # and honours `gen`.
            dev = B["plane_id"].device
            r = (torch.rand((), generator=gen, device=dev)
                 if (gen is not None and gen.device.type == dev.type)
                 else torch.rand((), device=dev))
            if float(r) < self.plane_frac:
                m = "plane"
        return make_mask(B, m,
                         self.mask_ratio if ratio is None else ratio,
                         self.n_planes if n_planes is None else n_planes, gen=gen)

    def require_batch_keys(self, B):
        """Raise a named KeyError for anything the objective needs and lacks.

        A METHOD, not inline in ``forward``, because ``forward`` is not the only
        entry point: the evaluator reproduces the objective from a single
        ``raw_heads`` call and must run the same check, so a missing ``tgt`` is
        named instead of surfacing as a bare KeyError from inside the loss.
        """
        gone = [k for k in ("occ", "valid", "inp", "tgt") if k not in B]
        if gone:
            raise KeyError(
                f"batch is missing {gone} required by the categorical objective "
                f"(losses_cat); "
                f"helix.model.tokenize.to_fm() emits all of them")

    def forward(self, batch, tok_mask=None):
        """pimm Trainer contract: ``model(batch) -> dict`` carrying ``loss``.

        Draws the mask (unless one is supplied), decodes the masked rows and
        applies the categorical objective to them."""
        B = dict(batch)
        B.setdefault("n_cells", B["plane_id"].shape[0])
        self.require_batch_keys(B)
        return self._forward_one(B, self.make_mask(B) if tok_mask is None else tok_mask)

    def _forward_one(self, B, m):
        if not getattr(self, "_bins_checked", False):
            # once per model, not per step: `.all()` into `assert` is a host sync
            self._bins_checked = True
            assert torch.isfinite(self.bin_edges).all(), \
                ("forward() requires set_bins(edges) first — the edges buffer is "
                 "still unset (NaN). They are training-set statistics the model "
                 "cannot invent.")
        # Only masked rows enter the objective, so only they are decoded, and the
        # value head is evaluated only on the slots it weights.
        feat, rows = self.forward_feat(B, m, masked_only=True)
        bce, vloss = cat_head_sparse(self, feat, B, rows)
        return {"loss": bce + vloss, "bce": bce.detach(), "val": vloss.detach(),
                "masked_frac": m.float().mean().detach()}


for _k, _v in _TRAIN_OPTS.items():       # class defaults; build_fm overrides per instance
    setattr(FMModel, _k, _v)


def fm_keys():
    """Every model-config key build_fm accepts: the FMModel constructor's
    parameters and the train options."""
    import inspect
    return (set(inspect.signature(FMModel.__init__).parameters) | set(_TRAIN_OPTS)) - {"self"}


def build_fm(cfg=None, **kw):
    """Construct an FM from a plain config dict (``kw`` overrides it). Every key
    must be an FMModel parameter or a train option: one that is neither would
    otherwise do nothing, silently."""
    opts = dict(cfg or {})
    opts.update(kw)
    unknown = sorted(set(opts) - fm_keys())
    if unknown:
        raise TypeError(f"unknown FM config key(s) {unknown}; known: {sorted(fm_keys())}")
    train = {k: opts.pop(k) for k in list(opts) if k in _TRAIN_OPTS}
    model = FMModel(**opts)
    for k, v in train.items():
        setattr(model, k, v)
    return model
