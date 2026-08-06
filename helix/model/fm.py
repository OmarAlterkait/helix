"""The coefficient foundation model.

``FMModel`` is extracted verbatim from the research tree
(``coeff_foundation_model/fm/model.py`` lines 138-369) with exactly one
documented edit: the research ``forward`` is renamed ``raw_heads``, and a new
``forward`` implements the pimm Trainer contract (``model(batch) -> dict``
with a ``loss`` key). The parameter tree is untouched, so research checkpoints
load key-for-key — see ``tests/test_model_fm.py``.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from helix.model.layers import (rope_angles, apply_rope, ResponseFiLM,
                                Block, CrossBlock)
from helix.model.loss import losses, losses_fused, losses_cat
from helix.model.mask import make_mask


# Training-policy options: how to mask and which objective to apply. These are
# deliberately NOT arguments of the extracted __init__ — they carry no
# parameters, so keeping them off the verbatim signature keeps a checkpoint's
# architecture metadata and the parameter tree in exact correspondence. They
# are applied by build_fm() as instance attributes; the values here are the
# defaults, and match the research CLI defaults in mae_ddp.py.
_TRAIN_OPTS = dict(mask_mode="random", mask_ratio=0.5, n_planes=1,
                   loss_fused=False, vis_w=0.0, noisy=False,
                   alpha=0.0, beta=0.0, varb=None)


class FMModel(nn.Module):
    def __init__(self, n_slot, n_band, n_plane, n_wirefeat=1, d=128, blocks=4,
                 dec_blocks=2, heads=4, film=("band", "plane", "wire"), nll=False, ffn_mult=4,
                 lam_t=(8.0, 4336.0), lam_w=(32.0, 2048.0), cond="film", dec_mode="self",
                 mup=False, d_base=128, wire_rope=True, n_bins=0):
        super().__init__()
        self.d, self.n_slot, self.nll, self.cond, self.heads = d, n_slot, nll, cond, heads
        self.lam_t, self.lam_w = lam_t, lam_w     # per-axis RoPE wavelength band (time / wire)
        # wire_rope=False: RoPE on TIME ONLY. wire_pos is a PER-PLANE projection axis (non-metric
        # across planes), so relative wire rotation injects nonsense phase onto cross-plane
        # attention pairs (the triangulation pairs). Wire identity still enters via FiLM. (audit bug 2)
        self.wire_rope = wire_rope
        self.n_bins = n_bins                      # >0 => categorical (discretized-bin) value head
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
        _vout = n_slot * (n_bins if n_bins > 0 else (2 if nll else 1))  # cat: K/slot; nll: mu+logvar; else mu
        self.val_head = nn.Linear(d, _vout)
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

    def raw_heads(self, B, tok_mask):
        """Raw head outputs — the research ``forward`` verbatim, renamed so
        ``forward`` can carry the pimm Trainer contract. (Not ``heads``:
        ``self.heads`` is already the attention-head count.)"""
        x = self.forward_feat(B, tok_mask)
        occ = self.occ_head(x) * self.readout_mult      # muP output multiplier (1/m; 1 if mup=False)
        val = self.val_head(x) * self.readout_mult
        if self.n_bins > 0:
            return occ, val.view(-1, self.n_slot, self.n_bins), None   # (occ, logits (n_cells,n_slot,K), None)
        if self.nll:
            val = val.view(-1, self.n_slot, 2)
            return occ, val[..., 0], val[..., 1]            # logit, mu, logvar
        return occ, val, None

    # ---- pimm integration (the documented delta; parameter tree untouched) ---

    def set_bins(self, edges, cent_asinh=None, cent_lin=None):
        """Register the categorical head's bin edges, shape (n_band, n_bins+1).

        Required before ``forward`` when ``n_bins > 0``: ``losses_cat`` needs
        them, and they are training-set statistics rather than learned
        parameters, so they ride as non-persistent buffers instead of living in
        the config. The research trainer kept them in a separate sidecar file
        that the checkpoint never referenced."""
        assert self.n_bins > 0, "set_bins() on a model built with n_bins=0"
        edges = torch.as_tensor(edges)
        assert edges.shape[1] == self.n_bins + 1, \
            f"edges {tuple(edges.shape)} inconsistent with n_bins={self.n_bins}"
        self.register_buffer("bin_edges", edges, persistent=False)
        for nm, v in (("bin_cent_asinh", cent_asinh), ("bin_cent_lin", cent_lin)):
            if v is not None:
                self.register_buffer(nm, torch.as_tensor(v), persistent=False)
        return self

    def make_mask(self, B, ratio=None, mode=None, n_planes=None, gen=None):
        """Draw a token mask for this batch under the model's configured policy."""
        return make_mask(B, self.mask_mode if mode is None else mode,
                         self.mask_ratio if ratio is None else ratio,
                         self.n_planes if n_planes is None else n_planes, gen=gen)

    def forward(self, batch, tok_mask=None):
        """pimm Trainer contract: ``model(batch) -> dict`` carrying ``loss``.

        The research entry point is ``raw_heads(B, tok_mask)``, which this
        wraps: draw the mask (unless one is supplied), run the heads, then apply
        whichever objective the head configuration implies."""
        B = dict(batch)
        B.setdefault("n_cells", B["plane_id"].shape[0])
        # The two objectives read different representations of the same tokens
        # (losses gathers sparse rows; losses_fused reads the dense grid), and a
        # missing key otherwise surfaces as a bare KeyError from inside the loss.
        need = ("occ", "valid", "inp")
        need += ("tgt",) if (self.loss_fused or self.n_bins > 0) else \
                ("cell", "slot", "target")
        gone = [k for k in need if k not in B]
        if gone:
            raise KeyError(
                f"batch is missing {gone} required by "
                f"{'losses_cat' if self.n_bins > 0 else 'losses_fused' if self.loss_fused else 'losses'}"
                f"; helix.tokenize.to_fm() emits all of them")
        m = self.make_mask(B) if tok_mask is None else tok_mask
        occ, val, logvar = self.raw_heads(B, m)
        if self.n_bins > 0:
            assert hasattr(self, "bin_edges"), \
                "n_bins > 0 requires set_bins(edges) before forward()"
            bce, vloss = losses_cat(occ, val, B, m, self.bin_edges, vis_w=self.vis_w)
        elif self.loss_fused:
            bce, vloss = losses_fused(occ, val, logvar, B, m, vis_w=self.vis_w,
                                      noisy=self.noisy, alpha=self.alpha,
                                      beta=self.beta, varb=self.varb)
        else:
            bce, vloss = losses(occ, val, logvar, B, m, vis_w=self.vis_w,
                                noisy=self.noisy)
        return {"loss": bce + vloss, "bce": bce.detach(), "val": vloss.detach(),
                "masked_frac": m.float().mean().detach()}


for _k, _v in _TRAIN_OPTS.items():       # class defaults; build_fm overrides per instance
    setattr(FMModel, _k, _v)


def build_fm(cfg=None, **kw):
    """Construct an FM from a plain config dict, ignoring unrelated keys.

    Accepts a checkpoint's own metadata dict directly, which is how
    ``tools/convert_fm_ckpt.py`` rebuilds a research model."""
    import inspect
    from helix.model.serial import SerialFMModel   # lazy: serial imports this module

    opts = dict(cfg or {})
    opts.update(kw)
    cls = SerialFMModel if opts.pop("serial", True) else FMModel
    if cls is SerialFMModel and opts.get("dec_mode", "self") != "cross":
        # SerialFMModel's decoder is grouped-CROSS attention: it reads
        # blk.q/kv/nq/nk, which only a CrossBlock has. With dec_mode="self" the
        # decoder is a plain Block and this dies as an AttributeError several
        # frames into attention. Refuse up front instead.
        raise ValueError(
            f"serial=True requires dec_mode='cross' (got "
            f"{opts.get('dec_mode', 'self')!r}); pass serial=False for the "
            f"full-attention model")
    train = {k: opts.pop(k) for k in list(opts) if k in _TRAIN_OPTS}
    arch = set(inspect.signature(cls.__init__).parameters) | \
        set(inspect.signature(FMModel.__init__).parameters)
    model = cls(**{k: v for k, v in opts.items() if k in arch and k != "self"})
    for k, v in train.items():
        setattr(model, k, v)
    return model
