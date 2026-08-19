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
# are applied by build_fm() as instance attributes.
#
# These track the research CLI defaults in mae_ddp.py. Two of them did NOT, under
# a comment asserting that they did: mask_ratio was 0.5 against `--mask 0.75`,
# and loss_fused was False against `--fused 1`. m113 trained at 0.75, so a helix
# run masked 48% of tokens where research masked 72% — a different task, not a
# different hyperparameter. `--fused` is MSE-benign (2.4e-7) but selects a
# different logvar clamp for NLL heads, (-12, 8) fused vs (-8, 8) gathered:
# measured 6379 vs 360 on one batch.
#
# plane_frac mixes the two masking modes PER STEP (research mae_ddp.py:187):
# whole-plane masking with that probability, mask_mode otherwise. It is NOT the
# same as mask_mode="plane", which would mask planes every step. m113 trained at
# 0.1; the default here is research's 0.0, so it must be asked for.
# loss_fused stays FALSE as the library default even though research's CLI
# default is --fused 1. It is not just a faster path: it requires a different set
# of batch keys, and for an NLL head it selects the (-12, 8) logvar clamp instead
# of (-8, 8) — measured 6379 vs 360 on one batch. A default that re-tasks every
# existing consumer is the wrong place to express a per-run choice, so the
# training recipe opts in explicitly instead.
_TRAIN_OPTS = dict(mask_mode="random", mask_ratio=0.75, n_planes=1,
                   plane_frac=0.0, loss_fused=False, vis_w=0.0, noisy=False,
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
        if n_bins > 0:
            # PERSISTENT. These are training-set statistics that inference is
            # WRONG without — the BatchNorm running_mean/running_var case, not
            # the causal-mask case. `persistent=False` is for tensors __init__
            # can regenerate from its own arguments (rotary inv_freq, attention
            # masks); bin edges cannot be regenerated from anything.
            #
            # They were non-persistent, so they vanished from every state_dict
            # the trainer wrote and the edges survived only as a PATH STRING in
            # a sidecar json. A checkpoint moved away from its experiment
            # directory was then unusable, and the probe could not read anything
            # we trained. 4 x 129 float32 = 2 kB; there is no cost argument.
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
            # supplied, so neither is ever NaN — an optional table is what let a
            # guard silently disable the metrics that depend on it.
            #
            # `bin_cent_lin` is deliberately absent. It was E[raw ADC | bin]
            # pooled across planes whose norm_sigma differ by 22%, which biases a
            # Y-plane read-back 13% low and a U/V one 6% high — cancelling under a
            # random mask and NOT under a plane mask. Nothing read it; it is
            # removed rather than carried.
            for _nm in ("bin_cent_asinh", "bin_cent_ratio"):
                self.register_buffer(_nm, torch.full((n_band, n_bins), float("nan")))
            # Provenance, so a checkpoint says whether its centroids were MEASURED
            # over a corpus or derived from the edges. Measured on the R1 corpus:
            # a derived cent_ratio under-reads sum|centroid| by 2.7-3.0% per band,
            # with the two open outer bins ~24% low.
            self.register_buffer("bin_cent_measured",
                                 torch.zeros(2, dtype=torch.uint8))
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

    def set_bins(self, edges, *, cent_asinh=None, cent_ratio=None):
        """Register the categorical head's bin edges and centroid tables.

        Required before ``forward`` when ``n_bins > 0``: ``losses_cat`` needs the
        edges. They are training-set statistics the model cannot invent, so they
        are PERSISTENT buffers and travel inside the state_dict — the research
        trainer kept them in a sidecar the checkpoint never referenced, which is
        the failure this closes.

        Centroids are DERIVED from the edges when not supplied, so no centroid
        buffer is ever NaN. That is the point: they used to be optional, and a
        consumer then had to ask "are these present?" — three files asked, one
        forgot, and the evaluator's charge metrics silently never ran. A measured
        table (from ``derive_coeff_bins``) is strictly better and is recorded as
        such in ``bin_cent_measured``, but its absence can no longer disable
        anything.

        The centroid arguments are KEYWORD-ONLY. Three call sites passed
        ``set_bins(edges, cent_asinh, cent_lin)`` positionally; when the signature
        grew a fourth table the third argument silently bound to the wrong one.
        Keyword-only turns that into a TypeError.
        """
        from helix.model.tokenize import bin_centroids_asinh, bin_centroids_ratio

        assert self.n_bins > 0, "set_bins() on a model built with n_bins=0"
        edges = torch.as_tensor(edges, dtype=self.bin_edges.dtype)
        assert edges.shape[1] == self.n_bins + 1, \
            f"edges {tuple(edges.shape)} inconsistent with n_bins={self.n_bins}"
        assert edges.shape[0] == self.bin_edges.shape[0], \
            f"edges has {edges.shape[0]} bands, model has {self.bin_edges.shape[0]}"
        self.bin_edges.copy_(edges.to(self.bin_edges.device))

        e_np = edges.detach().cpu().numpy()
        for k, (nm, given, derive) in enumerate(
                (("bin_cent_asinh", cent_asinh, bin_centroids_asinh),
                 ("bin_cent_ratio", cent_ratio, bin_centroids_ratio))):
            measured = given is not None
            v = torch.as_tensor(given if measured else derive(e_np),
                                dtype=self.bin_edges.dtype)
            assert v.shape == (self.bin_edges.shape[0], self.n_bins), \
                f"{nm} {tuple(v.shape)} != {(self.bin_edges.shape[0], self.n_bins)}"
            getattr(self, nm).copy_(v.to(self.bin_edges.device))
            self.bin_cent_measured[k] = 1 if measured else 0
        return self

    def make_mask(self, B, ratio=None, mode=None, n_planes=None, gen=None):
        """Draw a token mask for this batch under the model's configured policy.

        ``plane_frac`` mixes the two modes PER STEP: with that probability the
        step masks whole planes, otherwise it uses ``mask_mode``. That mix is the
        point — a run that only ever masks randomly never has to reconstruct a
        plane it cannot see, so nothing forces cross-plane triangulation, and a
        run that only ever masks planes never learns the within-plane task.
        m113 trained at 0.1 (research mae_ddp.py:187).

        An explicit ``mode=`` overrides the policy and skips the draw, so callers
        that want one specific mode (evaluation, tests) are unaffected.
        """
        m = self.mask_mode if mode is None else mode
        if mode is None and self.plane_frac > 0:
            # Research draws this from numpy's global RNG. Using torch keeps the
            # draw on the batch's device and honours `gen`, which is the same
            # trade already made and documented for randperm in mask.py: the
            # distribution is identical, the stream is not.
            dev = B["plane_id"].device
            r = (torch.rand((), generator=gen, device=dev)
                 if (gen is not None and gen.device.type == dev.type)
                 else torch.rand((), device=dev))
            if float(r) < self.plane_frac:
                m = "plane"
        return make_mask(B, m,
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
                f"; helix.model.tokenize.to_fm() emits all of them")
        m = self.make_mask(B) if tok_mask is None else tok_mask
        occ, val, logvar = self.raw_heads(B, m)
        if self.n_bins > 0:
            assert torch.isfinite(self.bin_edges).all(), \
                ("n_bins > 0 requires set_bins(edges) before forward() — the "
                 "edges buffer is still unset (NaN). They are training-set "
                 "statistics the model cannot invent.")
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
