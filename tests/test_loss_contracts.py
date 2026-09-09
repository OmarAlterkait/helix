"""What the losses COMPUTE, stated as the formula rather than a recorded number.

`test_training_parity` pinned these against the research implementation, and is
retiring with `research/`. Without it, `losses()` and `losses_fused()` had no
value-correctness test anywhere: `test_model_fm::test_forward_returns_loss_dict`
checks only that the loss is finite, zero-dim and carries grad, which a sign
flip, a `.sum()` for a `.mean()`, or MSE substituted for Gaussian NLL all pass.

These assert the MATH, deliberately, not a frozen output. The masking policy, the
head, and the model width are all expected to change; a pinned value would fire
on every legitimate change and be deleted in irritation, whereas the formula is
what has to stay true. Where a golden IS the only available statement -- the
legacy DSP chain, which is finished and has no analytic form -- one exists
(goldens_legacy_dsp.npz). This is the other case.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F                                    # noqa: E402

from helix.model.loss import losses, losses_fused                  # noqa: E402


def _batch(n_cells=6, n_slot=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    occ = (torch.rand(n_cells, n_slot, generator=g) < 0.6)
    tgt = torch.randn(n_cells, n_slot, generator=g)
    cell, slot = occ.nonzero(as_tuple=True)
    return dict(occ=occ.float(), tgt=tgt, target=tgt[cell, slot],
                valid=torch.ones(n_cells, n_slot, dtype=torch.bool),
                inp=torch.randn(n_cells, n_slot, generator=g),
                cell=cell, slot=slot, n_cells=n_cells)


def test_bce_is_masked_only_and_is_the_bce_formula():
    """Occupancy BCE is supervised on MASKED cells only.

    Visible occupancy is observed, so supervising it would leak the answer -- the
    docstring says so and nothing checked it.
    """
    B = _batch()
    occ_logit = torch.randn(6, 4, generator=torch.Generator().manual_seed(1))
    m = torch.zeros(6, dtype=torch.bool); m[::2] = True

    bce, _ = losses(occ_logit, torch.zeros(6, 4), None, B, m)
    sel = m[:, None] & B["valid"]
    want = F.binary_cross_entropy_with_logits(occ_logit[sel], B["occ"][sel])
    assert torch.allclose(bce, want), "BCE is not the masked-only BCE it claims"

    # and it must actually IGNORE the visible half: perturbing visible logits
    # cannot move it.
    pert = occ_logit.clone(); pert[~m] += 5.0
    bce2, _ = losses(pert, torch.zeros(6, 4), None, B, m)
    assert torch.allclose(bce, bce2), "visible occupancy leaked into the BCE"


def test_value_term_is_mse_without_logvar():
    B = _batch()
    mu = torch.randn(6, 4, generator=torch.Generator().manual_seed(2))
    m = torch.zeros(6, dtype=torch.bool); m[:3] = True

    _, val = losses(torch.zeros(6, 4), mu, None, B, m)
    rows = m[B["cell"]]
    want = F.mse_loss(mu[B["cell"], B["slot"]][rows], B["target"][rows])
    assert torch.allclose(val, want)


def test_value_term_is_gaussian_nll_with_logvar():
    """0.5 * ((pred-tgt)^2 * exp(-lv) + lv). An MSE substituted here is invisible
    to a finiteness check and changes what the model learns."""
    B = _batch()
    g = torch.Generator().manual_seed(3)
    mu = torch.randn(6, 4, generator=g)
    lv = torch.randn(6, 4, generator=g)
    m = torch.zeros(6, dtype=torch.bool); m[:3] = True

    _, val = losses(torch.zeros(6, 4), mu, lv, B, m)
    rows = m[B["cell"]]
    p = mu[B["cell"], B["slot"]][rows]
    t = B["target"][rows]
    l = lv[B["cell"], B["slot"]][rows].clamp(-8, 8)
    want = (0.5 * (((p - t) ** 2) * torch.exp(-l) + l)).mean()
    assert torch.allclose(val, want), "value term is not the Gaussian NLL"

    # the sign of the logvar term matters: +lv penalises uncertainty, -lv rewards
    # it and the model drives logvar to -inf. Assert we are on the right side.
    wrong = (0.5 * (((p - t) ** 2) * torch.exp(-l) - l)).mean()
    assert not torch.allclose(val, wrong), "logvar sign is inverted"


def test_vis_w_adds_the_denoising_term_and_zero_disables_it():
    """vis_w=0 must be masked-only; >0 must ADD the visible term, not replace it."""
    B = _batch()
    mu = torch.randn(6, 4, generator=torch.Generator().manual_seed(4))
    m = torch.zeros(6, dtype=torch.bool); m[:3] = True

    _, v0 = losses(torch.zeros(6, 4), mu, None, B, m, vis_w=0.0)
    _, v1 = losses(torch.zeros(6, 4), mu, None, B, m, vis_w=1.0)
    rows = ~m[B["cell"]]
    vis = F.mse_loss(mu[B["cell"], B["slot"]][rows], B["target"][rows])
    assert torch.allclose(v1, v0 + vis), "vis_w does not ADD the visible term"


def test_noisy_predicts_the_input_not_the_clean_target():
    """noisy=True is the self-supervised mode: no clean truth is used at all."""
    B = _batch()
    mu = torch.randn(6, 4, generator=torch.Generator().manual_seed(5))
    m = torch.zeros(6, dtype=torch.bool); m[:3] = True

    _, val = losses(torch.zeros(6, 4), mu, None, B, m, noisy=True)
    rows = m[B["cell"]]
    want = F.mse_loss(mu[B["cell"], B["slot"]][rows],
                      B["inp"][B["cell"], B["slot"]][rows])
    assert torch.allclose(val, want)


def test_fused_agrees_with_the_gather_implementation():
    """losses_fused is a PERFORMANCE rewrite of losses (dense + masks instead of
    advanced indexing). Its whole contract is that it computes the same thing."""
    B = _batch()
    g = torch.Generator().manual_seed(6)
    occ_logit, mu = torch.randn(6, 4, generator=g), torch.randn(6, 4, generator=g)
    m = torch.zeros(6, dtype=torch.bool); m[::2] = True

    for lv, vw in ((None, 0.0), (None, 0.5), (torch.randn(6, 4, generator=g), 0.0)):
        a = losses(occ_logit, mu, lv, B, m, vis_w=vw)
        b = losses_fused(occ_logit, mu, lv, B, m, vis_w=vw)
        for x, y, name in zip(a, b, ("bce", "val")):
            assert torch.allclose(x, y, atol=1e-6), \
                f"fused {name} disagrees with the gather path (logvar={lv is not None}, vis_w={vw})"
