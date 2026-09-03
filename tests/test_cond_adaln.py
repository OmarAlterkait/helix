"""cond='adaln' must actually condition the model it is built into.

AdaLN was carried over from the research model and never selected for a run, so
nothing ever checked that it RUNS. It did not. ``SerialFMModel`` — the class
``build_fm`` returns by default, i.e. every production run — overrides
``_emb``/``encode``/``forward_feat`` and re-implements the block bodies in
``_self``/``_cross``, and none of those had an AdaLN branch. Selecting
``cond='adaln'`` therefore disabled FiLM (``FMModel.__init__`` builds
``self.film`` only ``if film and not adaln``), allocated ``ada`` and
``cond_wire`` weights, and then never called any of them: a strictly weaker
model than ``cond='film'`` plus a few hundred thousand dead parameters carrying
optimizer state and an EMA shadow. Nothing raised.

The failure mode is "a parameter exists but no forward path reaches it", so the
tests assert on GRADIENT FLOW rather than on output values. A value test would
have passed against the broken code: AdaLN-Zero is identity at init, so the
broken and fixed models agree on step 0 and diverge only after training.
"""

import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm                                   # noqa: E402
from test_model_fm import SMALL, make_batch                        # noqa: E402


def _cond_params(model):
    """Every parameter that exists ONLY to carry conditioning."""
    return {n: p for n, p in model.named_parameters()
            if ".ada." in n or n.startswith("cond_wire.")}


def _backward(model, seed=0, off_zero=True):
    """One masked forward/backward, arranged so a live path cannot look dead.

    Two degeneracies have to be stepped around, both artefacts of init rather
    than of the model:

    * ``mask_tok`` is zeros and ``dec_norm.bias`` is zeros, so at init the
      decoded rows are EXACTLY zero and a ``feat.square()`` loss has zero
      gradient there — every decoder parameter then reads as dead. A random
      linear functional has a non-degenerate gradient everywhere instead.
    * AdaLN-Zero zero-inits ``ada``, and ``cond_wire`` reaches the loss only
      THROUGH ``ada.weight``, so its gradient is legitimately zero on step 0 and
      non-zero forever after. ``off_zero`` nudges ``ada`` off zero first, which
      is simply where training is after one step.
    """
    if off_zero:
        with torch.no_grad():
            for n, q in model.named_parameters():
                if ".ada." in n:
                    q.add_(torch.randn_like(q) * 0.02)
    B = make_batch(n_slot=SMALL["n_slot"], n_band=SMALL["n_band"],
                   n_plane=SMALL["n_plane"], seed=seed)
    tok_mask = torch.zeros(B["n_cells"], dtype=torch.bool)
    tok_mask[::3] = True                       # a third masked, both sets non-empty
    feat = model.forward_feat(B, tok_mask)
    torch.manual_seed(seed)
    (feat * torch.randn_like(feat)).sum().backward()
    return B


@pytest.mark.parametrize("serial", [True, False])
def test_every_adaln_parameter_receives_gradient(serial):
    """The regression itself: ada/cond_wire allocated but never reached.

    AdaLN-Zero zero-inits ``ada``, so the shift/scale slices legitimately have
    zero gradient at step 0 (the gate multiplies them by 0). The GATE slices do
    not, so the per-parameter grad norm is non-zero for a live path and exactly
    zero — or None — for a dead one.
    """
    model = build_fm(dict(SMALL, cond="adaln", serial=serial))
    params = _cond_params(model)
    assert params, "cond='adaln' built no conditioning parameters at all"
    _backward(model)
    dead = [n for n, p in params.items()
            if p.grad is None or p.grad.abs().sum().item() == 0.0]
    assert not dead, (
        f"serial={serial}: {len(dead)}/{len(params)} conditioning parameters got "
        f"no gradient — they are allocated but no forward path reaches them: {dead}")


@pytest.mark.parametrize("serial", [True, False])
def test_adaln_replaces_film_rather_than_silently_dropping_conditioning(serial):
    """adaln turns FiLM OFF, so it must supply the identity FiLM would have.

    Under the broken code the mask queries were the bare ``mask_tok`` — one
    identical vector for every masked position, separated only by RoPE — so
    band/plane identity reached the decoder through nothing at all.
    """
    model = build_fm(dict(SMALL, cond="adaln", serial=serial))
    assert model.film is None, "adaln and FiLM are mutually exclusive by construction"
    assert model.cond_wire is not None, "adaln must build the wire conditioner it uses"
    _backward(model)
    for i in range(len(model.dec)):
        g = model.dec[i].ada.weight.grad
        assert g is not None and g.abs().sum().item() > 0, (
            f"serial={serial}: decoder block {i} never applied its conditioning; "
            "the masked queries carry no band/plane identity")


@pytest.mark.parametrize("serial", [True, False])
def test_film_path_builds_no_adaln_parameters(serial):
    """The default path must not pay for the alternative it did not choose."""
    model = build_fm(dict(SMALL, cond="film", serial=serial))
    assert not _cond_params(model), "cond='film' allocated AdaLN weights"
    assert model.film is not None and model.cond_wire is None


def test_adaln_is_identity_at_init_but_not_after_a_step():
    """AdaLN-Zero's contract, and the reason a value test could not catch the bug.

    Zero-init makes both gates 0, so block outputs are the residual stream
    untouched; once ``ada`` moves off zero the outputs must move too.
    """
    model = build_fm(dict(SMALL, cond="adaln", serial=True))
    B = make_batch(n_slot=SMALL["n_slot"], n_band=SMALL["n_band"],
                   n_plane=SMALL["n_plane"])
    tok_mask = torch.zeros(B["n_cells"], dtype=torch.bool); tok_mask[::3] = True
    with torch.no_grad():
        before = model.forward_feat(B, tok_mask).clone()
        for p in _cond_params(model).values():
            p.add_(torch.randn_like(p) * 0.05)
        after = model.forward_feat(B, tok_mask)
    assert not torch.allclose(before, after, atol=1e-6), \
        "perturbing every conditioning weight changed nothing — the path is dead"
