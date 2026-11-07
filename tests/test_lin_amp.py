"""FMModel(lin_amp_scale): the linear-amplitude input channel, and loading a model without it.

Pinned: 0 is the model without the channel (same parameters, same output); S > 0 widens the
embedding by ``n_slot`` columns that read ``sinh(inp) / S``, start at zero, and leave every other
parameter drawn as at 0, so the new model computes the old one's output; ``pad_input_columns``
loads a checkpoint without the columns by copying the old ones and zeroing the new.
"""
import pytest

torch = pytest.importorskip("torch")

from helix.model import build_fm                                   # noqa: E402
from helix.model.denoise import pad_input_columns                  # noqa: E402
from test_model import SMALL, make_batch                           # noqa: E402

N = SMALL["n_slot"]


def _pair(scale, seed=0):
    torch.manual_seed(seed)
    a = build_fm(dict(SMALL)).eval()
    torch.manual_seed(seed)
    return a, build_fm(dict(SMALL, lin_amp_scale=scale)).eval()


def test_zero_scale_is_the_model_without_the_channel():
    a, b = _pair(0.0)
    assert b.embed.in_features == 2 * N
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert pa.keys() == pb.keys() and all(torch.equal(pa[k], pb[k]) for k in pa)
    B = make_batch(n_slot=N)
    with torch.no_grad():
        assert torch.equal(a.encode(B), b.encode(B))                # bit-identical


def test_a_positive_scale_adds_zero_columns_that_read_the_linear_amplitude():
    a, b = _pair(30.0)
    assert b.embed.in_features == 3 * N
    assert torch.equal(b.embed.weight[:, :2 * N], a.embed.weight) and torch.equal(b.embed.bias, a.embed.bias)
    assert (b.embed.weight[:, 2 * N:] == 0).all()
    pa, pb = dict(a.named_parameters()), dict(b.named_parameters())
    assert all(torch.equal(pa[k], pb[k]) for k in pa if k != "embed.weight")   # drawn as at 0
    B = make_batch(n_slot=N)
    seen = []
    h = b.embed.register_forward_hook(lambda m, i, o: seen.append(i[0]))
    with torch.no_grad():
        torch.testing.assert_close(b.encode(B), a.encode(B), atol=1e-5, rtol=0)   # starts as the old model
    h.remove()
    torch.testing.assert_close(seen[0][:, 2 * N:], torch.sinh(B["inp"]) / 30.0)
    with torch.no_grad():
        b.embed.weight[:, 2 * N:] = 0.1
        assert (b.encode(B) - a.encode(B)).abs().max() > 1e-3     # the channel reaches the encoder


def test_pad_input_columns_copies_the_old_columns_and_zeros_the_new():
    torch.manual_seed(1)
    old = build_fm(dict(SMALL)).eval()
    torch.manual_seed(2)
    new = build_fm(dict(SMALL, lin_amp_scale=30.0)).eval()
    sd = old.state_dict()
    padded = pad_input_columns(new, sd)
    w = padded["embed.weight"]
    assert w.shape == new.embed.weight.shape
    assert torch.equal(w[:, :2 * N], sd["embed.weight"]) and (w[:, 2 * N:] == 0).all()
    assert all(padded[k] is sd[k] for k in sd if k != "embed.weight")             # nothing else touched
    new.load_state_dict(padded, strict=True)
    B = make_batch(n_slot=N)
    with torch.no_grad():
        torch.testing.assert_close(new.encode(B), old.encode(B), atol=1e-5, rtol=0)
    assert pad_input_columns(old, new.state_dict())["embed.weight"].shape[1] == 3 * N   # never shrinks
