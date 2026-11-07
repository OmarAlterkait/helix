"""Every named source has what helix.build needs (``python -m helix.build SOURCE``)."""
from __future__ import annotations

import pytest

from helix.build import SOURCES, source


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_every_source_has_what_build_needs(name):
    """What :func:`helix.build.main` and :func:`helix.build.plan` read of a source class."""
    Source = source(name)
    for attr in ("systems", "dataset", "pattern", "events", "shard_attrs", "__len__"):
        assert hasattr(Source, attr), f"{name}: {Source.__name__} has no {attr}"
    assert hasattr(Source, "chain_config") or hasattr(Source, "optical_chain")
    assert not hasattr(Source, "chain_config") or hasattr(Source, "bind")


def test_an_unknown_source_is_refused():
    with pytest.raises(SystemExit, match="unknown source"):
        source("nope")
