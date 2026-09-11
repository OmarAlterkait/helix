# Deferred: the dense-chain integration tests

`test_dense_chain.py.wip` is 32 tests cut from five pimm-data test files when the
forward model moved to helix. It is NOT collected (the `.wip` suffix and this
directory keep it out of pytest's path).

**Why it is not finished.** Those tests were written as coherent units against a
chain that WAS one thing -- Densify -> AddNoise -> Digitize, all in pimm-data.
They share module-level fixtures, helpers and parametrize decorators across the
line the boundary now cuts. Three attempts to split them mechanically each fixed
some breakage and introduced other breakage; the file carries scar tissue from
every cut.

**What to do instead.** Write them fresh: read what each test asserts and write
it against the new API, with its own fixtures. The `.wip` file is the inventory
of WHAT to cover, not a draft to patch.

**Why this is not urgent.** They are integration coverage of the post-collate
dense path, which has no trainer hook -- `pimm/engines/train.py` has zero
references to `batch_transform` or `on_after_batch_transfer`, so the chain cannot
run under pimm today. The halves ARE covered: Densify by pimm-data's suite, the
forward model by helix's `test_forward_noise.py`.
