"""HELIX — Hierarchical Encoding for Learned Inference on eXperimental data.

  helix.core    — the coefficient shard codec, the wavelet transform, robust statistics
  helix.tpc     — the wire-plane chain (helix.tpc.chain.TPCChain): coherent noise removal and
                  wavelet threshold
  helix.optical — optical waveforms (SBND PMT / X-ARAPUCA, goop): one codec, optical shards
  helix.build   — any source's events through both into shards (python -m helix.build sim)
  helix.data    — the coeff corpus as a pimm-data family, its truth and its verifier
  helix.model   — the coefficient foundation model and its tokenizer
  helix.eval    — the evaluation of a trained encoder

Import the module you need; this package imports nothing, so ``import helix`` pulls in neither
pimm nor torch (tests/test_pimm_config.py). ``__version__`` is stamped into every corpus shard.
"""
__version__ = "0.3.0"
