"""What corpus is this, really?

A checkpoint and a corpus can be mismatched with nothing to say so. Each corpus
is internally consistent -- ``CoeffTPCReader`` rejects shards that disagree with
each other (coeff_reader.py:134) -- so a reader pointed at the WRONG corpus is
perfectly satisfied, and ``eval_checkpoint.py`` validates only that the split
name exists. The result is a plausible number computed against data the model
never trained on.

This is not hypothetical. Two corpora live side by side under the same run name:

    coeff_tpc     removal_json {... npass:2}            basis_digest 7f954a84..
    coeff_tpc_r1  removal_json {... npass:2, tau:0.05}  basis_digest 8c4542b6..

Identical wavelet, bands, gids, sigma_norm and noise model; they differ only in
the coherent-removal gate, which changes WHICH COEFFICIENTS SURVIVE. Different
signal, not different vintages of the same signal. Both read cleanly and both
pass ``verify_corpus``.

``basis_digest`` already hashes the full DSP identity (wavelet, level, mode,
n_ticks_raw, pad, band_lengths, removal, threshold, sigma_norm), so it is the
one field worth carrying: equal digests mean the same corpus recipe, and that is
exactly the question an evaluator needs answered.
"""
from __future__ import annotations

import glob
import os

import h5py


def corpus_identity(data_root, dataset_name="sim_wire", split=None):
    """The DSP identity of a corpus, read from the first shard's ``/config``.

    Returns ``{'basis_digest', 'removal_json', 'sigma_norm', 'shard'}``, or
    ``None`` when no shard is found -- callers distinguish "not a corpus" from
    "a corpus that disagrees", which are different failures.
    """
    roots = [os.path.join(data_root, split)] if split else [data_root]
    for r in roots:
        for pat in (f"{dataset_name}_coeff_*.h5", f"*/{dataset_name}_coeff_*.h5"):
            hits = sorted(glob.glob(os.path.join(r, pat)))
            if hits:
                with h5py.File(hits[0], "r") as f:
                    cfg = f["config"]
                    return {
                        "basis_digest": str(cfg.attrs.get("basis_digest", "")),
                        "removal_json": str(cfg.attrs.get("removal_json", "")),
                        "sigma_norm": float(cfg.attrs.get("sigma_norm", 0.0)),
                        "shard": os.path.basename(hits[0]),
                    }
    return None


def check_corpus_matches(recorded, actual, *, where=""):
    """Refuse when a run's recorded corpus identity disagrees with the one open.

    ``recorded`` is what the run stamped at train time (may be None for runs that
    predate the stamp); ``actual`` is :func:`corpus_identity` of the corpus about
    to be read. Returns a note for the caller to log.

    A MISSING record is not an error -- every checkpoint trained before this
    existed has none, and failing them would make the guard unadoptable. A
    PRESENT record that disagrees is always an error: that is the case where a
    number would otherwise come out plausible and wrong.
    """
    if actual is None:
        raise ValueError(f"{where}: no coeff shard found — this is not a corpus")
    if not recorded or not recorded.get("basis_digest"):
        return ("corpus identity NOT RECORDED by this run (predates the stamp); "
                f"reading basis_digest={actual['basis_digest'][:12]}… unchecked")
    a, b = str(recorded["basis_digest"]), str(actual["basis_digest"])
    if a != b:
        raise ValueError(
            f"{where}: corpus mismatch — this run trained on basis_digest={a}\n"
            f"  but the corpus being read is basis_digest={b}\n"
            f"  recorded removal: {recorded.get('removal_json')}\n"
            f"  actual   removal: {actual['removal_json']}\n"
            "  These are different DSP, so the coefficients differ. Point at the "
            "corpus this run trained on, or evaluate a checkpoint trained on this one.")
    return f"corpus identity OK (basis_digest={b[:12]}…)"


def corpus_runs(corpus_root=None) -> list[str]:
    """The run list, read from the corpus's own ``_calib/RUNS.txt``.

    ``corpus_root`` is the GENERATION directory (the parent of the run dirs and
    of ``_calib/``), defaulting to ``HELIX_CORPUS``'s parent.

    Why this exists. The eight run ids were retyped as a literal list in
    ``configs/pimm/coeff_fm_train_8run.py`` and again (the first three) in
    ``coeff_fm_cooldown.py``, while the BUILD side already reads them from a
    file: ``scripts/submit_coeff_corpus.sh`` uses ``$OUT/_calib/RUNS.txt`` and
    ``scripts/calibrate_norm_sigma.sh`` reads ``$CALIB/RUNS.txt``. So the corpus
    ships its own manifest and the training configs ignored it.

    That matters more than ordinary duplication because ``RUNS.txt`` is not
    derived -- ``docs/RUNBOOK.md`` calls it "step zero and hand-written", and
    both build phases abort without it. It is the closest thing the corpus has
    to a declaration of what it contains, and a copy that carries a different
    subset would train happily on whatever the literal named.

    The order is preserved as written: the split is keyed on
    ``blake2b(run/source_file) + event`` and so is order-free, but the run list
    also feeds ``N_TRAIN_EVENTS`` accounting and a stable order keeps two
    readings of the same corpus comparable.
    """
    import os as _os
    if corpus_root is None:
        from helix.paths import root as _root
        corpus_root = _root("HELIX_CORPUS").parent
    manifest = _os.path.join(str(corpus_root), "_calib", "RUNS.txt")
    if not _os.path.isfile(manifest):
        raise FileNotFoundError(
            f"no run manifest at {manifest}. _calib/RUNS.txt is hand-written and "
            f"is step zero of a corpus build (docs/RUNBOOK.md §1); a corpus "
            f"without it is incomplete, not merely undocumented.")
    with open(manifest, "r", encoding="utf-8") as fh:
        runs = fh.read().split()
    if not runs:
        raise ValueError(f"{manifest} is empty")
    return runs
