#!/usr/bin/env python
"""Build the coeff corpus from real doraemon sensor shards (the production plane_fn).

Composes pimm-data (geometry + noise — the COLORED incoherent spectrum the old
build omitted, which defaulted to white) with helix (the TPC chain,
helix.tpc.chain.sim_config + build_corpus). This is a BUILD-TIME composer: it imports both helix and pimm-data
(allowed at build time; the read path never does). Not part of the helix package.

plane_fn(event) -> (noisy_planes, clean_planes):
  noisy = digitize( clean_image + incoherent(colored) + coherent )
  clean = the noise-free digitized image (co-supported target at the noisy mask)

The DSP is helix.tpc.chain.TPCChain(sim_config(layout.num_wires, overrides)): the
sensor file's wire layout as the readout, HELIX's block gate, coif3 L4, hysteresis
threshold, D1 dropped. Backends: numpy, or torch on the GPU (densify, noise,
digitize and the chain all on the device).

Usage:
  python build_coeff_corpus.py --shard <sensor.h5> --out <dir> --events 100 \
      --npz <noise_spectrum.npz> --geom cubic_wireplane_geometry.json --run <run>

Verified on run_0027575715: ~39k coeffs/plane, D1 ~1.8%, coeff+coeff_clean
co-supported, reads back through pimm_data.CoeffTPCDataset.
"""
import argparse
import hashlib
import os
import sys
import time

import numpy as np
from pathlib import Path


def _parse_tau(v):
    """Occupancy tolerance. Refuses the legacy 'none' spelling by name.

    'none' meant the pre-2026-08-16 magnitude-only refusal rule. Silently
    accepting it here would let a legacy-rule corpus be built by the primary
    builder and be indistinguishable at the call site from a current one.
    """
    if str(v).lower() == "none":
        raise SystemExit(
            "--tau none is the LEGACY magnitude-only gate rule (what corpora "
            "built before 2026-08-16 used). It is not available here: use "
            "scripts/build_coeff_corpus_legacy.py on the legacy-corpus-repro "
            "branch if you are deliberately reproducing an old vintage.")
    return float(v)


class _RefuseWhite(argparse.Action):
    """--white by name, for the same reason --tau none is refused by name.

    White incoherent noise is the PRE-FIX forward model (the original build
    omitted the measured spectrum and defaulted to white). Left to argparse it
    dies as bare "unrecognized arguments", which is the silent-ish failure the
    tau refusal exists to avoid — and it is how the removal quietly broke
    scripts/compare_noise_bands.py, whose whole job is building one corpus each
    way.
    """

    def __init__(self, option_strings, dest, **kw):
        super().__init__(option_strings, dest, nargs=0, help=argparse.SUPPRESS, **kw)

    def __call__(self, parser, ns, values, option_string=None):
        raise SystemExit(
            "--white is the PRE-FIX white-noise forward model (what corpora built "
            "before the measured spectrum landed used). It is not available here: "
            "use scripts/build_coeff_corpus_legacy.py on the legacy-corpus-repro "
            "branch if you are deliberately reproducing an old vintage, or "
            "comparing against one.")


def _parse_kgate(text):
    """'4.0' -> 4.0 ; '4.0,3.0' -> [4.0, 3.0] (one entry per gate pass)."""
    parts = [p for p in str(text).split(",") if p.strip() != ""]
    vals = [float(p) for p in parts]
    return vals[0] if len(vals) == 1 else vals


def _add_repo_paths(helix_root, pimm_src):
    for p in (helix_root, pimm_src):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def _loader_stream(args, reg, noise_spec):
    """Mode A: pimm-data DataLoader workers + the torch dense tail -> the chain.

    The head (`Collect`) runs per-event in worker PROCESSES on sparse COO, so the
    HDF5 decode overlaps GPU work (measured DataLoader wait: 2.5 ms/event). The
    tail densifies/noises/digitizes on-device with pimm-data's tested torch ops;
    the grids stay on the device for the chain.

    Densify runs BEFORE AddNoise, so the clean target is simply the densified
    image captured before noise is added — no second pass over the event.
    """
    from torch.utils.data import DataLoader, Subset
    from pimm_data import JAXTPCDataset
    from pimm_data.collate import collate_fn
    from pimm_data.batch_transforms import content_seed
    from pimm_data.transform import Compose
    # Registers AddNoise/Digitize, which the Compose below names by string.
    # Before the forward model moved, `from pimm_data.transform import Compose`
    # registered them transitively via pimm_data/__init__.py; it no longer does,
    # and the Compose then raised "AddNoise is not in the transforms registry".
    import helix.data.transforms  # noqa: F401

    # --shard locates the RUN here, not a file: the sensor reader globs every
    # shard of the run and builds ONE joint index over all of them. Naming a
    # different shard of the same run therefore reads exactly the same events.
    root = args.data_root or str(Path(args.shard).parents[2])
    split = args.split or Path(args.shard).parent.name
    head = [dict(type="Collect", parts={"sensor": dict(
        keys=("wire", "time", "value", "plane_gid"))})]
    dense = Compose([dict(type="ToDevice", device="cuda"),
                     dict(type="Densify", geom=reg, modality="sensor")])
    noise = Compose([dict(type="AddNoise", geom=reg, modality="sensor", coherent=True,
                          incoherent=True, series_spectrum=noise_spec,
                          wire_lengths_m=2.33),
                     dict(type="Digitize", geom=reg, modality="sensor", n_bits=12)])
    ds = JAXTPCDataset(data_root=root, split=split, dataset_name=args.dataset_name,
                       modalities=("sensor",), transform=head)
    # Slice the joint index HERE. --event-start used to be ignored in this mode,
    # so every invocation re-read the front of the run: a per-shard build loop
    # produced byte-identical duplicate events in every output shard (noise is
    # seeded from the event NAME, so even the noise repeated).
    n_total = len(ds)
    sel = list(range(min(args.event_start, n_total),
                     min(args.event_start + args.events, n_total)))
    if not sel:
        raise SystemExit(
            f"--event-start {args.event_start} is past the end of the joint index "
            f"({n_total} events in {root}/{split}); nothing to build.")
    ds = Subset(ds, sel)
    # spawn, NOT fork: the parent has initialised CUDA, which a forked worker
    # cannot use safely. spawn re-imports cleanly in each worker.
    dl = DataLoader(ds, batch_size=1, num_workers=args.workers, collate_fn=collate_fn,
                    persistent_workers=args.workers > 0,
                    multiprocessing_context="spawn" if args.workers > 0 else None,
                    prefetch_factor=2 if args.workers > 0 else None)
    for b in dl:
        b = dense(b)
        grids = b["sensor_dense"]
        clean = {int(g): t[0].clone() for g, t in grids.items()}   # capture BEFORE noise
        b = noise(b)
        noisy = {int(g): t[0] for g, t in b["sensor_dense"].items()}
        # Carry the TRUE identity ("<file>_evt<NNN>"), which the batch already
        # holds. The builder used to fabricate provenance from --file-index, so a
        # shard recorded source_file="..._0001.h5" over bytes read from _0000.h5.
        nm = b["name"]
        nm = nm[0] if isinstance(nm, (list, tuple)) else str(nm)
        src, sep, ev = str(nm).rpartition("_evt")
        # Resolve the SAME seed AddNoise derived internally, so the one baked-in
        # noise realisation is recorded and reproducible. AddNoise seeds from
        # batch['name'] via content_seed with base_seed/epoch/rank all 0 here.
        seed = content_seed(str(nm), 0, 0, 0)
        yield (((src, int(ev), seed) if sep and ev.isdigit() else (str(nm), -1, seed)),
               noisy, clean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", required=True,
                    help="sensor HDF5 shard. serial mode reads exactly this file; "
                         "loader mode uses it only to locate the RUN (it reads the "
                         "run-wide joint index) -- use --event-start to select within it")
    ap.add_argument("--out", required=True, help="output corpus dir (<root>/coeff_tpc/<run>/)")
    # The PACKAGED spectrum by default. It used to hardcode
    # /sdf/group/neutrino/omara/JAXTPC/config/noise_spectrum.npz -- one person's
    # checkout on one cluster, and the only external input to the DSP that
    # helix.paths did not resolve. helix ships its own copy (declared in
    # pyproject's package-data), so a clone has it; HELIX_JAXTPC_ROOT still wins
    # when you deliberately want JAXTPC's.
    ap.add_argument("--npz", default=None,
                    help="colored incoherent series spectrum (freqs_hz, shape). "
                         "Default: helix's packaged copy, or "
                         "$HELIX_JAXTPC_ROOT/config/noise_spectrum.npz when set")
    ap.add_argument("--geom", default="cubic_wireplane_geometry.json",
                    help="plane registry (pimm_data.geometry.load_plane_registry)")
    # `sim_wire`, from the same declaration the readers and every other script
    # take it from. The old default was "wire_test_00_00_02" -- the name of a
    # SIMULATION DIRECTORY, not a dataset -- which produced shards that no reader
    # can glob (readers look for f"{dataset_name}_coeff_*.h5"). Every caller
    # passed --dataset-name sim_wire, so the default was reachable only by
    # someone who did not know to, which is the definition of a trap.
    # Default None, resolved after the sys.path bootstrap below. The old default
    # was "wire_test_00_00_02" -- the name of a SIMULATION DIRECTORY, not a
    # dataset -- and it produced shards no reader can glob (readers look for
    # f"{dataset_name}_coeff_*.h5", helix/data/coeff_reader.py). Every caller
    # passed --dataset-name sim_wire, so the default was reachable only by
    # someone who did not know to pass it: a trap, not a convenience.
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--kgate", type=str, default=None,
                    help="coherent-gate threshold in units of the per-band coherent "
                         "scale (chain coherent.kgate). None -> sim_config's default (3.0), which is "
                         "what run_0027575715 was built with. The research figures "
                         "and evals all used 4.0; at 3.0 genuine coherent excursions "
                         "are misclassified as signal and survive as block-wide "
                         "strips (~15x more off-signal residual, and slightly WORSE "
                         "F0). Recorded in removal_json either way. Accepts a "
                         "comma-separated PER-PASS sequence too (npass=2), e.g. "
                         "'4.0,3.0' = aggressive first pass then a permissive "
                         "second; the block gate indexes kgate by pass.")
    ap.add_argument("--tau", type=str, default=None,
                    help="occupancy tolerance on the gate's REFUSAL: refuse only "
                         "when |M| is large AND more than this fraction of the "
                         "block's wires were flagged (chain coherent.tau). Default "
                         "(unset) is sim_config's 0.05 = 3 of 64. Recorded in "
                         "removal_json. The legacy magnitude-only rule ('none', "
                         "what pre-2026-08-16 corpora used) lives on the "
                         "legacy-corpus-repro branch, not here.")
    ap.add_argument("--run", default="")
    ap.add_argument("--file-index", type=int, default=0)
    ap.add_argument("--events", type=int, default=100)
    ap.add_argument("--event-start", type=int, default=0)
    ap.add_argument("--cal-events", type=int, nargs="*", default=None,
                    help="event indices for the normalization table; default = ALL events")
    ap.add_argument("--norm-sigma", default=None,
                    help="load a FROZEN global norm_sigma .npy — use this for every shard "
                         "of a corpus so normalization is identical across shards")
    ap.add_argument("--save-norm-sigma", default=None,
                    help="write the computed norm_sigma to .npy (build shard 0 with this, "
                         "then pass it as --norm-sigma to every other shard)")
    ap.add_argument("--white", action=_RefuseWhite)      # refused by name; see _RefuseWhite
    ap.add_argument("--calibrate", action="store_true",
                    help="compute norm_sigma from these events and write it to "
                         "--save-norm-sigma WITHOUT writing shards. A corpus needs ONE "
                         "frozen table shared by every shard of every run, so calibrate "
                         "across a sample spanning the runs, average, then pass the "
                         "result as --norm-sigma to every build job.")
    ap.add_argument("--allow-index-mismatch", action="store_true",
                    help="serial mode: permit --file-index to disagree with --shard's "
                         "numeric suffix (mislabels provenance; use only deliberately)")
    ap.add_argument("--backend", choices=["numpy", "torch"], default="torch",
                    help="DSP backend: torch (the DEFAULT) runs densify, noise, "
                         "digitize and the TPC chain on the GPU (cuda); numpy runs "
                         "everything on the host.")
    ap.add_argument("--mode", choices=["serial", "loader"], default="serial",
                    help="loader: pimm-data DataLoader workers + the torch dense tail "
                         "(ToDevice/Densify/AddNoise/Digitize) fed to the chain on the "
                         "GPU; the read overlaps the GPU. Needs --backend torch.")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers (loader mode)")
    ap.add_argument("--split", default=None, help="run dir under data_root (loader mode)")
    ap.add_argument("--data-root", default=None, help="dataset root (loader mode)")
    # Default to the tree THIS script lives in, not a hardcoded worktree. The
    # old default (/sdf/group/neutrino/omara/helix-consolidate) was inserted at
    # sys.path[0], so running the builder from any other checkout silently used
    # helix-consolidate's code instead — including over PYTHONPATH.
    ap.add_argument("--helix-root",
                    default=str(Path(__file__).resolve().parent.parent))
    # Default None and resolved BELOW, not here: helix is not importable at
    # argparse time -- this script deliberately puts it on sys.path only in
    # _add_repo_paths, so that --helix-root governs which checkout is used. The
    # literal this replaces (/sdf/group/neutrino/omara/pimm-data/src) ignored
    # HELIX_PIMM_DATA_SRC, which exists in helix.paths for this one consumer.
    ap.add_argument("--pimm-src", default=None)
    args = ap.parse_args()

    # helix_root first, which is derived from __file__ and always available;
    # that makes helix.paths importable so the rest can be resolved.
    _add_repo_paths(args.helix_root, None)
    from helix.paths import resolve as _resolve                  # noqa: E402
    if args.pimm_src is None:
        _ps, _ = _resolve("HELIX_PIMM_DATA_SRC")
        args.pimm_src = str(_ps) if _ps else None
    _add_repo_paths(args.helix_root, args.pimm_src)

    # Same story for the dataset name: it is declared once in
    # helix/data/data/reference_bins.json and read by helix.data.bins.DEFAULTS,
    # and importing that needs pimm_data, which needs the sys.path above.
    if args.dataset_name is None:
        from helix.data.bins import DEFAULTS as _BIN_DEFAULTS    # noqa: E402
        args.dataset_name = _BIN_DEFAULTS["dataset_name"]

    # Resolve --npz here, because argparse cannot: the packaged copy's location
    # depends on where helix was installed, and $HELIX_JAXTPC_ROOT may not be
    # set. A literal default could not fall back, and the literal it used to
    # carry was one person's JAXTPC checkout.
    #
    # `resolve`, not `root`: root() RAISES when a site does not declare
    # HELIX_JAXTPC_ROOT, and "this site has no JAXTPC checkout" is the normal
    # case that must fall through to the packaged copy. resolve() returns None
    # for it instead. (This used to read os.environ directly, because root()
    # once returned an S3DF literal whenever the variable was unset -- that
    # default is gone, so the workaround is too.)
    if args.npz is None:
        from helix.paths import packaged, resolve as _resolve
        _jx, _ = _resolve("HELIX_JAXTPC_ROOT")
        _cand = Path(_jx) / "config" / "noise_spectrum.npz" if _jx else None
        args.npz = str(_cand) if (_cand and _cand.exists()) else str(packaged("noise_spectrum.npz"))
    if not Path(args.npz).exists():
        raise SystemExit(
            f"--npz {args.npz} does not exist. helix ships its own copy; pass "
            f"--npz explicitly, or unset HELIX_JAXTPC_ROOT to use the packaged one.")

    if args.mode == "loader" and args.backend != "torch":
        raise SystemExit("--mode loader runs the torch dense tail on the GPU: use --backend torch")
    from helix.core import backend
    backend.set_backend(args.backend)
    from helix.tpc.io import (sensor_layout, read_sensor_event,
                              read_sensor_event_coo, list_events)
    from helix.core.coeff_io import _code_version

    def _device_name(backend):
        if backend != "torch":
            return backend
        try:
            import torch
            return (f"torch:{torch.cuda.get_device_name(0)}"
                    if torch.cuda.is_available() else "torch:cpu")
        except Exception:
            return "torch:unknown"
    from helix.tpc.chain import TPCChain, sim_config
    from helix.tpc.corpus import build_corpus
    from helix.tpc.geometry import load_plane_registry
    from helix.tpc.noise import generate_noise, digitize

    # Always the MEASURED spectrum. White incoherent noise was an acknowledged
    # bug; the switch that reproduced it now lives only on legacy-corpus-repro.
    npz = np.load(args.npz, allow_pickle=True)
    noise_spec = (npz["spectrum_freqs_hz"], npz["spectrum_shape"])
    reg = load_plane_registry(args.geom)
    layout = sensor_layout(args.shard)
    if layout.num_wires is None:
        raise SystemExit(f"{args.shard}: no /config/num_wires and the planes do not form a "
                         f"volume x plane grid; the chain needs the wire layout")
    coherent = {}
    if args.kgate is not None:
        coherent["kgate"] = _parse_kgate(args.kgate)
    if args.tau is not None:
        coherent["tau"] = _parse_tau(args.tau)
    overrides = {"coherent": coherent} if coherent else None

    use_torch = args.backend == "torch"
    dev = None
    if use_torch:
        import torch
        from helix.tpc import dense_ops as _dops
        from pimm_data.dense_ops import densify as _densify  # going to dense
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    chain = TPCChain(sim_config(layout.num_wires, overrides), device=dev)
    # The coherent noise model's channel groups ARE the readout layout's groups (the
    # block gate removes per group): one number, read from the chain, never set twice.
    group_size = int(chain.cfg["readout"]["group_size"])
    _co, _dw = chain.cfg["coherent"], chain.cfg["dwt"]
    print(f"n_time={layout.num_time_steps} planes={len(layout.plane_labels)} "
          f"wavelet={_dw['wavelet']} L{_dw['level']} removal={_co['method']} "
          f"k{_co['kgate']}/np{_co['npass']}/tau{_co['tau']} group={group_size} "
          f"chain={chain.digest} device={dev or 'numpy'} noise=colored")

    _src_name = Path(args.shard).name

    def _seed(ev):
        """Noise seed for event ``ev`` of THIS shard.

        The shard name is load-bearing. Seeding on the event index alone means
        every per-file job — each of which starts at event 0 — draws the SAME
        noise realisations, so a 500-shard corpus would contain 500 copies of
        each of 200 noise patterns. That is a learnable artefact for a denoising
        or MAE model, and every check we have is blind to it: the corpus is
        deterministic and internally consistent, sigma varies healthily, and the
        coord_digest pairing is correct. Loader mode never had this (it seeds
        from the event NAME, which carries the file); serial mode did.
        """
        return int.from_bytes(
            hashlib.blake2b(f"{args.run}/{_src_name}/ev{ev}".encode(),
                            digest_size=8).digest(), "little") & 0xFFFFFFFF

    def _plane_fn_torch(ev):
        """Torch: densify + noise + digitize ON DEVICE, same ops as loader mode.

        Without this the torch backend read a dense NUMPY image and noised it on
        the host before handing it to the GPU DSP (19.6 s/event, nearly all of it
        the host prologue).
        Uses pimm-data's batched dense ops with B=1; those are the same
        implementations the loader tail runs, and the bounds-guarded ones.
        """
        spec = read_sensor_event_coo(args.shard, ev, layout)
        seed = _seed(ev)
        w_l, t_l, v_l, p_l = [], [], [], []
        gmap = {}
        for label, (w, t, v, nw, nt) in spec.items():
            gid = layout.plane_gid(label)
            gmap[gid] = label
            w_l.append(torch.as_tensor(np.asarray(w), dtype=torch.long, device=dev))
            t_l.append(torch.as_tensor(np.asarray(t), dtype=torch.long, device=dev))
            v_l.append(torch.as_tensor(np.asarray(v), dtype=torch.float32, device=dev))
            p_l.append(torch.full((len(w),), gid, dtype=torch.long, device=dev))
        wire = torch.cat(w_l); time_ = torch.cat(t_l)
        val = torch.cat(v_l); pid = torch.cat(p_l)
        offset = torch.tensor([wire.numel()], dtype=torch.long, device=dev)
        grids = _densify(wire, time_, val, pid, offset, reg)
        clean = {int(g): x[0].clone() for g, x in grids.items()}   # BEFORE noise
        grids = _dops.add_intrinsic_noise(
            grids, reg, seeds=[seed], incoherent=True, coherent=True,
            series_spectrum=noise_spec, group_size=group_size)
        peds = {int(g): layout.pedestal(gmap[int(g)]) for g in grids}
        grids = _dops.digitize(grids, peds, n_bits=12)
        return {int(g): x[0] for g, x in grids.items()}, clean

    def plane_fn(ev):
        if use_torch:
            return _plane_fn_torch(ev)
        planes = read_sensor_event(args.shard, ev, layout)
        rng = np.random.default_rng(_seed(ev))
        noisy, clean = {}, {}
        for label, img in planes.items():
            gid = layout.plane_gid(label)
            nw = img.shape[0]
            wl = np.asarray(reg.get(gid, {}).get("wire_lengths", []), np.float64)
            if wl.size != nw:
                wl = np.full(nw, 2.33, np.float64)
            noise = generate_noise(img.shape, rng=rng, wire_lengths_m=wl,
                                   incoherent=True, coherent=True,
                                   series_spectrum=noise_spec, group_size=group_size)
            noisy[gid] = digitize(img + noise, layout.pedestal(label))
            clean[gid] = img.astype(np.float32)
        return noisy, clean

    norm_in = np.load(args.norm_sigma) if args.norm_sigma else None
    if norm_in is not None:
        print(f"norm_sigma: FROZEN from {args.norm_sigma} {norm_in.shape}")
    # Recorded into /config/noise_json. Without it, a white-noise shard (which only
    # the legacy builder can still produce) is indistinguishable on disk from a
    # colored one, so a corpus accidentally mixed across vintages is undetectable
    # after the fact.
    # Serial mode names its output and its /ident/source_file from --file-index
    # alone, never from --shard. Omitting --file-index in a per-file job loop
    # therefore mislabels provenance AND collides on the output filename, both
    # silently. Require the two to agree when --shard carries a numeric suffix.
    if args.mode == "serial":
        _stem = Path(args.shard).stem
        _suf = _stem.rsplit("_", 1)[-1]
        if _suf.isdigit() and int(_suf) != args.file_index and not args.allow_index_mismatch:
            raise SystemExit(
                f"--shard {Path(args.shard).name} is shard {int(_suf)} but --file-index is "
                f"{args.file_index}. Serial mode derives BOTH /ident/source_file and the "
                f"output filename from --file-index, so this would label the data as shard "
                f"{args.file_index} and overwrite that shard's output. Pass "
                f"--file-index {int(_suf)} (or override deliberately with --allow-index-mismatch).")

    def _sha256(path):
        try:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return None

    # Pin the EXTERNAL inputs the DSP consumed but does not contain. basis_digest
    # covers the wavelet/gate/threshold; it says nothing about the plane geometry
    # or the noise spectrum, so without these a change to either makes old and new
    # shards silently incomparable.
    from helix.tpc.geometry import _resolve as _resolve_geom
    try:
        _geom_path = str(_resolve_geom(args.geom))   # registry resolves a bare name
    except Exception:
        _geom_path = args.geom
    provenance_meta = dict(
        geom=os.path.basename(str(_geom_path)), geom_sha256=_sha256(_geom_path),
        spectrum=os.path.basename(args.npz),
        spectrum_sha256=_sha256(args.npz),
        seed_formula=("loader:content_seed(name|0|0|0)" if args.mode == "loader"
                      else "serial:blake2b(run/source_file/ev{n})"),
        builder="build_coeff_corpus.py",
        # Stamped HERE, by the composer, not only inside helix's writer. This
        # script sys.path-injects a --helix-root and a --pimm-src, so the tree
        # that runs is chosen at the command line; the writer can only report
        # the helix it was imported from, and only if it is recent enough to try.
        # run_0027575715 shows why: 0 of its 100 shards carry `code`, so
        # verify_corpus skipped its mixed-tree check and reported the corpus
        # clean, while a surviving build log shows 12 shards imported corpus.py
        # from a different worktree.
        code=_code_version(),
        # May be None: --pimm-src defaults to HELIX_PIMM_DATA_SRC, and a site
        # need not declare one -- the image INSTALLS pimm-data at the pinned
        # revision, so a checkout is the deliberate override, not the norm.
        # abspath(None) raises, which is how this broke the CI smoke job at
        # stage 2 and masked everything after it.
        pimm_data_src=(os.path.abspath(args.pimm_src) if args.pimm_src else None),
        # The GPU generation is part of the build, not the environment: the
        # torch DSP path is architecture-sensitive (turing vs A100 differ on
        # 0.016% of surviving coefficients for identical code and input), so a
        # corpus spanning generations is not one corpus.
        device=_device_name(args.backend))

    noise_meta = dict(kind="colored",
                      incoherent=True, coherent=True,
                      group_size=group_size,
                      spectrum=os.path.basename(args.npz),
                      mode=args.mode, backend=args.backend)
    t0 = time.perf_counter()
    if args.mode == "loader":
        from helix.tpc.corpus import build_corpus_stream
        stream = _loader_stream(args, reg, noise_spec)
        noisy, clean, norm = build_corpus_stream(
            stream, chain, args.out, dataset_name=args.dataset_name, run=args.run,
            file_index=args.file_index, norm_sigma=norm_in, noise=noise_meta,
            provenance=provenance_meta, write=not args.calibrate,
            with_clean=not args.calibrate,
            cal_events=tuple(args.cal_events) if args.cal_events else None)
    else:
        # Select from the ids that EXIST, not from range(count): production files
        # can omit an id in the middle, so a count-derived range asks for a
        # missing event (hard failure) and drops a real one off the tail.
        # --event-start/--events stay positional into this list, so shards
        # partition the file without overlap however the ids are numbered.
        _ids = list_events(args.shard)
        _evs = _ids[args.event_start:args.event_start + args.events]
        if len(_ids) != (_ids[-1] - _ids[0] + 1 if _ids else 0):
            _absent = sorted(set(range(_ids[0], _ids[-1] + 1)) - set(_ids))
            print(f"note: {Path(args.shard).name} has non-contiguous event ids — "
                  f"{len(_absent)} absent in {_ids[0]}..{_ids[-1]}: {_absent[:10]}"
                  f"{'...' if len(_absent) > 10 else ''}", flush=True)
        # carry (source_file, event, resolved_seed) so /ident records the true
        # origin AND the one noise realisation this shard bakes in
        events = [(_src_name, e, _seed(e)) for e in _evs]
        noisy, clean, norm = build_corpus(events, plane_fn, chain, args.out,
                                          dataset_name=args.dataset_name, run=args.run,
                                          file_index=args.file_index, norm_sigma=norm_in,
                                          noise=noise_meta,
                                          provenance=provenance_meta,
                                          write=not args.calibrate,
                                          with_clean=not args.calibrate,
                                          cal_events=tuple(args.cal_events) if args.cal_events else None)
    dt = time.perf_counter() - t0
    print(f"built {len(noisy)} events in {dt:.1f}s ({dt/max(len(noisy),1):.1f}s/ev); "
          f"coeffs/event={[ce.n_coeff for ce in noisy]}")
    if args.save_norm_sigma and norm is not None:
        Path(args.save_norm_sigma).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_norm_sigma, norm)
        print(f"saved norm_sigma -> {args.save_norm_sigma}")
    print(f"norm_sigma {None if norm is None else norm.shape}"
          f"{' (frozen/global)' if norm_in is not None else ' (derived from this shard)'}"
          f"; wrote -> {args.out}")


if __name__ == "__main__":
    main()
