#!/usr/bin/env python3
"""What did each run ACTUALLY do? Join the run's own files to SLURM accounting.

Analysis here re-derived a run's identity from its FILENAME. A filename is a
label a human typed. `sweep/steps_b8` ran at lr 4.4e-3 -- B=16's rate -- under a
plot captioned "each at its own best LR", and nothing noticed for two days.

Two authoritative sources already existed and neither was read:

  * Each run directory holds `resolved_config.json`, `run_metadata.json` and
    `provenance.json`, written by pimm and helix themselves.
  * SLURM's accounting DB holds the exit status of every job step, via the
    `SLURM_JOB_ID` that `provenance.json` records. `sacct`'s DerivedExitCode is
    "the highest exit code returned by the job's job steps", so it reports a
    failed `srun` EVEN WHEN the batch script around it exited 0 -- which 7 of
    the harness's 10 wrappers do, by ending an arm with `echo "exit=$?"`.

An earlier version of this file invented a `status.json` sidecar for the exit
code. That was a reinvention and a worse one: it exists for no historical run,
whereas sacct answers for all of them, retroactively, and cannot be corrupted by
the broken wrapper whose failure it is reporting.

CAVEAT on DerivedExitCode: it is per JOB, and one job here produced up to nine
runs. A nonzero value means SOME step failed, not that THIS run did. Step-level
detail (`sacct -j <id>` without -X) attributes it, but the harness names its
steps `env`, so step->arm mapping needs `srun --job-name=<tag>` to be added
before that is automatic. Treat a nonzero derived code as "inspect this job",
not "discard this run".

Usage:  scripts/audit_runs.py <runs_dir> [<runs_dir> ...]

Each <runs_dir> holds one directory per run (a pimm save_path).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

B_IN_TAG = re.compile(r"_b(\d+)")
# No '-' in the class: `_lr2.2e-3-retry` would capture a trailing '-', float()
# would raise, and the tag/config comparison would be skipped in silence -- in a
# tool whose whole thesis is that silence is the enemy.
LR_IN_TAG = re.compile(r"_lr([0-9.]+(?:e[+-]?\d+)?)")


def _load(d: Path, name: str):
    p = d / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (ValueError, OSError):
        return None


def _sacct(job_ids):
    """One batched query. Returns {job_id: (state, exit, derived)}."""
    if not job_ids:
        return {}
    try:
        out = subprocess.run(
            ["sacct", "-n", "-X", "-P", "-j", ",".join(sorted(job_ids)),
             "--format=JobID,State,ExitCode,DerivedExitCode"],
            capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = {}
    for line in out.strip().splitlines():
        f = line.split("|")
        if len(f) >= 4:
            rows[f[0]] = (f[1].split()[0], f[2], f[3])
    return rows


def collect(d: Path) -> dict | None:
    cfg = _load(d, "resolved_config.json") or {}
    meta = _load(d, "run_metadata.json") or {}
    prov = _load(d, "provenance.json")
    if not cfg and not meta:
        return None
    rec = (prov[0] if isinstance(prov, list) and prov else prov) or {}
    batch, nw = cfg.get("batch_size"), cfg.get("num_worker")
    return dict(
        tag=d.name,
        batch=batch,
        nw=nw,
        # `nw and batch` would skip an explicit num_worker=0 -- the exact value
        # this check exists to catch -- because 0 is falsy.
        wpg=(nw // batch) if (nw is not None and batch) else None,
        lr=cfg.get("optimizer", {}).get("lr"),
        seed=cfg.get("seed"),
        job=str((rec.get("env") or {}).get("SLURM_JOB_ID") or ""),
        helix_dirty=bool(meta.get("git", {}).get("is_dirty")),
        # The framework's cleanliness is the one that was never checked, and it
        # is the worse of the two: 38/38 of this campaign ran a dirty pimm.
        pimm_dirty=bool((rec.get("pimm") or {}).get("dirty")),
    )


def problems(r: dict) -> list:
    out = []
    m = B_IN_TAG.search(r["tag"])
    if m and r["batch"] is not None and int(m.group(1)) != r["batch"]:
        out.append(f"tag says b{m.group(1)}, config says batch={r['batch']}")
    m = LR_IN_TAG.search(r["tag"])
    if m and r["lr"] is not None and abs(float(m.group(1)) - float(r["lr"])) > 1e-12:
        out.append(f"tag says lr={m.group(1)}, config says lr={r['lr']}")
    if r["wpg"] == 0:
        # The seed does collapse (pimm derives it as seed + rank*workers_per_gpu),
        # but the streams decorrelate within a step or two, so this is NOT a
        # reason to discard a loss curve. The measured cost is throughput:
        # In-process loading measured ~44% slower per step than 1+ worker at 16
        # GPUs. These runs are sound as loss measurements and not comparable as
        # step-time measurements.
        out.append("0 workers/GPU -- loss ok, step times not comparable")
    return out


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]] or [Path.cwd()]
    rows = []
    for root in roots:
        if not root.is_dir():
            print(f"skip {root}: not a directory", file=sys.stderr)
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            r = collect(d)
            if r:
                r["root"] = root.name
                rows.append(r)
    if not rows:
        print("no runs with sidecars found")
        return 0

    st = _sacct({r["job"] for r in rows if r["job"]})
    for r in rows:
        s = st.get(r["job"], ("?", "?", "?"))
        r["state"], r["exit"], r["derived"] = s
        r["problems"] = problems(r)

    hdr = (f"{'run':32s} {'B':>4s} {'w/gpu':>5s} {'lr':>8s} {'job':>9s} "
           f"{'state':>10s} {'deriv':>6s} {'dirty(h/p)':>10s}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['root']+'/'+r['tag']:32s} {str(r['batch']):>4s} {str(r['wpg']):>5s} "
              f"{str(r['lr']):>8s} {r['job']:>9s} {r['state']:>10s} {r['derived']:>6s} "
              f"{str(r['helix_dirty'])[0]+'/'+str(r['pimm_dirty'])[0]:>10s}")

    bad = [r for r in rows if r["problems"]]
    print(f"\n{len(rows)} runs | helix dirty {sum(r['helix_dirty'] for r in rows)} | "
          f"pimm dirty {sum(r['pimm_dirty'] for r in rows)} | "
          f"0 workers/GPU {sum(r['wpg'] == 0 for r in rows)}")
    if bad:
        print(f"\n{len(bad)} run(s) needing attention:")
        for r in bad:
            for p in r["problems"]:
                print(f"  * {r['root']}/{r['tag']}: {p}")

    # Per JOB, not per run: one job produced up to nine runs here, so printing
    # this on every row implies nine failures where there was one.
    jobs = {}
    for r in rows:
        if r["derived"] not in ("0:0", "", "?"):
            jobs.setdefault((r["job"], r["state"], r["derived"]), []).append(r["tag"])
    if jobs:
        print(f"\n{len(jobs)} job(s) with a failed step -- inspect, do not "
              f"assume every run below is bad (`sacct -j <id>` attributes it):")
        for (job, state, deriv), tags in sorted(jobs.items()):
            print(f"  * job {job} [{state}] DerivedExitCode {deriv} -> "
                  f"{len(tags)} run(s): {', '.join(sorted(tags)[:4])}"
                  f"{' ...' if len(tags) > 4 else ''}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
