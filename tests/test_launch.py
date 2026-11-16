"""The launch scripts: scripts/*.sh, scripts/helix_env.sh -> helix_run.sh, scripts/corpus.sh and
scripts/sweep.py, run for real with ``sbatch``, pimm's submit and the container stubbed out.

Nothing here submits a job: ``sbatch`` is a stub on PATH, ``submit.sh`` is replaced by a stub, and
the container runtime is ``none`` (helix_run.sh runs the command directly). The site profile is
nersc's, with $SCRATCH / $CFS pointed at a temporary directory, so this runs on any machine.
"""
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from _fixtures import write_task

H = Path(__file__).resolve().parent.parent


def _stub(path, body):
    path.write_text("#!/bin/bash\n" + body)
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def env(tmp_path):
    """A shell environment on the nersc profile with sbatch stubbed: every call appends its
    arguments, one per line, then a blank line, to ``sbatch.log``."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "sbatch.log"
    _stub(bin_ / "sbatch", f'printf "%s\\n" "$@" >> {log}; echo >> {log}; echo 4242\n')
    e = {k: v for k, v in os.environ.items() if not k.startswith(("HELIX_", "SLURM_"))}
    e.update(PATH=f"{bin_}:{e['PATH']}", HELIX_SITE="nersc", SCRATCH=str(tmp_path), CFS=str(tmp_path),
             HELIX_PYTHON=sys.executable, HELIX_CONTAINER_RUNTIME="none", HELIX_CONTAINER_IMAGE="none",
             HELIX_CONTAINER_PYTHON=sys.executable, HDF5_USE_FILE_LOCKING="FALSE")
    return e


def _sbatch_calls(tmp_path):
    log = tmp_path / "sbatch.log"
    return [c.split("\n") for c in log.read_text().strip("\n").split("\n\n")] if log.exists() else []


@pytest.mark.parametrize("script", sorted(str(p.relative_to(H)) for p in (H / "scripts").glob("*.sh"))
                         + ["container/install.sh"])
def test_every_shell_script_parses(script):
    r = subprocess.run(["bash", "-n", str(H / script)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_corpus_sh_plans_a_campaign_and_submits_every_task_once(tmp_path, env):
    """plan -> build -> task, through helix_env.sh and helix_run.sh: a task file per sensor file,
    the array jobs cover every task exactly once when the plan is longer than one array, and an
    array task hands helix.build its own plan line."""
    sim = tmp_path / "sim"
    for job in (0, 5, 10):                      # three sensor files -> shards 0, 5, 10 at 200 per shard
        write_task(str(sim), first_job=job, n_events=1000, n_tracks=1, n_volumes=1)
    out = tmp_path / "corpus"
    sh = str(H / "scripts" / "corpus.sh")
    r = subprocess.run([sh, "plan", "sim", str(out), str(sim / "j000000")], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = (out / "_calib" / "CHUNKS.txt").read_text().split("\n")[:-1]
    assert [ln.split()[1:5] for ln in lines] == [["0", "1000", "200", k] for k in ("0", "5", "10")]
    assert all(ln.split()[0] == f"{out}/j000000/sim_wire" for ln in lines)
    assert (out / "_calib" / "RUNS.txt").read_text().split() == ["j000000"]

    np.save(out / "_calib" / "norm_sigma_global.npy", np.ones((6, 5), np.float32))
    r = subprocess.run([sh, "build", "sim", str(out)], env=dict(env, ARRAY_MAX="2"), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    calls = _sbatch_calls(tmp_path)
    assert len(calls) == 2
    covered = []
    for c in calls:
        assert "--account=m5238_g" in c and "--gpus=1" in c and "--qos=shared" in c
        arr = next(a for a in c if a.startswith("--array=")).split("=")[1].split("%")[0]
        off = int(next(a for a in c if a.startswith("--export=")).split("OFFSET=")[1])
        lo, hi = map(int, arr.split("-"))
        covered += [off + i for i in range(lo, hi + 1)]
    assert covered == [0, 1, 2]

    # the array task: line OFFSET + SLURM_ARRAY_TASK_ID of the plan, to helix.build (a stub here)
    args = tmp_path / "build.args"
    py = _stub(tmp_path / "bin" / "fakepy", f'printf "%s\\n" "$@" > {args}\n')
    r = subprocess.run([sh, "task", "sim", str(out)], capture_output=True, text=True,
                       env=dict(env, HELIX_CONTAINER_PYTHON=py, OFFSET="2", SLURM_ARRAY_TASK_ID="0"))
    assert r.returncode == 0, r.stderr
    a = args.read_text().split("\n")
    assert a[:3] == ["-m", "helix.build", "sim"]
    assert a[a.index("--input") + 1] == lines[2].split()[5]
    assert a[a.index("--shard-start") + 1] == "10" and a[a.index("--count") + 1] == "1000"
    assert a[a.index("--norm-sigma") + 1] == str(out / "_calib" / "norm_sigma_global.npy")


@pytest.mark.parametrize("study", sorted(p.name for p in (H / "configs" / "sweeps").glob("*.yaml")))
def test_sweep_renders_every_arm_of_every_study(study, tmp_path, env, monkeypatch, capsys):
    """scripts/sweep.py --dry-run on each study file: every arm reaches submit.sh with a config that
    exists and its overrides; a ``then`` stage keeps its arm's overrides under its own run name; each
    ``after_each`` job takes the site's eval resources and its arm's last run name."""
    sys.path.insert(0, str(H / "scripts"))
    try:
        import sweep
    finally:
        sys.path.pop(0)
    log = tmp_path / "submit.log"
    monkeypatch.setattr(sweep, "SUBMIT", _stub(tmp_path / "submit.sh", f'echo "$@" >> {log}\n'))
    for k in [k for k in os.environ if k.startswith(("HELIX_", "SLURM_"))]:
        monkeypatch.delenv(k)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    path = H / "configs" / "sweeps" / study
    spec = yaml.safe_load(path.read_text())
    monkeypatch.setattr(sys, "argv", ["sweep.py", str(path), "--dry-run"])
    sweep.main()

    subs = [ln.split() for ln in log.read_text().splitlines()]
    then = spec.get("then")
    assert len(subs) == len(spec["arms"]) * (2 if then else 1)
    by_name = {s[s.index("--run.name") + 1]: s for s in subs}
    for tag, arm in spec["arms"].items():
        stages = [(tag, spec["config"], {})]
        if then:
            stages.append((tag + then.get("suffix", "_then"), then["config"], then.get("overrides") or {}))
        for name, config, own in stages:
            s = by_name[name]
            assert s[0] == config and (H / "configs" / "pimm" / f"{config}.py").is_file()
            assert "--dry-run" in s
            ov = s[s.index("--") + 1:]
            for k, v in {**(spec.get("overrides") or {}), **((arm or {}).get("overrides") or {})}.items():
                if k not in own:
                    assert f"{k}={v}" in ov, (name, k)
            assert not any("{run}" in x or "$HELIX" in x for x in ov), ov
    sb = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("+ sbatch")]
    if spec.get("after_each"):
        assert len(sb) == len(spec["arms"])
        for tag, ln in zip(spec["arms"], sb):
            last = tag + then.get("suffix", "_then") if then else tag
            assert "--gpus=2" in ln and "--qos=shared" in ln and "--account=m5238_g" in ln
            assert f"/{last}" in ln and "{run}" not in ln
