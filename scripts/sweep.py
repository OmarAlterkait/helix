#!/usr/bin/env python3
"""Submit a study: several arms of one training config, and the evaluation that follows them.

    python scripts/sweep.py configs/sweeps/<study>.yaml [--dry-run] [--only TAG[,TAG]]

A study file (configs/sweeps/*.yaml) is the whole study, in place of a launcher script:

    config: fm                          # configs/pimm/<config>.py
    recipe: preempt                     # configs/launch/<recipe>.yaml (default preempt)
    submit: [--resources.nnodes, "4"]   # pimm submit flags, every arm
    overrides: {epoch: 3}               # training overrides (key: value), every arm
    arms:                               # run name -> its own overrides (and optional submit flags)
      lr1.1e-3: {overrides: {optimizer.lr: 1.1e-3}}
      lr2.2e-3: {overrides: {optimizer.lr: 2.2e-3}, submit: [--chain.jobs, "4"]}
    then:                               # optional: a second training run per arm, after the first
      config: fm_cooldown               #   (here the WSD cooldown of each stable phase)
      suffix: _cool                     #   its run name: <arm><suffix>
      overrides: {weight: "$HELIX_EXP/{run}/model/last"}
    after_each:                         # optional: one job per arm, once that arm has succeeded
      command: python -m helix.eval --uprobe $HELIX_EXP/{run} --weights last --out $HELIX_EXP/{run}/eval.json
      time: "02:00:00"
    after_all:                          # optional: one job once every arm has succeeded
      command: python scripts/compare.py {runs}

Each arm goes through scripts/submit.sh (so through pimm's launcher, site facts and all) as run
``<arm>`` under $HELIX_EXP. The ``after_*`` commands run in the container (scripts/helix_run.sh) as
one-GPU batch jobs of the site's ``eval`` job kind (helix/sites/<site>.yaml), held by ``--dependency=afterok`` on the arm's
job: submitit requeues a preempted chain under the same job id, so the dependency waits for the
whole chain. ``then`` is submitted the same way, behind its arm, with the study's and the arm's
overrides carried over (a cooldown keeps its arm's LR and seed) under its own. In ``then`` overrides ``{run}`` is
the arm's first run; in ``after_*`` commands ``{run}`` is the arm's LAST run (its ``then`` run if
there is one) and ``{runs}`` all of them, space-separated. ``$VARS`` in overrides expand from the
site environment (scripts/helix_env.sh).
"""
import argparse
import os
import re
import shlex
import subprocess
import sys

import yaml

H = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMIT = os.path.join(H, "scripts", "submit.sh")
JOB_ID = re.compile(r"job id:\s+(\d+)")


def _flags(xs):
    return [str(x) for x in (xs or [])]


def _overrides(*ds):
    out = {}
    for d in ds:
        out.update(d or {})
    return [f"{k}={v}" for k, v in out.items()]


def submit_arm(study, tag, arm, dry, env, after=None):
    cmd = [SUBMIT, study["config"], "--recipe", study.get("recipe", "preempt"), "--run.name", tag,
           *_flags(study.get("submit")), *_flags(arm.get("submit"))]
    if after:
        cmd += ["--resources.dependency", f"afterok:{after}"]
    if dry:
        cmd.append("--dry-run")
    ov = [_expand(x, env) for x in _overrides(study.get("overrides"), arm.get("overrides"))]
    if ov:
        cmd += ["--", *ov]
    print("+", shlex.join(cmd), flush=True)
    out = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE).stdout
    sys.stdout.write(out)
    m = JOB_ID.search(out)
    if not m and not dry:
        raise SystemExit(f"sweep: no job id in pimm submit's output for {tag}")
    return m.group(1) if m else None


def _expand(s, env):
    """``$VAR`` / ``${VAR}`` from the site environment; anything else is left as written."""
    return re.sub(r"\$\{?(\w+)\}?", lambda m: env.get(m.group(1), m.group(0)), s)


def site_env():
    """helix_env.sh's exports, for the after-jobs' sbatch flags."""
    out = subprocess.run(["bash", "-c", f"source {shlex.quote(os.path.join(H, 'scripts', 'helix_env.sh'))} && env -0"],
                         check=True, stdout=subprocess.PIPE).stdout
    return dict(kv.split("=", 1) for kv in out.decode().split("\0") if "=" in kv)


def submit_after(spec, deps, name, fmt, env, dry):
    command = _expand(spec["command"], env)
    for k, v in fmt.items():                       # not str.format: commands may hold their own braces
        command = command.replace("{" + k + "}", v)
    sb = ["sbatch", "--parsable", "--nodes=1", f"--time={spec.get('time', '02:00:00')}", f"--job-name={name}",
          f"--output={env.get('HELIX_EXP', '.')}/{name}-%j.out"]
    # the site's `eval` job kind: one A100 with its cores and memory, on a QOS that does not
    # preempt (a preempted evaluation starts over)
    for flag, key in (("account", "ACCOUNT"), ("qos", "QOS"), ("constraint", "CONSTRAINT"),
                      ("partition", "PARTITION"), ("gpus", "GPUS"), ("cpus-per-task", "CPUS_PER_TASK"),
                      ("mem", "MEM")):
        v = env.get(f"HELIX_SLURM_EVAL_{key}")
        if v:
            sb.append(f"--{flag}={v}")
    if not env.get("HELIX_SLURM_EVAL_GPUS"):
        raise SystemExit("sweep: this site declares no scheduler.eval (helix/sites/<site>.yaml); "
                         "evaluation jobs need a GPU")
    if deps:
        sb.append("--dependency=afterok:" + ":".join(deps))
    wrap = (f"cd {shlex.quote(H)} && HELIX_INTERACTIVE=0 HDF5_USE_FILE_LOCKING=FALSE "
            f"scripts/helix_run.sh {command}")
    sb.append(f"--wrap={wrap}")
    print("+", shlex.join(sb), flush=True)
    if not dry:
        jid = subprocess.run(sb, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()
        print(f"  {name}: job {jid}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("study")
    ap.add_argument("--dry-run", action="store_true", help="render every submission, submit nothing")
    ap.add_argument("--only", default=None, help="comma-separated arm names")
    a = ap.parse_args()
    study = yaml.safe_load(open(a.study))
    arms = study["arms"]
    if a.only:
        keep = a.only.split(",")
        unknown = sorted(set(keep) - set(arms))
        if unknown:
            raise SystemExit(f"sweep: no arm(s) {unknown}; the study has {sorted(arms)}")
        arms = {k: arms[k] for k in keep}
    env = site_env()
    jobs = {}
    for tag, arm in arms.items():
        jid = submit_arm(study, tag, arm or {}, a.dry_run, env)
        then = study.get("then")
        if then:
            # the stage continues the arm: the study's and the arm's overrides (its LR, its seed),
            # then the stage's own on top -- without them a cooldown anneals from the config's LR
            ov = dict(study.get("overrides") or {})
            ov.update((arm or {}).get("overrides") or {})
            ov.update({k: str(v).replace("{run}", tag) for k, v in (then.get("overrides") or {}).items()})
            stage = dict(config=then["config"], recipe=then.get("recipe", study.get("recipe", "preempt")),
                         submit=then.get("submit"), overrides=ov)
            tag = tag + then.get("suffix", "_then")
            jid = submit_arm(stage, tag, {}, a.dry_run, env, after=jid)
        jobs[tag] = jid
    for tag, jid in jobs.items():
        if study.get("after_each"):
            submit_after(study["after_each"], [jid] if jid else [], f"{tag}-after", dict(run=tag), env, a.dry_run)
    if study.get("after_all"):
        deps = [j for j in jobs.values() if j]
        submit_after(study["after_all"], deps, f"{os.path.splitext(os.path.basename(a.study))[0]}-after",
                     dict(runs=" ".join(jobs)), env, a.dry_run)


if __name__ == "__main__":
    main()
