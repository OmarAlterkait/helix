"""Size an arm to fill its two-hour preempt slot.

A preempt slot is billed at its 2 h floor (--time-min) whether or not the work
fills it, and it cannot be preempted inside that floor -- so an arm sized to
finish inside it needs no resume, and every second it leaves unused is paid for
anyway. The run's wall time is modelled term by term:

    wall = startup + steps * step_s + n_eval * eval_s + n_ckpt * ckpt_s

and the step count is whatever makes that equal a fraction of the floor: 0.95
where the step time was measured for that (width, GPU count), 0.90 where it is
predicted. TIMEOUT does not requeue, and a cooldown cut off by the wall is a lost
run, so the margin is not optional.

Numbers are for the merged model (helix 3aa96b2: one implementation,
compile_blocks=True) on Perlmutter A100-40GB. The previous table, for the
shipped model, is in slot_old_shipped.py.
"""

WALL = 7200
TRAIN_EVENTS = 150_239

# d512, s/step by GPU count. 4 and 16 are measured (steady median, 2026-09-23);
# the rest are the shipped model's measured pace at that count times the speedup
# measured at 16 GPUs (0.217 -> 0.140), so +-10%. 512 is extrapolated.
STEP_D512 = {1: 0.096, 4: 0.117, 8: 0.128, 16: 0.140, 32: 0.150,
             64: 0.150, 128: 0.158, 256: 0.205, 512: 0.215}
# 256/512 GPUs are MEAN s/step, not the steady median: at 64 nodes stragglers
# stall the step (median 0.134-0.149, mean 0.19-0.30). Measured all-in 0.212-0.218
# s/step over 31-34k steps on wu256-12.4e-3 and lr256-6.2e-3 (2026-09-25/26),
# after which both wu256 arms, sized at 0.166, timed out at 31.4k of 35.9k.
# Measured s/step for (d, gpus), interactive A1 2026-09-23 (3,000 steps each,
# compiled, steady median). Where a (d, gpus) is not measured the d512 table is
# scaled by that width's 16-GPU factor -- the production shape.
MEASURED_STEP = {(256, 4): 0.0920, (512, 4): 0.1145, (768, 4): 0.1614,
                 (1024, 4): 0.2189,
                 (256, 16): 0.1013, (512, 16): 0.1352, (768, 16): 0.2007,
                 (1024, 16): 0.2720}
MEASURED = set(MEASURED_STEP)


def width_factor(d):
    if (d, 16) in MEASURED_STEP:
        return MEASURED_STEP[(d, 16)] / MEASURED_STEP[(512, 16)]
    raise KeyError(f"d={d} has no 16-GPU measurement; run interactive A1b first")

N_EVAL = 30           # points on the curve; EVERY = steps // N_EVAL
N_CKPT = 4


def step_s(d, gpus):
    if (d, gpus) in MEASURED_STEP:
        return MEASURED_STEP[(d, gpus)]
    return STEP_D512[gpus] * width_factor(d)


def startup_s(nodes):
    """Job start -> steady state: container, corpus index, torch.compile, its
    recompiles, and the first eval's own compile (~30-50 s). Measured 150-230 s
    (1 node) and 250-380 s (4 nodes) before that first eval; job start to the
    first logged step ~13 min at 16 nodes and ~20 min at 64 (2026-09-25/26)."""
    return 280 if nodes == 1 else 430 if nodes <= 4 else 800 if nodes <= 16 else 1250


def eval_s(gpus, d=512):
    """One evaluation after the first: 256 val events over the ranks plus the
    val workers' start-up (this pimm re-forks them every eval), scaled by width.
    Measured at B=16: 9.8 s (d512) and 17-18.5 s (d1024); at B=4 14 s (d512).
    Deliberately on the high side."""
    g16 = MEASURED_STEP[(d, 16)] / MEASURED_STEP[(512, 16)]
    return (6.0 + 0.2 * 256 / gpus) * g16


def ckpt_s(d):
    return 15.0 * (d / 512) ** 2


def margin(d, gpus):
    return 0.95 if (d, gpus) in MEASURED else 0.90


def steps_for(nodes, d=512, wall=WALL):
    g = nodes * 4
    budget = wall * margin(d, g) - startup_s(nodes) - N_EVAL * eval_s(g, d) - N_CKPT * ckpt_s(d)
    return int(budget / step_s(d, g))


def eval_every(steps):
    return max(100, steps // N_EVAL // 100 * 100)


CD_EVALS = 5          # evals per in-job cooldown; its endpoint is what is read


def surface_wall(nodes, d, S, frac=0.1, branches=(0.25, 0.5, 0.75)):
    """Wall time of a job that trains S stable steps, then cools the checkpoints
    at `branches` x S for frac x their step count each. The cooldowns run eagerly
    (each is its own launch, and compiling for ~1k steps would not pay back):
    ~1.21x the compiled step, ~150 s to start."""
    g = nodes * 4
    s = step_s(d, g)
    return (startup_s(nodes) + S * s + N_EVAL * eval_s(g, d) + N_CKPT * ckpt_s(d)
            + len(branches) * (150 + CD_EVALS * eval_s(g, d) + ckpt_s(d))
            + 1.21 * s * frac * sum(branches) * S)


def surface_steps(nodes, d, wall=WALL, **kw):
    fixed = surface_wall(nodes, d, 0, **kw)
    per_step = surface_wall(nodes, d, 1, **kw) - fixed
    return int((wall * margin(d, nodes * 4) - fixed) / per_step)


def wall_estimate(nodes, d, steps):
    g = nodes * 4
    return (startup_s(nodes) + steps * step_s(d, g) + N_EVAL * eval_s(g, d)
            + N_CKPT * ckpt_s(d))


if __name__ == "__main__":
    print(f"{'nodes':>6}{'B':>6}{'steps':>9}{'examples':>12}{'epochs':>8}{'est min':>9}")
    for n in (1, 2, 4, 8, 16, 32, 64, 128):
        st = steps_for(n)
        print(f"{n:6d}{n*4:6d}{st:9,}{st*n*4:12,}{st*n*4/TRAIN_EVENTS:8.1f}"
              f"{wall_estimate(n, 512, st)/60:9.1f}")
