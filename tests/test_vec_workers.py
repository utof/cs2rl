"""auto_vec_workers — divisible-by-construction worker autoselection.

WHY: pufferlib.vector.make hard-errors unless num_envs % num_workers == 0,
and the old default (min(num_envs, physical_cores)) ignored that, so any
box whose visible core count doesn't divide 256 (6-core VM, 12-core
laptop, ...) crashed at launch and needed a hand-picked --vec-num-workers
every single time (recurring since 2026-08; user asked for the auto-fix).

PITFALL: an EXPLICIT --vec-num-workers still passes through untouched —
if it doesn't divide num_envs, pufferlib's error is the right feedback
for a deliberate choice; only the automatic default must never crash.
"""


def test_auto_vec_workers_picks_largest_divisor_within_cores():
    from cs2rl.train import auto_vec_workers

    assert auto_vec_workers(256, 6) == 4               # the 6-core VM case that kept crashing
    assert auto_vec_workers(256, 8) == 8               # exact divisor cap is kept
    assert auto_vec_workers(256, 12) == 8              # 12 doesn't divide, fall to 8
    assert auto_vec_workers(10, 6) == 5
    assert auto_vec_workers(4, 16) == 4                # never exceed num_envs


def test_auto_vec_workers_degenerate_inputs_never_crash():
    from cs2rl.train import auto_vec_workers

    assert auto_vec_workers(7, 3) == 1                 # prime env count → serial-ish but valid
    assert auto_vec_workers(1, 64) == 1
    assert auto_vec_workers(256, 0) == 1               # psutil returning 0/None is guarded upstream
