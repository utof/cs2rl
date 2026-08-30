#!/usr/bin/env bash
# Rung 1 launcher (spec 2026-08-29 §4): 5 treatment seeds + 2 negative-control
# seeds, each retried with --resume-run on a non-zero exit (the training box's
# GPU falls off the bus mid-run — R0-C full-state resume exists for this).
# Results are NOT promoted through outputs/experiments/.
#
# Usage:  scripts/run_rung1.sh [OUT_ROOT] [MAX_RETRIES]
#   OUT_ROOT     per-seed dirs <OUT_ROOT>/rung1-s<k>, <OUT_ROOT>/rung1-neg-s<k>
#                (default outputs/checkpoints/rung1, relative to the CWD)
#   MAX_RETRIES  --resume-run attempts per seed after the fresh run (default 5)
# Env overrides (all optional):
#   RUNG1_EXTRA      extra train.py tokens appended LAST — argparse keeps the last
#                    occurrence, so "--timesteps 10240 --num_envs 16 --device cpu
#                    --vec-backend serial" turns this into a CPU smoke run.
#                    Unquoted on purpose: word-split into tokens (no spaces in values).
#   RUNG1_SEEDS      treatment seeds (default "0 1 2 3 4"); "" = none
#   RUNG1_NEG_SEEDS  negative-control seeds (default "0 1"); "" = none
#   RUNG1_TRAIN_CMD  command prefix (default "env UV_NO_SYNC=1 uv run python <repo>/src/train.py");
#                    tests point it at a fake trainer.
#
# Per-seed state machine (all decisions are on files, so re-running the script
# after a crash of the script itself is safe):
#   <dir>/DONE                          -> finished: skip.  Written by THIS script
#                                          after train.py exits 0 and holds the
#                                          PARTICIPATING --timesteps budget that run
#                                          used. If the budget now requested (last
#                                          --timesteps in the argv, i.e. RUNG1_EXTRA
#                                          wins) is LARGER, the seed is EXTENDED via
#                                          --resume-run instead of skipped (budget keys
#                                          are allowlisted on resume). An empty/legacy
#                                          DONE skips unconditionally — delete it to
#                                          extend. NOT dust2_policy.pt: that file is
#                                          rewritten every --save_every_sec, so a seed
#                                          that died after its first periodic save
#                                          would look "finished" and never resume.
#   <dir>/<label>/trainer_state.pt      -> a full-state checkpoint set exists
#                                          (what resolve_resume_run keys on): resume.
#   neither                             -> fresh start; if that dies before its first
#                                          checkpoint set there is nothing to resume.
# The resume leg re-sends the ORIGINAL argv + --resume-run <dir>: `env` (--map),
# `seed`, `pin_pitch`, gamma keys are NOT in RESUME_CONFIG_ALLOWLIST, so a bare
# `--resume-run` would be refused with "config.json mismatch". Budget flags are
# re-sent with identical values (allowlisted — harmless).
# One dead seed must not abort the sweep: failures are collected and reported
# at the end (exit 1).
# The retry counter is PER INVOCATION: a seed that exhausted MAX_RETRIES leaves
# no DONE, so re-running the script grants it MAX_RETRIES more resumes.
# Every train.py leg's stdout+stderr is appended to <dir>/train.log (tee'd, so
# it still streams to the console) — the crash traceback that justified a
# retry survives a 7-seed sweep.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_ROOT=$(realpath -m "${1:-outputs/checkpoints/rung1}")   # absolute BEFORE the cd below
MAX_RETRIES=${2:-5}
cd "$REPO"                                                  # uv run needs the project root

# Default command built as an explicit array (quotes inside an unquoted
# ${VAR:-default} expansion do NOT survive word-splitting). The env override is
# word-split on whitespace on purpose (see header) — no spaces in paths.
if [[ -n "${RUNG1_TRAIN_CMD:-}" ]]; then
  # shellcheck disable=SC2206
  TRAIN=($RUNG1_TRAIN_CMD)
else
  TRAIN=(env UV_NO_SYNC=1 uv run python "$REPO/src/train.py")
fi
SEEDS=${RUNG1_SEEDS-0 1 2 3 4}
NEG_SEEDS=${RUNG1_NEG_SEEDS-0 1}

# Spec §4 argv verbatim (self-play stays ENABLED — §5's window excludes
# self_play/used_past rows; --timesteps is a PARTICIPATING budget, §2.2;
# --no-dead-run-abort is mandatory: every kill-less round is a timeout).
# shellcheck disable=SC2206
COMMON=(--train --map arena-duel --n-active-per-team 1 --round-time-ticks 160
        --crouch-enabled 0 --gamma 0.99 --timesteps 10000000 --num_envs 256
        --checkpoint-interval 10 --eval-interval 10 --no-dead-run-abort
        --reward-win-t-elimination 1.0 --reward-win-ct-elimination 1.0
        --reward-win-ct-timeout 0 --reward-win-t-detonation 0 --reward-win-ct-defuse 0
        --reward-kill 0.3 --reward-death 0.1 --reward-shot-penalty 0 --reward-ct-survival 0
        --reward-inaction 0.0005 --pbrs-hp-weight 0.002 --pbrs-alive-weight 0.3
        --pbrs-site-weight 0 --pbrs-bomb-progress-weight 0 --pbrs-nav-weight-t 0 --pbrs-nav-weight-ct 0)

last_flag_value() {   # $1 = flag, rest = argv; prints the LAST value (argparse semantics)
  # Accepts BOTH argparse spellings: "--flag value" and "--flag=value". The
  # header only forbids spaces in RUNG1_EXTRA values, so "--timesteps=10240" is
  # legal input; missing the `=` form here would record COMMON's 10000000 in
  # DONE after a smoke run and make a later full-budget run skip that seed.
  local flag=$1 v=""; shift
  while (( $# > 0 )); do
    if [[ $1 == "$flag" && $# -gt 1 ]]; then v=$2
    elif [[ $1 == "$flag="* ]]; then v=${1#*=}; fi
    shift
  done
  printf '%s' "$v"
}

run_train() {   # $1 = log file, rest = train.py argv. Tees output; returns train.py's status.
  local log=$1; shift
  # PITFALL: under `pipefail` the pipeline's status is tee's unless we read
  # PIPESTATUS[0] right after it. Safe under `set -e` only because every caller
  # is an `if` condition (errexit is suspended inside such calls).
  "${TRAIN[@]}" "$@" 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

run_seed() {   # $1 = label, $2 = seed, rest = arm-specific flags
  local label=$1 seed=$2; shift 2
  local dir="$OUT_ROOT/$label"
  local log="$dir/train.log"
  local attempt=0
  mkdir -p "$dir"
  # Original argv, assembled ONCE and reused verbatim on every resume leg.
  # RUNG1_EXTRA goes last so its repeated options win in argparse.
  # shellcheck disable=SC2206
  local argv=("${COMMON[@]}" "$@" --seed "$seed" --checkpoint-dir "$dir" --run-id "$label"
              ${RUNG1_EXTRA:-})
  local budget; budget=$(last_flag_value --timesteps "${argv[@]}")
  if [[ -f "$dir/DONE" ]]; then
    local done_budget; done_budget=$(tr -d '[:space:]' < "$dir/DONE")
    if [[ -z "$done_budget" ]]; then
      echo "[run_rung1] $label already finished (DONE has no budget recorded; delete it to extend)"
      return 0
    fi
    if (( done_budget >= budget )); then
      echo "[run_rung1] $label already finished at --timesteps $done_budget (requested $budget)"
      return 0
    fi
    echo "[run_rung1] $label: extending --timesteps $done_budget -> $budget"
    if [[ ! -f "$dir/$label/trainer_state.pt" ]]; then
      echo "[run_rung1] $label has DONE but no checkpoint set — cannot extend" >&2; return 1
    fi
  fi
  if [[ ! -f "$dir/$label/trainer_state.pt" ]]; then
    echo "[run_rung1] $label: fresh start"
    if run_train "$log" "${argv[@]}"; then echo "$budget" > "$dir/DONE"; return 0; fi
    if [[ ! -f "$dir/$label/trainer_state.pt" ]]; then
      echo "[run_rung1] $label died before its first checkpoint set — not resumable" >&2; return 1
    fi
  fi
  while (( attempt < MAX_RETRIES )); do
    attempt=$((attempt + 1))
    echo "[run_rung1] $label: resume attempt $attempt"
    if run_train "$log" "${argv[@]}" --resume-run "$dir"; then echo "$budget" > "$dir/DONE"; return 0; fi
  done
  echo "[run_rung1] $label FAILED after $MAX_RETRIES resumes" >&2; return 1
}

failed=()
for s in $SEEDS; do
  run_seed "rung1-s$s" "$s" --aim-entropy-bonus off --aim-log-std-max -2.9957 \
    || failed+=("rung1-s$s")                    # one dead seed must not abort the sweep (set -e)
done
for s in $NEG_SEEDS; do   # negative control (spec §4): bonus on, sigma cap log 0.5
  run_seed "rung1-neg-s$s" "$s" --aim-entropy-bonus on --aim-log-std-max -0.6931 \
    || failed+=("rung1-neg-s$s")
done
if (( ${#failed[@]} )); then echo "[run_rung1] FAILED: ${failed[*]}" >&2; exit 1; fi
echo "[run_rung1] done -> $OUT_ROOT"
