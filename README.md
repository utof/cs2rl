# cs2rl

cs2rl trains Counter-Strike 2 bots with reinforcement learning, in a simulation rather than the
game. A 5v5 simulation written in C (`src/cs2rl/env/c/`) is compiled by zig into a Python
extension module, `binding`. `cs2rl.env.c.cs2_env` wraps it as a PufferLib environment, reading
the C structs through ctypes mirrors. `python -m cs2rl.train` trains a policy on it with
PufferLib's PPO trainer, subclassed as `Cs2PuffeRL`. `python -m cs2rl.deploy.export_policy`
exports a checkpoint to ONNX for the CS2 server plugin in `deploy/CS2RLBot/`; deploy work is
suspended since 2026-05-03.

Linux and WSL. The raylib first-person demo below does not run on WSL.

## From a fresh clone to a green smoke run

Every command runs from the repository root.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv runs all the Python here
git clone <repo> cs2rl && cd cs2rl
uv sync                                  # the venv, plus the C extension, built with zig
git config core.hooksPath .githooks      # ruff / yapf / clang-format / pyrefly on commit
uv run awpy get navs                     # the maps' nav meshes, into ~/.awpy/navs/
uv run awpy get tris                     # the maps' triangles, into ~/.awpy/tris/ (~480 MB)
uv run python -m cs2rl.train --smoke     # 20k env steps on dust2, no training
uv run python -m pytest tests/env/c/test_struct_sizes.py -q -n 0   # one test file
```

`--smoke` builds a dust2 env, steps it 20,000 times with all-zero actions, checks that every
observation and reward is finite, and prints steps/sec. It reads `~/.awpy/navs/de_dust2.json`
(or the file `CS2RL_NAV_PATH` names). The first dust2 load has no cache yet: it writes
`vis_cache*.npy` into `src/cs2rl/`, and the visibility half reads `~/.awpy/tris/de_dust2.tri`
with one worker process per CPU core, at about 900 MB each. Later runs load the cache.

The whole test suite is one session on two workers (several minutes):

```bash
uv run python -m pytest -n 2 --dist loadgroup tests -q
```

Always pass `tests` or files under it: `pyproject.toml` names no test paths. `CONTRIBUTING.md`
says why 2 workers, what the fast `-m "not slow"` loop drops, and how to add an action head.
`tests/CONTEXT.md` says where a new test goes and what the session guards check.

See the [public documentation index](docs/README.md) for architecture, ADRs, format contracts and validation limits.

## Training

`python -m cs2rl.train --help` lists every flag. `--train` runs PPO self-play training,
`--eval` evaluates a checkpoint, `--record` records one episode for rerun, and `--dump-config`
writes the run's `config.json` without training. Outputs go under `outputs/`.

## Where things live

| path | what | read first |
|------|------|------------|
| `src/cs2rl/env/` | the C simulation, its Python env, maps, nav mesh, env config, the env factory | `src/cs2rl/env/CONTEXT.md` |
| `src/cs2rl/train/` | `python -m cs2rl.train`: the CLI, the run loop, `Cs2PuffeRL` | `src/cs2rl/train/CONTEXT.md` |
| `src/cs2rl/experiment/` | gates and analyses that read a finished run's files | `src/cs2rl/experiment/CONTEXT.md` |
| `src/cs2rl/spec/` | the obs and action layouts generated from the C header, and the output paths | |
| `src/cs2rl/eval/`, `src/cs2rl/viz/`, `src/cs2rl/deploy/` | scripted baselines and the metrics registry; rerun and play viewers; ONNX export | |
| `src/cs2rl/policy.py` | the policy network | |
| `src/cs2rl/train_bc.py`, `src/cs2rl/bc_demos.py` | behaviour cloning and its scripted demos | |
| `scripts/` | command-line tools run by path | `scripts/CONTEXT.md` |
| `scripts/modal_runner/` | the optional Modal cloud-training runner | `scripts/modal_runner/CONTEXT.md` |
| `tests/` | the suite, mirroring `src/cs2rl/` | `tests/CONTEXT.md` |
| `deploy/` | the CS2 server plugin (C#) and `deploy/serversetup.md` | |

`pyproject.toml` holds the import-linter contracts that order the packages into layers. A new
module directly under `cs2rl`, `cs2rl.env` or `cs2rl.train` fails the suite until it has a
layer there.

## Git worktrees

A worktree shares the main checkout's `.venv`, whose editable install names one checkout's
`src/`. Put the worktree's own `src/` first, `env PYTHONPATH=<worktree>/src ...`, and never run
`uv run` or `uv sync` there: they re-point the shared install at the worktree. `CONTRIBUTING.md`
has the full recipe.

## When something fails

- `ImportError: cs2rl was imported from ..., but ... is inside another checkout`: the message
  prints the `PYTHONPATH` to use.
- `StaticData layout hash mismatch`: the built extension and `cs2_env.py` describe different
  structs, usually because the C sources changed after the build. `uv sync` does not rebuild it;
  run `uv run --with 'ziglang>=0.14,<0.15' python setup.py build_ext --inplace --force`.
- `.tri file not found`: run `uv run awpy get tris`.
- `awpy get` fails with HTTP 403: on 2026-09-30 awpy 2.0.2's download host refused it. Copy
  `de_dust2.json` into `~/.awpy/navs/` and `de_dust2.tri` into `~/.awpy/tris/` from a machine
  that has them.
- pytest stops with `checkout tripwire (tests/conftest.py)`: each problem line names its fix.

## The raylib demo

The demo is a separate zig step (`uv sync` builds only the Python extension). Install the X11
headers once, then build and run it:

```bash
sudo apt install libx11-dev x11proto-dev libxcursor-dev libxext-dev \
                 libxfixes-dev libxi-dev libxinerama-dev libxrandr-dev \
                 libxrender-dev libglx-dev libgl-dev
uv run python scripts/bake_nav.py     # writes nav_data.h (untracked), which the demo includes
cd src/cs2rl/env/c
uvx --from 'ziglang>=0.14,<0.15' python -m ziglang build cs2_demo
./zig-out/bin/cs2_demo
```

- The `ziglang>=0.14,<0.15` bound matches `pyproject.toml`'s `[build-system]`: Zig 0.15 dropped
  `addSharedLibrary`, which `build.zig` uses. The `ziglang` package puts no `zig` command on
  PATH, so call it as `python -m ziglang`; `uvx` runs it outside the project's `.venv`.
- The link step prints "archive member ... is neither ET_REL nor LLVM bitcode" warnings and may
  label a step "failure" while exiting 0. Check that `zig-out/bin/cs2_demo` exists.
- Without sudo: `apt-get download` the same packages, `dpkg -x` them into a directory `<root>`,
  and build with `--search-prefix <root>/prefix`, where `<root>/prefix/include` links to
  `../usr/include` and `<root>/prefix/lib` to `../usr/lib/x86_64-linux-gnu`. Repoint each dev
  `.so` symlink there at the installed runtime library by absolute path
  (`ln -sfn /usr/lib/x86_64-linux-gnu/libX11.so.6 <root>/usr/lib/x86_64-linux-gnu/libX11.so`):
  a dangling one does not fail the build, zig silently links the static `.a` instead.
- raylib is fetched on the first build (`build.zig.zon`) into the zig package cache.

## Watch the bots on a live CS2 server

With the server installed (`deploy/serversetup.md`), run `bash scripts/launch_server.sh`, then
open CS2 and `connect 127.0.0.1`.
