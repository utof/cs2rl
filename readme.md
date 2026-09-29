works on linux and wsl(but not the raylib first-person simulation)

# how to install

1. install uv (thing to run python painless)
`curl -LsSf https://astral.sh/uv/install.sh | sh`
2. `uv sync`
3. wire up the shared git hooks (ruff / yapf / clang-format on commit):
   `git config core.hooksPath .githooks`


# to run raylib demo:

`uv sync` already builds the Python C-extension (`binding.so`) via `zig build`.
The raylib demo is a separate target — build it explicitly.

Linux prerequisites (install once):

```
sudo apt install libx11-dev libxcursor-dev libxext-dev libxfixes-dev \
                 libxi-dev libxinerama-dev libxrandr-dev libxrender-dev \
                 libgl-dev
```

Then:

```
uv run python scripts/bake_nav.py
cd src/cs2rl/env/c
uv run --with 'ziglang>=0.14,<0.15' python -m ziglang build cs2_demo
./zig-out/bin/cs2_demo
```

Notes:
- `--with 'ziglang>=0.14,<0.15'` matches the pin in `pyproject.toml`'s
  `[build-system]`. Without the bound, uv pulls Zig 0.15+ which dropped
  `addSharedLibrary` and breaks `build.zig`.
- Raylib is fetched automatically on first build from `build.zig.zon`;
  subsequent runs use the Zig package cache (`~/.cache/zig`).
- If you have system Zig 0.14 installed, `zig build cs2_demo` from
  `src/cs2rl/env/c/` also works.

# quick launch: watch RL bots play on a live CS2 server

Prerequisites: server already installed (see `deploy/serversetup.md`).

```bash
bash scripts/launch_server.sh
```

Then open CS2 and `connect 127.0.0.1`.