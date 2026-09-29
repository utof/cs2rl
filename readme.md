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
sudo apt install libx11-dev x11proto-dev libxcursor-dev libxext-dev \
                 libxfixes-dev libxi-dev libxinerama-dev libxrandr-dev \
                 libxrender-dev libglx-dev libgl-dev
```

Then:

```
uv run python scripts/bake_nav.py
cd src/cs2rl/env/c
uvx --from 'ziglang>=0.14,<0.15' python -m ziglang build cs2_demo
./zig-out/bin/cs2_demo
```

Notes:
- `--from 'ziglang>=0.14,<0.15'` matches the pin in `pyproject.toml`'s
  `[build-system]`. Without the bound, uv pulls Zig 0.15+ which dropped
  `addSharedLibrary` and breaks `build.zig`.
- The `ziglang` package puts no `zig` command on PATH (it has no entry
  point; the binary sits inside the package): call it as `python -m ziglang`. `uvx` runs it in its own throwaway environment, so the
  project's `.venv` is not touched.
- The link step prints "archive member ... is neither ET_REL nor LLVM bitcode"
  warnings and may label a step "failure" while still exiting 0. They are
  harmless; check that `zig-out/bin/cs2_demo` exists.
- Without sudo: `apt-get download` the same packages and unpack them with
  `dpkg -x` into a user-local directory `<root>`. The files land in
  `<root>/usr/include` and `<root>/usr/lib/x86_64-linux-gnu`, but zig's
  `--search-prefix <prefix>` only adds `<prefix>/include` and `<prefix>/lib`,
  so make a symlink view: `<root>/prefix/include -> ../usr/include` and
  `<root>/prefix/lib -> ../usr/lib/x86_64-linux-gnu`. Then build with
  `--search-prefix <root>/prefix`.
- In that unpack, the dev `.so` symlinks point at runtime libraries the dev
  packages do not ship; repoint each at the installed library in
  `/usr/lib/x86_64-linux-gnu` (e.g. `libX11.so -> libX11.so.6`). A dangling
  one does not fail the build: zig silently falls back to the static `.a` in
  the same directory.
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