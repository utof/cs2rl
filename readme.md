works on linux and wsl(but not the raylib first-person simulation)

# how to install

1. install uv (thing to run python painless)
`curl -LsSf https://astral.sh/uv/install.sh | sh`
2. `uv sync`


# to run raylib demo:
```
uv run python scripts/bake_nav.py
mkdir -p build_demo && cd build_demo
cmake -DBUILD_CS2_DEMO=ON -DPython_EXECUTABLE=$(uv run python -c "import sys; print(sys.executable)") ../src/c_env
cmake --build . --target cs2_demo
./cs2_demo
```

# quick launch: watch RL bots play on a live CS2 server

Prerequisites: server already installed (see `deploy/serversetup.md`).

```bash
bash scripts/launch_server.sh
```

Then open CS2 and `connect 127.0.0.1`.