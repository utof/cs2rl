# CS2RLBot Server Setup Guide

This guide walks you through installing a CS2 dedicated server, deploying the CS2RLBot plugin with your trained model, and verifying everything works. Written for Linux (tested on Linux Mint 22 / Ubuntu 24.04).

**Prerequisites:** you have already trained a model (you have a `.pt` checkpoint file), run `uv sync`, and know how to run the environment.

---

## Overview of what we're installing

```
SteamCMD  →  CS2 dedicated server  →  Metamod:Source  →  CounterStrikeSharp  →  CS2RLBot plugin
```

Each layer sits on the one below it:
- **SteamCMD** is Valve's tool for downloading game servers.
- **CS2 dedicated server** is the actual game server (~60 GB).
- **Metamod:Source** is a low-level plugin loader that hooks into the Source engine. CounterStrikeSharp needs it.
- **CounterStrikeSharp** exposes a C# API on top of Metamod, so plugins can react to game events and control bots.
- **CS2RLBot** is our plugin — it runs ONNX inference each tick and feeds actions into bots.

---

## Step 0 — Export your trained model to ONNX

The plugin doesn't load PyTorch checkpoints directly — it uses ONNX, which is a cross-platform model format that runs without a Python environment. You need to export your `.pt` checkpoint to `.onnx` before setting up the server.

```bash
# From the repo root
uv run python deploy/export_policy.py --checkpoint /path/to/your/model_XXXXXX.pt
```

This writes two files to `deploy/models/`:
- `policy_lstm.onnx` — the model weights in ONNX format
- `policy_lstm.onnx.data` — external data file for large models (may or may not exist)
- `policy_lstm.json` — a sidecar that records `obs_dim`, `hidden_dim`, and `action_sizes`

The plugin reads the sidecar JSON to know the model's shape. If the sidecar is missing or has wrong dims, the plugin will fail to load.

**Verify the export is numerically correct** (optional but recommended — checks that ONNX and PyTorch produce the same outputs):

```bash
uv run python deploy/verify_onnx.py
```

All 100 steps should print `PASS`. If any fail, re-export.

---

## Step 1 — Pre-flight checks

Before downloading 60 GB, make sure you have space and the model files are ready.

```bash
# Need at least 70 GB free on /home
df -BG /home | awk 'NR==2 {gsub("G",""); if ($4 < 70) print "FAIL: only " $4 "G free, need 70G"; else print "OK: " $4 "G free"}'

# .NET SDK is required to build the plugin locally (build output is gitignored)
[ -x "$HOME/.dotnet/dotnet" ] && "$HOME/.dotnet/dotnet" --version && echo "dotnet OK" || \
  echo "MISSING: ~/.dotnet/dotnet (install .NET 8 SDK or adjust the build command in Step 6)"

# Model files must exist
ls deploy/models/policy_lstm.onnx deploy/models/policy_lstm.json && echo "models OK" || echo "MISSING model files"

# Required tools
for t in curl tar unzip; do command -v $t &>/dev/null && echo "$t OK" || echo "MISSING: $t"; done
```

If any tool is missing: `sudo apt-get install -y curl tar unzip`

---

## .NET 8 SDK

Linux Mint 22 is based on Ubuntu 24.04 (Noble). The Ubuntu main repo ships
dotnet-sdk-8.0 but Mint's package overrides sometimes shadow it — the
`ppa:dotnet/backports` PPA is more reliable.

```bash
sudo add-apt-repository ppa:dotnet/backports -y
sudo apt-get update
sudo apt-get install -y dotnet-sdk-8.0

# Verify
dotnet --version
# Should print 8.0.xxx
```

**Fallback** (if the PPA fails or isn't available):
```bash
curl -sSL https://dot.net/v1/dotnet-install.sh -o dotnet-install.sh
chmod +x dotnet-install.sh
./dotnet-install.sh --channel 8.0
echo 'export DOTNET_ROOT=$HOME/.dotnet' >> ~/.bashrc
echo 'export PATH=$PATH:$DOTNET_ROOT:$DOTNET_ROOT/tools' >> ~/.bashrc
source ~/.bashrc
dotnet --version
```

**Docs**: https://learn.microsoft.com/en-us/dotnet/core/install/linux-ubuntu

## Step 2 — SteamCMD

SteamCMD is a command-line tool that downloads and updates game servers. We install it manually under `~/steamcmd/` to avoid conflicts with any system package.

```bash
# CS2 server has 32-bit dependencies — add i386 architecture support
sudo dpkg --add-architecture i386
sudo apt-get update
sudo apt-get install -y lib32gcc-s1 lib32stdc++6 curl tar unzip

# Download and extract SteamCMD
mkdir -p ~/steamcmd && cd ~/steamcmd
curl -sqL "https://steamcdn-a.akamaihd.net/client/installer/steamcmd_linux.tar.gz" | tar xzf -

# Test it (this will self-update on first run — normal)
~/steamcmd/steamcmd.sh +quit && echo "SteamCMD OK"
```

Create a symlink to suppress a common runtime warning:

```bash
mkdir -p ~/.steam/sdk64
ln -sf ~/steamcmd/linux64/steamclient.so ~/.steam/sdk64/steamclient.so
```

---

## Step 3 — CS2 dedicated server (~60 GB, 20–40 minutes)

This is the long step. Just run it and wait. SteamCMD will print progress percentages.

```bash
mkdir -p $HOME/cs2rl-server
~/steamcmd/steamcmd.sh \
  +force_install_dir "$HOME/cs2rl-server" \
  +login anonymous \
  +app_update 730 validate \
  +quit
```

When it finishes, verify the key files exist:

```bash
ls $HOME/cs2rl-server/game/csgo/gameinfo.gi && echo "gameinfo.gi OK" || echo "FAIL"
ls $HOME/cs2rl-server/game/cs2.sh           && echo "cs2.sh OK"      || echo "FAIL"
```

> **Why `cs2.sh` and not the binary directly?** Valve's wrapper script sets up `LD_LIBRARY_PATH` and SteamRT environment variables. Launching the binary directly breaks on recent CS2 versions.

---

## Step 4 — Metamod:Source

Metamod is the foundation that CounterStrikeSharp sits on. You need the **2.x dev** branch — the 1.x stable release only works with older Source games, not CS2.

### 4a. Download

The "latest" permalink URL on AlliedModders CDN sometimes breaks. The reliable method is to scrape the CDN listing for the newest build:

```bash
cd /tmp

# Find the latest build number from the CDN directory listing
LATEST=$(curl -sL "https://mms.alliedmods.net/mmsdrop/2.0/" \
  | grep -o 'mmsource-2\.0\.0-git[0-9]*-linux\.tar\.gz' \
  | sort -t'-' -k4 -V | tail -1)

echo "Downloading: $LATEST"
curl -sSLO "https://mms.alliedmods.net/mmsdrop/2.0/$LATEST"

# Verify it's actually a gzip (not a 404 HTML page)
file "$LATEST" | grep -q gzip && echo "Download OK" || { echo "FAIL: got HTML instead of gzip — check URL manually"; exit 1; }
```

If the CDN is completely down, go to https://www.sourcemm.net/downloads.php?branch=master&all=1, download the latest Linux `.tar.gz` manually, and save it to `/tmp/`.

### 4b. Extract

```bash
tar xzf /tmp/mmsource-2.0.0-git*-linux.tar.gz -C $HOME/cs2rl-server/game/csgo/
echo "Metamod extracted OK"
```

### 4c. Patch gameinfo.gi

CS2 needs to be told to load Metamod. This is done by adding one line to `gameinfo.gi`. This file lives at `~/cs2rl-server/game/csgo/gameinfo.gi`.

```bash
GAMEINFO="$HOME/cs2rl-server/game/csgo/gameinfo.gi"

# Back up the original
cp "$GAMEINFO" "$GAMEINFO.bak"

# Add the Metamod line (only if not already there)
if grep -q 'csgo/addons/metamod' "$GAMEINFO"; then
  echo "Already patched"
else
  sed -i '/Game_LowViolence/a\            Game    csgo/addons/metamod' "$GAMEINFO"
  echo "Patched"
fi

# Verify
grep -q 'csgo/addons/metamod' "$GAMEINFO" && echo "Patch verified" || { echo "FAIL: patch did not apply"; exit 1; }
```

> **WARNING: CS2 updates overwrite `gameinfo.gi`** every time Valve pushes a patch. After any CS2 update, re-run this patch (or run `~/cs2rl-server/update.sh` — created in Step 8).

---

## Step 5 — CounterStrikeSharp

CounterStrikeSharp (CSS) is the C# plugin framework. Download the **with-runtime** Linux build — this bundles the .NET runtime so the server doesn't need .NET installed separately.

### 5a. Find the latest version

GitHub's API rate-limits unauthenticated requests. Use the redirect trick instead:

```bash
CSS_TAG=$(curl -sI "https://github.com/roflmuffin/CounterStrikeSharp/releases/latest" \
  | grep -i location | grep -o 'v[0-9.]*' | head -1)
echo "Latest CSS: $CSS_TAG"
```

### 5b. Find the correct filename

> **Gotcha:** the asset filename changed between versions — it used to include `build` in the name, now it doesn't. Always check the actual asset list rather than hardcoding the name.

```bash
CSS_ZIP=$(curl -sL "https://github.com/roflmuffin/CounterStrikeSharp/releases/expanded_assets/${CSS_TAG}" \
  | grep -o 'href="[^"]*counterstrikesharp-with-runtime-linux[^"]*\.zip"' \
  | grep -o 'counterstrikesharp[^"]*\.zip' | head -1)
echo "Asset: $CSS_ZIP"
```

### 5c. Download and extract

```bash
cd /tmp
curl -sSLO "https://github.com/roflmuffin/CounterStrikeSharp/releases/download/${CSS_TAG}/${CSS_ZIP}"

# Verify it's actually a zip
file "$CSS_ZIP" | grep -q Zip && echo "Download OK" || { echo "FAIL: got unexpected file type"; exit 1; }

cd $HOME/cs2rl-server/game/csgo
unzip -o /tmp/${CSS_ZIP}
echo "CounterStrikeSharp extracted OK"
```

### 5d. Verify

```bash
ls $HOME/cs2rl-server/game/csgo/addons/counterstrikesharp/bin/ && echo "CSS OK" || echo "FAIL"
```

---

## Step 6 — Build and deploy CS2RLBot plugin

Build the plugin from source, then copy the build output and model files into the server's plugin directory.

```bash
REPO="$(pwd)"  # run from repo root
PLUGIN_DIR="$HOME/cs2rl-server/game/csgo/addons/counterstrikesharp/plugins/CS2RLBot"

mkdir -p "$PLUGIN_DIR/models" "$PLUGIN_DIR/logs"

# Build a fresh Release copy of the plugin
"$HOME/.dotnet/dotnet" publish "$REPO/deploy/CS2RLBot/CS2RLBot.csproj" \
  -c Release -o "$REPO/deploy/CS2RLBot_build" -v quiet
echo "Plugin built"

# Copy all DLLs and config from the build output
cp -r "$REPO/deploy/CS2RLBot_build/"* "$PLUGIN_DIR/"
echo "Plugin files copied"

# Copy model files
cp "$REPO/deploy/models/policy_lstm.onnx"  "$PLUGIN_DIR/models/"
cp "$REPO/deploy/models/policy_lstm.json"  "$PLUGIN_DIR/models/"

# Copy external data file if it exists (large models split weights into a .data file)
[ -f "$REPO/deploy/models/policy_lstm.onnx.data" ] && \
  cp "$REPO/deploy/models/policy_lstm.onnx.data" "$PLUGIN_DIR/models/" && \
  echo "Copied .onnx.data"

echo "Model files copied"
```

Verify:

```bash
ls "$PLUGIN_DIR/CS2RLBot.dll" && echo "Plugin DLL OK" || { echo "FAIL: CS2RLBot.dll missing"; exit 1; }
ls "$PLUGIN_DIR/models/policy_lstm.onnx" && echo "ONNX OK" || { echo "FAIL: model missing"; exit 1; }
```

---

## Step 7 — Create server config files

These config files make the dedicated server auto-spawn one bot, keep the server awake without a human player, and shorten rounds so round-transition verification happens quickly.

```bash
mkdir -p "$HOME/cs2rl-server/game/csgo/cfg"

cat > "$HOME/cs2rl-server/game/csgo/cfg/server.cfg" << 'EOF'
sv_cheats 1
sv_hibernate_when_empty 0

// Skip warmup as much as possible before the plugin takes over.
mp_warmuptime 0
mp_warmup_pausetimer 0
mp_warmup_end

// Auto-spawn one bot even with no human player connected.
bot_quota_mode fill
bot_quota 1
bot_join_after_player 0
bot_add_t

// Short rounds for fast verification.
mp_roundtime 0.5
mp_roundtime_defuse 0.5
mp_round_restart_delay 5
mp_freezetime 3
EOF

cat > "$HOME/cs2rl-server/game/csgo/cfg/gamemode_casual_server.cfg" << 'EOF'
// Overrides that apply AFTER gamemode_casual.cfg loads.
sv_cheats 1
sv_hibernate_when_empty 0
mp_warmuptime 0
mp_warmup_pausetimer 0
mp_warmup_end
mp_roundtime 0.5
mp_roundtime_defuse 0.5
mp_round_restart_delay 3
mp_freezetime 2
bot_quota_mode fill
bot_quota 1
bot_join_after_player 0
bot_add_t
EOF
```

> **Important:** these configs are necessary, but they are not sufficient on their own to force a live round on a cold boot. The current plugin build also sends `mp_warmup_end` once after the first bot registers. If you remove that plugin behavior, cold starts can get stuck in warmup again.

---

## Step 8 — Create launch and update scripts

### Launch script

```bash
cat > $HOME/cs2rl-server/start.sh << 'EOF'
#!/bin/bash
# CS2RLBot development server — LAN only, casual mode for bot testing
cd "$(dirname "$0")"

GSLT=""  # optional: paste your Game Server Login Token here for public listing

cd game && ./cs2.sh \
  -dedicated \
  -console \
  -usercon \
  +game_type 0 \
  +game_mode 0 \
  +map de_dust2 \
  +sv_lan 1 \
  +sv_hibernate_when_empty 0 \
  ${GSLT:++sv_setsteamaccount "$GSLT"}
EOF
chmod +x $HOME/cs2rl-server/start.sh
```

### Update script (run after every Valve CS2 patch)

```bash
cat > $HOME/cs2rl-server/update.sh << 'EOF'
#!/usr/bin/env bash
set -e
echo "=== Updating CS2 server ==="
~/steamcmd/steamcmd.sh \
  +force_install_dir "$HOME/cs2rl-server" \
  +login anonymous \
  +app_update 730 validate \
  +quit

echo "=== Re-patching gameinfo.gi ==="
GAMEINFO="$HOME/cs2rl-server/game/csgo/gameinfo.gi"
if grep -q 'csgo/addons/metamod' "$GAMEINFO"; then
  echo "Already patched."
else
  sed -i '/Game_LowViolence/a\            Game    csgo/addons/metamod' "$GAMEINFO"
  echo "Patched."
fi
echo "=== Done ==="
EOF
chmod +x $HOME/cs2rl-server/update.sh
```

---

## Step 9 — Start the server and verify

### Start

```bash
~/cs2rl-server/start.sh
```

This runs in the foreground. Wait for these lines in the output:

```
SV:  64 player server started
[CS2RLBot] Plugin loaded
```

That means the server and plugin finished booting. Cold starts usually take 30–60 seconds.

### Verify Metamod loaded

In the server console, type:

```
meta version
```

Expected output: `Metamod:Source version X.X.X-devXXX`

If you get `Unknown command` — Metamod didn't load. The most common cause is the `gameinfo.gi` patch not being applied. Check: `grep metamod ~/cs2rl-server/game/csgo/gameinfo.gi`

### Verify CounterStrikeSharp loaded

```
meta list
```

Expected: CounterStrikeSharp listed as a Metamod plugin.

### Verify CS2RLBot loaded

```
css_plugins list
```

Expected: `CS2RLBot` listed.

Also look in the startup output for:

```
[CS2RLBot] LatencyTracker self-test passed
[CS2RLBot] Loaded config — obs_dim=72 action_sizes=[9,2,2,2] model=/.../policy_lstm.onnx
[CS2RLBot] Plugin loaded
[CS2RLBot] Bot registered: <name> team=3
[CS2RLBot] Requested mp_warmup_end after first bot registration
[CS2RLBot] RoundStart — LSTM reset for 1 bot(s)
```

If `Plugin loaded` doesn't appear, check the plugin log:

```bash
tail -50 ~/cs2rl-server/game/csgo/addons/counterstrikesharp/plugins/CS2RLBot/logs/cs2rlbot-*.log
```

> **Cold-start gotcha:** you may see an early `RoundStart` or `RoundEnd` for `0 bot(s)` before the first bot is registered. Ignore those. The real success signal is the later `Requested mp_warmup_end...` and `RoundStart/RoundEnd` entries for `1 bot(s)`.

### Check the auto-added bot and inference latency

The config from Step 7 auto-adds one bot. In a separate terminal, watch the plugin log:

```bash
tail -f ~/cs2rl-server/game/csgo/addons/counterstrikesharp/plugins/CS2RLBot/logs/cs2rlbot-*.log
```

Look for lines like:

```
[CS2RLBot] Stats bot=SomeName p50=120µs p99=340µs
[CS2RLBot] RoundEnd — LSTM reset for 1 bot(s)
[CS2RLBot] RoundStart — LSTM reset for 1 bot(s)
```

Target: `p50 < 500µs`. If p50 is over a few milliseconds, the ONNX model may be too large or CPU is overloaded.

If you want to add more bots manually, use:

```
bot_add_t
```

---

## Step 10 — Joining the server from CS2 client

Since the server runs with `+sv_lan 1` it's local-only — no internet or Steam account needed.

1. Launch CS2 on the same machine.
2. Enable the developer console: **Settings → Game → Enable Developer Console → Yes**
3. Press `` ` `` (tilde key, top-left) to open the console.
4. Type:

```
connect 127.0.0.1
```

Default port is 27015. If that doesn't work: `connect 127.0.0.1:27015`

You should connect and see the map loaded. One bot should already be present from the config in Step 7. Use `bot_add_t` or `bot_add_ct` only if you want extra bots.

---

## Troubleshooting

### "meta version" says unknown command

`gameinfo.gi` patch wasn't applied or was overwritten by a CS2 update.

```bash
grep metamod ~/cs2rl-server/game/csgo/gameinfo.gi
# If nothing printed, re-apply the patch from Step 4c
```

### CS2RLBot not in css_plugins list

Check the CSS log for load errors:

```bash
ls ~/cs2rl-server/game/csgo/addons/counterstrikesharp/logs/
cat ~/cs2rl-server/game/csgo/addons/counterstrikesharp/logs/*.log | tail -50
```

Common causes:
- `CS2RLBot.dll` is missing from the plugins folder — re-run Step 6
- `policy_lstm.onnx` or `policy_lstm.json` is missing from `plugins/CS2RLBot/models/` — re-run Step 6
- `policy_lstm.onnx.data` is missing — large models split weights into this sidecar file; make sure it was copied too

### Server boots but never reaches live rounds

Check the plugin log:

```bash
tail -f ~/cs2rl-server/game/csgo/addons/counterstrikesharp/plugins/CS2RLBot/logs/cs2rlbot-$(date +%Y%m%d).log
```

On a healthy cold boot you should see:

```text
[CS2RLBot] Bot registered: ...
[CS2RLBot] Requested mp_warmup_end after first bot registration
[CS2RLBot] RoundStart — LSTM reset for 1 bot(s)
```

If `Bot registered` appears but `Requested mp_warmup_end...` does not, you are running an old plugin build. Re-run Step 6 and restart the server.

### "obs_dim mismatch" or plugin crashes on load

The model in `deploy/models/` was exported from a different checkpoint than what the plugin expects. Re-export with `deploy/export_policy.py` and re-run Step 6.

### SteamCMD download URL for Metamod returns HTML (404)

The `mmsource-2.0-latest-linux.tar.gz` permalink is broken. Use the CDN scrape approach from Step 4a, or download manually from:

https://www.sourcemm.net/downloads.php?branch=master&all=1

Save the `.tar.gz` to `/tmp/` and extract manually:

```bash
tar xzf /tmp/mmsource-2.0.0-git*-linux.tar.gz -C $HOME/cs2rl-server/game/csgo/
```

### GitHub API rate limit when fetching CSS version

The unauthenticated GitHub API allows ~60 requests/hour per IP. Use the redirect trick instead:

```bash
curl -sI "https://github.com/roflmuffin/CounterStrikeSharp/releases/latest" | grep -i location
```

This returns the tag without hitting the API.

### Server takes forever to start / hangs

Normal. The first start loads shaders and precaches models — can take 2–3 minutes. Wait for `SV: 64 player server started` plus the plugin startup lines.

### CS2 update broke everything

Run the update script — it re-downloads the server and re-patches `gameinfo.gi`:

```bash
~/cs2rl-server/update.sh
```

Then re-deploy the plugin (Step 6) in case the update wiped `addons/`.

---

## Quick reference: paths

| Thing | Path |
|---|---|
| SteamCMD | `~/steamcmd/` |
| CS2 server root | `~/cs2rl-server/` |
| Launch script | `~/cs2rl-server/start.sh` |
| Update script | `~/cs2rl-server/update.sh` |
| server.cfg | `~/cs2rl-server/game/csgo/cfg/server.cfg` |
| gamemode_casual_server.cfg | `~/cs2rl-server/game/csgo/cfg/gamemode_casual_server.cfg` |
| gameinfo.gi | `~/cs2rl-server/game/csgo/gameinfo.gi` |
| Metamod | `~/cs2rl-server/game/csgo/addons/metamod/` |
| CSS plugins dir | `~/cs2rl-server/game/csgo/addons/counterstrikesharp/plugins/` |
| CS2RLBot plugin | `~/cs2rl-server/.../plugins/CS2RLBot/` |
| CS2RLBot models | `~/cs2rl-server/.../plugins/CS2RLBot/models/` |
| CS2RLBot logs | `~/cs2rl-server/.../plugins/CS2RLBot/logs/` |
| Plugin source | `deploy/CS2RLBot/` (this repo) |
| Built plugin | `deploy/CS2RLBot_build/` (this repo) |
| Export script | `deploy/export_policy.py` |
| Verify script | `deploy/verify_onnx.py` |

## Quick reference: server console commands

| Command | What it does |
|---|---|
| `meta version` | Confirm Metamod is loaded |
| `meta list` | List Metamod plugins (should include CSS) |
| `css_plugins list` | List CounterStrikeSharp plugins |
| `css_plugins reload CS2RLBot` | Hot-reload plugin without restarting server |
| `bot_add_t` | Add an extra T-side bot |
| `bot_add_ct` | Add a CT-side bot |
| `bot_kick` | Kick all bots |
| `mp_warmup_end` | Manually force warmup to end (debug fallback only; plugin should do this automatically on cold boot) |
| `mp_restartgame 1` | Restart the round in 1 second |
| `status` | Show connected players and server info |
| `quit` | Stop the server |
