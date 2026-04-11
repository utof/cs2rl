#!/usr/bin/env bash
# Starts a local CS2 server with 2 RL bots per side and tails the plugin log.
# Run from anywhere. Requires ~/cs2rl-server to be set up (see deploy/serversetup.md).

set -e
SERVER_DIR=~/cs2rl-server
PLUGIN_LOG_DIR="$SERVER_DIR/game/csgo/addons/counterstrikesharp/plugins/CS2RLBot/logs"

# Kill any running server (kill the cs2 binary, not just the cs2.sh wrapper)
pkill -9 -f "cs2 -dedicated" 2>/dev/null && sleep 3 || true

# Write a 2v2 config (overrides bot_quota in server.cfg)
cat > "$SERVER_DIR/game/csgo/cfg/cs2rl_match.cfg" << 'EOF'
sv_cheats 1
sv_hibernate_when_empty 0
mp_limitteams 0
mp_autoteambalance 0
mp_autokick 0
mp_warmuptime 0
mp_warmup_pausetimer 0
mp_warmup_end
bot_quota_mode fill
bot_quota 4
bot_join_after_player 0
bot_add_t
bot_add_t
bot_add_ct
bot_add_ct
bot_stop 1
rcon_password cs2rl
mp_roundtime 1.92
mp_roundtime_defuse 1.92
mp_round_restart_delay 5
mp_freezetime 3
EOF

# Start server in background; +exec loads our 2v2 config after gamemode cfgs
nohup bash -c "cd \"$SERVER_DIR/game\" && ./cs2.sh \
  -dedicated -console -usercon \
  +ip 0.0.0.0 \
  +game_type 0 +game_mode 0 \
  +map de_dust2 \
  +sv_lan 1 \
  +sv_hibernate_when_empty 0 \
  +exec cs2rl_match" \
  > /tmp/cs2rl-server.log 2>&1 &

echo "Server starting (PID $!). Cold start takes ~60s."
echo ""

# Wait for the plugin log to appear, then tail it
echo "Waiting for plugin log..."
for i in $(seq 1 40); do
  LOG=$(ls "$PLUGIN_LOG_DIR"/cs2rlbot-*.log 2>/dev/null | tail -1)
  [ -n "$LOG" ] && break
  sleep 3
done

if [ -z "$LOG" ]; then
  echo "Plugin log not found — check /tmp/cs2rl-server.log for server errors."
  exit 1
fi

echo "Plugin log: $LOG"
echo ""
echo "========================================"
echo " When you see 'RoundStart — LSTM reset'"
echo " open CS2 and run:  connect 127.0.0.1"
echo "========================================"
echo ""
echo "RCON: uvx --from rcon rconshell -c ~/.cs2rl_rcon 127.0.0.1:27015"
echo ""
tail -f -n +1 "$LOG"
