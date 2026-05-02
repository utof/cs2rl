// ── DEPLOY SUSPENDED 2026-05-03 ────────────────────────────────────────────
// Active development paused after Batch 3.5 (sim-only training take-priority).
// POC verified on a real CS2 server pre-suspend; resuming pending sim/RL
// showing promising emergent behaviour. Last-known-good schema: v2-105dim.
// Do NOT plumb new sim obs/action surface through here as the sim evolves —
// inevitable CSS API drift + ONNX I/O changes mean a from-scratch pass is
// likely on resume. Leave as reference, not living code. See gh #(filed).
// ───────────────────────────────────────────────────────────────────────────
using System.Text.Json;
using CounterStrikeSharp.API;
using RayTraceAPI;  // FUNPLAY-pro-CS2/Ray-Trace v1.0.7 — exposes CRayTraceInterface for LOS traces.
                    // Compile-time stub only; runtime assembly loaded by CSS from
                    // addons/counterstrikesharp/shared/RayTraceApi.dll (see csproj Private=false comment).
                    // Server must have RayTraceImpl (CSS plugin) + RayTrace.so (Metamod plugin) installed.
using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Core.Attributes;
using CounterStrikeSharp.API.Core.Attributes.Registration;
using CounterStrikeSharp.API.Core.Capabilities;
using CounterStrikeSharp.API.Modules.Cvars;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Events;

namespace CS2RLBot;

[MinimumApiVersion(80)]
public class CS2RLBotPlugin : BasePlugin
{
    public override string ModuleName    => "CS2RLBot";
    public override string ModuleVersion => "0.1.0";
    public override string ModuleAuthor  => "cs2rl";

    // ── Per-bot state ─────────────────────────────────────────────────────────
    private readonly Dictionary<CCSPlayerController, PolicyInference> _policies    = new();
    private readonly Dictionary<CCSPlayerController, ActionExecutor>  _executors   = new();
    private readonly Dictionary<CCSPlayerController, int[]>           _cachedActions = new();
    // Batch 3 fix: Δyaw is NO LONGER cached across server ticks. Training
    // applies Δyaw once per env step (= once per inference call) — re-applying
    // the cached value on the 3 non-inference ticks between inferences would
    // 4× over-rotate (e.g. 45° Δyaw → 180° per inference cycle). The
    // continuous-aim value flows through a local variable in OnTick directly
    // into ActionExecutor.Execute, which gates the Teleport call on
    // isInferenceTick=true. On the 3 non-inference ticks between, yaw stays
    // wherever the last inference Teleport set it.

    // ── Observation pipeline ──────────────────────────────────────────────────
    private readonly Dictionary<CCSPlayerController, EnemyMemory> _enemyMemories = new();
    private ObservationBuilder? _obsBuilder;
    private CRayTraceInterface? _rayTrace;   // RayTraceAPI namespace confirmed; null until Load() acquires it
    private BombState           _bombState;

    // ── Map / model metadata ──────────────────────────────────────────────────
    private string _obsVersion = string.Empty;

    // ── Tick counter ──────────────────────────────────────────────────────────
    private int _tickCounter;
    private bool _requestedInitialWarmupEnd;
    private bool _rayTraceAcquired;          // true once RayTrace capability resolved (Load or deferred)

    // ── Sidecar config ────────────────────────────────────────────────────────
    private string   _modelPath    = string.Empty;
    private int[]    _actionSizes  = Array.Empty<int>();
    private int      _obsDim;
    // Batch 3: continuous-aim head dimensionality. 0 = no aim head (Batch-2
    // checkpoint, legacy NumHeads+2 ONNX layout). >0 = continuous Δyaw head
    // (currently always 1 — single-scalar Δyaw). Drives PolicyInference's
    // hasAimHead constructor flag.
    private int      _aimDim;

    // ── Structured logger (Serilog) ───────────────────────────────────────────
    private Serilog.ILogger _slog = Serilog.Log.Logger;

    // ── Convars ───────────────────────────────────────────────────────────────
    public FakeConVar<int> LogObsConVar { get; } =
        new("cs2rl_log_obs", "Set to 1 to dump full obs vectors to log file each inference tick", 0);

    // ─────────────────────────────────────────────────────────────────────────
    // Load
    // ─────────────────────────────────────────────────────────────────────────
    public override void Load(bool hotReload)
    {
        // 1. Set up Serilog (rolling daily log file + console)
        string logDir = Path.Combine(ModuleDirectory, "logs");
        Directory.CreateDirectory(logDir);
        string logPath = Path.Combine(logDir, "cs2rlbot-.log");

        _slog = new LoggerConfiguration()
            .MinimumLevel.Debug()
            .WriteTo.Console(restrictedToMinimumLevel: LogEventLevel.Information)
            .WriteTo.File(logPath, rollingInterval: RollingInterval.Day,
                          outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff} [{Level:u3}] {Message:lj}{NewLine}{Exception}")
            .CreateLogger();

        // 2. Run LatencyTracker self-test before doing anything else
        string? selfTestErr = PolicyInference.LatencyTracker.SelfTest();
        if (selfTestErr != null)
        {
            Logger.LogError("[CS2RLBot] LatencyTracker self-test FAILED: {Err}", selfTestErr);
            _slog.Error("[CS2RLBot] LatencyTracker self-test FAILED: {Err}", selfTestErr);
        }
        else
        {
            Logger.LogInformation("[CS2RLBot] LatencyTracker self-test passed");
            _slog.Information("[CS2RLBot] LatencyTracker self-test passed");
        }

        // 3. Read sidecar JSON
        string jsonPath = Path.Combine(ModuleDirectory, "models", "policy_lstm.json");
        if (!File.Exists(jsonPath))
        {
            Logger.LogError("[CS2RLBot] Sidecar JSON not found at {Path} — copy deploy/models/ to plugin ModuleDirectory/models/", jsonPath);
            return;
        }
        using var stream = File.OpenRead(jsonPath);
        using var doc = JsonDocument.Parse(stream);
        _obsDim     = doc.RootElement.GetProperty("obs_dim").GetInt32();
        _actionSizes = doc.RootElement.GetProperty("action_sizes")
                         .EnumerateArray()
                         .Select(e => e.GetInt32())
                         .ToArray();

        // Batch 3: aim_dim is a top-level int field in the sidecar (added by T6
        // export_policy.py). Missing/0 means a Batch-2 checkpoint (no aim head),
        // and we fall back to the legacy NumHeads+2 output layout.
        _aimDim = doc.RootElement.TryGetProperty("aim_dim", out var aimProp)
            ? aimProp.GetInt32()
            : 0;
        // Cross-check the sidecar's aim_dim against the plugin's compiled AimDim.
        // A future Batch-3.5 export (aim_dim=2 for Δyaw + Δpitch) shipped against
        // a plugin compiled with AimDim=1 would otherwise hit an opaque ORT
        // shape mismatch at construction. Surface a clear deploy-time error.
        if (_aimDim != 0 && _aimDim != PolicyInference.AimDim)
        {
            Logger.LogError(
                "[CS2RLBot] sidecar aim_dim={Sidecar} but plugin compiled with " +
                "PolicyInference.AimDim={Compile}. Rebuild plugin or re-export " +
                "policy with matching dim.",
                _aimDim, PolicyInference.AimDim);
            return;
        }

        // Read obs_version from sidecar — required for version handshake with mapdata JSON
        _obsVersion = doc.RootElement.TryGetProperty("obs_version", out var vProp)
            ? vProp.GetString() ?? string.Empty
            : string.Empty;
        if (string.IsNullOrEmpty(_obsVersion))
        {
            Logger.LogWarning("[CS2RLBot] policy_lstm.json missing obs_version — run export_policy.py again");
            _obsVersion = "v1-105dim"; // assume current version (105-dim obs with carrier bit at obs[104]) if not present
        }

        // 4. Resolve model path
        _modelPath = Path.Combine(ModuleDirectory, "models", "policy_lstm.onnx");
        if (!File.Exists(_modelPath))
        {
            Logger.LogError("[CS2RLBot] ONNX model not found at {Path}", _modelPath);
            return;
        }

        Logger.LogInformation(
            "[CS2RLBot] Loaded config — obs_dim={ObsDim} action_sizes=[{Sizes}] aim_dim={AimDim} model={Model}",
            _obsDim, string.Join(",", _actionSizes), _aimDim, _modelPath);
        _slog.Information(
            "[CS2RLBot] Loaded config — obs_dim={ObsDim} action_sizes=[{Sizes}] aim_dim={AimDim} model={Model}",
            _obsDim, string.Join(",", _actionSizes), _aimDim, _modelPath);

        // Load map normalization constants (generated by deploy/export_mapdata.py)
        // Path: two levels up from ModuleDirectory (plugins/CS2RLBot/) → addons/counterstrikesharp/ → mapdata/
        string mapDataPath = Path.Combine(ModuleDirectory, "..", "..", "mapdata", "de_dust2.json");
        if (!File.Exists(mapDataPath))
        {
            Logger.LogError("[CS2RLBot] Map data not found at {Path} — run: python deploy/export_mapdata.py --map de_dust2", mapDataPath);
            return;
        }
        var mapConstants = MapConstants.Load(mapDataPath);

        // Version handshake: obs_version in model sidecar must match mapdata JSON
        if (mapConstants.ObsVersion != _obsVersion)
        {
            Logger.LogError(
                "[CS2RLBot] obs_version mismatch: model={ModelVer} mapdata={MapVer}. " +
                "Re-run export_policy.py and export_mapdata.py with matching versions.",
                _obsVersion, mapConstants.ObsVersion);
            return;
        }
        Logger.LogInformation("[CS2RLBot] ObsVersion match: {Ver}", _obsVersion);
        _slog.Information("[CS2RLBot] ObsVersion match: {Ver}", _obsVersion);

        // Acquire Ray-Trace LOS interface — provided at runtime by RayTraceImpl (CSS plugin) + RayTrace.so (Metamod)
        // PluginCapability.Get() throws KeyNotFoundException if the capability isn't registered yet (e.g. RayTraceImpl
        // loads after CS2RLBot, or isn't deployed).  Catch and treat as "not available" rather than crashing Load().
        try
        {
            _rayTrace = new PluginCapability<CRayTraceInterface>("raytrace:craytraceinterface").Get();
        }
        catch (KeyNotFoundException)
        {
            _rayTrace = null;
        }
        if (_rayTrace == null)
            Logger.LogWarning("[CS2RLBot] RayTrace not available at Load() — will retry in OnTick (RayTraceImpl loads after CS2RLBot alphabetically).");
        else
        {
            _rayTraceAcquired = true;
            Logger.LogInformation("[CS2RLBot] RayTrace interface acquired at Load()");
            _slog.Information("[CS2RLBot] RayTrace interface acquired at Load()");
        }

        // Build ObservationBuilder — throws if obs_version mismatch (belt-and-suspenders)
        _obsBuilder = new ObservationBuilder(_obsDim, _obsVersion, mapConstants);
        Logger.LogInformation("[CS2RLBot] ObservationBuilder ready: obs_dim={ObsDim}", _obsDim);
        _slog.Information("[CS2RLBot] ObservationBuilder ready: obs_dim={ObsDim}", _obsDim);

        // 5. Register tick listener ([GameEventHandler] attributes handle event registration)
        RegisterListener<Listeners.OnTick>(OnTick);

        // 6. bot_stop 1 is issued in OnTick once the first bot registers (same timing as
        // mp_warmup_end), where sv_cheats is guaranteed active. Also set in cs2rl_match.cfg.
        // bot_stop suppresses native AI (buying, pathfinding, strategic movement).
        // Tested: EventRoundEnd fires normally — bots can still be killed by players.
        // NOTE: bot_stop also suppresses native weapon-firing; whether plugin Attack bit
        // still fires weapons is an open research question (see research-brief Q2).
        Logger.LogInformation("[CS2RLBot] Plugin loaded");
        _slog.Information("[CS2RLBot] Plugin loaded");
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Unload
    // ─────────────────────────────────────────────────────────────────────────
    public override void Unload(bool hotReload)
    {
        foreach (var p in _policies.Values) p.Dispose();
        _policies.Clear();
        _executors.Clear();
        _cachedActions.Clear();
        _enemyMemories.Clear();
        Logger.LogInformation("[CS2RLBot] Plugin unloaded — all PolicyInference instances disposed");
        _slog.Information("[CS2RLBot] Plugin unloaded");
        (_slog as IDisposable)?.Dispose();
        Serilog.Log.CloseAndFlush();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // OnTick — runs every server tick (64 Hz)
    // ─────────────────────────────────────────────────────────────────────────
    private void OnTick()
    {
        _tickCounter++;
        bool isInferenceTick = (_tickCounter % 4  == 0); // 16 Hz
        bool isStatsTick     = (_tickCounter % 64 == 0); // ~1 Hz
        bool shouldEndWarmup = false;

        // Deferred RayTrace acquisition: CS2RLBot loads before RayTraceImpl (alphabetical CSS
        // plugin order), so Load() gets null. Retry every ~1s for the first 10s of server life.
        if (!_rayTraceAcquired && _tickCounter <= 640)
        {
            if (_tickCounter % 64 == 1)
            {
                try
                {
                    _rayTrace = new PluginCapability<CRayTraceInterface>("raytrace:craytraceinterface").Get();
                }
                catch (KeyNotFoundException) { /* still not registered */ }

                if (_rayTrace != null)
                {
                    _rayTraceAcquired = true;
                    Logger.LogInformation("[CS2RLBot] RayTrace interface acquired (deferred, tick={Tick})", _tickCounter);
                    _slog.Information("[CS2RLBot] RayTrace interface acquired (deferred, tick={Tick})", _tickCounter);
                }
            }
        }

        // Materialize player lists once per tick — GetPlayers() is O(N); calling inside the bot
        // loop would make inference O(N²). Filter: IsBot && !IsHLTV && IsValid && PawnIsAlive.
        var allPlayers = Utilities.GetPlayers().Where(p => p.IsValid && !p.IsHLTV).ToList();
        var allBots    = allPlayers.Where(p => p.IsBot && p.PawnIsAlive).ToList(); // !IsHLTV already enforced by allPlayers

        foreach (var bot in allBots)
        {
            var pawn = bot.PlayerPawn?.Value;
            if (pawn == null) continue;

            // Lazy-initialise dicts on first encounter
            if (!_policies.ContainsKey(bot))
            {
                // Batch 3: hasAimHead is driven by sidecar `aim_dim > 0`. Older
                // checkpoints (aim_dim=0 or field absent) get the legacy layout.
                _policies[bot]      = new PolicyInference(_modelPath, _actionSizes, Logger, hasAimHead: _aimDim > 0);
                _executors[bot]     = new ActionExecutor();
                // Batch 3: ACTION_DIM 8→7 (move/shoot/reload/weapon/use/crouch/jump);
                // HEAD_AIM was index 1 in the old 8-element layout and is now a
                // separate continuous mu_aim output, not in this int[] cache.
                _cachedActions[bot] = new int[7];
                _enemyMemories[bot] = new EnemyMemory();
                Logger.LogInformation("[CS2RLBot] Bot registered: {Name} team={Team}",
                    bot.PlayerName, bot.TeamNum);
                _slog.Information("[CS2RLBot] Bot registered: {Name} team={Team}",
                    bot.PlayerName, bot.TeamNum);

                // Dedicated casual startup still leaves the server in warmup until this
                // command is issued after at least one bot exists. Doing it from config
                // or during Load() fires too early, before bots are present.
                if (!_requestedInitialWarmupEnd)
                {
                    _requestedInitialWarmupEnd = true;
                    shouldEndWarmup = true;
                }
            }

            // Δyaw flows directly from this inference into Execute(...) below,
            // ONLY on inference ticks. On non-inference ticks we pass 0f — the
            // executor won't use it (its yaw block is gated on isInferenceTick),
            // but 0f makes the data-flow explicit and removes any risk of stale
            // values being silently re-applied if the gate were ever loosened.
            float deltaYawRad = 0f;

            if (isInferenceTick)
            {
                // Build enemy lists for this inference tick — reuse allBots materialized above
                var teammates = allBots
                    .Where(p => p != bot && p.TeamNum == bot.TeamNum)
                    .ToList();
                // Enemies sorted by distance (nearest first) — includes human players so bots
                // can see/react to humans during testing. Uses allPlayers, not allBots.
                var enemies = allPlayers
                    .Where(p => p != bot && p.TeamNum != bot.TeamNum && p.TeamNum > 1 && p.PawnIsAlive)
                    .OrderBy(p => {
                        var ep = p.PlayerPawn?.Value?.AbsOrigin;
                        var sp = pawn.AbsOrigin;
                        if (ep == null || sp == null) return float.MaxValue;
                        float dx = ep.X - sp.X, dy = ep.Y - sp.Y;
                        return dx * dx + dy * dy; // squared distance — no sqrt needed for sort
                    })
                    .ToList();

                // Update enemy memory (Ray-Trace LOS) before building obs — null rayTrace = no LOS, stale pos only
                _enemyMemories[bot].Update(pawn, enemies, _rayTrace, _tickCounter);

                // Diagnostic: log canSee + distances once per second per bot
                if (isStatsTick)
                {
                    int seenCount = 0;
                    for (int ei = 0; ei < Math.Min(enemies.Count, 5); ei++)
                    {
                        var (_, _, _, canSee, _) = _enemyMemories[bot].Get(ei);
                        if (canSee) seenCount++;
                    }
                    var selfPos = pawn.AbsOrigin;
                    float nearestDist = float.MaxValue;
                    string nearestName = "none";
                    foreach (var e in enemies)
                    {
                        var ep = e.PlayerPawn?.Value?.AbsOrigin;
                        if (ep == null || selfPos == null) continue;
                        float dx = ep.X - selfPos.X, dy = ep.Y - selfPos.Y, dz = ep.Z - selfPos.Z;
                        float d = MathF.Sqrt(dx * dx + dy * dy + dz * dz);
                        if (d < nearestDist) { nearestDist = d; nearestName = e.PlayerName; }
                    }
                    _slog.Information(
                        "[CS2RLBot] LOS bot={Bot} sees={Seen}/{Total} nearest={Name}@{Dist:F0}u rayTrace={HasRT}",
                        bot.PlayerName, seenCount, enemies.Count, nearestName, nearestDist,
                        _rayTrace != null ? "yes" : "NULL");
                }

                float[] obs;
                if (_obsBuilder != null)
                {
                    // No single "RoundTimeRemaining" property in CSS — must compute from game rules
                    // RoundTime = configured duration (seconds); RoundStartTime = server time at round start
                    var grProxy = Utilities.FindAllEntitiesByDesignerName<CCSGameRulesProxy>("cs_gamerules")
                        .FirstOrDefault();
                    var gameRules = grProxy?.GameRules;
                    float roundTimeTotal = gameRules?.RoundTime ?? 115f;
                    float roundTicksLeft = gameRules != null
                        ? Math.Max(0f, gameRules.RoundStartTime + gameRules.RoundTime - Server.CurrentTime)
                        : roundTimeTotal;
                    obs = _obsBuilder.Build(bot, pawn, teammates, enemies, _enemyMemories[bot],
                                            _bombState, allPlayers, roundTimeTotal, roundTicksLeft);
                }
                else
                {
                    obs = new float[_obsDim]; // fallback: obsBuilder failed to init (mapdata missing or version mismatch)
                }

                // Batch 3: RunInference now returns (logits, muAim). muAim is
                // an empty array when the loaded checkpoint has no aim head.
                var (logits, muAim) = _policies[bot].RunInference(obs, isDone: false);

                int[] cached = _cachedActions[bot];
                int limit = Math.Min(logits.Length, cached.Length);
                for (int i = 0; i < limit; i++)
                    cached[i] = ActionExecutor.Argmax(logits[i]);

                // Pass the fresh Δyaw through to Execute below. Not cached:
                // applied exactly once this inference cycle, then discarded.
                // 0f when no aim head present.
                deltaYawRad = muAim.Length > 0 ? muAim[0] : 0f;

                _slog.Debug("[CS2RLBot] Inference tick={Tick} bot={Bot} actions=[{Actions}] dYaw={DYaw:F4}",
                    _tickCounter, bot.PlayerName, string.Join(",", cached), deltaYawRad);

                if (LogObsConVar.Value == 1)
                    _slog.Debug("[CS2RLBot] ObsDump tick={Tick} obs=[{Obs}]",
                        _tickCounter, string.Join(",", obs));
            }

            _executors[bot].Execute(bot, pawn, _cachedActions[bot], deltaYawRad, isInferenceTick);
        }

        if (shouldEndWarmup)
        {
            Server.ExecuteCommand("mp_warmup_end");
            Logger.LogInformation("[CS2RLBot] Requested mp_warmup_end after first bot registration");
            _slog.Information("[CS2RLBot] Requested mp_warmup_end after first bot registration");
        }

        // Rolling latency stats — logged ~once per second
        if (isStatsTick)
        {
            foreach (var (bot, policy) in _policies)
            {
                var stats = policy.GetStats();
                _slog.Information(
                    "[CS2RLBot] Stats bot={Bot} p50={P50}µs p99={P99}µs max={Max}µs tick={Tick}",
                    bot.PlayerName, stats.P50Us, stats.P99Us, stats.MaxUs, _tickCounter);
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Event handlers
    // ─────────────────────────────────────────────────────────────────────────
    [GameEventHandler]
    public HookResult OnRoundStart(EventRoundStart @event, GameEventInfo info)
    {
        _tickCounter = 0;
        foreach (var (bot, policy) in _policies)
        {
            policy.ResetLstmState();
            Logger.LogInformation("[CS2RLBot] RoundStart — LSTM reset for {Name}", bot.PlayerName);
        }
        _slog.Information("[CS2RLBot] RoundStart — LSTM reset for {Count} bot(s)", _policies.Count);
        // Reset all enemy memories and bomb state — matches LSTM reset timing
        foreach (var mem in _enemyMemories.Values)
            mem.Reset();
        _bombState = default;
        _obsBuilder?.ClearReloadCache(); // _reloadStartTimes keys are entity indices; stale across rounds
        // Defensive mirror of OnRoundEnd: if OnRoundEnd is skipped (warmup, plugin reload,
        // server crash recovery, freezetime abort), latches would otherwise drift across rounds.
        // ObservationBuilder.cs:296 docstring explicitly anticipates RoundStart usage.
        _obsBuilder?.ClearCarrierLatches();
        return HookResult.Continue;
    }

    [GameEventHandler]
    public HookResult OnRoundEnd(EventRoundEnd @event, GameEventInfo info)
    {
        foreach (var (bot, policy) in _policies)
        {
            policy.ResetLstmState();
            Logger.LogInformation("[CS2RLBot] RoundEnd — LSTM reset for {Name}", bot.PlayerName);
        }
        _slog.Information("[CS2RLBot] RoundEnd — LSTM reset for {Count} bot(s)", _policies.Count);
        _obsBuilder?.ClearReloadCache(); // prevent stale reload tracking across round boundary
        // Batch 2: reset designated-carrier latches so each new round can re-latch.
        // Mirrors the round-fixed sim semantics: obs[104] is set once per round, then frozen.
        _obsBuilder?.ClearCarrierLatches();
        return HookResult.Continue;
    }

    [GameEventHandler]
    public HookResult OnPlayerDeath(EventPlayerDeath @event, GameEventInfo info)
    {
        var player = @event.Userid;
        if (player == null || !player.IsBot) return HookResult.Continue;

        if (_policies.TryGetValue(player, out var policy))
        {
            policy.ResetLstmState();
            Logger.LogInformation("[CS2RLBot] Bot died — LSTM reset for {Name}", player.PlayerName);
            _slog.Information("[CS2RLBot] Bot died — LSTM reset for {Name}", player.PlayerName);
        }
        // Reset the dead bot's enemy memory — stale knowledge irrelevant after respawn
        if (_enemyMemories.TryGetValue(player, out var mem))
            mem.Reset();
        _obsBuilder?.ClearReloadCache(); // weapon may respawn with different entity index
        return HookResult.Continue;
    }

    [GameEventHandler]
    public HookResult OnPlayerDisconnect(EventPlayerDisconnect @event, GameEventInfo info)
    {
        var player = @event.Userid;
        if (player == null || !player.IsBot) return HookResult.Continue;

        if (_policies.TryGetValue(player, out var policy))
        {
            policy.Dispose();
            _policies.Remove(player);
            _executors.Remove(player);
            _cachedActions.Remove(player);
            _enemyMemories.Remove(player);
            Logger.LogInformation("[CS2RLBot] Bot disconnected — disposed {Name}", player.PlayerName);
        }
        return HookResult.Continue;
    }

    [GameEventHandler]
    public HookResult OnBombPlanted(EventBombPlanted @event, GameEventInfo info)
    {
        // Grab planted C4 entity for timer and position — C4Blow = server time of detonation
        var plantedC4 = Utilities.FindAllEntitiesByDesignerName<CPlantedC4>("planted_c4").FirstOrDefault();
        _bombState.BombX       = plantedC4?.AbsOrigin?.X ?? 0f;
        _bombState.BombY       = plantedC4?.AbsOrigin?.Y ?? 0f;
        _bombState.BlowTime    = plantedC4?.C4Blow ?? 0f;
        _bombState.TimerLength = plantedC4?.TimerLength ?? 40f; // TimerLength confirmed on CPlantedC4
        _bombState.Planted     = true;
        _bombState.Dropped     = false;
        _bombState.CarrierId   = -1;
        Logger.LogInformation("[CS2RLBot] Bomb planted");
        _slog.Information("[CS2RLBot] Bomb planted at ({X:F0},{Y:F0}) blow={BlowTime:F1}s",
            _bombState.BombX, _bombState.BombY, _bombState.TimerLength);
        return HookResult.Continue;
    }

    [GameEventHandler]
    public HookResult OnBombDropped(EventBombDropped @event, GameEventInfo info)
    {
        // Record drop position from the dropping player's pawn — no dedicated entity at drop point
        _bombState.Dropped   = true;
        _bombState.Planted   = false;
        _bombState.CarrierId = -1;
        _bombState.BombX = @event.Userid?.PlayerPawn?.Value?.AbsOrigin?.X ?? 0f;
        _bombState.BombY = @event.Userid?.PlayerPawn?.Value?.AbsOrigin?.Y ?? 0f;
        Logger.LogInformation("[CS2RLBot] Bomb dropped");
        _slog.Information("[CS2RLBot] Bomb dropped at ({X:F0},{Y:F0})", _bombState.BombX, _bombState.BombY);
        return HookResult.Continue;
    }

    [GameEventHandler]
    public HookResult OnBombPickup(EventBombPickup @event, GameEventInfo info)
    {
        // CarrierId = player Slot (integer) — used in ObservationBuilder to look up carrier position
        _bombState.Dropped   = false;
        _bombState.Planted   = false;
        _bombState.CarrierId = @event.Userid?.Slot ?? -1;
        Logger.LogInformation("[CS2RLBot] Bomb picked up by slot={Slot}", _bombState.CarrierId);
        _slog.Information("[CS2RLBot] Bomb picked up by slot={Slot}", _bombState.CarrierId);
        return HookResult.Continue;
    }

}
