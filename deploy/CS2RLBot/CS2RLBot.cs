using System.Text.Json;
using CounterStrikeSharp.API;
using RayTraceAPI;  // FUNPLAY-pro-CS2/Ray-Trace v1.0.7 — exposes CRayTraceInterface for LOS traces.
                    // Compile-time stub only; runtime assembly loaded by CSS from
                    // addons/counterstrikesharp/shared/RayTraceApi.dll (see csproj Private=false comment).
                    // Server must have RayTraceImpl (CSS plugin) + RayTrace.so (Metamod plugin) installed.
using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Core.Attributes;
using CounterStrikeSharp.API.Core.Attributes.Registration;
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

    // ── Tick counter ──────────────────────────────────────────────────────────
    private int _tickCounter;
    private bool _requestedInitialWarmupEnd;

    // ── Sidecar config ────────────────────────────────────────────────────────
    private string   _modelPath    = string.Empty;
    private int[]    _actionSizes  = Array.Empty<int>();
    private int      _obsDim;

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

        // 4. Resolve model path
        _modelPath = Path.Combine(ModuleDirectory, "models", "policy_lstm.onnx");
        if (!File.Exists(_modelPath))
        {
            Logger.LogError("[CS2RLBot] ONNX model not found at {Path}", _modelPath);
            return;
        }

        Logger.LogInformation(
            "[CS2RLBot] Loaded config — obs_dim={ObsDim} action_sizes=[{Sizes}] model={Model}",
            _obsDim, string.Join(",", _actionSizes), _modelPath);
        _slog.Information(
            "[CS2RLBot] Loaded config — obs_dim={ObsDim} action_sizes=[{Sizes}] model={Model}",
            _obsDim, string.Join(",", _actionSizes), _modelPath);

        // 5. Register tick listener ([GameEventHandler] attributes handle event registration)
        RegisterListener<Listeners.OnTick>(OnTick);

        // 6. NOTE: bot_stop intentionally NOT set here — it prevents round timers from
        // expiring (bots can't die), blocking EventRoundEnd. Native AI runs alongside
        // plugin button writes for now. Phase 7D will re-evaluate once obs builder exists.
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

        foreach (var bot in GetControlledBots())
        {
            var pawn = bot.PlayerPawn?.Value;
            if (pawn == null) continue;

            // Lazy-initialise dicts on first encounter
            if (!_policies.ContainsKey(bot))
            {
                _policies[bot]      = new PolicyInference(_modelPath, _actionSizes, Logger);
                _executors[bot]     = new ActionExecutor();
                _cachedActions[bot] = new int[7]; // always 7 slots; extras default to 0
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

            if (isInferenceTick)
            {
                var obs = new float[_obsDim]; // zero obs — Phase 7D will replace with ObservationBuilder

                float[][] logits = _policies[bot].RunInference(obs, isDone: false);

                int[] cached = _cachedActions[bot];
                int limit = Math.Min(logits.Length, cached.Length);
                for (int i = 0; i < limit; i++)
                    cached[i] = ActionExecutor.Argmax(logits[i]);

                _slog.Debug("[CS2RLBot] Inference tick={Tick} bot={Bot} actions=[{Actions}]",
                    _tickCounter, bot.PlayerName, string.Join(",", cached));

                if (LogObsConVar.Value == 1)
                    _slog.Debug("[CS2RLBot] ObsDump tick={Tick} obs=[{Obs}]",
                        _tickCounter, string.Join(",", obs));
            }

            _executors[bot].Execute(bot, pawn, _cachedActions[bot], isInferenceTick);
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
            Logger.LogInformation("[CS2RLBot] Bot disconnected — disposed {Name}", player.PlayerName);
        }
        return HookResult.Continue;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────
    private static IEnumerable<CCSPlayerController> GetControlledBots() =>
        Utilities.GetPlayers()
            .Where(p => p.IsBot && !p.IsHLTV && p.IsValid && p.PawnIsAlive);
}
