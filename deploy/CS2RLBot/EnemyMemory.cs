// deploy/CS2RLBot/EnemyMemory.cs
using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Modules.Utils;
using RayTraceAPI;

namespace CS2RLBot;

/// <summary>
/// Per-bot enemy state tracker. Stores last-known world position for each
/// of up to 5 enemies. Updated once per inference tick (16 Hz) via Ray-Trace LOS.
///
/// Enemy position strategy: we store actual world position (not nav centroid).
/// The sim uses nav area centroid as a proxy; world pos is strictly more precise.
///
/// Call order: Update() must be called BEFORE ObservationBuilder.Build() on the same tick.
/// Reset() must be called on RoundStart, RoundEnd, and focal bot death to match LSTM reset timing.
/// </summary>
internal sealed class EnemyMemory
{
    private const int MaxEnemies = 5;

    // World-only trace options — we only care if geometry blocks LOS, not players.
    // Using InteractionLayers.MASK_WORLD_ONLY so player hitboxes don't occlude.
    // TraceOptions constructor: (interactsAs, interactsWith, interactsExclude, drawBeam)
    // All three layer params are InteractionLayers enum, NOT ulong — verified from RayTraceApi.dll.
    private static readonly TraceOptions WorldOnlyTrace = new(
        InteractionLayers.None,
        InteractionLayers.MASK_WORLD_ONLY,
        InteractionLayers.None,
        false);

    private struct EnemyEntry
    {
        public Vector LastKnownPos;   // world position when last visible
        public float  LastKnownYaw;   // enemy yaw (degrees) when last visible
        public bool   IsAlive;
        public int    LastSeenTick;
        public bool   CanSee;
        /// <summary>
        /// False until the enemy has been seen at least once this round.
        /// When false, position slot in obs stays zero — matches sim INVALID_AREA_IDX path.
        /// </summary>
        public bool   EverSeen;
    }

    // Keyed by slot offset within the enemy team (0..4), stable per round.
    // Slot assignment is caller's responsibility (ObservationBuilder passes sorted list).
    private readonly EnemyEntry[] _entries = new EnemyEntry[MaxEnemies];

    /// <summary>
    /// Update enemy memory for one inference tick.
    /// Must be called before ObservationBuilder.Build on the same tick.
    /// </summary>
    /// <param name="selfPawn">The observing bot's pawn (trace origin).</param>
    /// <param name="enemies">Enemy team players in slot order (0..4). Caller maintains order.</param>
    /// <param name="rayTrace">Ray-Trace interface, or null if not yet acquired.</param>
    /// <param name="currentTick">Server.TickCount at this update.</param>
    public void Update(
        CCSPlayerPawn selfPawn,
        List<CCSPlayerController> enemies,
        CRayTraceInterface? rayTrace,
        int currentTick)
    {
        for (int slot = 0; slot < MaxEnemies; slot++)
        {
            if (slot >= enemies.Count)
            {
                // No enemy in this slot — team has fewer than 5 players
                _entries[slot].IsAlive = false;
                _entries[slot].CanSee  = false;
                continue;
            }

            var enemy     = enemies[slot];
            var enemyPawn = enemy.PlayerPawn?.Value;

            if (enemyPawn == null || !enemy.PawnIsAlive)
            {
                _entries[slot].IsAlive = false;
                _entries[slot].CanSee  = false;
                continue;
            }

            // Ray-Trace LOS: fire a world-only ray from self eye to enemy eye.
            // AbsOrigin is at feet level — must add ViewOffset to get eye position,
            // otherwise the ray clips the floor and DidHit is always true.
            // If rayTrace is null (plugin still starting up), treat as not visible.
            bool canSee = false;
            if (rayTrace != null && selfPawn.AbsOrigin != null && enemyPawn.AbsOrigin != null)
            {
                // Eye position = AbsOrigin + ViewOffset (standing ~64u, crouching ~46u)
                var selfEye = new Vector(
                    selfPawn.AbsOrigin.X,
                    selfPawn.AbsOrigin.Y,
                    selfPawn.AbsOrigin.Z + (selfPawn.ViewOffset?.Z ?? 64f));
                var enemyEye = new Vector(
                    enemyPawn.AbsOrigin.X,
                    enemyPawn.AbsOrigin.Y,
                    enemyPawn.AbsOrigin.Z + (enemyPawn.ViewOffset?.Z ?? 64f));

                rayTrace.TraceEndShape(
                    selfEye,
                    enemyEye,
                    selfPawn,           // ignore self so the trace doesn't hit the bot's own hitbox
                    WorldOnlyTrace,
                    out var result);
                // canSee = ray reached the enemy without hitting geometry
                canSee = !result.DidHit;
            }

            if (canSee)
            {
                // Verified: V_angle.Y is yaw (EyeAngles broke Aug 2025, Issue #1023)
                _entries[slot].LastKnownPos  = enemyPawn.AbsOrigin!;
                _entries[slot].LastKnownYaw  = enemyPawn.V_angle.Y;
                _entries[slot].IsAlive       = true;
                _entries[slot].LastSeenTick  = currentTick;
                _entries[slot].CanSee        = true;
                _entries[slot].EverSeen      = true;
            }
            else
            {
                _entries[slot].IsAlive = true; // still alive, just not visible
                _entries[slot].CanSee  = false;
                // LastKnownPos/Yaw intentionally stale — this matches sim enemy_mem_idx behavior:
                // the sim uses the last-known nav area until the enemy is seen again.
            }
        }
    }

    /// <summary>
    /// Get the memory entry for enemy at team slot (0..4).
    /// Returns (null, 0, false, false, false) for slots never seen.
    /// </summary>
    public (Vector? LastPos, float LastYaw, bool IsAlive, bool CanSee, bool EverSeen)
        Get(int slot)
    {
        var e = _entries[slot];
        return (e.EverSeen ? e.LastKnownPos : null,
                e.LastKnownYaw, e.IsAlive, e.CanSee, e.EverSeen);
    }

    /// <summary>
    /// Clear all entries. Call on RoundStart, RoundEnd, and focal bot death.
    /// Matches the LSTM reset that also fires at these events — stale memory from
    /// a previous round would corrupt the hidden state.
    /// </summary>
    public void Reset()
    {
        for (int i = 0; i < MaxEnemies; i++)
            _entries[i] = default;
    }
}
