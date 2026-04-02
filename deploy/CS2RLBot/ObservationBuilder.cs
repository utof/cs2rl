// deploy/CS2RLBot/ObservationBuilder.cs
// Builds the 104-dim observation vector used by the RL policy, matching
// cs2_observations.h exactly. See that file for the ground-truth formula
// and dimension layout.
using System.Text.Json;
using CounterStrikeSharp.API;
using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Modules.Utils;

namespace CS2RLBot;

// ── Map constants (from deploy/mapdata/de_dust2.json) ────────────────────────
// Loaded once at plugin startup from the sidecar JSON.
// Formula: x_norm = x * inv_x_range - x_offset  (cs2_observations.h:33-34)
// CRITICAL: NOT (x - x_offset) * inv_x_range — the offset is applied after multiplication.

internal sealed record MapConstants(
    string Map,
    string ObsVersion,
    float InvXRange,
    float InvYRange,
    float XOffset,
    float YOffset,
    float MapDiag)
{
    internal static MapConstants Load(string jsonPath)
    {
        using var stream = File.OpenRead(jsonPath);
        using var doc    = JsonDocument.Parse(stream);
        var root = doc.RootElement;
        return new MapConstants(
            Map:        root.GetProperty("map").GetString()!,
            ObsVersion: root.GetProperty("obs_version").GetString()!,
            InvXRange:  root.GetProperty("inv_x_range").GetSingle(),
            InvYRange:  root.GetProperty("inv_y_range").GetSingle(),
            XOffset:    root.GetProperty("x_offset").GetSingle(),
            YOffset:    root.GetProperty("y_offset").GetSingle(),
            MapDiag:    root.GetProperty("map_diag").GetSingle());
    }
}

// ── BombState (lightweight struct, updated by event handlers in CS2RLBot.cs) ─
// Updated by: BombPlanted, BombDefuseStart, BombDefuseAborted, BombExploded,
//             BombDropped, BombPickup events, and per-tick carrier scan.

internal struct BombState
{
    public bool  Planted;
    public bool  Dropped;
    public float BombX;
    public float BombY;
    public float BlowTime;      // Server.CurrentTime when bomb detonates (CPlantedC4.C4Blow)
    public float TimerLength;   // total fuse duration in seconds (CPlantedC4.TimerLength)
    public int   CarrierId;     // player Slot of carrier (-1 = no carrier)
}

// ── ObservationBuilder ───────────────────────────────────────────────────────
// One instance is shared across all bots (stateless per-build, except reload tracking).
// Not thread-safe — all calls must occur from the server tick thread.

internal sealed class ObservationBuilder
{
    private const string SupportedVersion = "v1-104dim";
    private const int    TeamSize         = 5;
    private const int    Terrorist        = 2; // CS2 TeamNum for T-side
    private const int    CounterTerrorist = 3; // CS2 TeamNum for CT-side

    private readonly int          _obsDim;
    private readonly MapConstants _map;
    // Reused across Build calls to avoid per-tick heap allocation (important at 64 Hz).
    private readonly float[]      _buf;

    // Reload tracking: CSS does not expose a reload-start-time field, so we track it
    // manually. Key = entity Index (UInt32, stable within a round).
    // Pitfall: do NOT use EntityHandle.Raw — CEntityHandle is an opaque CSS type and
    // its .Raw is not publicly exposed in v1.0.364. Use CBaseEntity.Index instead.
    private readonly Dictionary<uint, float> _reloadStartTimes = new();

    public ObservationBuilder(int obsDim, string obsVersion, MapConstants map)
    {
        if (obsVersion != SupportedVersion)
            throw new InvalidOperationException(
                $"ObservationBuilder supports obs_version={SupportedVersion}, got {obsVersion}");

        _obsDim = obsDim;
        _map    = map;
        _buf    = new float[obsDim];
    }

    /// <summary>
    /// Build the 104-dim obs vector for one bot.
    /// Returns the internal buffer — caller must consume it before the next Build call (shared buffer).
    /// All FillX helpers write into _buf; ClipAll runs last to enforce [-5, 5].
    /// </summary>
    public float[] Build(
        CCSPlayerController       bot,
        CCSPlayerPawn             pawn,
        List<CCSPlayerController> teammates,    // other bots on same team, in slot order (no sort)
        List<CCSPlayerController> enemies,      // opposite team, in slot order
        EnemyMemory               enemyMem,
        BombState                 bomb,
        List<CCSPlayerController> allPlayers,
        float                     roundTimeTotal,
        float                     roundTicksLeft)
    {
        Array.Clear(_buf, 0, _obsDim);
        FillSelf(pawn, bot);
        FillTeammates(pawn, teammates);
        FillEnemies(pawn, enemies, enemyMem);
        FillBombAndGlobal(bot, pawn, bomb, allPlayers, roundTimeTotal, roundTicksLeft);
        ClipAll();
        return _buf;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Self state (dims 0-22) — mirrors cs2_observations.h:29-55
    // ─────────────────────────────────────────────────────────────────────────

    private void FillSelf(CCSPlayerPawn pawn, CCSPlayerController bot)
    {
        _buf[0] = pawn.Health / 100f;
        _buf[1] = pawn.ArmorValue / 100f;
        // Helmet: confirmed controller.PawnHasHelmet (not pawn.HasHelmet — that field doesn't exist)
        _buf[2] = bot.PawnHasHelmet ? 1f : 0f;

        // Absolute position — formula: x * inv_x_range - x_offset  (cs2_observations.h:33-34)
        if (pawn.AbsOrigin != null)
        {
            _buf[3] = pawn.AbsOrigin.X * _map.InvXRange - _map.XOffset;
            _buf[4] = pawn.AbsOrigin.Y * _map.InvYRange - _map.YOffset;
        }
        _buf[5] = 0f; // z placeholder (matches sim)

        // Velocity / 250 — confirmed: AbsVelocity (not pawn.Velocity = CNetworkVelocityVector)
        _buf[6] = pawn.AbsVelocity.X / 250f;
        _buf[7] = pawn.AbsVelocity.Y / 250f;
        _buf[8] = 0f; // vz placeholder

        // Facing: sin/cos of yaw. Confirmed: V_angle.Y is yaw (EyeAngles broke Aug 2025, Issue #1023).
        // V_angle is QAngle (X=pitch, Y=yaw, Z=roll) — null-safe: AbsOrigin may be null but V_angle
        // is a value-type QAngle embedded in the pawn, never null.
        float yawRad = ObsMath.YawToRad(pawn.V_angle.Y);
        _buf[9]  = MathF.Sin(yawRad);
        _buf[10] = MathF.Cos(yawRad);

        // Crouch — confirmed: MovementServices.Ducked. Do NOT use Flags arithmetic (unreliable in CSS).
        _buf[11] = IsCrouching(pawn) ? 1f : 0f;

        // Active weapon: slot one-hot (0=rifle, 1=pistol, 2=knife) — cs2_observations.h:43-45
        // ActiveWeapon.Value returns CBasePlayerWeapon; DesignerName is on base class (no cast).
        var weapon = pawn.WeaponServices?.ActiveWeapon?.Value;
        int slot = GetWeaponSlot(weapon);
        _buf[12] = (slot == 0) ? 1f : 0f;
        _buf[13] = (slot == 1) ? 1f : 0f;
        _buf[14] = (slot == 2) ? 1f : 0f;

        // Ammo dims (15-19) — cs2_observations.h:44-52
        // IMPORTANT: normalization uses WEAPON_DEFS slot-based constants, NOT per-weapon values.
        // The sim's WEAPON_DEFS (cs2_weapons.h) uses one table per slot (rifle/pistol/knife),
        // not per specific weapon. All rifles share mag_size=25, reserve_mags=3, cycle_ticks=2 (at 16Hz).
        if (weapon != null)
        {
            var weaponBase = weapon.As<CCSWeaponBase>();
            // Slot-based constants matching WEAPON_DEFS (cs2_weapons.h):
            // rifle: mag=25, reserveMags=3, cycleTicks=8 (2×4 for 64Hz)
            // pistol: mag=16, reserveMags=2, cycleTicks=12 (3×4 for 64Hz)
            int slotMag     = slot == 0 ? 25 : (slot == 1 ? 16 : -1);
            int slotResMags = slot == 0 ? 3  : (slot == 1 ? 2  : -1);

            // obs[15]: clip fraction — ammo_clip[slot] / def->mag_size (cs2_observations.h:44-45)
            _buf[15] = slotMag > 0 ? weapon.Clip1 / (float)slotMag : 1f;

            // obs[16]: reserve fraction — ammo_reserve[slot] / def->reserve_mags (cs2_observations.h:46-47)
            // CSS ReserveAmmo[0] is total bullets; convert to mag count first:
            //   magCount = ReserveAmmo[0] / actualClipSize  (per-weapon clip for unit conversion only)
            int actualClip = weaponBase?.VData?.MaxClip1 ?? GetMaxClipFallback(weapon);
            float reserveMagCount = actualClip > 0 ? weapon.ReserveAmmo[0] / (float)actualClip : 0f;
            _buf[16] = slotResMags > 0 ? reserveMagCount / slotResMags : 0f;

            _buf[17] = (weaponBase?.InReload == true) ? 1f : 0f;
            _buf[18] = GetReloadProgress(weapon, weaponBase);
            _buf[19] = GetFireCooldown(weapon, slot);
        }
        // No active weapon → ammo dims stay 0.

        // Has bomb (T-side only) — matches sim obs[20] = (team==0 && has_bomb)
        _buf[20] = (bot.TeamNum == Terrorist && HasC4(pawn)) ? 1f : 0f;
        _buf[21] = bot.PawnIsAlive ? 1f : 0f;
        // Team: T=1.0, CT=0.0 — matches sim obs[22]=(a->team==0) where sim team 0 = T
        _buf[22] = (bot.TeamNum == Terrorist) ? 1f : 0f;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Weapon helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Classify active weapon into slot: 0=rifle, 1=pistol, 2=knife.
    /// DesignerName is on CBasePlayerWeapon (no cast needed).
    /// </summary>
    private static int GetWeaponSlot(CBasePlayerWeapon? weapon)
    {
        if (weapon == null) return 2;
        string name = weapon.DesignerName ?? "";
        if (name is "weapon_knife" or "weapon_bayonet") return 2;
        // Pistols — if new weapons are added to CS2, extend this list.
        if (name is "weapon_glock"         or "weapon_usp_silencer" or "weapon_p2000"
                 or "weapon_p250"          or "weapon_deagle"       or "weapon_fiveseven"
                 or "weapon_tec9"          or "weapon_cz75a"        or "weapon_revolver"
                 or "weapon_hkp2000"       or "weapon_elite"        or "weapon_mp5sd") return 1;
        return 0; // rifles, SMGs, shotguns, LMGs, snipers
    }

    /// <summary>
    /// Fallback max clip size lookup by weapon name.
    /// Used only when VData is unavailable (e.g. entity partially constructed).
    /// Prefer weaponBase.VData.MaxClip1 which is the authoritative source from game data.
    /// </summary>
    private static int GetMaxClipFallback(CBasePlayerWeapon weapon)
    {
        return weapon.DesignerName switch
        {
            "weapon_ak47"          => 30, "weapon_m4a1"          => 30,
            "weapon_m4a1_silencer" => 20, "weapon_awp"           => 10,
            "weapon_glock"         => 20, "weapon_usp_silencer"  => 12,
            "weapon_deagle"        => 7,  "weapon_p250"          => 13,
            "weapon_fiveseven"     => 20, "weapon_tec9"          => 18,
            "weapon_sg556"         => 30, "weapon_aug"           => 30,
            "weapon_famas"         => 25, "weapon_galil"         => 35,
            "weapon_mp9"           => 30, "weapon_mac10"         => 30,
            "weapon_p90"           => 50, "weapon_bizon"         => 64,
            "weapon_ump45"         => 25, "weapon_mp5sd"         => 30,
            "weapon_nova"          => 8,  "weapon_xm1014"        => 7,
            "weapon_sawedoff"      => 7,  "weapon_mag7"          => 5,
            "weapon_m249"          => 100,"weapon_negev"         => 150,
            "weapon_ssg08"         => 10, "weapon_g3sg1"         => 20,
            "weapon_scar20"        => 20,
            _                      => 0,  // knife/unknown
        };
    }

    /// <summary>
    /// Max reserve magazine count.
    /// The sim tracks reserve_mags (number of spare magazines), not total bullet count.
    /// CS2 weapons typically carry 2-4 extra mags. Knives carry 0.
    /// </summary>
    private static int GetMaxReserve(CBasePlayerWeapon weapon)
    {
        if (weapon.DesignerName is "weapon_knife" or "weapon_bayonet") return 0;
        return weapon.DesignerName switch
        {
            "weapon_awp"   => 2,
            "weapon_ssg08" => 2,
            _              => 4,
        };
    }

    /// <summary>
    /// Reload progress in [0,1] since reloading started.
    /// CSS does not expose a reload-start-time field, so we record the time we first
    /// observed InReload==true and compute elapsed / total.
    /// Duration proxy: CCSWeaponBaseVData.DisallowAttackAfterReloadStartDuration
    ///   (no direct "total reload time" field exists in the API).
    /// Key: CBaseEntity.Index (UInt32) — stable for the entity's lifetime within a round.
    /// Returns 0 if not currently reloading.
    /// </summary>
    private float GetReloadProgress(CBasePlayerWeapon weapon, CCSWeaponBase? weaponBase)
    {
        if (weaponBase?.InReload != true)
        {
            // Clear stale entry when not reloading so next reload starts fresh.
            _reloadStartTimes.Remove(weapon.Index);
            return 0f;
        }

        uint key = weapon.Index;
        if (!_reloadStartTimes.TryGetValue(key, out float startTime))
        {
            _reloadStartTimes[key] = Server.CurrentTime;
            return 0f;
        }

        float totalDuration = weaponBase.VData?.DisallowAttackAfterReloadStartDuration ?? 2.0f;
        if (totalDuration <= 0f) return 0f;
        return Math.Clamp((Server.CurrentTime - startTime) / totalDuration, 0f, 1f);
    }

    /// <summary>
    /// Clear reload start-time cache. Call on RoundStart and RoundEnd to prevent
    /// stale entries if weapon entity indices are reused across rounds.
    /// </summary>
    public void ClearReloadCache() => _reloadStartTimes.Clear();

    /// <summary>
    /// Fire cooldown normalized to [0,1] using slot-based cycle_ticks from WEAPON_DEFS.
    /// WEAPON_DEFS cycle_ticks at 16Hz: rifle=2, pistol=3, knife=0.
    /// At 64Hz (CS2 server rate): multiply by 4 → rifle=8, pistol=12.
    /// Verified: NextPrimaryAttackTick (int) compared to Server.TickCount (both in server ticks).
    /// </summary>
    private static float GetFireCooldown(CBasePlayerWeapon weapon, int weaponSlot)
    {
        int cycleTicks64Hz = weaponSlot == 0 ? 8 : (weaponSlot == 1 ? 12 : 0);
        if (cycleTicks64Hz == 0) return 0f; // knife has no fire cooldown
        int ticksRemaining = weapon.NextPrimaryAttackTick - Server.TickCount;
        if (ticksRemaining <= 0) return 0f;
        return Math.Clamp(ticksRemaining / (float)cycleTicks64Hz, 0f, 1f);
    }

    /// <summary>
    /// Returns true if the pawn is fully crouched.
    /// Confirmed: use MovementServices.Ducked, not Flags & FL_DUCKING (flag arithmetic was unreliable).
    /// </summary>
    private static bool IsCrouching(CCSPlayerPawn pawn)
    {
        return pawn.MovementServices?.As<CCSPlayer_MovementServices>()?.Ducked ?? false;
    }

    /// <summary>Returns true if the pawn is carrying the C4 bomb.</summary>
    private static bool HasC4(CCSPlayerPawn pawn)
    {
        return pawn.WeaponServices?.MyWeapons
            .Any(w => w.IsValid && (w.Value?.DesignerName?.Contains("c4") ?? false))
            ?? false;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ── Teammates (dims 23-50): 4×7 — mirrors cs2_observations.h:57-77 ──────
    // Teammates are in slot order (spec correction #3: original spec said distance-sorted,
    // but the header iterates in index order). Contrast: FillEnemies DOES sort by distance.
    // Dead teammates → slot stays zero (matches sim "dead teammate: all zeros").
    // ─────────────────────────────────────────────────────────────────────────

    private void FillTeammates(CCSPlayerPawn selfPawn, List<CCSPlayerController> teammates)
    {
        float selfX = selfPawn.AbsOrigin?.X ?? 0f;
        float selfY = selfPawn.AbsOrigin?.Y ?? 0f;

        int filled = 0;
        foreach (var tm in teammates)
        {
            if (filled >= 4) break;
            int baseIdx = 23 + filled * 7;
            filled++;

            var tmPawn = tm.PlayerPawn?.Value;
            if (tmPawn == null || !tm.PawnIsAlive || tmPawn.AbsOrigin == null)
                continue; // dead/invalid teammate → slot zero (already cleared)

            float dx    = tmPawn.AbsOrigin.X - selfX;
            float dy    = tmPawn.AbsOrigin.Y - selfY;
            // Angle is FROM self TO teammate — atan2(dy, dx), NOT teammate's own facing.
            // Spec correction #2: cs2_observations.h:71 uses atan2f(dy, dx).
            float angle = ObsMath.AngleTo(dx, dy);

            _buf[baseIdx + 0] = ObsMath.NormRel(dx, _map.MapDiag);
            _buf[baseIdx + 1] = ObsMath.NormRel(dy, _map.MapDiag);
            _buf[baseIdx + 2] = 0f; // z placeholder
            _buf[baseIdx + 3] = tmPawn.Health / 100f;
            _buf[baseIdx + 4] = 1f; // alive
            _buf[baseIdx + 5] = MathF.Sin(angle);
            _buf[baseIdx + 6] = MathF.Cos(angle);
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Enemies (dims 51-90): 5 × 8 — mirrors cs2_observations.h:79-126
    // Uses EnemyMemory for LOS + last-known position.
    // ─────────────────────────────────────────────────────────────────────────

    private void FillEnemies(
        CCSPlayerPawn             selfPawn,
        List<CCSPlayerController> enemies,
        EnemyMemory               enemyMem)
    {
        // ── Enemies (51-90): 5 × 8 — mirrors cs2_observations.h:79-126 ──────────
        // IMPORTANT: enemies are sorted by distance (nearest = slot 0).
        // This matches the sim's insertion sort at cs2_observations.h:82-98.
        // Contrast with FillTeammates which uses slot order (spec correction #3).
        // mem_s (EnemyMemory key) is the ORIGINAL team-slot index, not the distance rank.
        float selfX = selfPawn.AbsOrigin?.X ?? 0f;
        float selfY = selfPawn.AbsOrigin?.Y ?? 0f;

        // Build distance-sorted index array (mirrors sim's order[] array)
        // Dead enemies are included in the sort — the sim sorts all 5 regardless of alive status.
        int[] order = new int[TeamSize];
        float[] distsSq = new float[TeamSize];
        for (int s = 0; s < TeamSize; s++)
        {
            order[s] = s;
            if (s < enemies.Count)
            {
                var ep = enemies[s].PlayerPawn?.Value?.AbsOrigin;
                float dx = (ep?.X ?? selfX) - selfX;
                float dy = (ep?.Y ?? selfY) - selfY;
                distsSq[s] = dx * dx + dy * dy;
            }
            else
            {
                distsSq[s] = float.MaxValue; // empty slot → sorted to end
            }
        }
        // Insertion sort (matches sim, fine for N=5)
        for (int s = 1; s < TeamSize; s++)
        {
            int   ko = order[s];
            float kd = distsSq[s];
            int t = s - 1;
            while (t >= 0 && distsSq[t] > kd) { order[t + 1] = order[t]; distsSq[t + 1] = distsSq[t]; t--; }
            order[t + 1] = ko; distsSq[t + 1] = kd;
        }

        for (int distRank = 0; distRank < TeamSize; distRank++)
        {
            int   memSlot = order[distRank]; // original team-slot index → EnemyMemory key
            int   baseIdx = 51 + distRank * 8;
            var (lastPos, _, isAlive, canSee, everSeen) = enemyMem.Get(memSlot);

            _buf[baseIdx + 4] = isAlive ? 1f : 0f;
            _buf[baseIdx + 3] = canSee  ? 1f : 0f;

            if (canSee && memSlot < enemies.Count)
            {
                var enemyPawn = enemies[memSlot].PlayerPawn?.Value;
                if (enemyPawn?.AbsOrigin != null)
                {
                    float dx   = enemyPawn.AbsOrigin.X - selfX;
                    float dy   = enemyPawn.AbsOrigin.Y - selfY;
                    float dist = MathF.Sqrt(dx * dx + dy * dy);
                    float ang  = ObsMath.AngleTo(dx, dy); // atan2(dy,dx) — angle FROM self TO enemy

                    _buf[baseIdx + 0] = ObsMath.NormRel(dx, _map.MapDiag);
                    _buf[baseIdx + 1] = ObsMath.NormRel(dy, _map.MapDiag);
                    _buf[baseIdx + 2] = 0f; // z placeholder
                    _buf[baseIdx + 5] = MathF.Sin(ang);
                    _buf[baseIdx + 6] = MathF.Cos(ang);
                    _buf[baseIdx + 7] = ObsMath.NormRel(dist, _map.MapDiag);
                }
            }
            else if (!canSee && everSeen && lastPos != null)
            {
                // Use last-known position (stale). No angle/dist for stale — stays 0.
                // Matches sim's enemy_mem_idx branch (cs2_observations.h:119-124).
                float mx = lastPos.X - selfX;
                float my = lastPos.Y - selfY;
                _buf[baseIdx + 0] = ObsMath.NormRel(mx, _map.MapDiag);
                _buf[baseIdx + 1] = ObsMath.NormRel(my, _map.MapDiag);
            }
            // !everSeen → entire slot stays zero (matches sim INVALID_AREA_IDX path)
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Global / bomb (dims 91-103) — mirrors cs2_observations.h:128-169
    // ─────────────────────────────────────────────────────────────────────────

    private void FillBombAndGlobal(
        CCSPlayerController       bot,
        CCSPlayerPawn             selfPawn,
        BombState                 bomb,
        List<CCSPlayerController> allPlayers,
        float                     roundTimeTotal,
        float                     roundTicksLeft)
    {
        float selfX = selfPawn.AbsOrigin?.X ?? 0f;
        float selfY = selfPawn.AbsOrigin?.Y ?? 0f;
        bool  selfIsT = bot.TeamNum == Terrorist;

        // obs[91]: round time remaining fraction (cs2_observations.h:129)
        _buf[91] = roundTimeTotal > 0f ? roundTicksLeft / roundTimeTotal : 0f;

        // obs[92-95]: bomb status one-hot (carried-by-self, by-teammate, dropped, planted)
        // Mirrors sim logic at cs2_observations.h:131-139
        if (!bomb.Planted && !bomb.Dropped)
        {
            if (selfIsT && HasC4(selfPawn))
                _buf[92] = 1f; // carried by self
            else if (selfIsT && bomb.CarrierId >= 0)
                _buf[93] = 1f; // carried by a T teammate
        }
        else if (bomb.Dropped)
            _buf[94] = 1f;
        else if (bomb.Planted)
            _buf[95] = 1f;

        // obs[96-98]: bomb position relative to self (cs2_observations.h:141-152)
        if (bomb.Planted || bomb.Dropped)
        {
            _buf[96] = ObsMath.NormRel(bomb.BombX - selfX, _map.MapDiag);
            _buf[97] = ObsMath.NormRel(bomb.BombY - selfY, _map.MapDiag);
        }
        else if (selfIsT && bomb.CarrierId >= 0 && !HasC4(selfPawn))
        {
            // Teammate carrying: show their position (cs2_observations.h:145-150)
            var carrier     = allPlayers.FirstOrDefault(p => p.IsValid && p.Slot == bomb.CarrierId);
            var carrierPawn = carrier?.PlayerPawn?.Value;
            if (carrierPawn?.AbsOrigin != null)
            {
                _buf[96] = ObsMath.NormRel(carrierPawn.AbsOrigin.X - selfX, _map.MapDiag);
                _buf[97] = ObsMath.NormRel(carrierPawn.AbsOrigin.Y - selfY, _map.MapDiag);
            }
        }
        _buf[98] = 0f; // z placeholder

        // obs[99]: bomb timer remaining fraction (cs2_observations.h:153-154)
        // Confirmed: CPlantedC4.C4Blow = server time at detonation; TimerLength = total fuse seconds.
        _buf[99] = bomb.Planted && bomb.BlowTime > 0f && bomb.TimerLength > 0f
            ? Math.Clamp((bomb.BlowTime - Server.CurrentTime) / bomb.TimerLength, 0f, 1f)
            : 0f;

        // obs[100]: bomb plant-in-progress fraction (cs2_observations.h:155-156)
        // Planting is very brief (~3 s) and not easily tracked via CSS events alone;
        // leave at 0 for now (safe: the policy was trained with sparse plant-progress signal).

        // obs[101]: defuse progress (cs2_observations.h:157-167)
        // Confirmed: CPlantedC4.BeingDefused (bool), DefuseCountDown (float, time remaining),
        // DefuseLength (float, total defuse duration). Progress = 1 - countDown / length.
        if (bomb.Planted)
        {
            var c4 = Utilities.FindAllEntitiesByDesignerName<CPlantedC4>("planted_c4").FirstOrDefault();
            if (c4 != null && c4.BeingDefused && c4.DefuseLength > 0f)
                _buf[101] = Math.Clamp(1f - (c4.DefuseCountDown / c4.DefuseLength), 0f, 1f);
        }

        // obs[102-103]: alive counts normalized by team size (cs2_observations.h:168-169)
        int tAlive  = allPlayers.Count(p => p.IsValid && p.TeamNum == Terrorist        && p.PawnIsAlive);
        int ctAlive = allPlayers.Count(p => p.IsValid && p.TeamNum == CounterTerrorist && p.PawnIsAlive);
        _buf[102] = tAlive  / (float)TeamSize;
        _buf[103] = ctAlive / (float)TeamSize;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Post-processing
    // ─────────────────────────────────────────────────────────────────────────

    private void ClipAll()
    {
        // Clip every dimension to [-5, 5] — cs2_observations.h:171-175
        for (int k = 0; k < _obsDim; k++)
            _buf[k] = ObsMath.Clip(_buf[k]);
    }
}
