using CounterStrikeSharp.API;
using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Modules.Utils;

namespace CS2RLBot;

/// <summary>
/// Translates 8 discrete action integers into CS2 bot inputs.
/// Stateful: owns aim interpolation state for one bot instance.
/// </summary>
public sealed class ActionExecutor
{
    // ── Static lookup tables (built once for all instances) ───────────────────

    // AimBinsDeg[i] = -180 + i * 22.5°  →  range [-180°, +168.75°]
    // Index 8 = 0° (no change). With the current 2-element aim head, only indices 0 and 1 are used.
    private static readonly float[] AimBinsDeg = Enumerable
        .Range(0, 16)
        .Select(i => -180f + i * 22.5f)
        .ToArray();

    // MoveLUT[moveAction] = (bits to set, bits to clear) in ButtonStates[0]
    private static readonly (ulong Set, ulong Clear)[] MoveLUT;
    private static readonly ulong MoveMask;

    static ActionExecutor()
    {
        ulong F = (ulong)PlayerButtons.Forward;
        ulong B = (ulong)PlayerButtons.Back;
        ulong L = (ulong)PlayerButtons.Moveleft;
        ulong R = (ulong)PlayerButtons.Moveright;
        MoveMask = F | B | L | R;

        MoveLUT = new (ulong, ulong)[9];
        MoveLUT[0] = (0,     MoveMask); // stop
        MoveLUT[1] = (F,     MoveMask); // N  (forward)
        MoveLUT[2] = (F | R, MoveMask); // NE
        MoveLUT[3] = (R,     MoveMask); // E  (strafe right)
        MoveLUT[4] = (B | R, MoveMask); // SE
        MoveLUT[5] = (B,     MoveMask); // S  (backward)
        MoveLUT[6] = (B | L, MoveMask); // SW
        MoveLUT[7] = (L,     MoveMask); // W  (strafe left)
        MoveLUT[8] = (F | L, MoveMask); // NW
    }

    // ── Per-bot aim interpolation state ──────────────────────────────────────
    private float _prevYaw;
    private float _targetYaw;
    private int   _aimStep;   // increments each tick; resets to 0 on each inference tick

    // ── Execute ───────────────────────────────────────────────────────────────
    /// <param name="bot">Needed for ExecuteClientCommandFromServer (weapon_switch).</param>
    /// <param name="pawn">Needed for movement, angle, button state writes.</param>
    /// <param name="actions">8-element int array (move, aim, shoot, reload, weapon, use, crouch, jump).</param>
    /// <param name="isInferenceTick">True every 4th server tick (16 Hz). Controls aim target update + weapon_switch gate.</param>
    public void Execute(
        CCSPlayerController bot,
        CCSPlayerPawn       pawn,
        int[]               actions,
        bool                isInferenceTick)
    {
        var ms = pawn.MovementServices;
        if (ms == null) return;

        if (actions.Length < 8) return;

        ref ulong btns = ref ms.Buttons.ButtonStates[0];

        // ── Head 0: move (9 options) ─────────────────────────────────────────
        int moveAction = Math.Clamp(actions[0], 0, MoveLUT.Length - 1);
        var (set, clear) = MoveLUT[moveAction];
        btns = (btns & ~clear) | set;

        // ── Head 1: aim (16 bins) ────────────────────────────────────────────
        // Guard: if current checkpoint has a smaller aim head (e.g. size 2), actions[1] is 0 or 1.
        // AimBinsDeg is 16 elements; indices 0 and 1 map to -180° and -157.5°. Still valid.
        if (isInferenceTick)
        {
            _prevYaw   = pawn.EyeAngles.Y;
            _targetYaw = _prevYaw + AimBinsDeg[Math.Min(actions[1], AimBinsDeg.Length - 1)];
            _aimStep   = 0;
        }
        _aimStep++;
        float t   = Math.Clamp(_aimStep / 4f, 0f, 1f);
        float yaw = _prevYaw + t * (_targetYaw - _prevYaw); // lerp
        // Normalize to [-180, 180]
        yaw = yaw - 360f * MathF.Floor((yaw + 180f) / 360f);
        pawn.Teleport(null, new QAngle(pawn.EyeAngles.X, yaw, 0f), null);

        // ── Head 2: shoot (2) ────────────────────────────────────────────────
        if (actions[2] == 1)
            btns |= (ulong)PlayerButtons.Attack;
        else
            btns &= ~(ulong)PlayerButtons.Attack;

        // ── Head 3: reload (2) ───────────────────────────────────────────────
        if (actions[3] == 1)
            btns |= (ulong)PlayerButtons.Reload;
        else
            btns &= ~(ulong)PlayerButtons.Reload;

        // ── Head 4: weapon_switch (3): 0=none 1=primary 2=secondary ──────────
        // Only fire on inference ticks to avoid spamming the command buffer.
        if (isInferenceTick && actions[4] != 0)
            bot.ExecuteClientCommandFromServer(actions[4] == 1 ? "slot1" : "slot2");

        // ── Head 5: use (2) — plant/defuse ───────────────────────────────────
        if (actions[5] == 1)
            btns |= (ulong)PlayerButtons.Use;
        else
            btns &= ~(ulong)PlayerButtons.Use;

        // ── Head 6: crouch (2) ───────────────────────────────────────────────
        if (actions[6] == 1)
            btns |= (ulong)PlayerButtons.Duck;
        else
            btns &= ~(ulong)PlayerButtons.Duck;

        // ── Head 7: jump (2) ─────────────────────────────────────────────────
        if (actions.Length > 7 && actions[7] == 1)
            btns |= (ulong)PlayerButtons.Jump;
        else
            btns &= ~(ulong)PlayerButtons.Jump;
    }

    /// <summary>Returns the argmax of a logit array.</summary>
    public static int Argmax(float[] arr)
    {
        int best = 0;
        for (int i = 1; i < arr.Length; i++)
            if (arr[i] > arr[best]) best = i;
        return best;
    }
}
