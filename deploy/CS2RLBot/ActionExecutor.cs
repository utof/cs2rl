using CounterStrikeSharp.API;
using CounterStrikeSharp.API.Core;
using CounterStrikeSharp.API.Modules.Utils;

namespace CS2RLBot;

/// <summary>
/// Translates 7 discrete action integers + 1 continuous Δyaw into CS2 bot inputs.
/// Batch 3: aim is a continuous head — no more 16-bin LUT, no more 4-tick lerp.
/// Stateless: per-tick Δyaw makes host-side smoothing redundant.
/// </summary>
public sealed class ActionExecutor
{
    // ── Static lookup tables (built once for all instances) ───────────────────
    //
    // Batch 3 deletions (intentionally absent):
    //   - AimBinsDeg LUT: replaced by continuous Δyaw radians from the policy.
    //   - _prevYaw / _targetYaw / _aimStep: 4-tick lerp dropped — sim emits a
    //     clamped per-inference-tick Δyaw, so host smoothing is redundant.

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

    // ── Execute ───────────────────────────────────────────────────────────────
    /// <summary>
    /// Translates 7 discrete action integers + 1 continuous Δyaw into CS2 bot inputs.
    /// Batch 3: aim is a continuous head — no more 16-bin LUT, no more 4-tick lerp.
    /// The sim emits a per-inference-tick clamped Δyaw and the policy's tanh-squashed
    /// μ stays in [-max_turn_speed, +max_turn_speed] = [-π/4, +π/4] rad = [-45°, +45°].
    /// </summary>
    /// <param name="bot">Needed for ExecuteClientCommandFromServer (weapon_switch).</param>
    /// <param name="pawn">Needed for movement, angle, button state writes.</param>
    /// <param name="actions">7-element int array (move, shoot, reload, weapon, use, crouch, jump).
    ///   Note: HEAD_AIM removed in Batch 3 — was index 1 in the old 8-element layout.</param>
    /// <param name="deltaYawRad">Continuous Δyaw output from the policy, in radians,
    ///   tanh-squashed and scaled to [-π/4, +π/4]. Applied as: new_yaw = wrap(current_yaw + Δyaw).
    ///   Pitch is preserved from pawn.EyeAngles.X (NOT zeroed — would visibly snap the bot's view).</param>
    /// <param name="isInferenceTick">True every 4th server tick (16 Hz). Gates Δyaw application
    ///   and weapon_switch. Δyaw MUST only be applied once per inference cycle to match
    ///   training semantics (see comment block below).</param>
    public void Execute(
        CCSPlayerController bot,
        CCSPlayerPawn       pawn,
        int[]               actions,
        float               deltaYawRad,
        bool                isInferenceTick)
    {
        var ms = pawn.MovementServices;
        if (ms == null) return;
        if (actions.Length < 7) return;
        ref ulong btns = ref ms.Buttons.ButtonStates[0];

        // ── Head 0: move (9 options) ─────────────────────────────────────────
        int moveAction = Math.Clamp(actions[0], 0, MoveLUT.Length - 1);
        var (set, clear) = MoveLUT[moveAction];
        btns = (btns & ~clear) | set;

        // ── Aim (continuous Δyaw, Batch 3) ───────────────────────────────────
        // Apply Δyaw ONLY on inference ticks (every 4th server tick at 16 Hz)
        // to match training semantics: the policy is trained with one Δyaw per
        // env step (= one inference call), and the C env consumes Δyaw exactly
        // once per step (see src/cs2rl/env/c/cs2_env.h aim consumption block:
        //   a->facing = wrap_pi(a->facing + clamped);  // fires once per step).
        // Re-applying the cached Δyaw on every server tick would 4× over-rotate
        // (e.g. 45° Δyaw → 180° per inference cycle). On non-inference ticks
        // the bot's yaw stays at whatever the previous inference Teleport set.
        if (isInferenceTick)
        {
            // Defensive clamp: ONNX should emit only tanh-squashed values, but
            // belt-and-braces against numeric drift / quantisation.
            const float MAX_TURN_RAD = (float)(Math.PI / 4.0);
            deltaYawRad = Math.Clamp(deltaYawRad, -MAX_TURN_RAD, MAX_TURN_RAD);
            float deltaYawDeg = deltaYawRad * 180f / MathF.PI;
            QAngle current    = pawn.EyeAngles;                       // X=pitch, Y=yaw (Source convention)
            float newYaw      = current.Y + deltaYawDeg;
            newYaw            = newYaw - 360f * MathF.Floor((newYaw + 180f) / 360f); // wrap to [-180, +180]
            pawn.Teleport(null, new QAngle(current.X, newYaw, 0f), null);
            //                              ^^^^^^^^^ pitch preserved — NOT 0.
            // (current.X stays from the previous frame; sim is 2D yaw-only and
            // doesn't drive pitch, so it tracks whatever the human/admin/bot AI
            // set it to. Batch 3.5 #24 adds Δpitch as a second continuous head
            // and this becomes `current.X + deltaPitchDeg` instead.)
        }

        // ── Head 1: shoot (2) — was Head 2 in the 8-element layout ───────────
        if (actions[1] == 1) btns |=  (ulong)PlayerButtons.Attack;
        else                 btns &= ~(ulong)PlayerButtons.Attack;

        // ── Head 2: reload (2) — was Head 3 ──────────────────────────────────
        if (actions[2] == 1) btns |=  (ulong)PlayerButtons.Reload;
        else                 btns &= ~(ulong)PlayerButtons.Reload;

        // ── Head 3: weapon_switch (3) — was Head 4 ───────────────────────────
        // Only fire on inference ticks to avoid spamming the command buffer.
        if (isInferenceTick && actions[3] != 0)
            bot.ExecuteClientCommandFromServer(actions[3] == 1 ? "slot1" : "slot2");

        // ── Head 4: use (2) — was Head 5 ─────────────────────────────────────
        if (actions[4] == 1) btns |=  (ulong)PlayerButtons.Use;
        else                 btns &= ~(ulong)PlayerButtons.Use;

        // ── Head 5: crouch (2) — was Head 6 ──────────────────────────────────
        if (actions[5] == 1) btns |=  (ulong)PlayerButtons.Duck;
        else                 btns &= ~(ulong)PlayerButtons.Duck;

        // ── Head 6: jump (2) — was Head 7 ────────────────────────────────────
        if (actions[6] == 1) btns |=  (ulong)PlayerButtons.Jump;
        else                 btns &= ~(ulong)PlayerButtons.Jump;
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
