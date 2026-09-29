// deploy/CS2RLBot/ObsMath.cs
namespace CS2RLBot;

/// <summary>
/// Pure math helpers for observation normalization.
/// All methods are stateless and side-effect free — no CSS types.
/// Matches formulas in src/cs2rl/env/c/cs2_observations.h exactly.
/// </summary>
internal static class ObsMath
{
    /// <summary>Degrees-to-radians factor used for yaw conversion.</summary>
    internal const float Deg2Rad = MathF.PI / 180f;

    /// <summary>
    /// Clip to [-5, 5] — applied to every obs dim after building, matches sim.
    /// Formula from cs2_observations.h:172-175.
    /// </summary>
    internal static float Clip(float v) => Math.Clamp(v, -5f, 5f);

    /// <summary>
    /// Absolute position normalization: x * inv_range - offset.
    /// Formula from cs2_observations.h:33-34 and cs2_env.py:312-314.
    /// CRITICAL: NOT (x - offset) * inv_range — the offset is applied after multiplication.
    /// These two forms are NOT equivalent (different results when inv_range ≠ 1).
    /// </summary>
    internal static float NormAbs(float world, float invRange, float offset)
        => world * invRange - offset;

    /// <summary>
    /// Normalize a delta (relative position or distance) by dividing by a range value.
    /// Most commonly called with mapDiag for entity-to-entity distances (cs2_observations.h:66).
    /// Can also be called with any fixed divisor (e.g. 250f for velocity).
    /// Guard: returns 0 if divisor is zero — happens if MapConstants were never loaded
    /// (uninitialized map data on first tick before plugin fully starts up).
    /// </summary>
    internal static float NormRel(float delta, float divisor)
        => divisor > 0f ? delta / divisor : 0f;

    /// <summary>
    /// Angle FROM self TO target, in radians, using atan2(dy, dx).
    /// Matches cs2_observations.h:71 and :115 — this is NOT the target's own facing angle.
    /// </summary>
    internal static float AngleTo(float dx, float dy) => MathF.Atan2(dy, dx);

    /// <summary>
    /// Convert CS2 yaw (degrees) to radians for sin/cos facing.
    /// CS2 yaw convention: clockwise-positive — east=0°, south=90°, west=±180°, north=-90°.
    /// This is the opposite of standard math convention; keep this in mind when interpreting
    /// sin/cos results.
    /// </summary>
    internal static float YawToRad(float yawDeg) => yawDeg * Deg2Rad;
}
