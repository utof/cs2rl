// deploy/CS2RLBot/ObsMath.cs
namespace CS2RLBot;

/// <summary>
/// Pure math helpers for observation normalization.
/// All methods are stateless and side-effect free — no CSS types.
/// Matches formulas in src/c_env/cs2_observations.h exactly.
/// </summary>
internal static class ObsMath
{
    internal const float Deg2Rad = MathF.PI / 180f;

    /// <summary>Clip to (-5, 5) — applied to every obs dim, matches sim.</summary>
    internal static float Clip(float v) => Math.Clamp(v, -5f, 5f);

    /// <summary>
    /// Absolute position normalization: x * inv_range - offset.
    /// Formula from cs2_observations.h:32 and cs2_env.py:312-314.
    /// Note: NOT (x - offset) * inv_range — the offset is applied after multiplication.
    /// </summary>
    internal static float NormAbs(float world, float invRange, float offset)
        => world * invRange - offset;

    /// <summary>Relative position/distance normalized by map diagonal.</summary>
    internal static float NormRel(float delta, float mapDiag)
        => mapDiag > 0f ? delta / mapDiag : 0f;

    /// <summary>
    /// Angle FROM self TO target, in radians, using atan2(dy, dx).
    /// Matches cs2_observations.h:71 and :115 — this is NOT the target's facing angle.
    /// </summary>
    internal static float AngleTo(float dx, float dy) => MathF.Atan2(dy, dx);

    /// <summary>Convert CS2 yaw (degrees) to radians for sin/cos facing.</summary>
    internal static float YawToRad(float yawDeg) => yawDeg * Deg2Rad;
}
