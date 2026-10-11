"""The sim's hit-model geometry for Python aimers: eye heights and the point to aim at.

WHAT: ``hit_geometry(hitboxes)`` returns a ``HitGeometry`` for one of the two hit models
(Dust2Env.hit_model, set from ``EnvConfig.hitboxes``): the shooter's eye above its feet per
stance, and the aim point on a target per stance, in the target's (fwd, left, up) frame.
  * the capsule hitboxes (EnvConfig's default): CS:GO's eye heights (cs2_hitboxes.h
    HB_EYE_*) and the midpoint of the head capsule's axis in the target's pose
    (cs2_hitbox_table.h);
  * the legacy ellipsoid: cs2_combat.h's EYE_HEIGHT_* and the torso centre TORSO_OFFSET_*
    on the target's axis.
Every number is read from ``binding.hitbox_table()``, so no Python copy of a C constant exists
to drift (eval.baselines held one, pinned by a regex over the header, until #391).

WHY the head midpoint: rays from the CS eye aimed exactly at it hit the head on every ray in
all four stance pairs, d in [100, 1000] u at every target facing; a level (pinned-pitch) ray
at a same-stance target hit every time too, on the head or neck (#391, first capsule entered).
A target's eye on its vertical axis does not: the crouched head sits 6.2 u behind the axis,
and a standing shooter aiming there hit 0.3151 of the rays.

Consumers: eval.baselines' ``OracleActor``, experiment.oracle_statue's ``ObsOracleActor``,
experiment.aim_eval's miss decomposition, viz.render's eye pose and
scripts/sim_fingerprint.py's track aim. ``eval.baselines.StateReader`` snapshots carry the
env's switch as ``st["hitboxes"]``.

PITFALL: an aimer that does not know the target's facing (the observation carries none) can
only use ``target_z`` on the axis. That is exact for a standing target; on a crouched one it
hit 0.4579 of the rays (#391; a KNOWN LIMIT for ObsOracleActor, whose arena runs crouch off).
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass

from cs2rl.env.c import binding

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class HitGeometry:
    """One hit model's eye heights and aim points; index 0 is standing, 1 crouched.

    ``eye`` is the eye above the feet. ``target`` is agent-local (fwd, left, up) with the feet
    at the origin and fwd = (cos facing, sin facing), the frame of cs2_hitbox_table.h.
    """
    hitboxes: bool
    eye: tuple[float, float]
    target: tuple[Vec3, Vec3]

    def eye_z(self, z: float, crouched) -> float:
        """The world z of an agent's eye: its feet ``z`` plus the stance's eye height."""
        return z + self.eye[1 if crouched else 0]

    def target_xyz(self, x: float, y: float, z: float, facing: float, crouched) -> Vec3:
        """The world aim point on a target at feet (x, y, z) facing ``facing``.

        For the legacy model the point is on the axis, so this returns (x, y, z + torso)
        exactly: the rotated offsets are zeros.
        """
        f, left, up = self.target[1 if crouched else 0]
        c, s = math.cos(facing), math.sin(facing)
        return x + f * c - left * s, y + f * s + left * c, z + up

    def target_z(self, z: float, crouched) -> float:
        """The aim point's world z alone, for an aimer that cannot see the target's facing."""
        return z + self.target[1 if crouched else 0][2]


def hit_geometry(hitboxes) -> HitGeometry:
    """The geometry of the capsule model (truthy) or the legacy ellipsoid (falsy), from C.

    Any truthy value works (a snapshot's bool, a numpy bool, a config int): it is bool()-ed
    before the cache, which a numpy 0-d array could not key.
    """
    return _hit_geometry(bool(hitboxes))


@functools.cache
def _hit_geometry(hitboxes: bool) -> HitGeometry:
    table = binding.hitbox_table()
    if not hitboxes:
        eye_s, eye_c = table["legacy_eye"]
        torso_s, torso_c = table["legacy_torso"]
        return HitGeometry(False, (float(eye_s), float(eye_c)),
                           ((0.0, 0.0, float(torso_s)), (0.0, 0.0, float(torso_c))))
    head = table["head"]

    def midpoint(pose: int) -> Vec3:
        ax, ay, az, bx, by, bz, _r = table["caps"][pose][head]
        return (ax + bx) / 2.0, (ay + by) / 2.0, (az + bz) / 2.0

    eye_s, eye_c = table["eye"]
    return HitGeometry(True, (float(eye_s), float(eye_c)), (midpoint(0), midpoint(1)))
