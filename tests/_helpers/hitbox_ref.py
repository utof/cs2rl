"""A numpy reference for the hit model's ray test (cs2_hitboxes.h), the oracle of test_hitboxes.

WHAT
  ray_capsule_t(ro, rd, a, b, r): the entry distance t >= 0 of each ray into ONE capsule, np.inf
    for a miss. Rays are vectorised: ro, rd (N, 3), rd unit length. The infinite cylinder around
    a-b first; its entry counts when it lands between the two end planes; otherwise the nearer
    of the two end spheres (Inigo Quilez's capsule intersection, the form cs2_hitboxes.h
    hb_capsule_t ports). A zero-length capsule is a sphere, and a ray parallel to the axis enters
    only the caps.
  first_region(ro, rd, caps, head, neck): the region the hit model gives each ray: the capsule
    entered first (the lower index on an exact tie), then the neck rule (a neck-first ray that
    also enters the head counts as the head); -1 for a miss. Also returns the entry t and a
    margin per ray, for excluding rays that float32 and float64 may decide differently.
WHY a second implementation: the C test is static inline float32 under -ffast-math, reachable
  from Python only through binding.hitbox_first_region. A reference written independently in
  float64 catches an error in the C branches (a dropped cap test, a wrong end-plane check), which
  a test that reads its expectations off the same C would not. Phase 1 of #391 checked this form
  against brute-force ray marching on 1000 rays (0 disagreements) and the C form against it on
  199712 rays (0 mismatches away from a boundary).
PITFALL: the capsules come from binding.hitbox_table(), the C table itself, so this checks the
  ray test, not the table's numbers (scripts/hitboxes/gen_table.py --check owns those). Rays
  within the margin of a capsule surface or of a two-capsule tie are excluded by the caller:
  float32 and float64 can disagree there, and either answer is right.
"""
import numpy as np


def ray_capsule_t(ro, rd, a, b, r):
    """Entry distance (N,) of rays ro + t rd into the capsule (a, b, r); np.inf on a miss."""
    ro = np.atleast_2d(np.asarray(ro, dtype=np.float64))
    rd = np.atleast_2d(np.asarray(rd, dtype=np.float64))
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    t_out = np.full(len(ro), np.inf)

    def sphere(c):
        oc = ro - c
        bb = np.einsum("ij,ij->i", rd, oc)
        h = bb * bb - (np.einsum("ij,ij->i", oc, oc) - r * r)
        t = -bb - np.sqrt(np.maximum(h, 0.0))
        return np.where((h >= 0.0) & (t >= 0.0), t, np.inf)

    ba = b - a
    baba = float(ba @ ba)
    if baba < 1e-12:                   # a == b: a sphere
        return sphere(a)
    oa = ro - a
    bard = rd @ ba
    baoa = oa @ ba
    rdoa = np.einsum("ij,ij->i", rd, oa)
    oaoa = np.einsum("ij,ij->i", oa, oa)
    qa = baba - bard * bard
    qb = baba * rdoa - baoa * bard
    qc = baba * oaoa - baoa * baoa - r * r * baba
    h = qb * qb - qa * qc
    par = qa < 1e-9 * baba             # parallel to the axis: only the caps can be entered
    with np.errstate(divide="ignore", invalid="ignore"):
        t_body = (-qb - np.sqrt(np.maximum(h, 0.0))) / qa
    y = baoa + t_body * bard
    body = (~par) & (h >= 0.0) & (y > 0.0) & (y < baba) & (t_body >= 0.0)
    t_out[body] = t_body[body]
    rest = ~body & (par | (h >= 0.0))
    if rest.any():
        t_out[rest] = np.minimum(sphere(a), sphere(b))[rest]
    return t_out


def _surface_gap(ro, rd, a, b, r, samples=33):
    """|closest distance from each ray to the segment a-b - r| (the segment sampled at
    `samples` points, the ray clamped to t >= 0): how near the ray passes to the surface."""
    q = a[None, :] + np.linspace(0.0, 1.0, samples)[:, None] * (b - a)[None, :]
    w = q[None, :, :] - ro[:, None, :]
    t = np.maximum(np.einsum("nki,ni->nk", w, rd), 0.0)
    dist = np.linalg.norm(w - t[:, :, None] * rd[:, None, :], axis=2).min(axis=1)
    return np.abs(dist - r)


def first_region(ro, rd, caps, head, neck):
    """(region (N,), t (N,), margin (N,)) for agent-local rays against one pose's ``caps``
    (rows ax, ay, az, bx, by, bz, r). region is -1 and t np.inf on a miss. margin is the
    smaller of the gap between the two nearest entries and the nearest pass by any capsule
    surface (np.inf when neither exists)."""
    ro = np.atleast_2d(np.asarray(ro, dtype=np.float64))
    rd = np.atleast_2d(np.asarray(rd, dtype=np.float64))
    rows = [np.asarray(c, dtype=np.float64) for c in caps]
    t_all = np.stack([ray_capsule_t(ro, rd, c[0:3], c[3:6], c[6]) for c in rows], axis=1)
    region = np.argmin(t_all, axis=1)                  # argmin returns the first minimum: the lower index wins
    t = t_all[np.arange(len(ro)), region]
    region = np.where(np.isfinite(t), region, -1)
    region = np.where((region == neck) & np.isfinite(t_all[:, head]), head, region)
    ts = np.sort(t_all, axis=1)
    with np.errstate(invalid="ignore"):                # inf - inf where both miss; np.where drops it
        margin = np.where(np.isfinite(ts[:, 1]), ts[:, 1] - ts[:, 0], np.inf)
    for c in rows:
        margin = np.minimum(margin, _surface_gap(ro, rd, c[0:3], c[3:6], c[6]))
    return region, t, margin
