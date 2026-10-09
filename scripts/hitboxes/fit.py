"""CS2-like per-region capsule hitboxes fitted to the Quaternius Universal Animation Library
mannequin (CC0 1.0), posed for the sim's two stances, in sim units (#391).

CLI, from the repository root (prints the skinning self-check, the scale and the region table):
    python -m scripts.hitboxes.fit --glb <dir>/AnimationLibrary_Godot_Standard.glb

WHAT
  * load_model(): reads the GLB after checking its sha256 against ASSET_GLB_SHA256, so a different
    download can never silently produce a different table.
  * Model.poses(): bone world matrices (model space: metres, +Y up, +Z forward, +X = the
    character's left) for each sim stance. "stand" = Pistol_Aim_Neutral at t=0, both hands
    forward on the grip: the closest free clip to CS's weapon-holding upper body (the pack has no
    rifle pose). "crouch" is LAYERED like CS's aim layer over locomotion: the legs
    (thigh/shin/foot/toe chains) and the hips TRANSLATION come from Crouch_Idle_Loop at t=0, every
    other bone keeps the stand pose's world rotation and is shifted by the hips' translation delta.
  * fit_regions(): one capsule per region, fitted ONCE on the bind (rest) pose to the region's
    dominant-joint vertex cluster and stored in its anchor bone's frame (CS2 stores a hitbox as two
    points in its parent bone's space plus a radius, per a decompiled CS2 agent model), then
    carried to any pose by that bone's world matrix -- a rigid hitbox, exactly what LBS does to a
    point weighted 100% to one bone.
  * model_to_sim() / sim_table(): model space -> agent-local sim units (fwd, left, up), feet at the
    origin, with ONE uniform scale s chosen so the stand pose's mesh top is 72 u (scale_for).
WHY rest-pose fitting: the dominant-joint clusters are least deformed in the bind pose (straight
  limbs); fitting per pose would give each pose different radii, which CS2 does not (one hitbox set).
WHY these numbers (measured in #391 phase 1):
  * 72 u is CS:GO's standing hull top (VEC_HULL_MAX). One scale for every pose keeps the head the
    same size in both stances; the layered crouch then tops out at 55.45 u, near CS:GO's 54 u
    crouch hull, where the free crouch clip alone (deeper than CS's crouch) tops out at 44.34 u.
  * RADIUS_PERCENTILE 75 was pre-registered as "the percentile with the highest mean orthographic
    silhouette IoU against the skinned mesh over stand/crouch x front/side": p75 0.8253, p90
    0.8013, p97 0.7615 (today's ellipsoid scored 0.36, standing front-on).
  * CS2's real per-hitbox dimensions are not public, so these are anatomy fits, not Valve data.
PITFALLS
  * gltf_pose's joint names keep the dots ("DEF-spine.001"); three.js renames them, we do not.
  * The mannequin's .L side is +X (model) = sim LEFT; region "_l" = the character's left.
  * Layering keeps the stand pose's upper-body WORLD rotations; it does not re-solve the legs, so
    the pelvis/thigh seam can be slightly discontinuous. Irrelevant for a hit model.
  * REGIONS' order is the C table's region order, and the C hit test breaks an exact tie in entry
    distance by the LOWER index: head must stay first.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from . import gltf_pose as gp

# The asset is not committed: download the zip, unpack it, and pass the GLB with --glb.
ASSET_ZIP_URL = "https://opengameart.org/sites/default/files/universal_animation_librarystandard.zip"
ASSET_ZIP_SHA256 = "18ff1a7215f4852b320203e8aaf02a1578b5c8eef9027fbaedfcedc7b85a3ac2"
ASSET_GLB_MEMBER = "Animation Library[Standard]/Godot/AnimationLibrary_Godot_Standard.glb"
ASSET_GLB_SHA256 = "1b7bf67866360665426bb99e4c71bd619f19b408453c24e30f0c3071601eee5c"
ASSET_NAME = "Quaternius Universal Animation Library [Standard]"
ASSET_LICENSE = "CC0 1.0 Universal (Public Domain Dedication), Models by @Quaternius"
HULL_TOP_STAND = 72.0                  # CS:GO VEC_HULL_MAX z
STAND_CLIP, STAND_T = "Pistol_Aim_Neutral", 0.0
CROUCH_CLIP, CROUCH_T = "Crouch_Idle_Loop", 0.0
RADIUS_PERCENTILE = 75.0
END_PERCENTILES = (1.0, 99.0)
LEG_PREFIXES = ("DEF-thigh", "DEF-shin", "DEF-foot", "DEF-toe")

# name, hit group, dominant-joint cluster, anchor bone, axis rule
#   ("seg", A, B): axis A->B (rest pose joints), line through the cluster's robust centre
#   ("bone", axis_index): the anchor bone's local axis (0 = X lateral, 1 = Y along the bone)
_FINGERS = ("f_index", "f_middle", "f_ring", "f_pinky", "thumb")


def _hand(side):
    return [f"DEF-hand.{side}"] + [f"DEF-{f}.0{k}.{side}" for f in _FINGERS for k in (1, 2, 3)]


REGIONS = [
    ("head", "HEAD", ["DEF-head"], "DEF-head", ("bone", 1)),
    ("neck", "NECK", ["DEF-neck"], "DEF-neck", ("seg", "DEF-neck", "DEF-head")),
    ("upper_chest", "CHEST", ["DEF-spine.003", "DEF-shoulder.L",
                              "DEF-shoulder.R"], "DEF-spine.003", ("bone", 0)),
    ("chest", "CHEST", ["DEF-spine.002"], "DEF-spine.002", ("bone", 0)),
    ("stomach", "STOMACH", ["DEF-spine.001"], "DEF-spine.001", ("bone", 0)),
    ("pelvis", "STOMACH", ["DEF-hips"], "DEF-hips", ("bone", 0)),
]
for _s, _side in (("l", "L"), ("r", "R")):
    _arm = "LEFTARM" if _side == "L" else "RIGHTARM"
    _leg = "LEFTLEG" if _side == "L" else "RIGHTLEG"
    REGIONS += [
        (f"upper_arm_{_s}", _arm, [f"DEF-upper_arm.{_side}"], f"DEF-upper_arm.{_side}",
         ("seg", f"DEF-upper_arm.{_side}", f"DEF-forearm.{_side}")),
        (f"forearm_{_s}", _arm, [f"DEF-forearm.{_side}"], f"DEF-forearm.{_side}",
         ("seg", f"DEF-forearm.{_side}", f"DEF-hand.{_side}")),
        (f"hand_{_s}", _arm, _hand(_side), f"DEF-hand.{_side}", ("seg", f"DEF-hand.{_side}",
                                                                 f"DEF-f_middle.03.{_side}")),
        (f"thigh_{_s}", _leg, [f"DEF-thigh.{_side}"], f"DEF-thigh.{_side}",
         ("seg", f"DEF-thigh.{_side}", f"DEF-shin.{_side}")),
        (f"calf_{_s}", _leg, [f"DEF-shin.{_side}"], f"DEF-shin.{_side}",
         ("seg", f"DEF-shin.{_side}", f"DEF-foot.{_side}")),
        (f"foot_{_s}", _leg, [f"DEF-foot.{_side}", f"DEF-toe.{_side}"], f"DEF-foot.{_side}",
         ("seg", f"DEF-foot.{_side}", f"DEF-toe.{_side}")),
    ]

# Pelvis -> STOMACH rests on one community page only (no CS2 source maps CS2's pelvis hitbox to
# a hit group): INFERRED. The damage rule per group lives in the C hit model, not here.


def load_model(path, expected_sha256=ASSET_GLB_SHA256):
    """Model of the GLB at `path`, refused unless its sha256 is `expected_sha256`."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError(f"{path}: sha256 {digest} is not {expected_sha256}. Download "
                         f"{ASSET_ZIP_URL} (sha256 {ASSET_ZIP_SHA256}) and pass its member "
                         f"{ASSET_GLB_MEMBER!r}.")
    return Model(path)


class Model:

    def __init__(self, path):
        self.doc = gp.GltfDoc(str(path))
        self.idx = {n: i for i, n in enumerate(self.doc.node_names)}
        mesh_node = gp.skinned_mesh_nodes(self.doc)[0]
        self.joints, self.joint_names, self.ibm = gp.skin_info(self.doc,
                                                               self.doc.nodes[mesh_node]["skin"])
        mesh = self.doc.json["meshes"][self.doc.nodes[mesh_node]["mesh"]]
        self.prims = [gp._primitive_arrays(self.doc, p) for p in mesh["primitives"]]
        self.tris = gp.triangles(self.doc)
        self.parent_name = {}
        for i, n in enumerate(self.doc.node_names):
            p = self.doc.parent[i]
            self.parent_name[n] = self.doc.node_names[p] if p >= 0 else None
        self.leg_nodes = [self.idx[n] for n in self.joint_names if n.startswith(LEG_PREFIXES)]
        self.rest_world = gp.world_matrices(self.doc, None, 0.0)

    def world(self, clip, t):
        return gp.world_matrices(self.doc, clip, t)

    def layered(self, upper, lower):
        """Legs + hips translation from `lower`, the rest from `upper` shifted by the hips delta."""
        h = self.idx["DEF-hips"]
        out = upper.copy()
        delta = lower[h][:3, 3] - upper[h][:3, 3]
        out[:, :3, 3] += delta
        for j in self.leg_nodes:
            out[j] = lower[j]
        return out

    def skin(self, world):
        """LBS vertices for arbitrary bone world matrices; same maths as gp.skinned_vertices."""
        jm = np.einsum("jab,jbc->jac", world[self.joints], self.ibm)
        vs, doms = [], []
        for pos, J, W in self.prims:
            ph = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)
            m = np.einsum("nk,nkab->nab", W, jm[J])
            vs.append(np.einsum("nab,nb->na", m, ph)[:, :3])
            doms.append(J[np.arange(len(J)), np.argmax(W, axis=1)])
        return np.concatenate(vs), np.concatenate(doms)

    def poses(self):
        """{"stand", "crouch": world matrices}, in the C table's pose order. Run-cycle poses are
        deferred (#394)."""
        stand = self.world(STAND_CLIP, STAND_T)
        return {"stand": stand, "crouch": self.layered(stand, self.world(CROUCH_CLIP, CROUCH_T))}


def _robust_centre(P, basis):
    """Centre of the p5..p95 box of points P projected on the orthonormal `basis` rows."""
    c = np.zeros(3)
    for e in basis:
        s = P @ e
        c += 0.5 * (np.percentile(s, 5) + np.percentile(s, 95)) * e
    return c


def fit_regions(model, pct=RADIUS_PERCENTILE, end_pct=END_PERCENTILES):
    """Fit every REGIONS capsule on the rest pose. Returns a list of dicts with the capsule in its
    anchor bone's local frame: a_local, b_local (homogeneous 4-vectors), r (metres)."""
    verts, dom = model.skin(model.rest_world)
    name_of = np.array(model.joint_names)[dom]
    fitted = []
    for name, group, cluster, anchor, axis in REGIONS:
        V = verts[np.isin(name_of, cluster)].astype(np.float64)
        if len(V) == 0:
            raise ValueError(f"region {name}: empty cluster {cluster}")
        Wa = model.rest_world[model.idx[anchor]]
        if axis[0] == "seg":
            pa = model.rest_world[model.idx[axis[1]]][:3, 3]
            pb = model.rest_world[model.idx[axis[2]]][:3, 3]
            u = (pb - pa) / np.linalg.norm(pb - pa)
        else:
            u = Wa[:3, axis[1]] / np.linalg.norm(Wa[:3, axis[1]])
            pa = Wa[:3, 3]
        # orthonormal basis of the plane perpendicular to u
        tmp = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
        v = np.cross(u, tmp)
        v /= np.linalg.norm(v)
        w = np.cross(u, v)
        rel = V - pa
        centre_perp = _robust_centre(rel, [v, w])
        origin = pa + centre_perp
        rel = V - origin
        s = rel @ u
        perp = rel - np.outer(s, u)
        d = np.linalg.norm(perp, axis=1)
        r = float(np.percentile(d, pct))
        lo, hi = np.percentile(s, end_pct[0]), np.percentile(s, end_pct[1])
        a_s, b_s = lo + r, hi - r
        if a_s > b_s:
            a_s = b_s = 0.5 * (lo + hi)
        A = origin + a_s * u
        B = origin + b_s * u
        inv = np.linalg.inv(Wa)
        fitted.append(
            dict(name=name,
                 group=group,
                 anchor=anchor,
                 n_verts=int(len(V)),
                 r=r,
                 a_local=inv @ np.append(A, 1.0),
                 b_local=inv @ np.append(B, 1.0)))
    return fitted


def posed_capsules(model, fitted, world):
    """[(name, group, A (3,), B (3,), r)] in model space for bone world matrices `world`."""
    out = []
    for f in fitted:
        W = world[model.idx[f["anchor"]]]
        out.append((f["name"], f["group"], (W @ f["a_local"])[:3], (W @ f["b_local"])[:3], f["r"]))
    return out


def model_to_sim(p, s):
    """Model point (X left, Y up, Z forward; metres) -> agent-local (fwd, left, up) in sim units."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 2], p[..., 0], p[..., 1]], axis=-1) * s


def sim_table(model, fitted, poses, s):
    """{pose: [(name, group, A_sim, B_sim, r_sim)]} in agent-local sim units (feet at the origin)."""
    tab = {}
    for pname, world in poses.items():
        tab[pname] = [(n, g, model_to_sim(A, s), model_to_sim(B, s), r * s)
                      for n, g, A, B, r in posed_capsules(model, fitted, world)]
    return tab


def scale_for(model, world_stand):
    """Sim units per metre that put the stand pose's mesh top at HULL_TOP_STAND."""
    v, _ = model.skin(world_stand)
    return HULL_TOP_STAND / float(v[:, 1].max())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--glb", required=True, help=f"{ASSET_GLB_MEMBER} from {ASSET_ZIP_URL}")
    args = ap.parse_args(argv)
    model = load_model(args.glb)
    poses = model.poses()
    v_ref, dom_ref, _ = gp.skinned_vertices(model.doc, STAND_CLIP, STAND_T)
    v_me, dom_me = model.skin(model.world(STAND_CLIP, STAND_T))
    print(
        f"skin self-check vs gltf_pose.skinned_vertices({STAND_CLIP}, {STAND_T}): n={len(v_me)} "
        f"max|dv|={np.abs(v_me - v_ref).max():.3e} dominant_equal={bool((dom_me == dom_ref).all())}"
    )
    s = scale_for(model, poses["stand"])
    print(f"scale s = {s:.4f} u/m (stand mesh top -> {HULL_TOP_STAND} u)")
    for p, world in poses.items():
        v, _ = model.skin(world)
        print(f"pose {p}: mesh top {model_to_sim(v, s)[:, 2].max():.2f} u")
    fitted = fit_regions(model)
    print(f"region table (pct {RADIUS_PERCENTILE:g}; agent-local fwd, left, up; feet at 0; r in u)")
    for p, caps in sim_table(model, fitted, poses, s).items():
        for n, g, A, B, r in caps:
            print(f"{p:6s} {n:12s} {g:8s} A=({A[0]:6.2f},{A[1]:6.2f},{A[2]:6.2f}) "
                  f"B=({B[0]:6.2f},{B[1]:6.2f},{B[2]:6.2f}) r={r:5.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
