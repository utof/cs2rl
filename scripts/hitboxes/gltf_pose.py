"""A small stdlib (json/struct/base64) + numpy glTF 2.0 reader for a rigged, animated humanoid.

    world_matrices(doc, clip, t)      -> (n_nodes, 4, 4) world matrices after the FULL node hierarchy
    skinned_vertices(doc, clip, t)    -> (verts (N,3) float32, dominant_joint (N,) int, joint_names)

plus the pieces scripts/hitboxes/fit.py skins with (skin_info, skinned_mesh_nodes, triangles,
_primitive_arrays):
    from scripts.hitboxes import gltf_pose as gp
    doc = gp.GltfDoc("<file>.glb")
    verts, dom, names = gp.skinned_vertices(doc, "Crouch_Idle_Loop", 0.0)
`clip` may be a clip name, a clip index, or None (= the rest pose: every node's own TRS/matrix).

WHY CUSTOM (#391 phase 1, existing-capability check): the project's environment has no glTF
reader (trimesh, pyembree and open3d are not installed). pygltflib 1.16.5 is a JSON data model
plus the binary blob, with no animation-sampling, hierarchy or skinning call; trimesh 5.1.1 loads
the bind-pose mesh only, has no clip/time argument, and its glTF loader never mentions skins or
animations. Both were run in isolation with uvx. three.js r186 (GLTFLoader + AnimationMixer + SkinnedMesh) does all of it and
was the ORACLE this reader was checked against, not a dependency: joints and every skinned vertex
of the Quaternius mannequin agreed within 3e-7 m, and four knock-outs went red.

glTF PITFALLS HANDLED (spec 2.0, registry.khronos.org/glTF/specs/2.0/glTF-2.0.html):
  * Matrices (node.matrix, MAT4 accessors such as inverseBindMatrices) are COLUMN-major: reshape to
    (4,4) and transpose. Reading them row-major silently moves every joint.
  * Quaternions are (x, y, z, w), w last (scipy's default order too; NOT Eigen/Blender's w-first).
  * Accessor byteStride: when the bufferView defines byteStride the elements are NOT tightly packed;
    an absent byteStride means tightly packed. Element offset = bufferView.byteOffset +
    accessor.byteOffset + i * stride.
  * normalized integer accessors (WEIGHTS_0 as u8/u16, quantized rotations) must be decoded:
    u8 c/255, u16 c/65535, i8 max(c/127,-1), i16 max(c/32767,-1). JOINTS_n are NOT normalized.
  * Sparse accessors overwrite the listed elements of the dense (or zero) base.
  * MAT2/MAT3 (and MAT4 never) with 1- or 2-byte components have each column padded to 4 bytes:
    NOT implemented here, refused with NotImplementedError rather than misread.
  * Animation: times outside [first, last] key clamp to the end value; LINEAR rotations use slerp
    on the shortest arc; STEP holds the left key; CUBICSPLINE stores (in-tangent a_k, value v_k,
    out-tangent b_k) per key and the tangents are scaled by the key interval dt; a cubic rotation
    is renormalised.
  * Skinning: jointMatrix_j = world(joint_j) @ inverseBindMatrix_j, and the TRANSFORM OF THE SKINNED
    MESH NODE IS IGNORED (spec: "Only the joint transforms are applied to the skinned mesh"). The
    skeleton root's PARENT (e.g. an "Armature"/"Rig" node carrying a 0.01 scale or a Z-up->Y-up
    rotation) IS applied, because it is part of each joint's world matrix.
  * skin.skeleton is optional and NOT needed for skinning; do not root the hierarchy walk at it.
  * A skin without inverseBindMatrices means identity IBMs.
KNOWN LIMITS: morph targets (weights channels) are ignored; KHR_* extensions (e.g. meshopt or Draco
compression, KHR_texture_transform) are not decoded -- a file that REQUIRES an extension is refused.
"""
import base64
import json
import struct
from pathlib import Path

import numpy as np

_DTYPE = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4")
}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}
_NORM_DIV = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}


class GltfDoc:
    """A parsed .glb/.gltf: JSON, binary buffers, node parents, and an accessor cache."""

    def __init__(self, path):
        self.path = Path(path)
        data = self.path.read_bytes()
        bin_chunk = None
        if data[:4] == b"glTF":
            _magic, version, length = struct.unpack_from("<III", data, 0)
            if version != 2:
                raise ValueError(f"GLB version {version} is not 2")
            off = 12
            gj = None
            while off < length:
                clen, ctype = struct.unpack_from("<II", data, off)
                chunk = data[off + 8:off + 8 + clen]
                if ctype == 0x4E4F534A:                # 'JSON'
                    gj = json.loads(chunk.decode("utf-8"))
                elif ctype == 0x004E4942:              # 'BIN\0'
                    bin_chunk = chunk
                off += 8 + clen
        else:
            gj = json.loads(data.decode("utf-8"))
        if gj is None:
            raise ValueError(f"{self.path}: GLB has no JSON chunk")
        self.json = gj
        req = gj.get("extensionsRequired") or []
        if req:
            raise NotImplementedError(f"file requires extensions {req}")
        self.buffers = []
        for i, b in enumerate(gj.get("buffers", [])):
            uri = b.get("uri")
            if uri is None:
                if bin_chunk is None:
                    raise ValueError(f"buffer {i} has no uri and the file has no BIN chunk")
                self.buffers.append(bin_chunk)
            elif uri.startswith("data:"):
                self.buffers.append(base64.b64decode(uri.split(",", 1)[1]))
            else:
                self.buffers.append((self.path.parent / uri).read_bytes())
        self.nodes = gj.get("nodes", [])
        self.parent = [-1] * len(self.nodes)
        for i, n in enumerate(self.nodes):
            for c in n.get("children", []):
                self.parent[c] = i
        self.node_names = [n.get("name", f"node{i}") for i, n in enumerate(self.nodes)]
        self.animations = gj.get("animations", [])
        self.clip_names = [a.get("name", f"anim{i}") for i, a in enumerate(self.animations)]
        self._acc = {}
                                                       # parents before children: a depth-first order from every root
        order, stack = [], [i for i in range(len(self.nodes)) if self.parent[i] < 0][::-1]
        while stack:
            i = stack.pop()
            order.append(i)
            stack.extend(self.nodes[i].get("children", [])[::-1])
        self.order = order

    # ---------------------------------------------------------------- accessors
    def accessor(self, index):
        """Decode accessor `index` to a (count, ncomp) float64 array (ints for non-normalized ints).
        Handles byteStride, normalized ints and sparse substitution. Cached."""
        if index in self._acc:
            return self._acc[index]
        a = self.json["accessors"][index]
        ctype, atype, count = a["componentType"], a["type"], a["count"]
        dt, ncomp = _DTYPE[ctype], _NCOMP[atype]
        if atype in ("MAT2", "MAT3") and dt.itemsize < 4:
            raise NotImplementedError("column-padded MAT2/MAT3 with 1/2-byte components")
        if "bufferView" in a:
            bv = self.json["bufferViews"][a["bufferView"]]
            buf = self.buffers[bv["buffer"]]
            elem = ncomp * dt.itemsize
            stride = bv.get("byteStride") or elem
            off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
            if count and off + stride * (count - 1) + elem > len(buf):
                raise ValueError(f"accessor {index} overruns its buffer")
            arr = np.ndarray(shape=(count, ncomp),
                             dtype=dt,
                             buffer=buf,
                             offset=off,
                             strides=(stride, dt.itemsize)).copy()
        else:
            arr = np.zeros((count, ncomp), dtype=dt)
        sp = a.get("sparse")
        if sp:
            ib = self.json["bufferViews"][sp["indices"]["bufferView"]]
            idt = _DTYPE[sp["indices"]["componentType"]]
            idx = np.frombuffer(self.buffers[ib["buffer"]],
                                dtype=idt,
                                count=sp["count"],
                                offset=ib.get("byteOffset", 0) + sp["indices"].get("byteOffset", 0))
            vb = self.json["bufferViews"][sp["values"]["bufferView"]]
            vals = np.frombuffer(self.buffers[vb["buffer"]],
                                 dtype=dt,
                                 count=sp["count"] * ncomp,
                                 offset=vb.get("byteOffset", 0) + sp["values"].get("byteOffset", 0))
            arr = arr.copy()
            arr[idx.astype(np.int64)] = vals.reshape(sp["count"], ncomp)
        if a.get("normalized") and ctype in _NORM_DIV:
            out = arr.astype(np.float64) / _NORM_DIV[ctype]
            if ctype in (5120, 5122):
                out = np.maximum(out, -1.0)
        elif ctype == 5126:
            out = arr.astype(np.float64)
        else:
            out = arr.astype(np.int64)
        self._acc[index] = out
        return out

    def mat4_accessor(self, index):
        """MAT4 accessor as (count, 4, 4) row-major numpy matrices (glTF stores column-major)."""
        return self.accessor(index).reshape(-1, 4, 4).transpose(0, 2, 1)

    def clip_index(self, clip):
        if clip is None or isinstance(clip, int):
            return clip
        return self.clip_names.index(clip)


# -------------------------------------------------------------------- math
def quat_to_mat3(q):
    """(x, y, z, w) unit quaternion -> 3x3 rotation matrix (normalised first)."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def trs_to_mat4(t, r, s):
    m = np.eye(4)
    m[:3, :3] = quat_to_mat3(r) * np.asarray(s, dtype=np.float64)[None, :]
    m[:3, 3] = t
    return m


def slerp(q0, q1, u):
    """Shortest-arc slerp of (x, y, z, w) quaternions."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1, d = -q1, -d
    if d > 0.9995:                     # nearly parallel: lerp + renormalise
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    a = np.arccos(min(d, 1.0))
    return (np.sin((1 - u) * a) * q0 + np.sin(u * a) * q1) / np.sin(a)


def _sample(doc, sampler, path, t):
    """Value of one animation sampler at time t (spec: clamp outside the key range)."""
    times = doc.accessor(sampler["input"])[:, 0]
    vals = doc.accessor(sampler["output"])
    interp = sampler.get("interpolation", "LINEAR")
    n = vals.shape[1] if vals.ndim == 2 else 1
    if path == "weights":
        return None                                                                     # morph weights: not used (KNOWN LIMIT)
    if interp == "CUBICSPLINE":
        trip = vals.reshape(len(times), 3, n)                                           # [a_k (in-tangent), v_k (value), b_k (out-tangent)]
        if t <= times[0]:
            return trip[0, 1]
        if t >= times[-1]:
            return trip[-1, 1]
        k = int(np.searchsorted(times, t, side="right") - 1)
        dt = times[k + 1] - times[k]
        s = (t - times[k]) / dt
        s2, s3 = s * s, s * s * s
        v = ((2 * s3 - 3 * s2 + 1) * trip[k, 1] + (s3 - 2 * s2 + s) * dt * trip[k, 2] +
             (-2 * s3 + 3 * s2) * trip[k + 1, 1] + (s3 - s2) * dt * trip[k + 1, 0])
        if path == "rotation":
            v = v / np.linalg.norm(v)
        return v
    vals = vals.reshape(len(times), n)
    if t <= times[0]:
        return vals[0]
    if t >= times[-1]:
        return vals[-1]
    k = int(np.searchsorted(times, t, side="right") - 1)
    if interp == "STEP":
        return vals[k]
    s = (t - times[k]) / (times[k + 1] - times[k])
    if path == "rotation":
        return slerp(vals[k], vals[k + 1], s)
    return vals[k] + s * (vals[k + 1] - vals[k])


def local_trs_overrides(doc, clip, t):
    """{node: {path: value}} for every channel of `clip` at time t (empty for clip=None)."""
    ci = doc.clip_index(clip)
    if ci is None:
        return {}
    an = doc.animations[ci]
    out = {}
    for ch in an["channels"]:
        tgt = ch["target"]
        if "node" not in tgt:
            continue
        v = _sample(doc, an["samplers"][ch["sampler"]], tgt["path"], float(t))
        if v is not None:
            out.setdefault(tgt["node"], {})[tgt["path"]] = v
    return out


def world_matrices(doc, clip=None, t=0.0):
    """(n_nodes, 4, 4) world matrices of every node for (clip, t), parents first."""
    over = local_trs_overrides(doc, clip, t)
    world = np.zeros((len(doc.nodes), 4, 4))
    for i in doc.order:
        n = doc.nodes[i]
        o = over.get(i, {})
        if "matrix" in n and not o:
            local = np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T            # column-major
        else:
            local = trs_to_mat4(o.get("translation", n.get("translation", [0, 0, 0])),
                                o.get("rotation", n.get("rotation", [0, 0, 0, 1])),
                                o.get("scale", n.get("scale", [1, 1, 1])))
        p = doc.parent[i]
        world[i] = local if p < 0 else world[p] @ local
    return world


def skin_info(doc, skin=0):
    s = doc.json["skins"][skin]
    joints = list(s["joints"])
    if "inverseBindMatrices" in s:
        ibm = doc.mat4_accessor(s["inverseBindMatrices"])[:len(joints)]
    else:
        ibm = np.tile(np.eye(4), (len(joints), 1, 1))
    return joints, [doc.node_names[j] for j in joints], ibm


def skinned_mesh_nodes(doc):
    return [i for i, n in enumerate(doc.nodes) if "mesh" in n and "skin" in n]


def _primitive_arrays(doc, prim):
    attr = prim["attributes"]
    pos = doc.accessor(attr["POSITION"])
    js, ws = [], []
    k = 0
    while f"JOINTS_{k}" in attr:
        js.append(doc.accessor(attr[f"JOINTS_{k}"]))
        ws.append(doc.accessor(attr[f"WEIGHTS_{k}"]).astype(np.float64))
        k += 1
    return pos, np.concatenate(js, axis=1), np.concatenate(ws, axis=1)


def triangles(doc, mesh_node=None):
    """(T, 3) vertex indices into the concatenation of all primitives (mode 4 only)."""
    mesh_node = skinned_mesh_nodes(doc)[0] if mesh_node is None else mesh_node
    mesh = doc.json["meshes"][doc.nodes[mesh_node]["mesh"]]
    tris, base = [], 0
    for p in mesh["primitives"]:
        npos = doc.json["accessors"][p["attributes"]["POSITION"]]["count"]
        if p.get("mode", 4) != 4:
            raise NotImplementedError(f"primitive mode {p.get('mode')}")
        idx = doc.accessor(p["indices"])[:, 0] if "indices" in p else np.arange(npos)
        tris.append(idx.reshape(-1, 3) + base)
        base += npos
    return np.concatenate(tris)


def skinned_vertices(doc, clip=None, t=0.0, mesh_node=None):
    """Linear-blend-skinned world vertices of the skinned mesh at (clip, t).

    Returns (verts (N,3) float32, dominant_joint (N,) int = index into joint_names of the highest
    weight, joint_names list in skin order). Weights are used as stored (not renormalised)."""
    mesh_node = skinned_mesh_nodes(doc)[0] if mesh_node is None else mesh_node
    node = doc.nodes[mesh_node]
    joints, names, ibm = skin_info(doc, node["skin"])
    world = world_matrices(doc, clip, t)
    jm = np.einsum("jab,jbc->jac", world[joints], ibm)                 # world(joint) @ IBM; mesh node ignored
    mesh = doc.json["meshes"][node["mesh"]]
    vs, doms = [], []
    for prim in mesh["primitives"]:
        pos, J, W = _primitive_arrays(doc, prim)
        ph = np.concatenate([pos, np.ones((len(pos), 1))], axis=1)     # (N,4)
        skin_m = np.einsum("nk,nkab->nab", W, jm[J])                   # (N,4,4)
        v = np.einsum("nab,nb->na", skin_m, ph)[:, :3]
        vs.append(v)
        doms.append(J[np.arange(len(J)), np.argmax(W, axis=1)])
    return np.concatenate(vs).astype(np.float32), np.concatenate(doms).astype(np.int64), names
