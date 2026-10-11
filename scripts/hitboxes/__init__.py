"""Generator for the sim's deterministic capsule hitboxes (#391).

Three modules, run from the repository root with `python -m`:
  gltf_pose   a stdlib + numpy glTF 2.0 reader: node hierarchy, clip sampling, skinning;
  fit         fits one capsule per hit region to the CC0 Quaternius mannequin and poses it;
  gen_table   writes src/cs2rl/env/c/cs2_hitbox_table.h from that fit.
The asset is downloaded by hand, not committed: fit.ASSET_ZIP_URL names it and its sha256 is
checked before anything is fitted. No module here imports cs2rl, and no module under src/
imports these.
"""
