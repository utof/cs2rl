"""L3 of `cs2rl layers` (pyproject.toml), beside eval: the viewers.

Members:
  render       -- the rerun recorder behind `train.py --record` (init_recording,
                  log_tick, ...); imports rerun at module scope;
  play         -- the play-vs-bots window, launched as `python -m cs2rl.viz.play`
                  (cs2_demo's `--policy` execs it); loads libcs2_play.so by ctypes;
  play_actions -- play's helpers with no ctypes and no raylib, safe to import from tests.

play_actions imports train at module scope (init_policy_state), and train imports
render inside record_episode. As siblings the two closed no loop; inside this one
package they close a viz <-> train cycle, so `cs2rl acyclic siblings` ignores
train -> viz.render, the same function-local edge the layers contract ignores. #92
retires it.

WHY this file holds a docstring and nothing else: a re-export here would put
render's rerun import, or play's ctypes loader, into the import chain of every
importer of any member. Without the file, grimp would not see the package and the
wheel would drop it (`namespaces = false`).
"""
