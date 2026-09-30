"""L3 of `cs2rl layers` (pyproject.toml), beside eval: the viewers.

Members:
  render       -- the rerun recorder behind `python -m cs2rl.train --record` (init_recording,
                  log_tick, ...); imports rerun at module scope;
  play         -- the play-vs-bots window, launched as `python -m cs2rl.viz.play`
                  (cs2_demo's `--policy` execs it); loads libcs2_play.so by ctypes;
  play_actions -- play's helpers with no ctypes and no raylib, safe to import from tests.

play_actions imports `cs2rl.policy` at module scope (init_policy_state), a layer
below viz, and `cs2rl.train.record` imports render inside record_episode, from the
layer above. Both edges point down, so no contract ignores either. Until #205 part 3
(#92) the policy lived in train, and this pair closed a viz <-> train cycle that
both cs2rl contracts had to ignore.

WHY this file holds a docstring and nothing else: a re-export here would put
render's rerun import, or play's ctypes loader, into the import chain of every
importer of any member. Without the file, grimp would not see the package and the
wheel would drop it (`namespaces = false`).
"""
