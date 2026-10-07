"""AST census of the metrics keys this repo's own emitters write (W4, spec 2026-08-31).

WHAT: a no-import, source-only extractor. Given the NAMED island of emitter
functions (``EMITTER_SITES``) it returns, for every metrics key those functions
write, the key itself plus the *shape* of the write — which is what
``src/cs2rl/eval/metrics_schema.py``'s declared ``aggregation`` is checked against in
tests/eval/test_metrics_schema.py.

WHY AST and not import-and-run: the emitters are gated on flags
(``--tag-diagnostic``), on architecture (split heads), on epoch parity
(``epoch % 5``) and on an episode having ended. A census built by running a
short training job sees maybe half the surface, and is *structurally blind* to
the omission of the other half — a key that never gets emitted in the sampled
configuration simply does not appear, so "the census matches the registry"
stays green while the registry silently rots. Reading the source sees every
branch.

WHY it is not a hand list either: the spec's own hand-derived census (review
round 7) missed ``actions/use_at_site_frac``, three presence-gated ``game/*``
keys and the eight split-architecture ``policy/aim_log_std_*`` keys. Anything
hand-maintained here drifts; the point of W4 is that adding a key to a LISTED
emitter FAILS a test until the key is registered.

SCOPE OF THAT PROMISE — read this before relying on it. ``census()`` sees the
NAMED island and nothing else, so on its own it can only be wrong by omission: a
key added to a function nobody put in ``EMITTER_SITES`` is invisible to it and to
every test built on it. ``_find_qualname`` makes a RENAMED emitter fail loudly; a
BRAND NEW one would be silent. ``metrics_write_sites()`` and
``island_merge_sources()`` below are the guard for that boundary — an independent
sweep of all of ``src/`` that does not consult ``EMITTER_SITES``, plus a one-hop
resolution of every non-literal dict merged into an island container. What even
those two do not catch is written down at ``metrics_write_sites``.

THE THREE THINGS THIS FILE PRODUCES, per emitter site:

  * ``EmittedKey``   — a concrete key literal, with its write shape.
  * ``KeyFamily``    — an f-string-built key template with every placeholder
                       replaced by ``*``. CLOSED when the placeholders are
                       statically resolvable — a literal iterable
                       (``for idx in range(9)``, ``for k in ("a", "b")``), a
                       name one hop back from one, or a PARAMETER every call
                       site in ``src/`` passes a constant for
                       (``emitter_param_bindings``, which is what closes the
                       seven ``tag/*`` families on ``mb_label``): ``members``
                       then holds the exact concrete keys. OPEN otherwise (a
                       ``zip`` over runtime objects, ``named_parameters()``):
                       ``members`` is empty and coverage falls back to the glob.
                       Open is the weaker state — an open template glob-matches,
                       so it alibis any registry entry underneath it.
  * ``write shape``  — one of the structural classes in ``SHAPES`` below, from
                       which the expected aggregation follows.

PITFALL — why the extractor is keyed on WRITE POSITIONS and not on "every
string constant in the function": the trainer's update methods mention dozens
of identifier-shaped strings that are config keys, tensor group names and dict
labels. Collecting all of them and then exempting the
false positives is how an extractor gets loosened until it enforces nothing.
Only subscript-assignment targets, dict literals that reach a metrics
container, and ``<container>.update({...})`` arguments count.
"""
import ast
import re
from pathlib import Path
from typing import NamedTuple

# parents[2]: this file is tests/_helpers/metrics_census.py (#207 moved it from tests/).
# tests/integration/test_path_constants_exist.py pins it against the conftest's REPO_ROOT, because
# a wrong root fails `census()` only by accident and leaves the SRC sweep vacuous.
REPO_ROOT = Path(__file__).resolve().parents[2]
# The package root: every EMITTER_SITES / NON_ISLAND_WRITES path below is relative to it,
# and the sweep walks it. src/ holds nothing else first-party.
SRC = REPO_ROOT / "src" / "cs2rl"

# ── The named emitter island ──────────────────────────────────────────────
#
# (relative path under src/cs2rl/, qualname, container names that hold metrics keys,
#  prefix applied to slash-free keys).
#
# The prefix column is the difference between a key as WRITTEN and the key as it
# reaches metrics.jsonl: PufferLib's mean_and_log re-keys `self.stats` under
# `environment/` and `self.losses` under `losses/` (pufferl.py mean_and_log),
# and Cs2Env._build_terminal_info's summary dict becomes `self.stats` entries via
# the info-collection loop, `Cs2PuffeRL._collect_infos`. Writing the prefix down
# here — rather than registering the bare names — is what makes the registry's keys
# the keys an analyst actually greps for.
#
# `self_play_used_past_metric` is on the list although it emits nothing: it is
# named by the spec as one of "our emitters", and an empty result from it is a
# fact worth re-checking rather than an omission worth wondering about. Its
# caller (`_log_selfplay` in train/loop.py) is what writes `self_play/used_past`.
#
# `Cs2Env.step` is listed SEPARATELY from `Cs2Env._build_terminal_info` because
# the step-stats merge is not in the helper: it is an inline
# `summary["step_stats"] = ...` in `step`, gated on include_step_stats_in_info.
# That is exactly the "inline write outside the helper" class the spec's own hand
# census missed once already, so deriving the island from the helpers alone is
# the mistake this line exists to not repeat.


class EmitterSite(NamedTuple):
    path: str
    qualname: str
    containers: dict                   # container expression -> key prefix


# gh#92: `Cs2PuffeRL.train` writes no key itself; its phase methods do. The minibatch
# SUMS go into `sums` (`_UpdateState.sums`, see GH90_ACCUMULATOR below), the per-update
# absolutes into the `losses` dict `_finish_update` builds from them. The trainer sites'
# (key, shape) multiset is the one the single pre-gh#92 `Cs2PuffeRL.train` site produced.
EMITTER_SITES = (
    EmitterSite("train/metrics.py", "compute_network_health", {"metrics": ""}),
    EmitterSite("train/metrics.py", "log_aim_log_std", {"logs": ""}),
    EmitterSite("train/metrics.py", "compute_head_divergence", {"out": ""}),
    EmitterSite("train/metrics.py", "compute_trunk_divergence", {"out": ""}),
    EmitterSite("train/metrics.py", "ScheduledEval.after_train", {"self.pending": ""}),
    EmitterSite("train/metrics.py", "compute_game_metrics", {"game_metrics": ""}),
    EmitterSite("train/metrics.py", "_inject_tag_metrics", {"logs": ""}),
    EmitterSite("train/trainer.py", "Cs2PuffeRL._accumulate_minibatch", {"sums": "losses/"}),
    EmitterSite("train/trainer.py", "Cs2PuffeRL._finish_update", {"losses": "losses/"}),
    EmitterSite("train/trainer.py", "Cs2PuffeRL._warmstart_metrics", {"losses": "losses/"}),
    EmitterSite("train/trainer.py", "Cs2PuffeRL._record_tag", {"self._tag_metrics": ""}),
    EmitterSite("train/trainer.py", "Cs2PuffeRL._log_and_checkpoint",
                {"self.stats": "environment/"}),
    EmitterSite("train/update.py", "tag_grad_cossim", {"out": ""}),
                                                                                                 # #92 part 2 split the epoch loop out of `train`; these four hold its writes.
    EmitterSite("train/loop.py", "_run_epochs", {"logs": ""}),
    EmitterSite("train/loop.py", "_log_epoch", {"logs": ""}),
    EmitterSite("train/loop.py", "_log_selfplay", {"logs": ""}),
    EmitterSite("train/loop.py", "_persist_row", {"log_entry": ""}),
    EmitterSite("train/selfplay.py", "self_play_used_past_metric", {"logs": ""}),
    EmitterSite("env/c/cs2_env.py", "Cs2Env._build_terminal_info", {"summary": "environment/"}),
    EmitterSite("env/c/cs2_env.py", "Cs2Env.step", {"summary": "environment/"}),
)

# ── The island's NEGATIVE list ────────────────────────────────────────────
#
# `metrics_write_sites()` sweeps all of src/ for metrics-SHAPED writes without
# consulting EMITTER_SITES, and every hit it finds outside the island has to be
# named here with the reason it is not an emitter. This is the audit that
# otherwise gets redone from scratch by every reviewer — three of the six
# pre-#204 entries look exactly like emitters to a grep (`metrics[...]`,
# `summary = {...}`, `stats = {...}` with the same bare key names cs2_env uses)
# and are not. The last two are experiment and BC-demo code that #204 moved into
# src/cs2rl/, which puts it inside this sweep: each writes into a container that
# shares a NAME with an island container (`stats`, `out`), and neither is a row.
#
# The list cannot rot into a blanket exemption: an entry that matches no hit fails
# too, so deleting an emitter-shaped site here is as loud as adding one.


class NonIslandWrite(NamedTuple):
    path: str                          # relative to src/cs2rl/
    qualname: str                      # "" = the whole file, including module scope
    reason: str


NON_ISLAND_WRITES = (
    NonIslandWrite(
        "eval/metrics_schema.py", "", "The registry ITSELF. Its ~200 dict-literal keys are "
        "declarations, not writes into a metrics row — they are the thing the census is "
        "compared against, so counting them as emissions would make every completeness "
        "test compare the registry with itself."),
    NonIslandWrite(
        "eval/baselines.py", "BaselineEvaluator.evaluate",
        "A REAL source of row keys, censused by its own extractor (`eval_output_keys()`) "
        "rather than as an island site: ScheduledEval merges the returned dict wholesale, "
        "so there is no per-key write for `_walk` to classify a shape from."),
    NonIslandWrite(
        "eval/baselines.py", "BaselineEvaluator._episode",
        "Per-episode RETURN VALUE of the evaluator's inner loop (shots_fired, "
        "shots_with_enemy_in_los, timed_out), consumed by evaluate() to build the eval/* "
        "numbers. Never written into a row itself."),
    NonIslandWrite(
        "train/evaluate.py", "evaluate_checkpoint",
        "`--eval` mode's local Counter, PRINTED TO STDOUT. Its 13 bare keys are the same "
        "names cs2_env's terminal info uses, which is why a grep-based audit reads it as "
        "an emitter; nothing here reaches metrics.jsonl."),
    NonIslandWrite(
        "train/resume.py", "convert_legacy_state_dict_to_split",
        "state_dict TENSOR names (`aim_log_std_t`/`_ct`), not metrics keys — checkpoint "
        "surgery for the legacy→split migration."),
    NonIslandWrite(
        "train_bc.py", "eval_plant_rate",
        "The BC-eval report dict (`summary = {...}`) with its own printer — a separate "
        "analysis path, never merged into a training row."),
    NonIslandWrite(
        "bc_demos.py", "generate_demos",
        "Demo-GENERATION stats (kept/discarded episodes, ticks min/median/max, spawn "
        "coverage), printed and returned to the CLI and the demo tests; never written into "
        "a metrics row. Hit only because `stats` is also an island container name."),
    NonIslandWrite(
        "experiment/analyze_tplant.py", "tag_summary",
        "An OFFLINE analysis summary: the `_vf`/`_n_raw`/backstop bookkeeping of the report "
        "dict tag_summary builds from a FINISHED run's metrics.jsonl. It reads rows and never "
        "writes one — not a trainer row. Hit only because `out` is also an island container "
        "name."),
)

# Non-literal dicts merged into an island container that resolve, one hop back, to
# something OUTSIDE the island. Keyed on the exact source expression so the excuse
# cannot widen: rewriting the expression fails here rather than staying excused.
# Everything else must resolve to an EMITTER_SITES member — see
# `island_merge_sources()` for why a merge is the one write shape the census is
# structurally unable to see.


class IslandMerge(NamedTuple):
    qualname: str                      # the EMITTER_SITES entry containing the merge
    source: str                        # ast.unparse of the expression it resolves to
    reason: str


ISLAND_MERGE_SOURCES = (
    IslandMerge(
        "ScheduledEval.after_train", "self.evaluator.evaluate(self.policy, self.device)",
        "BaselineEvaluator.evaluate's `out = {...}` contract, censused by "
        "`eval_output_keys()` and pinned by "
        "test_eval_keys_match_the_evaluate_output_contract."),
    IslandMerge(
        "_inject_tag_metrics", "trainer._tag_metrics",
        "A re-read of the island's OWN `trainer._tag_metrics` container: every key in it "
        "was written under Cs2PuffeRL._record_tag / tag_grad_cossim, both of which are "
        "EMITTER_SITES entries, so the merge adds no key the census has not already seen."),
)

# The two frozen gate readers (spec: never migrated, so the registry has to
# chase THEM). Only their own key literals are in scope — not their row loader
# (`src/cs2rl/experiment/analyze_tplant.py`), which #155 records as future-reader
# residue. Their consumer LABELS stay `rung1_gate` / `rung1a_smoke_read` (the names
# metrics_schema's consumers column uses); only their paths moved, in #204.
FROZEN_READERS = ("src/cs2rl/experiment/gate.py", "src/cs2rl/experiment/smoke_read.py")

# A frozen reader's key literal: slash-namespaced, or the one bare key the gate
# scripts read (`agent_steps`, PufferLib's own step counter — no *_KEY constant
# exists for it, it appears inline at three sites in experiment/smoke_read.py).
# Deliberately NOT "any identifier-shaped string": the readers are full of
# verdict labels ("PASS"), column kinds ("median") and dict keys ("fail") that
# are not metrics keys, and widening the predicate to swallow them would force
# junk registry entries.
READER_BARE_KEYS = ("agent_steps", )
KEY_SHAPED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z0-9_.\-]+)*$")

# `reader_key_literals` collects key-shaped strings in POSITIONS where a
# metrics key is used: a module-level constant table (the `*_KEY` names,
# EVAL_T/EVAL_CT, REPORT_EXTRA's argument tuples), a call argument
# (`weighted_sum(window, "game/shots_fired")`), a subscript slice, or an
# `in`/`not in` comparison. That is a positive rule, so the failure mode is a
# missed key rather than a junk registry entry — and it is broad enough that
# every read in both frozen scripts today lands in it.
#
# These three CALLS are excluded because their string argument provably is not
# a key. Excluded by position, so a real key literal cannot hide by being
# written next to one:
#   .replace  — the gate's `print_report` shortens a COLUMN HEADING with
#               `.replace('eval/win_vs_random_', 'eval_')`; the spec puts that
#               literal out of surface entirely.
#   .compile  — smoke_read.py's MOVE_KEY_RE. Its family gets a
#               hand-written registry entry naming the regex; there is no key
#               literal to extract from a pattern.
#   Path      — a filesystem path (RUN_DIR_DEFAULT), not a metrics key.
# The `"n/a"` a `_fmt` helper returns for a missing value is key-SHAPED but
# sits in none of the collected positions, so it never becomes a candidate.
NON_KEY_CALLS = ("replace", "compile", "Path")

# ── Write shapes → the structural aggregation classes ─────────────────────
#
# Each shape is a fact about the emission site that fixes how the value is
# aggregated before it reaches a metrics row. See metrics_schema.AGGREGATIONS.
SHAPES = {
                                                       # `self.stats[k] = [scalar]` — a ONE-ELEMENT list, so PufferLib's np.mean
                                                       # over the window list is an identity. `environment/episodes` is written
                                                       # this way on purpose (Cs2PuffeRL._log_and_checkpoint, see its comment).
    "stats-one-element-list": "last",
                                                       # `self.stats[k]` fed by the append/extend collection loop — PufferLib
                                                       # np.means the whole collection window.
    "stats-window": "window-mean-pufferlib",
                                                       # `sums[k] += ...` into the gh#90 accumulator: `_finish_update` divides the
                                                       # sum by the executed-minibatch count, i.e. a mean over minibatches.
    "losses-accumulated": "mean",
                                                       # `losses[k] = ...` into the dict that division produced: an absolute
                                                       # per-update scalar. gh#90 is the trap this distinction encodes — an
                                                       # absolute written into the sums is silently scaled by 1/minibatches.
    "losses-absolute": "last",
                                                       # Re-keyed inside compute_game_metrics out of values `_get()` read from
                                                       # `logs`, i.e. out of PufferLib window means that mean_and_log already
                                                       # computed. Pass-through: the aggregation belongs to the upstream pipeline.
    "game-passthrough": "window-mean-pufferlib",
                                                       # Written straight into the outer `logs` dict (or a dict merged into it)
                                                       # AFTER mean_and_log returned — one value per logged row, no windowing.
    "logs-post-mean": "last",
                                                       # Written straight into the persisted `log_entry` row in train/loop.py, never
                                                       # through PufferLib at all (run_id, step, epoch, team_spirit,
                                                       # resumed_from_step). Same "one value per row" contract as the above; kept
                                                       # separate so the registry says which pipeline a key belongs to.
    "row-literal": "last",
                                                       # A `summary[k] = <bare attribute>` write — the ONE shape in the terminal-info
                                                       # dict that is not wrapped in int()/float(): `summary["step_stats"] =
                                                       # self._step_stats_view`. mean_and_log's np.mean raises on a list of those
                                                       # and train/loop.py's isinstance(v, (int, float)) persist filter drops the key, so
                                                       # calling it a window mean would be the registry lying about a ctypes view.
                                                       # This is a TIGHTENING, not an exemption: the key is still censused and still
                                                       # requires a registry entry — it just gets an honest one.
    "stats-nonnumeric": "dropped-non-numeric",
}


class EmittedKey(NamedTuple):
    key: str
    shape: str
    site: str
    lineno: int


class KeyFamily(NamedTuple):
    template: str                      # placeholders replaced by `*`
    members: tuple                     # exact concrete keys, or () when not resolvable
    shape: str
    site: str
    lineno: int


# ── Parse cache ───────────────────────────────────────────────────────────
#
# Every extractor in this file reads source through `_parse_file`, so each file
# under src/ (and each frozen reader) is read and parsed AT MOST ONCE per
# process. That is not micro-optimization: `emitter_param_bindings` needs every
# call site in src/, and `census()` calls it once per EMITTER_SITES entry, so the
# uncached version re-parsed the whole ~600 KB tree (232 KB of it train.py) 14
# times over — measured 9.3 s for one `census()` on the drive this repo lives on,
# against 0.13 s before parameter binding existed. `test_metrics_schema.py` runs
# `census()` at module import, so the whole suite paid it on every run.
#
# SAFE TO SHARE because nothing in this module mutates an AST node — the trees
# are read-only inputs to the walks below. The cache is process-lifetime and
# keyed on the resolved path, so it goes stale only if a source file changes
# mid-process. No test does that: the mutation probes that check these gates edit
# a file and then run a FRESH pytest, which is a new interpreter and a cold cache.
_PARSE_CACHE = {}


def _parse_file(path):
    """The parsed module at `path`, parsed once per process. See _PARSE_CACHE."""
    key = str(path)
    tree = _PARSE_CACHE.get(key)
    if tree is None:
        tree = ast.parse(Path(key).read_text(), filename=key)
        _PARSE_CACHE[key] = tree
    return tree


_CALL_INDEX = None


def _src_call_index():
    """{callee short name: [Call nodes]} over every ``*.py`` under src/, built once.

    The sweep `emitter_param_bindings` used to redo per emitter site. Built in
    `sorted(SRC.rglob("*.py"))` order and, within a file, in `ast.walk` order, so
    a lookup here returns exactly the list — same nodes, same order — that the
    per-site sweep produced. Calls whose callee is neither a Name nor an
    Attribute (`f()()`, `(a or b)()`) get `_callee_name` == "" and are dropped,
    which is what the old `_callee_name(node) == fname` comparison did too, since
    an emitter's short name is never empty.
    """
    global _CALL_INDEX
    if _CALL_INDEX is None:
        index = {}
        for path in sorted(SRC.rglob("*.py")):
            for node in ast.walk(_parse_file(path)):
                if isinstance(node, ast.Call):
                    name = _callee_name(node)
                    if name:
                        index.setdefault(name, []).append(node)
        _CALL_INDEX = index
    return _CALL_INDEX


def _module_ast(rel_path):
    return _parse_file(SRC / rel_path)


_DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _bindings_in_scope(scope, name):
    """Statements in `scope`'s OWN body that bind `name`: def/class, a Name assignment
    target (Assign/AnnAssign/AugAssign), an import alias.

    Recurses through if/for/while/try/with/match blocks (same scope) but never into a
    nested def/class body, which is its own scope. The same rule as
    ast_oracle._scope_bindings in the gh#168 SDD folder and
    tests/train/test_trainer_composition.py::_bindings_in_scope: a `train = None` after the def
    is a binding Python honours, so it must count.
    """
    hits: list[ast.stmt] = []          # annotated: pyrefly infers list[def] from the first append

    def walk(stmts):
        for s in stmts:
            if isinstance(s, _DEFS):
                if s.name == name:
                    hits.append(s)
                continue
            if isinstance(s, (ast.Import, ast.ImportFrom)):
                if any((a.asname or a.name.split(".")[0]) == name for a in s.names):
                    hits.append(s)
            elif isinstance(s, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = s.targets if isinstance(s, ast.Assign) else [s.target]
                if any(
                        isinstance(n, ast.Name) and n.id == name for t in targets
                        for n in ast.walk(t)):
                    hits.append(s)
            for field in ("body", "orelse", "finalbody"):
                walk(getattr(s, field, []) or [])
            for h in getattr(s, "handlers", []) or []:
                walk(h.body)
            for c in getattr(s, "cases", []) or []:
                walk(c.body)

    walk(scope.body)
    return hits


def _enclosing_scope(tree, target):
    """The nearest Module/def/class whose body (transitively, through compound statements)
    contains `target`."""
    parents = {}
    for p in ast.walk(tree):
        for c in ast.iter_child_nodes(p):
            parents[c] = p
    n = parents[target]
    while not isinstance(n, (ast.Module, *_DEFS)):
        n = parents[n]
    return n


def _find_qualname(tree, qualname):
    """The FunctionDef/ClassDef node at `qualname` ('A.b.c'), or raise.

    Raising rather than returning None is deliberate: a renamed emitter must
    break this file loudly. Silently censusing zero keys for a site that moved
    is the failure mode that would make every downstream assertion vacuous.

    EXACTLY-ONCE (gh#168 W2a, spec §W2a hazard; PR #261 review): each part must be
    BOUND once in the scope it is looked up in, where a binding is a def/class, a
    Name assignment or an import alias (`_bindings_in_scope`). Python keeps the LAST
    binding of a name, so a first-match lookup would census a dead `def train` while
    a later duplicate def, or a later `train = None`, is what runs; any second binding
    is therefore an error, the same rule as ast_oracle.find_def. The first part is
    still searched at any depth because some emitters are nested defs: its def/class
    node is found anywhere in the tree, and the exactly-once rule is then applied in
    THAT node's enclosing scope; every later part is looked up in its parent's body.
    """
    parts = qualname.split(".")
    first = [n for n in ast.walk(tree) if isinstance(n, _DEFS) and n.name == parts[0]]
    if not first:
        raise AssertionError(f"emitter {qualname!r} not found — it was renamed or moved; "
                             "update EMITTER_SITES in tests/_helpers/metrics_census.py")
    if len(first) > 1:
        raise AssertionError(
            f"emitter {qualname!r}: {parts[0]!r} is defined {len(first)} times (lines "
            f"{[h.lineno for h in first]}); Python keeps the LAST, so the census refuses to "
            "pick one — delete the dead duplicate")
    scope = _enclosing_scope(tree, first[0])
    node = tree
    for part in parts:
        hits = _bindings_in_scope(scope, part)
        defs = [h for h in hits if isinstance(h, _DEFS)]
        if not defs:
            raise AssertionError(f"emitter {qualname!r} not found — it was renamed or moved; "
                                 "update EMITTER_SITES in tests/_helpers/metrics_census.py")
        if len(hits) > 1:
            raise AssertionError(
                f"emitter {qualname!r}: {part!r} is defined {len(hits)} times (lines "
                f"{[h.lineno for h in hits]}); Python keeps the LAST, so the census refuses to "
                "pick one — delete the dead duplicate (a def, an assignment or an import)")
        node = defs[0]
        scope = node
    return node


def _container_name(node):
    """'self.stats' / 'logs' / 'trainer._tag_metrics' for a subscript base."""
    try:
        return ast.unparse(node)
    except Exception:                  # pragma: no cover - ast always unparses
        return ""


# ── Symbolic resolution of f-string placeholders ──────────────────────────
#
# A loop variable binds when its iterable is a LITERAL (`for idx in range(9)`,
# `for k in ("a", "b")`) or a bare Name resolvable ONE HOP back to a literal
# assigned exactly once in the same function (`for g in pg_group_names`, where
# `pg_group_names = ("trunk", "policy_heads")` sits eight lines above). One hop
# is the same depth `losses_entropy_head_source` and `island_merge_sources`
# already resolve at; two hops is where this would start guessing, so
# `zip(_head_names, _dists)` and `model.named_parameters()` stay unresolvable and
# their families stay honestly OPEN.
#
# Emitter PARAMETERS bind the same way, from the call sites in src/ — see
# `emitter_param_bindings`. Both widenings exist for one measured reason: the
# seven `tag/*` families are built from `f"tag/<stat>/{g}/{mb_label}"`, and
# without them every member of the repo's only cross-team-gradient diagnostic is
# statically unknown, which makes `tag/*` an OPEN template that alibis any
# `tag/anything` registry entry.


def _literal_strings(node):
    """The string/int constants of a literal tuple/list/range, else None."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
        args = node.args
        if len(args) == 1 and isinstance(args[0], ast.Constant) and isinstance(args[0].value, int):
            return [str(i) for i in range(args[0].value)]
        return None
    if isinstance(node, (ast.Tuple, ast.List)):
        out = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, (str, int)):
                out.append(str(elt.value))
            else:
                return None
        return out
    return None


def _binding_counts(fn):
    """{name: how many times `fn` BINDS it}, over every binding construct Python has.

    The one primitive behind both static resolutions below, because both rest on
    the same claim — "this name holds one statically known value everywhere it is
    read" — and that claim is only true if nothing rebinds the name. Counting one
    construct (plain `ast.Assign`, which is all this used to do) leaves every
    other one invisible, and an invisible rebinding does not reopen the family: it
    keeps the STALE member list, silently. `pg_group_names += ("extra_group",)`
    and `mb_label = f"live_{len(groups)}"` each did exactly that with 29/29 green.

    Counted: the signature's own parameters (a parameter IS a binding, which is
    what makes `count > 1` mean "the body rebinds it"), `=`, `+=`, annotated
    assignment, `for` targets, `with ... as`, `:=`, `del`, `except ... as`,
    `import`, nested `def`/`class` names, `global`/`nonlocal`, and `match`
    captures. Tuple unpacking, `*rest` and attribute/subscript targets fall out of
    reading STORE/DEL context off the target expression rather than pattern-
    matching shapes: `d[k] = v` binds nothing (`d` and `k` are loads), `a, *b = x`
    binds both.

    NOT counted, on purpose: a comprehension's loop variable. `[g for g in xs]`
    has its own scope in Python 3 and does not touch an enclosing `g`, so counting
    it would assert a rebinding that does not happen — and a rule that is wrong in
    a visible case is a rule someone later deletes. The walrus inside a
    comprehension DOES leak to the enclosing scope and IS counted.

    OVER-COUNTS a name bound in a nested `def`, which is a separate scope. That is
    the safe direction (it un-resolves a name, reopening the family loudly) and it
    costs nothing today: the only two names this file resolves live in
    `tag_grad_cossim`, whose three nested helpers bind neither.
    """
    counts = {}

    def _bump(name, n=1):
        if name:
            counts[name] = counts.get(name, 0) + n

    def _bump_target(node):
        for sub in ast.walk(node) if node is not None else ():
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                _bump(sub.id)

    def _bump_params(fn_node):
        a = fn_node.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [a.vararg, a.kwarg]:
            if arg is not None:
                _bump(arg.arg)

    _bump_params(fn)
    for node in ast.walk(fn):
        if isinstance(node, (ast.Assign, ast.Delete)):
            for tgt in node.targets:
                _bump_target(tgt)
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign, ast.NamedExpr)):
            _bump_target(node.target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            _bump_target(node.target)
        elif isinstance(node, ast.withitem):
            _bump_target(node.optional_vars)
        elif isinstance(node, ast.ClassDef):
            _bump(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
            _bump(node.name)
            _bump_params(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                _bump((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.ExceptHandler):
            _bump(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                _bump(name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)):
            _bump(node.name)
        elif isinstance(node, ast.MatchMapping):
            _bump(node.rest)
    return counts


def _local_literal_bindings(fn):
    """{name: (values,)} for names bound ONCE to a literal iterable inside `fn`.

    Bound-once is the whole safety rule: a name rebound anywhere is not a
    constant, so it resolves to nothing and its family stays OPEN. `_binding_counts`
    is what makes that sentence TRUE rather than aspirational — this used to count
    `ast.Assign` targets only, so `pg_group_names += ("extra_group",)` was invisible
    and the census went on reporting the two-element axis while the emitter wrote
    three. Measured blast radius across the island today: exactly one name,
    `tag_grad_cossim`'s `pg_group_names` — every other non-literal iterable in an
    emitter is a method call (`model.named_parameters()`, `subsets.items()`) or a
    `zip`/`enumerate`, none of which this reaches.
    """
    counts = _binding_counts(fn)
    values = {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                values[tgt.id] = _literal_strings(node.value)
    return {n: tuple(v) for n, v in values.items() if v is not None and counts.get(n) == 1}


def _bind_for(node, env, literals=None):
    """Extend `env` with the loop variables `node` binds to literal values.

    `literals` is `_local_literal_bindings` of the enclosing emitter; it is what
    makes `for g in pg_group_names` resolvable. Omitted (the reads walk) it is
    empty and only in-statement literals bind.
    """
    env = dict(env)
    literals = literals or {}
    target, it = node.target, node.iter
    if isinstance(target, ast.Name):
        vals = _literal_strings(it)
        if vals is None and isinstance(it, ast.Name):
            vals = literals.get(it.id)
        env[target.id] = tuple(vals) if vals is not None else None
    elif isinstance(target, ast.Tuple) and isinstance(it, (ast.Tuple, ast.List)):
        # `for name, a, b in (("action_heads", x, y), ...)` — bind position-wise,
        # taking only the slots whose element is a constant in EVERY row.
        cols: dict[str, tuple[str, ...] | None] = {}
        for i, name in enumerate(target.elts):
            if not isinstance(name, ast.Name):
                continue
            vals = []
            for row in it.elts:
                if isinstance(row, (ast.Tuple, ast.List)) and i < len(row.elts):
                    elt = row.elts[i]
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, (str, int)):
                        vals.append(str(elt.value))
                        continue
                vals = None
                break
            cols[name.id] = tuple(vals) if vals else None
        env.update(cols)
    else:
        for name in ast.walk(target):
            if isinstance(name, ast.Name):
                env[name.id] = None
    return env


def _resolve(node, env):
    """Possible string values of `node`, or None when not statically known.

    Handles Constant, Name (from `env`), JoinedStr (cartesian product over its
    parts) and IfExp (the union of both branches).
    """
    if isinstance(node, ast.Constant):
        return (str(node.value), ) if isinstance(node.value, (str, int)) else None
    if isinstance(node, ast.Name):
        return env.get(node.id)
    if isinstance(node, ast.IfExp):
        a, b = _resolve(node.body, env), _resolve(node.orelse, env)
        return None if a is None or b is None else tuple(dict.fromkeys(a + b))
    if isinstance(node, ast.FormattedValue):
        return _resolve(node.value, env)
    if isinstance(node, ast.JoinedStr):
        acc = ("", )
        for part in node.values:
            vals = _resolve(part, env)
            if vals is None:
                return None
            acc = tuple(a + v for a in acc for v in vals)
        return tuple(dict.fromkeys(acc))
    return None


def _template(node, env):
    """The `*`-placeholder template for an f-string key."""
    out = []
    for part in node.values:
        if isinstance(part, ast.Constant):
            out.append(str(part.value))
        else:
            vals = _resolve(part, env)
            out.append("|".join(vals) if vals is not None and len(vals) == 1 else "*")
    return "".join(out)


# ── Shape classification ──────────────────────────────────────────────────

# The gh#90 accumulator (gh#92 made it a container instead of a divisor loop). A
# `losses/*` key is a minibatch MEAN when it is summed into `sums` and an absolute when
# it is written into the dict `_finish_update` divides those sums into, so the split
# is by container and `_classify` reads it off the container name. That is only true
# while the accumulator keeps the shape `gh90_accumulator_violations` checks.
GH90_ACCUMULATOR = "sums"
_GH90_SUMS_SITE = "Cs2PuffeRL._accumulate_minibatch"
_GH90_DIVIDE_SITE = "Cs2PuffeRL._finish_update"


def gh90_accumulator_violations():
    """Why `sums` is not (only) the minibatch-mean accumulator; [] when it is.

    The facts the mean/last split rests on, each a way it was or could be broken:

      * `.sums` and `.minibatches_run` are touched only in `_accumulate_minibatch`
        (alias `sums = update.sums`, then `+=` writes and the count) and in
        `_finish_update` (the division). A write anywhere else would be divided as
        a mean, or dropped if it lands after the division.
      * every use of the `sums` alias is the target of a `+=`: `sums[k] = v` would
        replace a sum with one minibatch's value and `sums.update(...)` would hide
        keys from the census.
      * `.minibatches_run` is incremented exactly once, beside the sums, so the
        divisor counts the minibatches that added to them.
      * `_finish_update` divides first: `losses` is bound to a dict built by
        dividing every `update.sums` item by `update.minibatches_run`, before any
        `losses[k] = ...` write. That divisor is the gh#90 fix (a KL-truncated
        update divides by the minibatches it ran, not by total_minibatches).
    """
    tree = _module_ast("train/trainer.py")
    sums_fn = _find_qualname(tree, _GH90_SUMS_SITE)
    divide_fn = _find_qualname(tree, _GH90_DIVIDE_SITE)
    out = []
    allowed = {id(n) for fn in (sums_fn, divide_fn) for n in ast.walk(fn)}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and node.attr in ("sums", "minibatches_run")
                and id(node) not in allowed):
            out.append(f"line {node.lineno}: `.{node.attr}` used outside "
                       f"{_GH90_SUMS_SITE} / {_GH90_DIVIDE_SITE}")

    alias = [
        n for n in ast.walk(sums_fn) if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id == GH90_ACCUMULATOR
    ]
    if len(alias) != 1 or ast.unparse(alias[0].value) != "update.sums":
        out.append(f"{_GH90_SUMS_SITE}: expected exactly one `{GH90_ACCUMULATOR} = update.sums`")
    added = {
        id(n.target.value)
        for n in ast.walk(sums_fn) if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add)
        and isinstance(n.target, ast.Subscript)
    }
    for n in ast.walk(sums_fn):
        if (isinstance(n, ast.Name) and n.id == GH90_ACCUMULATOR and isinstance(n.ctx, ast.Load)
                and id(n) not in added):
            out.append(f"line {n.lineno}: `{GH90_ACCUMULATOR}` used other than as "
                       f"`{GH90_ACCUMULATOR}[k] += ...`")
    counts = [
        n for n in ast.walk(sums_fn) if isinstance(n, ast.AugAssign) and isinstance(n.op, ast.Add)
        and ast.unparse(n.target) == "update.minibatches_run"
    ]
    stores = [
        n for n in ast.walk(sums_fn) if isinstance(n, ast.Attribute) and n.attr == "minibatches_run"
        and not isinstance(n.ctx, ast.Load)
    ]
    if len(counts) != 1 or len(stores) != 1:
        out.append(f"{_GH90_SUMS_SITE}: expected exactly one `update.minibatches_run += ...` "
                   f"(found {len(counts)} increments, {len(stores)} stores)")

    divide = next((n
                   for n in ast.walk(divide_fn) if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "losses"), None)
    comps = [c for c in ast.walk(divide) if isinstance(c, ast.DictComp)] if divide else []
    comp = comps[0] if len(comps) == 1 and len(comps[0].generators) == 1 else None
    quotient = (comp.value if comp is not None
                and ast.unparse(comp.generators[0].iter) == "update.sums.items()" else None)
    divisor = ""
    if isinstance(quotient, ast.BinOp) and isinstance(quotient.op, ast.Div):
        divisor = ast.unparse(quotient.right)
        bound = [
            ast.unparse(n.value) for n in ast.walk(divide_fn) if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == divisor for t in n.targets)
        ]
        if bound == ["update.minibatches_run"]:
            divisor = "update.minibatches_run"
    if divide is None or divisor != "update.minibatches_run":
        out.append(f"{_GH90_DIVIDE_SITE}: `losses` is not built as "
                   "`{key: total / update.minibatches_run for key, total in update.sums.items()}`")
        return out
    sums_reads = [
        n for n in ast.walk(divide_fn) if isinstance(n, ast.Attribute) and n.attr == "sums"
    ]
    if len(sums_reads) != 1:
        out.append(f"{_GH90_DIVIDE_SITE}: `update.sums` read {len(sums_reads)} times; "
                   "only the division may read it")
    early = [
        w for n in ast.walk(divide_fn) for _, _, _, w in site_write_targets(n, {"losses": ""})
        if w < divide.lineno
    ]
    if early:
        out.append(f"{_GH90_DIVIDE_SITE}: `losses` written at {early}, before the division")
    return out


def _game_passthrough_names(fn):
    """Locals in compute_game_metrics bound to a `_get(...)` read of `logs`.

    Used to prove — rather than assume — that a `game/*` value is a
    pass-through of a PufferLib window mean. A value built from anything else
    (a wall clock, a policy attribute) is NOT a window mean and must not be
    declared one.
    """
    names = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(
                node.targets[0], ast.Name):
            if any(
                    isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "_get"
                    for c in ast.walk(node.value)):
                names.add(node.targets[0].id)
    return names


def _is_passthrough(value, allowed_names):
    """True when every leaf of `value` is a `_get(...)`, an allowed local or a number.

    Descent STOPS at a `_get(...)` call — its arguments are the source-key name
    and a default, not part of the value expression. Walking into them would
    reject every pass-through in the function (measured: it does), and the
    tempting "fix" of allowing string constants everywhere would let a
    genuinely non-window value be declared a window mean.
    """
    if isinstance(value, ast.Call):
        return isinstance(value.func, ast.Name) and value.func.id == "_get"
    if isinstance(value, ast.Name):
        return value.id in allowed_names
    if isinstance(value, ast.Constant):
        return isinstance(value.value, (int, float)) and not isinstance(value.value, bool)
    return all(
        _is_passthrough(child, allowed_names) for child in ast.iter_child_nodes(value)
        if isinstance(child, ast.expr))


def _classify(site, container, value, lineno, ctx):
    """The structural write shape of one key assignment."""
    if container.endswith(".stats"):
        one_element_list = (isinstance(value, (ast.List, ast.Tuple)) and len(value.elts) == 1)
        return "stats-one-element-list" if one_element_list else "stats-window"
    if container == GH90_ACCUMULATOR:
        return "losses-accumulated"
    if container == "losses":
        return "losses-absolute"
    if container == "summary":
        # cs2_env's terminal-info dict is appended into self.stats by
        # Cs2PuffeRL._collect_infos, one entry per episode → a PufferLib window mean.
        # Exception, by SHAPE not by name: every numeric write in that dict
        # coerces with int()/float() (or is a local bound to such a coercion); a
        # BARE ATTRIBUTE read is the non-numeric payload case. Adding a numeric
        # `summary["x"] = stats.x` write would land here and fail loudly, which
        # is the right direction to be wrong in.
        if isinstance(value, ast.Attribute):
            return "stats-nonnumeric"
        return "stats-window"
    if container == "log_entry":
        return "row-literal"
    if site.qualname == "compute_game_metrics":
        return ("game-passthrough"
                if _is_passthrough(value, ctx["passthrough_names"]) else "logs-post-mean")
    return "logs-post-mean"


# ── The census walk ───────────────────────────────────────────────────────


def _dict_items(node):
    """(key_node, value_node) pairs of a dict literal, flattening `**{...}`."""
    for k, v in zip(node.keys, node.values, strict=True):
        if k is None and isinstance(v, ast.Dict):
            yield from _dict_items(v)
        elif k is not None:
            yield k, v


# A `c.setdefault(k)` with no default writes None. Kept as an explicit node so
# the one-argument form is censused like any other write instead of being a
# silent hole — `_classify` reads the value, and None is what it really is.
_IMPLICIT_NONE = ast.Constant(value=None)


def site_write_targets(node, containers):
    """[(container, key_node, value_node, lineno)] for writes into `containers` at `node`.

    THE FOUR WRITE SHAPES the census counts, in one place so that `_walk` and the
    unit test that knocks each of them out cannot drift apart:

        c[k] = v / c[k] += v      subscript assignment
        c = {k: v, ...}           a dict literal bound to a container NAME, with or
                                  without an annotation (`c: T = {...}`)
        c.update({k: v, ...})     a literal update
        c.setdefault(k, v)        (final review I-3)

    WHY the annotated form. `summary: dict[str, Any] = {...}` in
    Cs2Env._build_terminal_info (#354) is an ast.AnnAssign, not an ast.Assign;
    before it was accepted the census silently dropped that dict's six keys and
    only the registry's reverse check noticed.

    WHY setdefault is here at all. It was missing, and the miss was invisible
    twice over: `census()` did not produce the key, so the registry never had to
    carry it, AND `metrics_write_sites()` did not treat it as a write shape
    either, so the island-blind sweep could not report it as an unexplained hit.
    A probe writing `game_metrics.setdefault("game/probe", 1.0)` inside
    `compute_game_metrics` left all 67 tests green. It is a plausible spelling —
    "fill in a default only if the epoch did not compute one" is exactly what a
    metrics emitter reaches for — not a contrived one.

    `containers` is passed in rather than read off a site so the shapes can be
    tested against a synthetic snippet.
    """
    writes = []
    # An AnnAssign without a value (`c: T`) declares a name and writes nothing.
    if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and node.value is not None:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for tgt in targets:
            if isinstance(tgt, ast.Subscript) and _container_name(tgt.value) in containers:
                writes.append((_container_name(tgt.value), tgt.slice, node.value, node.lineno))
            elif (isinstance(tgt, ast.Name) and tgt.id in containers
                  and isinstance(node.value, ast.Dict)):
                for k, v in _dict_items(node.value):
                    writes.append((tgt.id, k, v, getattr(k, "lineno", node.lineno)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        base = _container_name(node.func.value)
        if base in containers and node.args:
            if node.func.attr == "update" and isinstance(node.args[0], ast.Dict):
                for k, v in _dict_items(node.args[0]):
                    writes.append((base, k, v, getattr(k, "lineno", node.lineno)))
            elif node.func.attr == "setdefault":
                value = node.args[1] if len(node.args) > 1 else _IMPLICIT_NONE
                key = node.args[0]
                writes.append((base, key, value, getattr(key, "lineno", node.lineno)))
    return writes


def _walk(node, site, env, ctx, keys, families, seen):
    """Recursive collector; `env` carries loop-variable bindings."""
    if isinstance(node, ast.For):
        env = _bind_for(node, env, ctx["local_literals"])
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not site._fn:
        # A nested helper (compute_game_metrics._maybe) is walked once per call
        # site, with its parameters bound to that call's constant arguments.
        for args in ctx["nested_calls"].get(node.name, ()):
            inner = dict(env)
            for param, arg in zip(node.args.args, args, strict=False):
                inner[param.arg] = _resolve(arg, env)
            for child in node.body:
                _walk(child, site, inner, ctx, keys, families, seen)
        return

    for container, key_node, value, lineno in site_write_targets(node, site.containers):
        shape = _classify(site, container, value, lineno, ctx)
        prefix = site.containers[container]
        if isinstance(key_node, ast.JoinedStr):
            resolved = _resolve(key_node, env)
            template = prefix + _template(key_node, env)
            members = tuple(sorted(prefix + r for r in resolved)) if resolved else ()
            if template not in seen:
                seen.add(template)
                families.append(KeyFamily(template, members, shape, site.qualname, lineno))
        else:
            for raw in _resolve(key_node, env) or ():
                key = prefix + raw
                if (key, site.qualname) not in seen:
                    seen.add((key, site.qualname))
                    keys.append(EmittedKey(key, shape, site.qualname, lineno))

    for child in ast.iter_child_nodes(node):
        _walk(child, site, env, ctx, keys, families, seen)


def emitter_param_bindings(site, fn):
    """Constant values `site`'s PARAMETERS take, unioned over every call in src/.

    WHY an emitter's parameters matter to a key census: `tag_grad_cossim` builds
    its keys as ``f"tag/cossim_cross/{g}/{mb_label}"``, and `mb_label` is a
    parameter. Read from the function alone it is unknowable, so all seven
    `tag/*` families come out OPEN — and an OPEN template is a glob alibi in
    `test_every_registered_emitted_key_is_actually_emitted`, so the registry
    could carry any `tag/...` key it liked. The single call site passes
    ``mb_label="mb0" if is_mb0 else "mbL"``: two constants, an `ast.IfExp` that
    `_resolve` already unions. The label axis is a static fact; it just is not
    written down inside the emitter.

    CONSERVATIVE IN THE SAFE DIRECTION. A parameter binds only when EVERY call
    site resolves it to constants and at least one call site exists. Add a caller
    passing a runtime label and the parameter unbinds, the families reopen, and
    the registry's now-unconfirmed `members` fail
    `test_closed_family_members_match_the_census_exactly`'s unpinned check —
    loudly, rather than by quietly describing a member set that moved.

    Both argument forms are read (positional by signature index, keyword by
    name), so switching a call from one to the other does not silently drop the
    binding. `*args`/`**kwargs` call sites resolve to nothing and unbind.

    A parameter the emitter REBINDS is dropped (`_binding_counts` — the signature
    contributes 1, so any body binding pushes the count past 1). Without that
    check the census reports the CALL SITE's values while the emitter writes keys
    built from something else, and it does so silently: one line —
    ``mb_label = f"live_{len(groups)}"`` at the top of `tag_grad_cossim` — moved
    every emitted key to `tag/<stat>/<g>/live_2` with the whole suite still green,
    leaving the registry documenting 26 keys nothing emits while the 26 real ones
    went unregistered. That is the exact defect W4 exists to catch. Dropping the
    parameter instead reopens the families, which
    `test_tag_families_are_census_closed_on_both_axes` reports by name.
    """
    fname = site.qualname.split(".")[-1]
    params = [a.arg for a in fn.args.posonlyargs] + [a.arg for a in fn.args.args]
    kwonly = [a.arg for a in fn.args.kwonlyargs]

    calls = _src_call_index().get(fname, ())
    if not calls:
        return {}

    rebound = {n for n, c in _binding_counts(fn).items() if c > 1}
    out = {}
    for pos, name in enumerate(params + kwonly):
        if name in rebound:
            continue
        vals = []
        for call in calls:
            arg = next((kw.value for kw in call.keywords if kw.arg == name), None)
            if arg is None and name in params and pos < len(call.args):
                arg = call.args[pos]
            resolved = _resolve(arg, {}) if arg is not None else None
            if resolved is None:
                vals = None
                break
            vals.extend(resolved)
        if vals:
            out[name] = tuple(dict.fromkeys(vals))
    return out


def _nested_call_args(fn):
    """{helper name: [(const arg nodes), ...]} for helpers defined inside `fn`."""
    local = {n.name for n in ast.walk(fn) if isinstance(n, ast.FunctionDef) and n is not fn}
    calls = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in local:
            calls.setdefault(node.func.id, []).append(node.args)
    return calls


class _SiteWithFn:
    """EmitterSite + the AST node it was resolved to (attribute passthrough).

    `_walk` needs the emitter's own node to tell it from a helper nested inside
    it: a nested def is walked once per call site with its parameters bound,
    the emitter body itself exactly once.
    """

    def __init__(self, site, fn):
        self._site = site
        self._fn = fn

    def __getattr__(self, name):
        return getattr(self._site, name)


def census():
    """(emitted keys, key families) over the whole emitter island.

    Raises when the gh#90 accumulator lost its shape: every losses/* aggregation in
    eval/metrics_schema.py is classified by it, so that is a registry-wide event.
    """
    violations = gh90_accumulator_violations()
    if violations:
        raise AssertionError("the gh#90 accumulator lost its shape; the losses/* mean/last "
                             "split is unchecked:\n  " + "\n  ".join(violations))
    keys, families = [], []
    for site in EMITTER_SITES:
        fn = _find_qualname(_module_ast(site.path), site.qualname)
        ctx = {
            "passthrough_names":
            (_game_passthrough_names(fn) if site.qualname == "compute_game_metrics" else set()),
            "nested_calls":
            _nested_call_args(fn),
            "local_literals":
            _local_literal_bindings(fn),
        }
        seen = set()
        # The site's own parameters are the starting environment: `mb_label` is
        # a `tag_grad_cossim` argument, not a local, and without it the seven
        # tag/* families have no statically known members. See
        # emitter_param_bindings for why this can only ever narrow, not invent.
        env0 = emitter_param_bindings(site, fn)
        for child in ast.iter_child_nodes(fn):
            _walk(child, _SiteWithFn(site, fn), env0, ctx, keys, families, seen)
    return keys, families


# ── Island completeness: the boundary guard ───────────────────────────────


class MetricsWrite(NamedTuple):
    path: str                          # relative to src/cs2rl/
    qualname: str                      # enclosing def/class chain; "" at module scope
    container: str                     # write target's container expr; "" for a bare dict literal
    key: str                           # the key literal, f-string placeholders as `*`
    lineno: int


def _key_text(node):
    """The literal text of a key node — an f-string's placeholders become `*`."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(str(p.value) if isinstance(p, ast.Constant) else "*" for p in node.values)
    return None


def metrics_write_sites(src=None):
    """Every metrics-SHAPED write in src/, found WITHOUT consulting EMITTER_SITES.

    WHY this exists: EMITTER_SITES is a named island, so `census()` can only be
    wrong by omission. A brand-new emitter — say a helper returning
    ``{"game/x": 1.0}`` that `train()` merges with ``logs.update(...)`` — writes a
    key into every row while all of test_metrics_schema.py stays green. This sweep
    is the independent second opinion: it finds writes by SHAPE across the whole
    tree, and test_metrics_schema.py requires every hit to be inside the island or
    on the reasoned NON_ISLAND_WRITES list.

    THE PREDICATE (deliberately broader than any grep — the review that prompted
    this used `logs[` / `.stats[` / `losses[` / `summary[` / `log_entry[` and
    missed `metrics[`, which occurs outside the island):

      write POSITIONS  ``c[k] = ...`` / ``c[k] += ...``, a dict literal bound to a
                       name (annotated or not), a bare dict literal anywhere, ``c.update({...})`` and
                       ``c.setdefault(k, ...)`` (the last added by final review
                       I-3 — it was a write shape NEITHER walk knew about, so a
                       `setdefault` emission was invisible tree-wide, not just
                       inside the island).
      key PREDICATE    a key-shaped string that is either slash-namespaced (any
                       `a/b`, so a NEW namespace is caught too — nothing here reads
                       the registry, which is what keeps the guard independent of
                       the thing it guards), or written into a container whose name
                       matches one the island itself uses (`logs`, `summary`,
                       `stats`, `losses`, `metrics`, `log_entry`, ...). The second
                       clause is what covers cs2_env's BARE keys, which carry no
                       slash until PufferLib prefixes them.

    WHAT IT STILL DOES NOT CATCH, precisely:

      * a computed key — ``logs[some_var]`` or an f-string whose every segment is
        a placeholder. No source-level extractor can name those; the registry's
        `family` entries are the mechanism for that class.
      * a write that is BOTH bare-keyed AND into a container named nothing like
        the island's (``row["sneaky"] = ...``). Adding that container name to an
        EMITTER_SITES entry — which is what makes it an emitter — also widens this
        predicate, so the gap only exists for a container that is never declared.
      * anything outside ``src/``. Scripts and notebooks are out of scope by spec.
        Experiment and BC-demo code that #204 moved INTO ``src/cs2rl/``
        (``experiment/``, ``bc_demos.py``, ``deploy/``) is in scope, and its two
        metrics-shaped writes are declared in NON_ISLAND_WRITES.
      * a key that reaches a row without a write shape at all, e.g. PufferLib's
        own `mean_and_log` literals (covered instead by PUFFERLIB_OWNED and a
        cross-package AST pin) or `**` splats of a non-literal mapping (covered by
        `island_merge_sources()` for island containers only).

    AND THE ONE THAT IS NOT ABOUT THE PREDICATE AT ALL — read this before adding
    a function to EMITTER_SITES. A hit INSIDE an island site is not reported as
    unexplained; that is the whole point of the cross-walk, and it means listing a
    function DISARMS this sweep for everything written inside it. Three shapes
    lived in that gap (final review I-3), and two of the three are now caught by
    ``undeclared_container_writes()``, which requires an in-island write to land
    in a container the site declares:

      * ``def _emit(out): out["game/x"] = ...`` — container as a PARAMETER. CAUGHT.
      * ``alias = logs; alias["game/x"] = ...`` — container under a second name. CAUGHT.
      * ``logs.setdefault("game/x", ...)`` — CAUGHT, by adding the shape to both
        walks (see ``site_write_targets``).

    What is left of that gap, precisely: an in-island write into an undeclared
    container whose keys are BARE and whose name resembles no island container
    (``tmp["kills"] = ...``) — dropped by the key predicate above before the
    container check can see it — and the one-argument ``c.setdefault(k)``, which
    writes None and is therefore dropped by train/loop.py's numeric persist filter
    before it can reach a row.
    """
    containers = {c.split(".")[-1] for s in EMITTER_SITES for c in s.containers}
    found = {}

    def _record(path, qualname, container, key_node, fallback_lineno: int):
        text = _key_text(key_node)
        if text is None or not KEY_SHAPED.match(text.replace("*", "x")):
            return
        if "/" not in text and container.split(".")[-1] not in containers:
            return
        lineno = getattr(key_node, "lineno", fallback_lineno)
        # First record wins: the container-bearing shape is always visited before
        # the bare dict literal nested inside it, so the richer one is kept.
        found.setdefault((path, qualname, text, lineno),
                         MetricsWrite(path, qualname, container, text, lineno))

    def _walk_all(path, node, qualname):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qualname = f"{qualname}.{node.name}" if qualname else node.name
        # The same assignment shapes as site_write_targets. Without AnnAssign a bare-keyed
        # `stats: dict = {...}` reached _record only as the nested literal, with no
        # container name, and the bare keys were dropped.
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if isinstance(tgt, ast.Subscript):
                    _record(path, qualname, _container_name(tgt.value), tgt.slice, node.lineno)
                elif isinstance(tgt, ast.Name) and isinstance(node.value, ast.Dict):
                    for k, _ in _dict_items(node.value):
                        _record(path, qualname, tgt.id, k, node.lineno)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update" and node.args
                and isinstance(node.args[0], ast.Dict)):
            for k, _ in _dict_items(node.args[0]):
                _record(path, qualname, _container_name(node.func.value), k, node.lineno)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setdefault" and node.args):
            _record(path, qualname, _container_name(node.func.value), node.args[0], node.lineno)
        if isinstance(node, ast.Dict):
            for k, _ in _dict_items(node):
                _record(path, qualname, "", k, node.lineno)
        for child in ast.iter_child_nodes(node):
            _walk_all(path, child, qualname)

    root = Path(src) if src is not None else SRC
    for path in sorted(root.rglob("*.py")):
        rel = str(path.relative_to(root))
        _walk_all(rel, _parse_file(path), "")
    return sorted(found.values())


def _island_site(write):
    """The EMITTER_SITES entry `write` sits inside, or None.

    A def NESTED in an emitter counts as that emitter, matching `_walk`, which
    descends into nested helpers (compute_game_metrics._maybe) with their
    parameters bound to the call site's constants.
    """
    for site in EMITTER_SITES:
        if write.path == site.path and (write.qualname == site.qualname
                                        or write.qualname.startswith(site.qualname + ".")):
            return site
    return None


def island_site_of(write):
    """The EMITTER_SITES qualname `write` belongs to, or None if it is outside."""
    site = _island_site(write)
    return site.qualname if site else None


def undeclared_container_writes(sweep=None):
    """Sweep hits INSIDE an emitter that write into a container the site never declares.

    THE HOLE THIS CLOSES (final review I-3). Listing a function in EMITTER_SITES
    is not only how a key gets censused — it is also what makes the island-blind
    sweep STOP asking questions about that function, because
    test_every_metrics_write_in_src_is_inside_the_island_or_declared_not_a_metric
    treats "inside the island" as the explanation for a hit. So a write inside a
    listed emitter that goes somewhere `census()` does not look was invisible to
    both walks at once, and listing the function is what made it invisible. Two
    ordinary spellings land there:

        def _emit(out):  out["game/x"] = ...   # container arrives as a PARAMETER
        alias = logs;    alias["game/x"] = ... # container reached under another name

    Both were probe-confirmed: dropped into `compute_game_metrics`, they left all
    67 metrics tests green while writing an unregistered key into every row.

    THE RULE, and why it is the right shape: a write inside an emitter must go
    into a container that emitter DECLARES. `_walk` keys on exactly those
    container names, so "declared" and "censused" are the same set — which makes
    this a cross-check between the two walks rather than a third opinion that can
    drift from either. The remedy for a failure is to add the container name to
    the site's `containers` (with its key prefix), which does not just silence
    this — it is what makes `census()` see the keys, so the registry then has to
    carry them.

    NOT covered, deliberately: a container that is neither declared nor named
    like any island container, written with BARE keys. `metrics_write_sites`'s
    own predicate drops those before they get here; see its residual list.
    """
    out = []
    for write in (metrics_write_sites() if sweep is None else sweep):
        site = _island_site(write)
        if site is not None and write.container not in site.containers:
            out.append(write)
    return sorted(out)


class MergeSource(NamedTuple):
    qualname: str                      # the EMITTER_SITES entry containing the merge
    lineno: int
    expr: str                          # the merged expression as written
    source: str                        # one hop back: what that expression is bound to
    resolved: str                      # callee name of `source`; "" when it is not a call


def _callee_name(node):
    """'compute_game_metrics' / 'evaluate' for a Call node, else ''."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
    return ""


def island_merge_sources():
    """Every NON-LITERAL dict merged into an island container, resolved one hop back.

    A merge is the one write shape `census()` is structurally unable to see: it
    ignores ``logs.update(<Call>)`` and ``logs.update(<Name>)`` on purpose, because
    the keys are not in the argument. That is correct only while every merged dict
    comes from a function that is ITSELF an emitter site — and nothing said so.
    Seven such merges exist today; five resolve to island members
    (compute_game_metrics, compute_network_health, compute_head_divergence,
    compute_trunk_divergence, tag_grad_cossim) and two are declared in
    ISLAND_MERGE_SOURCES.

    ONE HOP, exactly as `losses_entropy_head_source()` resolves `_head_names`: a
    Call gives its callee name directly; a Name is chased to the assignments that
    bind it in the same function. Two hops is where this would start guessing, so a
    Name bound to something that is not a call resolves to `""` and has to be
    declared rather than inferred. ``|=`` and keyword-form ``update(**x)`` are
    collected as well — they are merges with no argument to resolve at all, so they
    can only ever be declared.
    """
    out = []
    for site in EMITTER_SITES:
        fn = _find_qualname(_module_ast(site.path), site.qualname)
        for node in ast.walk(fn):
            merged = None
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "update"
                    and _container_name(node.func.value) in site.containers):
                if node.keywords or not node.args:
                    merged = [node]                                                  # update(**x) — no positional dict to resolve
                elif not isinstance(node.args[0], ast.Dict):
                    merged = [node.args[0]]
            elif (isinstance(node, ast.AugAssign) and isinstance(node.op, ast.BitOr)
                  and _container_name(node.target) in site.containers):
                merged = [node.value]
            else:
                continue
            for arg in merged or ():
                for source in _one_hop_bindings(fn, arg):
                    out.append(
                        MergeSource(site.qualname, node.lineno, ast.unparse(arg),
                                    ast.unparse(source), _callee_name(source)))
    return out


def _one_hop_bindings(fn, node):
    """`node` itself, or — when it is a Name — the values assigned to it in `fn`.

    A Name with no visible binding yields the Name back rather than nothing, so an
    unresolvable merge produces a record that must be declared instead of silently
    disappearing from the check.
    """
    if not isinstance(node, ast.Name):
        return [node]
    binds = [
        a.value for a in ast.walk(fn) if isinstance(a, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == node.id for t in a.targets)
    ]
    return binds or [node]


# ── Frozen-reader key literals ────────────────────────────────────────────


def reader_key_literals(rel_path):
    """Metrics-key string literals in one frozen gate script.

    AST, not a hardcoded snapshot: nothing enforces that the scripts stay
    frozen, so a snapshot would go stale silently and the completeness test
    would keep passing against a reader that has moved on.
    """
    path = REPO_ROOT / rel_path
    tree = _parse_file(path)
    candidates: list[ast.expr] = []

    def _collect_literal(node):
        """A module-level constant table: the `*_KEY` names and REPORT_EXTRA."""
        if isinstance(node, ast.Constant):
            candidates.append(node)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                _collect_literal(elt)

    for node in tree.body:
        if isinstance(node, ast.Assign):
            _collect_literal(node.value)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (node.func.attr if isinstance(node.func, ast.Attribute) else
                    node.func.id if isinstance(node.func, ast.Name) else "")
            if name not in NON_KEY_CALLS:
                candidates.extend(node.args)
                candidates.extend(kw.value for kw in node.keywords)
        elif isinstance(node, ast.Subscript):
            candidates.append(node.slice)
        elif isinstance(node, ast.Compare):
            candidates.append(node.left)
            candidates.extend(node.comparators)

    out = {}
    for node in candidates:
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        val = node.value
        if KEY_SHAPED.match(val) and ("/" in val or val in READER_BARE_KEYS):
            out.setdefault(val, node.lineno)
    return out


def _reader_module_env(tree):
    """{name: (values,)} for a frozen script's module-level string constants.

    `EPISODES_KEY = "environment/episodes"` and
    `EVAL_T, EVAL_CT = "eval/...", "eval/..."` are how the gate names half the
    keys it reads; an extractor that only saw string literals in place would miss
    every use of them.
    """
    env = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if isinstance(node.value, ast.Constant):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    env[tgt.id] = (str(node.value.value), )
        elif isinstance(node.value, ast.Tuple):
            vals = [_resolve(e, env) for e in node.value.elts]
            for tgt in node.targets:
                if isinstance(tgt, ast.Tuple):
                    for name, v in zip(tgt.elts, vals, strict=True):
                        if isinstance(name, ast.Name):
                            env[name.id] = v
    return env


def _is_reader_key_shaped(val):
    """The frozen-reader key predicate: slash-namespaced, or a declared bare key."""
    return bool(KEY_SHAPED.match(val)) and ("/" in val or val in READER_BARE_KEYS)


def reader_hidden_call_literals(rel_path):
    """Key-shaped literals a frozen reader keeps ONLY inside a `NON_KEY_CALLS` call.

    `reader_key_literals` excludes `.replace` / `.compile` / `Path` arguments by
    POSITION, which is the right call — a display prefix and a filesystem path are
    not metrics keys. But a positional exclusion is exactly the kind of rule that
    gets widened to make a red test green: append `"get"` to `NON_KEY_CALLS` and
    every real key read in both scripts leaves the surface silently.

    This is the other side of that exclusion. It returns what the exclusion HIDES,
    so `test_metrics_schema` can require each hidden literal to be named in
    `READER_OUT_OF_SURFACE` with a reason. Widening `NON_KEY_CALLS` then moves keys
    into an unexplained set and fails, and a literal that stops being hidden makes
    its declaration stale and fails too — the exemption is enumerated, not blanket.

    `smoke_read.py`'s `MOVE_KEY_RE` pattern is not returned: a regex with `^`
    and `(\\d+)` in it fails `KEY_SHAPED`, so it is out by SHAPE and never needed
    the positional rule. The registry entry for `environment/action_move_*` names
    the regex in prose instead.
    """
    path = REPO_ROOT / rel_path
    tree = _parse_file(path)
    visible = set(reader_key_literals(rel_path))
    out = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = (node.func.attr if isinstance(node.func, ast.Attribute) else
                node.func.id if isinstance(node.func, ast.Name) else "")
        if name not in NON_KEY_CALLS:
            continue
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                    and _is_reader_key_shaped(arg.value) and arg.value not in visible):
                out.setdefault(arg.value, arg.lineno)
    return out


# Key-shaped literals the frozen readers contain that are NOT metrics keys at all,
# each with the reason. Gated in BOTH directions by
# test_every_frozen_reader_key_is_registered_or_declared_out_of_surface: an entry
# matching no hidden literal is stale and fails, and a hidden literal with no entry
# is unexplained and fails.


class ReaderOutOfSurface(NamedTuple):
    script: str                        # relative to the repo root
    literal: str
    reason: str


READER_OUT_OF_SURFACE = (
    ReaderOutOfSurface(
        "src/cs2rl/experiment/gate.py", "eval/win_vs_random_",
        "A DISPLAY PREFIX. print_report shortens the two eval column HEADINGS with "
        "`.replace('eval/win_vs_random_', 'eval_')` so they fit a 22-char field. It is a "
        "fragment of two real keys, not a key — `eval/win_vs_random_as_t` and `_as_ct` are "
        "registered separately and read from the row under their full names."),
    ReaderOutOfSurface(
        "src/cs2rl/experiment/smoke_read.py", "outputs/checkpoints/rung1a/s0",
        "A FILESYSTEM PATH (RUN_DIR_DEFAULT), slash-namespaced by coincidence. It names the "
        "registered-evidence run directory the smoke read defaults to, not a metrics key."),
)


def reader_derived_column_sources(rel_path="src/cs2rl/experiment/gate.py"):
    """{report column: frozenset(emitted keys the script reads to compute it)}.

    WHY, given that `reader_report_columns` already forces every column to be
    registered: registration alone says a NAME exists. The registry additionally
    claims, per derived column, WHICH emitted keys it is computed from — and that
    `inputs` column is hand-written documentation about code in a frozen script,
    with nothing tying the two together. Swap `hit_per_on_target`'s numerator in
    the gate and the registry keeps describing the old ratio; every test stays
    green. This derives the same fact from the script's own source so the claim is
    checked instead of asserted.

    It also makes the brief's classification of the five bare `REPORT_ONLY`
    literals structural rather than prose: `shots_fired` resolves to
    `game/shots_fired` + the episode weight, and `rows` resolves to NOTHING, which
    is what "window bookkeeping, not an emitted key" means when a test says it.

    HOW, and where it deliberately stops:
      * `REPORT_EXTRA` rows carry `(column, kind, (source keys...))`, so the ARGUMENT
        tuple is read, not just the column name. The `kind` is resolved against
        `report_extra`'s own if/elif chain to find which helper that kind dispatches
        to, so the episode weight `ratio` adds — and `median`/`p90`/`last` do not —
        comes from the source rather than from a hardcoded assumption here.
      * `GATES` / `REPORT_ONLY` columns are computed in `seed_metrics`'s `m = {...}`
        literal. A call to a helper DEFINED IN THE SAME SCRIPT is followed into its
        body (that is how `weighted_sum`'s `EPISODES_KEY` is found); a call to
        anything else is not.
      * A local Name is chased to its binding ONLY at the ROOT of a column's value
        expression (`m["shots_fired"] = shots_fired`), never as a call ARGUMENT.
        That line matters: `window` is an argument everywhere, and following it
        reaches `gate_window`, whose `self_play/used_past` and `agent_steps` are how
        the WINDOW is selected, not what any column is computed from. Chasing it
        credits every column with both.
    """
    path = REPO_ROOT / rel_path
    tree = _parse_file(path)
    env = _reader_module_env(tree)
    mod_fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}

    def _named_keys(node):
        """Keys a Constant or module-constant Name denotes."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value} if _is_reader_key_shaped(node.value) else set()
        if isinstance(node, ast.Name):
            return {v for v in env.get(node.id) or () if _is_reader_key_shaped(v)}
        return set()

    def _body_keys(fn, seen):
        """Every key literal reachable in a same-script helper's body."""
        out = set()
        for sub in ast.walk(fn):
            out |= _named_keys(sub)
            if isinstance(sub, ast.Name) and sub.id in mod_fns and sub.id not in seen:
                out |= _body_keys(mod_fns[sub.id], seen | {sub.id})
        return out

    def _sources(node, fn, seen, root):
        out = set()
        if isinstance(node, ast.Constant):
            return _named_keys(node)
        if isinstance(node, ast.Name):
            named = _named_keys(node)
            if named or not root or node.id in seen:
                return named
            for bind in _one_hop_bindings(fn, node):
                if bind is not node:
                    out |= _sources(bind, fn, seen | {node.id}, True)
            return out
        if isinstance(node, ast.Call):
            callee = _callee_name(node)
            if callee in mod_fns and callee not in seen:
                out |= _body_keys(mod_fns[callee], seen | {callee})
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                out |= _sources(arg, fn, seen, False)
            return out
        # `root` survives only through arithmetic, so `a / b` keeps chasing both
        # operands while a comprehension's `for r in window` does not.
        deeper = root and isinstance(node, (ast.BinOp, ast.UnaryOp))
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.expr):
                out |= _sources(child, fn, seen, deeper)
        return out

    out = {}
    seed_metrics = mod_fns.get("seed_metrics")
    if seed_metrics is None:
        raise AssertionError("src/cs2rl/experiment/gate.py has no seed_metrics — every GATES and "
                             "REPORT_ONLY column's provenance was read from its `m = {...}` "
                             "literal; this extractor now measures nothing")
    for node in ast.walk(seed_metrics):
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict)
                and any(isinstance(t, ast.Name) and t.id == "m" for t in node.targets)):
            continue
        for key_node, value in _dict_items(node.value):
            for col in _resolve(key_node, env) or ():
                out[col] = frozenset(_sources(value, seed_metrics, set(), True))

    # REPORT_EXTRA: (column, kind, (args...)). The kind→helper map comes from
    # report_extra's own `if kind == "..."` chain.
    report_extra = mod_fns.get("report_extra")
    kind_callees = {}
    for node in ast.walk(report_extra) if report_extra else ():
        if not (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)):
            continue
        lits = [
            c.value for c in [node.test.left] + node.test.comparators
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        ]
        if lits:
            kind_callees.setdefault(lits[0], set()).update(
                _callee_name(c) for c in ast.walk(node) if isinstance(c, ast.Call))

    for node in tree.body:
        if not (isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple)
                and any(isinstance(t, ast.Name) and t.id == "REPORT_EXTRA" for t in node.targets)):
            continue
        for row in node.value.elts:
            # Each row is a (column, "kind", (args...)) literal; any other shape fails here.
            assert isinstance(row, (ast.Tuple, ast.List)), ast.unparse(row)
            kind_node = row.elts[1]
            assert isinstance(kind_node, ast.Constant), ast.unparse(row)
            assert isinstance(kind_node.value, str), ast.unparse(row)
            col = (_resolve(row.elts[0], env) or (None, ))[0]
            kind = kind_node.value
            keys = set()
            for elt in ast.walk(row.elts[2]):
                keys |= _named_keys(elt)
            for callee in kind_callees.get(kind, ()):
                if callee in mod_fns:
                    keys |= _body_keys(mod_fns[callee], {callee})
            out[col] = frozenset(keys)
    return out


def reader_report_columns(rel_path="src/cs2rl/experiment/gate.py"):
    """Column names of the gate's (experiment/gate.py) GATES / REPORT_ONLY / REPORT_EXTRA tables.

    These are the DERIVED report columns (`kills_per_episode` the ratio, not
    `game/kills_per_episode` the emitted key) plus the two eval/* keys the gate
    reads directly. Extracting them is what forces `derived` registry entries to
    exist; extracting ONLY them would enforce nothing, which is why
    reader_key_literals above exists alongside.
    """
    path = REPO_ROOT / rel_path
    tree = _parse_file(path)
    env = _reader_module_env(tree)
    cols = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Tuple):
            continue
        names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        if not names & {"GATES", "REPORT_ONLY", "REPORT_EXTRA"}:
            continue
        for elt in node.value.elts:
            first = elt.elts[0] if isinstance(elt, ast.Tuple) else elt
            vals = _resolve(first, env)
            cols.update(vals or ())
    return cols


# In-repo consumers of the metrics row, and how each one names the keys it
# reads: (consumer name, module, qualname, receiver of the .get call, prefix).
# `compute_game_metrics` reads through its own `_get` helper, which tries
# `environment/<k>` before the bare `<k>` — hence the prefix.
IN_REPO_CONSUMERS = (
    ("format_train_status", "train/metrics.py", "format_train_status", "logs", ""),
    ("elimination_only_win_rates", "train/metrics.py", "elimination_only_win_rates", "logs", ""),
    ("compute_game_metrics", "train/metrics.py", "compute_game_metrics", "_get", "environment/"),
)


def consumer_key_reads():
    """{consumer name: {key: lineno}} for every named reader of a metrics row.

    Derived, so `metrics_schema`'s `consumers` column is checked in BOTH
    directions: a key a reader reads must name that reader, and a key that
    names a reader must actually be read by it. A hand-written consumers column
    is the part of a registry that rots first — it is documentation about code
    somewhere else, with nothing tying the two together.
    """
    out = {
        name: dict(reader_key_literals(path))
        for name, path in (
            ("rung1_gate", "src/cs2rl/experiment/gate.py"),
            ("rung1a_smoke_read", "src/cs2rl/experiment/smoke_read.py"),
        )
    }
    for name, path, qualname, receiver, prefix in IN_REPO_CONSUMERS:
        fn = _find_qualname(_module_ast(path), qualname)
        keys = {}
        nested = _nested_call_args(fn)
        _walk_reads(fn, fn, receiver, prefix, {}, nested, keys)
        out[name] = keys
    return out


def _walk_reads(node, fn, receiver, prefix, env, nested, keys):
    """Collect `<receiver>.get("key")` / `_get("key")` reads under loop bindings.

    The env-aware walk matters: compute_game_metrics reads seven reward keys and
    nine combat counters through `for src, dst in (...)` / `for k in (...)`
    tables and three more through a nested `_maybe(src, dst)` helper. A
    constants-only scan sees none of those — it would find 13 of the 29 keys the
    function actually reads and declare the consumers column complete.
    """
    if isinstance(node, ast.For):
        env = _bind_for(node, env)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not fn:
        for args in nested.get(node.name, ()):
            inner = dict(env)
            for param, arg in zip(node.args.args, args, strict=False):
                inner[param.arg] = _resolve(arg, env)
            for child in node.body:
                _walk_reads(child, fn, receiver, prefix, inner, nested, keys)
        return
    if isinstance(node, ast.Call) and node.args:
        is_read = ((isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                    and _container_name(node.func.value) == receiver)
                   or (isinstance(node.func, ast.Name) and node.func.id == receiver))
        if is_read:
            for val in _resolve(node.args[0], env) or ():
                if KEY_SHAPED.match(val):
                    keys.setdefault(prefix + val, node.lineno)
    for child in ast.iter_child_nodes(node):
        _walk_reads(child, fn, receiver, prefix, env, nested, keys)


def eval_output_keys():
    """Keys of the dict literal BaselineEvaluator.evaluate() returns.

    The runtime guard in eval.baselines (`set(out) != set(EVAL_KEYS)` → raise)
    only fires when an eval actually runs, and the §3 gate runs with
    `--eval-interval 0`. This is the same contract checked from source, so it
    holds in a suite that never constructs an evaluator.
    """
    tree = _module_ast("eval/baselines.py")
    fn = _find_qualname(tree, "BaselineEvaluator.evaluate")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "out"
                and isinstance(node.value, ast.Dict)):
            return {k.value for k, _ in _dict_items(node.value) if isinstance(k, ast.Constant)}
    raise AssertionError("no `out = {...}` dict literal in BaselineEvaluator.evaluate")


def losses_entropy_head_source():
    """Name of the constant the per-head `losses/entropy/<head>` loop iterates.

    `losses/entropy/*` is an OPEN family to this extractor — the loop zips an
    imported Name with the policy's distributions, and the rule that only LITERAL
    iterables bind a loop variable stops short of it on purpose. metrics_schema
    therefore declares that family's members FROM `spec.action.ACTION_HEAD_NAMES`,
    which is circular unless something pins that the emitter reads the same tuple.
    This is that pin: it returns the constant's name, so re-pointing the emitter at
    a different head list fails the test instead of silently leaving the registry
    describing the old heads. A local copy (`names = list(ACTION_HEAD_NAMES)`) is
    resolved one hop, to the constant it copies.
    """
    fn = _find_qualname(_module_ast("train/trainer.py"), _GH90_SUMS_SITE)
    # The write we are anchored on: `sums[f"entropy/{name}"] += ...` inside a
    # `for ... in zip(<names>, ...)`. Walk outwards from the write to its loop.
    for loop in (n for n in ast.walk(fn) if isinstance(n, ast.For)):
        writes = [
            n for n in ast.walk(loop)
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript)
            and _container_name(n.target.value) == GH90_ACCUMULATOR and isinstance(
                n.target.slice, ast.JoinedStr) and "entropy/" in ast.unparse(n.target.slice)
        ]
        if not writes:
            continue
        it = loop.iter
        if isinstance(it, ast.Call) and _callee_name(it) == "enumerate" and it.args:
            it = it.args[0]
        if (isinstance(it, ast.Call) and _callee_name(it) == "zip" and it.args
                and isinstance(it.args[0], ast.Name)):
            for bound in _one_hop_bindings(fn, it.args[0]):
                inner = [n.id for n in ast.walk(bound) if isinstance(n, ast.Name)]
                if inner:
                    return inner[-1]
    raise AssertionError(
        f"no `{GH90_ACCUMULATOR}[f\"entropy/{{...}}\"] += ...` loop over `zip(<names>, ...)` found "
        f"in {_GH90_SUMS_SITE} — the per-head entropy family moved, and metrics_schema's "
        "losses/entropy/* member list is no longer tied to anything")


def tag_key_axes():
    """The two placeholder axes of `tag_grad_cossim`'s keys, read from source.

    Returns ``{"group": (...), "label": (...)}`` — the values `g` and `mb_label`
    take in ``f"tag/<stat>/{g}/{mb_label}"``. `census()` already resolves both
    (that is what makes the seven families closed); this names them separately so
    a test can say WHICH axis moved, and so a silent reopening — both the census
    and the registry losing the member list in the same edit, leaving `tag/*` an
    open glob that alibis any invented key — fails instead of passing quietly.

    An axis that stops resolving comes back as `()`, which is a failure for the
    caller rather than something to paper over here.
    """
    fn = _find_qualname(_module_ast("train/update.py"), "tag_grad_cossim")
    site = next(s for s in EMITTER_SITES if s.qualname == "tag_grad_cossim")
    literals = _local_literal_bindings(fn)
    loops = [(n, n.target.id) for n in ast.walk(fn)
             if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and any(
                 isinstance(w, ast.Subscript) and _container_name(w.value) == "out"
                 for w in ast.walk(n))]
    group = ()
    for loop, target in loops:
        bound = _bind_for(loop, {}, literals).get(target)
        if bound:
            group = tuple(bound)
            break
    return {"group": group, "label": tuple(emitter_param_bindings(site, fn).get("mb_label", ()))}


def stats_collection_is_append_shaped():
    """True iff Cs2PuffeRL._collect_infos accumulates episode infos into `self.stats` as LISTS.

    The `window-mean-pufferlib` aggregation of every `environment/*` key rests
    on this loop appending to a list that PufferLib later np.means. If it were
    ever rewritten to `self.stats[k] = v`, every one of those declarations
    would become wrong at once — and no other default-tier test would notice:
    the behaviour check (a real rollout, in tests/train/test_trainer_composition.py)
    runs only under `-m training`. Both halves are required: an `append` (the scalar branch, which is what the
    terminal infos' scalars take) and no store into `self.stats[...]`. Requiring
    only "some append or extend" stayed green with the scalar branch turned into
    an assignment (gh#92 knock-out K7).
    """
    tree = _module_ast("train/trainer.py")
    fn = _find_qualname(tree, "Cs2PuffeRL._collect_infos")
    appends = stores = 0
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append" and isinstance(node.func.value, ast.Subscript)
                and _container_name(node.func.value.value) == "self.stats"):
            appends += 1
        if (isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load)
                and _container_name(node.value) == "self.stats"):
            stores += 1
    return appends > 0 and stores == 0
