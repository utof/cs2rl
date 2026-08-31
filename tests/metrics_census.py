"""AST census of the metrics keys this repo's own emitters write (W4, spec 2026-08-31).

WHAT: a no-import, source-only extractor. Given the NAMED island of emitter
functions (``EMITTER_SITES``) it returns, for every metrics key those functions
write, the key itself plus the *shape* of the write — which is what
``src/metrics_schema.py``'s declared ``aggregation`` is checked against in
tests/test_metrics_schema.py.

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
string constant in the function": ``_train_with_return_norm`` is ~700 lines and
mentions dozens of identifier-shaped strings that are config keys, tensor group
names and dict labels. Collecting all of them and then exempting the
false positives is how an extractor gets loosened until it enforces nothing.
Only subscript-assignment targets, dict literals that reach a metrics
container, and ``<container>.update({...})`` arguments count.
"""
import ast
import re
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# ── The named emitter island ──────────────────────────────────────────────
#
# (relative path under src/, qualname, container names that hold metrics keys,
#  prefix applied to slash-free keys).
#
# The prefix column is the difference between a key as WRITTEN and the key as it
# reaches metrics.jsonl: PufferLib's mean_and_log re-keys `self.stats` under
# `environment/` and `self.losses` under `losses/` (pufferl.py mean_and_log),
# and Cs2Env._build_terminal_info's summary dict becomes `self.stats` entries via
# train.py's info-collection loop. Writing the prefix down here — rather than
# registering the bare names — is what makes the registry's keys the keys an
# analyst actually greps for.
#
# `self_play_used_past_metric` is on the list although it emits nothing: it is
# named by the spec as one of "our emitters", and an empty result from it is a
# fact worth re-checking rather than an omission worth wondering about. Its
# caller (train.py `train`) is what writes `self_play/used_past`.
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


EMITTER_SITES = (
    EmitterSite("train_metrics.py", "compute_network_health", {"metrics": ""}),
    EmitterSite("train_metrics.py", "log_aim_log_std", {"logs": ""}),
    EmitterSite("train_metrics.py", "compute_head_divergence", {"out": ""}),
    EmitterSite("train_metrics.py", "compute_trunk_divergence", {"out": ""}),
    EmitterSite("train_metrics.py", "ScheduledEval.after_train", {"self.pending": ""}),
    EmitterSite("train_metrics.py", "compute_game_metrics", {"game_metrics": ""}),
    EmitterSite("train_metrics.py", "_inject_tag_metrics", {"logs": ""}),
    EmitterSite("train_update.py", "_patch_trainer_with_return_norm._train_with_return_norm", {
        "losses": "losses/",
        "self.stats": "environment/",
        "trainer._tag_metrics": "",
    }),
    EmitterSite("train_update.py", "tag_grad_cossim", {"out": ""}),
    EmitterSite("train.py", "train", {
        "logs": "",
        "log_entry": ""
    }),
    EmitterSite("train.py", "_patch_trainer_with_timing._timed_train", {"result": ""}),
    EmitterSite("train.py", "self_play_used_past_metric", {"logs": ""}),
    EmitterSite("c_env/cs2_env.py", "Cs2Env._build_terminal_info", {"summary": "environment/"}),
    EmitterSite("c_env/cs2_env.py", "Cs2Env.step", {"summary": "environment/"}),
)

# ── The island's NEGATIVE list ────────────────────────────────────────────
#
# `metrics_write_sites()` sweeps all of src/ for metrics-SHAPED writes without
# consulting EMITTER_SITES, and every hit it finds outside the island has to be
# named here with the reason it is not an emitter. This is the audit that
# otherwise gets redone from scratch by every reviewer — three of these six look
# exactly like emitters to a grep (`metrics[...]`, `summary = {...}`,
# `stats = {...}` with the same bare key names cs2_env uses) and are not.
#
# The list cannot rot into a blanket exemption: an entry that matches no hit fails
# too, so deleting an emitter-shaped site here is as loud as adding one.


class NonIslandWrite(NamedTuple):
    path: str                          # relative to src/
    qualname: str                      # "" = the whole file, including module scope
    reason: str


NON_ISLAND_WRITES = (
    NonIslandWrite(
        "metrics_schema.py", "", "The registry ITSELF. Its ~200 dict-literal keys are "
        "declarations, not writes into a metrics row — they are the thing the census is "
        "compared against, so counting them as emissions would make every completeness "
        "test compare the registry with itself."),
    NonIslandWrite(
        "eval_baselines.py", "BaselineEvaluator.evaluate",
        "A REAL source of row keys, censused by its own extractor (`eval_output_keys()`) "
        "rather than as an island site: ScheduledEval merges the returned dict wholesale, "
        "so there is no per-key write for `_walk` to classify a shape from."),
    NonIslandWrite(
        "eval_baselines.py", "BaselineEvaluator._episode",
        "Per-episode RETURN VALUE of the evaluator's inner loop (shots_fired, "
        "shots_with_enemy_in_los, timed_out), consumed by evaluate() to build the eval/* "
        "numbers. Never written into a row itself."),
    NonIslandWrite(
        "train.py", "evaluate_checkpoint",
        "`--eval` mode's local Counter, PRINTED TO STDOUT. Its 13 bare keys are the same "
        "names cs2_env's terminal info uses, which is why a grep-based audit reads it as "
        "an emitter; nothing here reaches metrics.jsonl."),
    NonIslandWrite(
        "train.py", "convert_legacy_state_dict_to_split",
        "state_dict TENSOR names (`aim_log_std_t`/`_ct`), not metrics keys — checkpoint "
        "surgery for the legacy→split migration."),
    NonIslandWrite(
        "train_bc.py", "eval_plant_rate",
        "The BC-eval report dict (`summary = {...}`) with its own printer — a separate "
        "analysis path, never merged into a training row."),
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
        "_inject_tag_metrics", "getattr(trainer, '_tag_metrics', None)",
        "A re-read of the island's OWN `trainer._tag_metrics` container: every key in it "
        "was written under _train_with_return_norm / tag_grad_cossim, both of which are "
        "EMITTER_SITES entries, so the merge adds no key the census has not already seen."),
)

# The two frozen gate readers (spec: never migrated, so the registry has to
# chase THEM). Only their own key literals are in scope — not their row loader
# (`scripts/analyze_tplant.py`), which #155 records as future-reader residue.
FROZEN_READERS = ("scripts/rung1_gate.py", "scripts/rung1a_smoke_read.py")

# A frozen reader's key literal: slash-namespaced, or the one bare key the gate
# scripts read (`agent_steps`, PufferLib's own step counter — no *_KEY constant
# exists for it, it appears inline at three sites in rung1a_smoke_read.py).
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
#   .replace  — `rung1_gate.print_report` shortens a COLUMN HEADING with
#               `.replace('eval/win_vs_random_', 'eval_')`; the spec puts that
#               literal out of surface entirely.
#   .compile  — rung1a_smoke_read's MOVE_KEY_RE. Its family gets a
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
                                                       # this way on purpose (train_update.py, see its comment).
    "stats-one-element-list": "last",
                                                       # `self.stats[k]` fed by the append/extend collection loop — PufferLib
                                                       # np.means the whole collection window.
    "stats-window": "window-mean-pufferlib",
                                                       # `losses[k] += ...` BEFORE the gh#90 divisor loop: the sum is divided by
                                                       # the executed-minibatch count, i.e. a mean over minibatches.
    "losses-accumulated": "mean",
                                                       # `losses[k] = ...` AFTER the divisor loop: an absolute epoch scalar. The
                                                       # gh#90 comment at that loop names the trap this distinction encodes —
                                                       # anything written before the loop is silently scaled by 1/minibatches.
    "losses-absolute": "last",
                                                       # Re-keyed inside compute_game_metrics out of values `_get()` read from
                                                       # `logs`, i.e. out of PufferLib window means that mean_and_log already
                                                       # computed. Pass-through: the aggregation belongs to the upstream pipeline.
    "game-passthrough": "window-mean-pufferlib",
                                                       # Written straight into the outer `logs` dict (or a dict merged into it)
                                                       # AFTER mean_and_log returned — one value per logged row, no windowing.
    "logs-post-mean": "last",
                                                       # Written straight into the persisted `log_entry` row in train.py, never
                                                       # through PufferLib at all (run_id, step, epoch, team_spirit,
                                                       # resumed_from_step). Same "one value per row" contract as the above; kept
                                                       # separate so the registry says which pipeline a key belongs to.
    "row-literal": "last",
                                                       # A `summary[k] = <bare attribute>` write — the ONE shape in the terminal-info
                                                       # dict that is not wrapped in int()/float(): `summary["step_stats"] =
                                                       # self._step_stats_view`. mean_and_log's np.mean raises on a list of those
                                                       # and train.py's isinstance(v, (int, float)) persist filter drops the key, so
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


def _module_ast(rel_path):
    return ast.parse((SRC / rel_path).read_text(), filename=str(SRC / rel_path))


def _find_qualname(tree, qualname):
    """The FunctionDef/ClassDef node at `qualname` ('A.b.c'), or raise.

    Raising rather than returning None is deliberate: a renamed emitter must
    break this file loudly. Silently censusing zero keys for a site that moved
    is the failure mode that would make every downstream assertion vacuous.
    """
    parts = qualname.split(".")
    node = tree
    for part in parts:
        nxt = None
        for child in ast.walk(node) if node is tree else ast.iter_child_nodes(node):
            if isinstance(child,
                          (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and (child.name
                                                                                      == part):
                nxt = child
                break
        if nxt is None:
            raise AssertionError(f"emitter {qualname!r} not found — it was renamed or moved; "
                                 "update EMITTER_SITES in tests/metrics_census.py")
        node = nxt
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


def _local_literal_bindings(fn):
    """{name: (values,)} for names bound ONCE to a literal iterable inside `fn`.

    Assigned-once is the whole safety rule: a name rebound in a branch is not a
    constant, so it resolves to nothing and its family stays OPEN. Measured blast
    radius across the island today: exactly one name, `tag_grad_cossim`'s
    `pg_group_names` — every other non-literal iterable in an emitter is a
    method call (`model.named_parameters()`, `subsets.items()`) or a `zip`/
    `enumerate`, none of which this reaches.
    """
    counts, values = {}, {}
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                counts[tgt.id] = counts.get(tgt.id, 0) + 1
                values[tgt.id] = _literal_strings(node.value)
    return {n: tuple(v) for n, v in values.items() if v is not None and counts[n] == 1}


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
        cols = {}
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


def _divisor_lineno(fn):
    """Line of `for _lk in list(losses): losses[_lk] /= ...` (gh#90).

    Every `losses` key written before this line is divided by the executed
    minibatch count; every key written after it is an absolute. Located by
    shape, so moving the loop moves the classification with it — and deleting
    it fails here rather than silently reclassifying every loss key.
    """
    for node in ast.walk(fn):
        if not isinstance(node, ast.For):
            continue
        for sub in node.body:
            if (isinstance(sub, ast.AugAssign) and isinstance(sub.op, ast.Div)
                    and isinstance(sub.target, ast.Subscript)
                    and _container_name(sub.target.value) == "losses"):
                return node.lineno
    raise AssertionError(
        "the gh#90 `for _lk in list(losses): losses[_lk] /= ...` divisor loop is gone from "
        "_train_with_return_norm — every losses/* aggregation in metrics_schema.py is "
        "classified relative to it, so its removal is a registry-wide event, not a refactor")


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
    if container == "losses":
        return "losses-accumulated" if lineno < ctx["divisor_lineno"] else "losses-absolute"
    if container == "summary":
        # cs2_env's terminal-info dict is appended into self.stats by train.py's
        # collection loop, one entry per episode → a PufferLib window mean.
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

    writes = []
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for tgt in targets:
            if isinstance(tgt, ast.Subscript) and _container_name(tgt.value) in site.containers:
                writes.append((_container_name(tgt.value), tgt.slice, node.value, node.lineno))
            elif (isinstance(tgt, ast.Name) and tgt.id in site.containers
                  and isinstance(node.value, ast.Dict)):
                for k, v in _dict_items(node.value):
                    writes.append((tgt.id, k, v, getattr(k, "lineno", node.lineno)))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if (node.func.attr == "update" and _container_name(node.func.value) in site.containers
                and node.args and isinstance(node.args[0], ast.Dict)):
            base = _container_name(node.func.value)
            for k, v in _dict_items(node.args[0]):
                writes.append((base, k, v, getattr(k, "lineno", node.lineno)))

    for container, key_node, value, lineno in writes:
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
    ``mb_label="mb0" if _tag_mb0 else "mbL"``: two constants, an `ast.IfExp` that
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
    """
    fname = site.qualname.split(".")[-1]
    params = [a.arg for a in fn.args.posonlyargs] + [a.arg for a in fn.args.args]
    kwonly = [a.arg for a in fn.args.kwonlyargs]

    calls = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _callee_name(node) == fname:
                calls.append(node)
    if not calls:
        return {}

    out = {}
    for pos, name in enumerate(params + kwonly):
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
    """(emitted keys, key families) over the whole emitter island."""
    keys, families = [], []
    for site in EMITTER_SITES:
        fn = _find_qualname(_module_ast(site.path), site.qualname)
        ctx = {
            "divisor_lineno":
            _divisor_lineno(fn) if "losses" in site.containers else 0,
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
    path: str                          # relative to src/
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


def metrics_write_sites():
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
                       name, a bare dict literal anywhere, and ``c.update({...})``.
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
      * a key that reaches a row without a write shape at all, e.g. PufferLib's
        own `mean_and_log` literals (covered instead by PUFFERLIB_OWNED and a
        cross-package AST pin) or `**` splats of a non-literal mapping (covered by
        `island_merge_sources()` for island containers only).
    """
    containers = {c.split(".")[-1] for s in EMITTER_SITES for c in s.containers}
    found = {}

    def _record(path, qualname, container, key_node, fallback_lineno):
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
        if isinstance(node, (ast.Assign, ast.AugAssign)):
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
        if isinstance(node, ast.Dict):
            for k, _ in _dict_items(node):
                _record(path, qualname, "", k, node.lineno)
        for child in ast.iter_child_nodes(node):
            _walk_all(path, child, qualname)

    for path in sorted(SRC.rglob("*.py")):
        rel = str(path.relative_to(SRC))
        _walk_all(rel, ast.parse(path.read_text(), filename=str(path)), "")
    return sorted(found.values())


def island_site_of(write):
    """The EMITTER_SITES qualname `write` belongs to, or None if it is outside.

    A def NESTED in an emitter counts as that emitter, matching `_walk`, which
    descends into nested helpers (compute_game_metrics._maybe) with their
    parameters bound to the call site's constants.
    """
    for site in EMITTER_SITES:
        if write.path == site.path and (write.qualname == site.qualname
                                        or write.qualname.startswith(site.qualname + ".")):
            return site.qualname
    return None


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
    tree = ast.parse(path.read_text(), filename=str(path))
    candidates = []

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


def reader_report_columns(rel_path="scripts/rung1_gate.py"):
    """Column names of rung1_gate's GATES / REPORT_ONLY / REPORT_EXTRA tables.

    These are the DERIVED report columns (`kills_per_episode` the ratio, not
    `game/kills_per_episode` the emitted key) plus the two eval/* keys the gate
    reads directly. Extracting them is what forces `derived` registry entries to
    exist; extracting ONLY them would enforce nothing, which is why
    reader_key_literals above exists alongside.
    """
    path = REPO_ROOT / rel_path
    tree = ast.parse(path.read_text(), filename=str(path))
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name):
                    env[tgt.id] = (str(node.value.value), )
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Tuple):
            vals = [_resolve(e, env) for e in node.value.elts]
            for tgt in node.targets:
                if isinstance(tgt, ast.Tuple):
                    for name, v in zip(tgt.elts, vals, strict=True):
                        if isinstance(name, ast.Name):
                            env[name.id] = v
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
    ("format_train_status", "train.py", "format_train_status", "logs", ""),
    ("elimination_only_win_rates", "train.py", "elimination_only_win_rates", "logs", ""),
    ("compute_game_metrics", "train_metrics.py", "compute_game_metrics", "_get", "environment/"),
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
            ("rung1_gate", "scripts/rung1_gate.py"),
            ("rung1a_smoke_read", "scripts/rung1a_smoke_read.py"),
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

    The runtime guard in eval_baselines (`set(out) != set(EVAL_KEYS)` → raise)
    only fires when an eval actually runs, and the §3 gate runs with
    `--eval-interval 0`. This is the same contract checked from source, so it
    holds in a suite that never constructs an evaluator.
    """
    tree = _module_ast("eval_baselines.py")
    fn = _find_qualname(tree, "BaselineEvaluator.evaluate")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "out"
                and isinstance(node.value, ast.Dict)):
            return {k.value for k, _ in _dict_items(node.value) if isinstance(k, ast.Constant)}
    raise AssertionError("no `out = {...}` dict literal in BaselineEvaluator.evaluate")


def losses_entropy_head_source():
    """Name of the constant the per-head `losses/entropy/<head>` loop iterates.

    `losses/entropy/*` is an OPEN family to this extractor — the loop runs over a
    local bound to `list(ACTION_HEAD_NAMES)`, an imported Name, and the rule that
    only LITERAL iterables bind a loop variable stops one hop short of it on
    purpose. metrics_schema therefore declares that family's members FROM
    `_action_spec.ACTION_HEAD_NAMES`, which is circular unless something pins that
    the emitter reads the same tuple. This is that pin: it returns the constant's
    name, so re-pointing the emitter at a different head list fails the test
    instead of silently leaving the registry describing the old heads.
    """
    fn = _find_qualname(_module_ast("train_update.py"),
                        "_patch_trainer_with_return_norm._train_with_return_norm")
    # The write we are anchored on: `losses[f"entropy/{_hn}"] += ...` inside a
    # `for ... in zip(<names>, ...)`. Walk outwards from the write to its loop.
    loops = [n for n in ast.walk(fn) if isinstance(n, ast.For)]
    for loop in loops:
        writes = [
            n for n in ast.walk(loop)
            if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript)
            and _container_name(n.target.value) == "losses" and isinstance(
                n.target.slice, ast.JoinedStr) and "entropy/" in ast.unparse(n.target.slice)
        ]
        if not writes:
            continue
        # `for _hi, (_hn, _hd) in enumerate(zip(_head_names, _dists, strict=True))`
        names = [n.id for n in ast.walk(loop.iter) if isinstance(n, ast.Name)]
        for local in names:
            # Resolve one hop: the `_head_names = list(ACTION_HEAD_NAMES)` binding.
            for node in ast.walk(fn):
                if (isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == local):
                    inner = [n.id for n in ast.walk(node.value) if isinstance(n, ast.Name)]
                    if inner:
                        return inner[-1]
    raise AssertionError(
        "no `losses[f\"entropy/{...}\"] += ...` loop found in _train_with_return_norm — the "
        "per-head entropy family moved, and metrics_schema's losses/entropy/* member list "
        "is no longer tied to anything")


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
    fn = _find_qualname(_module_ast("train_update.py"), "tag_grad_cossim")
    site = next(s for s in EMITTER_SITES if s.qualname == "tag_grad_cossim")
    literals = _local_literal_bindings(fn)
    loops = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and any(
            isinstance(w, ast.Subscript) and _container_name(w.value) == "out" for w in ast.walk(n))
    ]
    group = ()
    for loop in loops:
        bound = _bind_for(loop, {}, literals).get(loop.target.id)
        if bound:
            group = tuple(bound)
            break
    return {"group": group, "label": tuple(emitter_param_bindings(site, fn).get("mb_label", ()))}


def stats_collection_is_append_shaped():
    """True iff train.py accumulates episode infos into `self.stats` as LISTS.

    The `window-mean-pufferlib` aggregation of every `environment/*` key rests
    on this loop appending to a list that PufferLib later np.means. If it were
    ever rewritten to `self.stats[k] = v`, every one of those declarations
    would become wrong at once — and nothing else in the suite would notice.
    """
    tree = _module_ast("train.py")
    fn = _find_qualname(tree, "_patch_trainer_with_selfplay._evaluate_with_selfplay")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("append", "extend")
                and isinstance(node.func.value, ast.Subscript)
                and _container_name(node.func.value.value) == "self.stats"):
            return True
    return False
