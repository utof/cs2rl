"""src/cs2rl/eval/metrics_schema.py checked against the SOURCE of every emitter and reader.

WHAT is enforced, and why each half exists:

  * AGGREGATION (the headline assert). Every registered `emitted`/`family` entry's
    declared aggregation must equal the one implied by the SHAPE of its write, as
    extracted by `metrics_census`. Declaring `environment/episodes` a window mean,
    or moving a `losses/*` write across the gh#90 divisor loop, fails BY KEY NAME.
  * COMPLETENESS, BOTH DIRECTIONS. A key an emitter writes but the registry does
    not carry fails; a key the registry carries but nothing writes fails too. A
    one-directional check would let the registry rot by accumulation.
  * CONSUMERS, BOTH DIRECTIONS. A reader that reads a key the entry does not name
    fails; an entry that names a reader which does not read it fails. The
    consumers column is the part of a registry that rots first — it is
    documentation about code somewhere else with nothing tying the two together.
  * THE ISLAND BOUNDARY. All three of the above compare the registry against
    `census()`, which reads only the NAMED emitter island — so all three are blind
    to a metrics write in a function nobody listed. Two tests below close that: an
    island-blind sweep of every metrics-shaped write in `src/`, and a one-hop
    resolution of the dict merges the census does not look inside. Their residual
    is stated at `metrics_census.metrics_write_sites`.

WHY the checks read SOURCE and not a training run: the emitters are gated on flags
(`--tag-diagnostic`, `--eval-interval`), on architecture (split heads), on epoch
parity (`epoch % 5`) and on an episode having ended. A census taken from a short
run sees roughly half the surface and is STRUCTURALLY BLIND to the other half —
an unemitted key simply does not appear, so "the census matches the registry"
stays green while the registry silently rots. The §3 gate's two-epoch row, for
instance, carries none of `game/plant_tick`, `actions/use_at_site_frac`, the
eight split `policy/aim_log_std_*` keys or `eval/epoch`.

WHY the frozen readers are parsed rather than migrated: `cs2rl/experiment/gate.py`
and `cs2rl/experiment/smoke_read.py` are registered evidence and must not change.
The registry chases them; a hardcoded snapshot of their keys would go stale
silently, since nothing enforces that they stay frozen.
"""
import ast
import importlib.util
from pathlib import Path

import pytest

from cs2rl.eval import metrics_schema as ms

# `metrics_census`, the AST extractor this file checks the registry against, is test
# code, so it lives in tests/_helpers/ rather than in the shipped package. It is
# imported under its one name, through `tests`, which pyproject.toml's pytest
# `pythonpath = ["."]` makes importable; `test_w1_modules` below is imported the same
# way, as `tests.test_w1_modules`, the name pytest collects it under. A bare
# `import metrics_census` would need tests/_helpers/ on sys.path and would load a
# second copy; TID251 bans it and the old `tests.metrics_census`.
from tests._helpers import metrics_census as census

REPO_ROOT = Path(__file__).resolve().parents[1]

# One census per session: every test below reads the same extraction, so a
# disagreement between two tests is a registry fact, never a re-parse artifact.
EMITTED, FAMILIES = census.census()
EMITTED_BY_KEY = {k.key: k for k in EMITTED}
FAMILY_BY_TEMPLATE = {f.template: f for f in FAMILIES}
READS = census.consumer_key_reads()

# ── Well-formedness ───────────────────────────────────────────────────────


def test_every_entry_uses_the_declared_vocabularies():
    """kind / aggregation / units / consumers are closed sets, not free text.

    A free-text units column is the part of a registry nobody can check, so
    nobody maintains it; making the vocabulary closed turns "invent a unit" into
    a deliberate edit of `metrics_schema.UNITS`.
    """
    bad = []
    for key, s in sorted(ms.REGISTRY.items()):
        if s.kind not in ms.KINDS:
            bad.append(f"{key}: kind {s.kind!r}")
        if s.aggregation not in ms.AGGREGATIONS:
            bad.append(f"{key}: aggregation {s.aggregation!r}")
        if s.units not in ms.UNITS:
            bad.append(f"{key}: units {s.units!r}")
        for c in s.consumers:
            if c not in ms.CONSUMERS:
                bad.append(f"{key}: consumer {c!r}")
    assert not bad, "registry entries outside the declared vocabularies:\n  " + "\n  ".join(bad)


def test_kind_specific_fields_are_used_consistently():
    """`inputs`/`producer` belong to derived, `members` to family, and only there."""
    bad = []
    for key, s in sorted(ms.REGISTRY.items()):
        if s.inputs and s.kind != "derived":
            bad.append(f"{key}: kind={s.kind} carries inputs={s.inputs}")
        if s.producer and s.kind != "derived":
            bad.append(f"{key}: kind={s.kind} carries producer={s.producer!r}")
        if s.producer and s.producer not in ms.PRODUCERS:
            bad.append(f"{key}: producer {s.producer!r} is not in PRODUCERS")
        if s.members and s.kind != "family":
            bad.append(f"{key}: kind={s.kind} carries members")
        if s.kind == "family" and "*" not in key:
            bad.append(f"{key}: family entries are keyed by a `*` template")
        if s.kind != "family" and "*" in key:
            bad.append(f"{key}: `*` in a non-family key")
    assert not bad, "\n  ".join([""] + bad)


# ── THE aggregation assert ────────────────────────────────────────────────


def test_declared_aggregation_matches_the_emission_shape():
    """Every emitted key's declared aggregation == the one its write SHAPE implies.

    This is the T1/I1 bug class: a comment says a key is a per-epoch absolute
    while the write sits before the gh#90 divisor loop and is silently scaled by
    1/minibatches. Structure decides, not prose.
    """
    bad = []
    for key, ek in sorted(EMITTED_BY_KEY.items()):
        s = ms.REGISTRY.get(key)
        if s is None:
            continue                                                                        # completeness test reports this
        expected = census.SHAPES[ek.shape]
        if s.aggregation != expected:
            bad.append(f"{key}: registry says {s.aggregation!r} but {ek.site}:{ek.lineno} "
                       f"writes it as {ek.shape!r} ⇒ {expected!r}")
    assert not bad, ("declared aggregation contradicts the emission site:\n  " + "\n  ".join(bad))


def test_declared_aggregation_matches_the_emission_shape_for_families():
    """Same assert for f-string families, and for every member of a closed one.

    Members matter separately: a closed family's members carry their own entries
    (so a reader can be attached to `environment/action_move_0`), and nothing
    else would notice a member declared differently from its template.
    """
    bad = []
    for template, fam in sorted(FAMILY_BY_TEMPLATE.items()):
        expected = census.SHAPES[fam.shape]
        s = ms.REGISTRY.get(template)
        if s is not None and s.aggregation != expected:
            bad.append(f"{template}: registry says {s.aggregation!r} but {fam.site}:"
                       f"{fam.lineno} writes it as {fam.shape!r} ⇒ {expected!r}")
        for member in fam.members:
            m = ms.REGISTRY.get(member)
            if m is not None and m.aggregation != expected:
                bad.append(f"{member}: registry says {m.aggregation!r} but its family "
                           f"{template} is written as {fam.shape!r} ⇒ {expected!r}")
    assert not bad, "\n  ".join([""] + bad)


def test_environment_episodes_is_the_one_element_list_identity():
    """The named special case: `self.stats["episodes"] = [float(...)]`.

    A ONE-element list means PufferLib's np.mean over the window list is an
    identity, so the key is the window's episode COUNT and aggregates as `last`.
    It is the denominator of every `weighted_sum(...)/episodes` ratio in both
    frozen gate scripts, so a silent change to a bare write would divide every
    gate number by the window length while every one of those scripts kept
    printing a plausible value.
    """
    ek = EMITTED_BY_KEY.get("environment/episodes")
    assert ek is not None, ("environment/episodes is no longer censused — it moved out of "
                            "trainer.Cs2PuffeRL.train; update EMITTER_SITES")
    assert ek.shape == "stats-one-element-list", (
        f"environment/episodes is written as {ek.shape!r} at {ek.site}:{ek.lineno}, not as a "
        "one-element list. PufferLib would now MEAN it over the collection window and every "
        "gate ratio that divides by it would be wrong by a factor of the window length.")
    assert ms.REGISTRY["environment/episodes"].aggregation == "last"


def test_environment_star_window_means_still_rest_on_an_append_shaped_collector():
    """Every `window-mean-pufferlib` environment/* claim needs the collector to append.

    train.py accumulates each episode's terminal info into `self.stats[k]` as a
    LIST that mean_and_log later np.means. Rewritten to `self.stats[k] = v`, all
    ~70 of those declarations become wrong at once and nothing else in the suite
    would notice — the values would still be numbers of a plausible size.
    """
    assert census.stats_collection_is_append_shaped(), (
        "train.py's info-collection loop no longer appends/extends into self.stats — every "
        "environment/* `window-mean-pufferlib` declaration in eval/metrics_schema.py is now a "
        "claim about a pipeline that does not exist")


def test_the_gh90_divisor_loop_still_separates_mean_from_last():
    """`losses/*` splits into `mean` (before the divisor) and `last` (after) — both sides
    must be non-empty, or the classification has quietly collapsed to one class."""
    means = {k for k, e in EMITTED_BY_KEY.items() if e.shape == "losses-accumulated"}
    lasts = {k for k, e in EMITTED_BY_KEY.items() if e.shape == "losses-absolute"}
    assert means and lasts, (
        f"losses/* no longer splits across the gh#90 divisor loop "
        f"(accumulated={len(means)}, absolute={len(lasts)}) — the shape that makes this "
        "distinction enforceable is gone, so every losses/* aggregation is unchecked")
    assert "losses/minibatches_run" in lasts, (
        "losses/minibatches_run must be inserted AFTER the divisor loop: it IS the divisor")
    assert "losses/entropy" in means


# ── Completeness, both directions ─────────────────────────────────────────


def test_every_emitted_key_has_a_registry_entry():
    """Forward direction: adding a key to an emitter fails until it is registered."""
    missing = sorted(k for k in EMITTED_BY_KEY if k not in ms.REGISTRY)
    assert not missing, ("emitted but unregistered — add them to metrics_schema.REGISTRY:\n  " +
                         "\n  ".join(f"{k}  ({EMITTED_BY_KEY[k].site}:"
                                     f"{EMITTED_BY_KEY[k].lineno})" for k in missing))


def test_every_emitted_family_has_a_registry_entry():
    missing = sorted(t for t in FAMILY_BY_TEMPLATE if t not in ms.REGISTRY)
    assert not missing, ("f-string key families with no `family` registry entry:\n  " +
                         "\n  ".join(missing))


# ── The island boundary ───────────────────────────────────────────────────
#
# Everything above compares the registry against `census()`, which reads the NAMED
# island in EMITTER_SITES. Those checks are therefore only as complete as the
# island is: a metrics write added to a function nobody listed is invisible to all
# of them, and a renamed emitter fails loudly while a NEW one is silent. These two
# tests are the boundary — one sweeps all of src/ without consulting the island,
# the other resolves the merges the census deliberately does not look inside.


def test_every_metrics_write_in_src_is_inside_the_island_or_declared_not_a_metric():
    """A metrics-shaped write anywhere in src/ is an emitter site or a named exception.

    Probe this is built against: a helper returning `{"game/sneaky": 1.0}` merged
    into `logs` by train(). Two unregistered keys, one in EVERY row, and before
    this test the whole file stayed green — because `census()` only ever looked at
    the fourteen functions it was told about.

    `census.metrics_write_sites` documents exactly what the sweep still cannot see;
    it is a real residual, not a complete proof.
    """
    sweep = census.metrics_write_sites()
    assert sweep, "the src/ metrics-write sweep returned nothing — the predicate is vacuous"

    unexplained = []
    matched = {i: 0 for i in range(len(census.NON_ISLAND_WRITES))}
    for w in sweep:
        if census.island_site_of(w) is not None:
            continue
        hits = [
            i for i, x in enumerate(census.NON_ISLAND_WRITES) if x.path == w.path and
            (not x.qualname or w.qualname == x.qualname or w.qualname.startswith(x.qualname + "."))
        ]
        for i in hits:
            matched[i] += 1
        if not hits:
            unexplained.append(f"{w.path}::{w.qualname or '<module>'}:{w.lineno}  "
                               f"{w.container or '{...}'}[{w.key!r}]")
    assert not unexplained, (
        "metrics-shaped writes outside the emitter island and outside "
        "metrics_census.NON_ISLAND_WRITES:\n  " + "\n  ".join(sorted(unexplained)) +
        "\nEither add the enclosing function to EMITTER_SITES (and register its keys), or "
        "add a NON_ISLAND_WRITES entry saying why it is not a metrics emitter.")

    stale = [
        census.NON_ISLAND_WRITES[i].path + "::" + (census.NON_ISLAND_WRITES[i].qualname or "*")
        for i, n in matched.items() if n == 0
    ]
    assert not stale, ("NON_ISLAND_WRITES entries that match no write in src/ any more:\n  " +
                       "\n  ".join(stale) +
                       "\nA negative list that keeps entries it no longer needs drifts "
                       "into a blanket exemption; delete them.")

    # Anti-vacuity, and the strongest statement available here: two independent
    # walks — the island-driven census and this island-blind sweep — must agree on
    # WHICH emitter sites write keys at all. If the sweep's predicate silently
    # stopped matching, or an island site stopped emitting, this diverges.
    sweep_sites = {q for q in (census.island_site_of(w) for w in sweep) if q is not None}
    census_sites = {e.site for e in EMITTED} | {f.site for f in FAMILIES}
    assert sweep_sites == census_sites, (
        f"the island-blind sweep and census() disagree on which emitter sites write keys: "
        f"sweep-only {sorted(sweep_sites - census_sites)}, "
        f"census-only {sorted(census_sites - sweep_sites)}")


def test_writes_inside_an_emitter_go_into_a_container_that_site_declares():
    """The other half of the boundary: listing a function must not create a blind spot.

    The test above explains a sweep hit by "it is inside the island", which is
    what makes EMITTER_SITES membership a way to go dark rather than a way to be
    censused — see `census.undeclared_container_writes` for the two probe-
    confirmed spellings that exploited it. A write into a container the site does
    not declare is a write `census()` never classifies, so its keys never reach
    the registry and never fail anything.
    """
    stray = census.undeclared_container_writes()
    assert not stray, (
        "metrics-shaped writes inside an emitter, into a container that emitter does not "
        "declare — invisible to census() AND excused by the island cross-walk:\n  " +
        "\n  ".join(f"{w.path}::{w.qualname}:{w.lineno}  {w.container}[{w.key!r}]" for w in stray) +
        "\nAdd the container (with its key prefix) to the site's EMITTER_SITES entry, which "
        "is what makes census() see the keys and the registry carry them.")


def test_knockout_the_three_island_interior_blind_shapes_are_now_reported(tmp_path):
    """Plant the I-3 probes in a fake src/ and require every one of them back.

    THE KNOCK-OUT for the test above and for the `setdefault` write shape. It
    plants a file at the REAL relative path and qualname of an emitter site
    (`train_metrics.py` :: `compute_game_metrics`), so the planted writes are
    genuinely "inside the island" as far as EMITTER_SITES is concerned — which is
    the condition that made all three invisible. Reproduces the three shapes
    exactly as they were probed against the live tree, where they left all 67
    tests in this file green.

    Deleting the setdefault branch of `site_write_targets`/`metrics_write_sites`,
    or the container check in `undeclared_container_writes`, fails here.
    """
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "metrics.py").write_text(
        "def compute_game_metrics(logs):\n"
        "    game_metrics = {}\n"
        "    def _emit(out):\n"
        "        out['game/probe_param'] = 1.0\n"
        "    _emit(game_metrics)\n"
        "    alias = game_metrics\n"
        "    alias['game/probe_alias'] = 1.0\n"
        "    game_metrics.setdefault('game/probe_sd', 1.0)\n"
        "    return game_metrics\n")
    sweep = census.metrics_write_sites(src=tmp_path)
    assert {w.key
            for w in sweep} == {"game/probe_param", "game/probe_alias", "game/probe_sd"
                                }, (f"the sweep did not see all three planted writes: {sweep}")

    stray = {w.key for w in census.undeclared_container_writes(sweep)}
    assert stray == {
        "game/probe_param", "game/probe_alias"
    }, (f"the undeclared-container check reported {stray}, not the two writes that reach "
        "`game_metrics` under another name")

    # The third is caught the other way: `game_metrics` IS declared, so the check
    # above correctly leaves it alone and `census()` has to produce the key. Pin
    # that the census walk now recognises the shape.
    fn = ast.parse("def f():\n    game_metrics.setdefault('game/probe_sd', 1.0)\n").body[0]
    targets = census.site_write_targets(fn.body[0].value, {"game_metrics": ""})
    assert [census._key_text(k) for _, k, _, _ in targets
            ] == ["game/probe_sd"
                  ], (f"site_write_targets no longer recognises setdefault: {targets}")


@pytest.mark.parametrize("snippet,expected", [
    ("logs['game/a'] = 1", "game/a"),
    ("logs['game/a'] += 1", "game/a"),
    ("logs = {'game/a': 1}", "game/a"),
    ("logs.update({'game/a': 1})", "game/a"),
    ("logs.setdefault('game/a', 1)", "game/a"),
])
def test_the_census_recognises_each_write_shape_it_claims_to(snippet, expected):
    """One assertion per shape named in `site_write_targets`'s docstring.

    A list of supported shapes in a docstring is the kind of claim that is true
    when written and silently wrong after the next edit; this is the claim being
    checked rather than described. `setdefault` is the shape that was documented
    nowhere and implemented nowhere until final review I-3.
    """
    node = ast.parse(snippet).body[0]
    node = node.value if isinstance(node, ast.Expr) else node
    targets = census.site_write_targets(node, {"logs": ""})
    assert [census._key_text(k) for _, k, _, _ in targets
            ] == [expected], (f"{snippet!r} is not recognised as a write into `logs`: {targets}")


def test_dicts_merged_into_island_containers_come_from_island_emitters():
    """`logs.update(<Call>)` is the one write shape `census()` cannot look inside.

    It ignores non-literal `.update()` arguments deliberately — the keys are not in
    the argument — which is correct only while every merged dict is built by a
    function that is itself an emitter site. Nothing said so, so a new helper
    merged into `logs` used to add keys to every row invisibly. This resolves each
    merge ONE HOP and requires the result to be an EMITTER_SITES member or to carry
    a reason in `metrics_census.ISLAND_MERGE_SOURCES`.
    """
    merges = census.island_merge_sources()
    assert merges, ("no non-literal merge found in any island body — either the extractor "
                    "broke or the merges moved; this check is vacuous as it stands")
    island_names = {s.qualname.split(".")[-1] for s in census.EMITTER_SITES}
    declared = {(m.qualname, m.source) for m in census.ISLAND_MERGE_SOURCES}

    used, bad = set(), []
    for m in merges:
        if m.resolved in island_names:
            continue
        if (m.qualname, m.source) in declared:
            used.add((m.qualname, m.source))
            continue
        bad.append(f"{m.qualname}:{m.lineno} merges {m.expr} — resolves to "
                   f"{m.source} ({m.resolved or 'not a call'})")
    assert not bad, (
        "dicts merged into an island metrics container from OUTSIDE the island:\n  " +
        "\n  ".join(sorted(bad)) + "\nEvery key in such a dict lands in the row while the "
        "census sees none of them. Add the producing function to EMITTER_SITES, or declare "
        "it in metrics_census.ISLAND_MERGE_SOURCES with the reason it needs no census.")
    stale = sorted(declared - used)
    assert not stale, ("ISLAND_MERGE_SOURCES entries matching no merge in the island — the "
                       f"expression changed or the merge is gone: {stale}")


# Registry families that declare `members` the AST census CANNOT confirm, and the
# test that pins each one instead. A declared member list is accepted provenance in
# test_every_registered_emitted_key_is_actually_emitted, so an UNPINNED one would
# let the registry vouch for its own invented keys — the exact accumulation the
# reverse-completeness half exists to stop. Membership is asserted below in both
# directions, so a new registry-closed family is a failure until it is either
# census-resolvable or pinned and named here.
MEMBERS_PINNED_ELSEWHERE = {
    "losses/entropy/*": "test_losses_entropy_family_members_track_the_action_spec",
}


def test_closed_family_members_match_the_census_exactly():
    """A statically-resolvable family declares its members, and they are checked.

    This is what makes a tenth `action_move` bin, or a fourth split head, a test
    failure rather than an unregistered key nobody notices — those keys never
    appear in a two-epoch gate row, so the emission half sees nothing.
    """
    bad = []
    for template, fam in sorted(FAMILY_BY_TEMPLATE.items()):
        if not fam.members:
            continue                                                             # OPEN: nothing to compare against
        s = ms.REGISTRY.get(template)
        if s is None:
            continue
        if tuple(sorted(s.members)) != tuple(sorted(fam.members)):
            bad.append(f"{template}: registry declares {sorted(s.members)} but "
                       f"{fam.site}:{fam.lineno} builds {sorted(fam.members)}")
        for member in fam.members:
            if member not in ms.REGISTRY:
                bad.append(f"{member}: closed family member with no entry of its own")
    assert not bad, "\n  ".join([""] + bad)

    # Every members-declaring family is either compared above or pinned elsewhere.
    census_closed = {t for t, f in FAMILY_BY_TEMPLATE.items() if f.members}
    declaring = {k for k, s in ms.REGISTRY.items() if s.kind == "family" and s.members}
    unpinned = sorted(declaring - census_closed - set(MEMBERS_PINNED_ELSEWHERE))
    assert not unpinned, (
        "registry families declaring `members` that no census family confirms and no named "
        "test pins:\n  " + "\n  ".join(unpinned) + "\nTheir members are accepted as provenance "
        "by test_every_registered_emitted_key_is_actually_emitted, so an unpinned list is the "
        "registry vouching for keys it invented. Make the family census-resolvable, or pin it "
        "and add it to MEMBERS_PINNED_ELSEWHERE.")
    stale = sorted(k for k in MEMBERS_PINNED_ELSEWHERE if k not in declaring or k in census_closed)
    assert not stale, ("MEMBERS_PINNED_ELSEWHERE names families that no longer need it (the "
                       f"census now resolves them, or they declare no members): {stale}")


def test_losses_entropy_family_members_track_the_action_spec():
    """`losses/entropy/*` is OPEN to the AST census, so its members are declared from
    `spec.action.ACTION_HEAD_NAMES` — this pins that the EMITTER iterates the same
    tuple, which is the only thing making that declaration non-circular."""
    source = census.losses_entropy_head_source()
    assert source == "ACTION_HEAD_NAMES", (
        f"the per-head entropy loop iterates {source!r}, not ACTION_HEAD_NAMES; "
        "metrics_schema's losses/entropy/* member list is derived from ACTION_HEAD_NAMES "
        "and is now describing a different set of heads")
    from cs2rl.spec.action import ACTION_HEAD_NAMES
    declared = ms.REGISTRY["losses/entropy/*"].members
    assert tuple(sorted(declared)) == tuple(sorted(f"losses/entropy/{h}"
                                                   for h in ACTION_HEAD_NAMES))
    for h in ACTION_HEAD_NAMES:
        assert f"losses/entropy/{h}" in ms.REGISTRY


def test_tag_families_are_census_closed_on_both_axes():
    """The seven `tag/*` families must stay CLOSED, with both axes read from source.

    `tag_grad_cossim` is the only emitter of `tag/*`, and it does not write those
    keys anywhere a reader can see them: `_inject_tag_metrics`, the function that
    puts them in the row, has the body `logs.update(pending)` and emits nothing.
    Neither placeholder is visible in the emitter body either — the group axis is
    a loop over a Name, the label axis is a parameter — so before both were
    resolved every family was OPEN, and an OPEN template is accepted as
    provenance by `test_every_registered_emitted_key_is_actually_emitted` for any
    key matching its glob. `tag/whatever/i/like` passed.

    WHY this is not covered by `test_closed_family_members_match_the_census_exactly`:
    that test compares two lists, and skips a family whose census members are
    empty. An edit that reopens the census AND drops the registry's `members` in
    the same breath leaves both sides empty and it stays green — with `tag/*`
    back to being a glob alibi. This asserts the closure itself.
    """
    axes = census.tag_key_axes()
    assert axes["group"] == ms.TAG_PARAM_GROUPS, (
        f"tag_grad_cossim's key loop runs over {axes['group']}, but metrics_schema declares "
        f"TAG_PARAM_GROUPS={ms.TAG_PARAM_GROUPS} — the registry is describing parameter "
        "groups the emitter no longer uses")
    assert axes["label"] == ms.TAG_MB_LABELS, (
        f"tag_grad_cossim is called with mb_label in {axes['label']}, but metrics_schema "
        f"declares TAG_MB_LABELS={ms.TAG_MB_LABELS}")

    tag_families = sorted(t for t in FAMILY_BY_TEMPLATE if t.startswith("tag/"))
    assert len(tag_families) == 7, (
        f"expected the seven tag/* families tag_grad_cossim builds, censused {tag_families}")
    open_now = [t for t in tag_families if not FAMILY_BY_TEMPLATE[t].members]
    assert not open_now, (
        "tag/* families the census can no longer enumerate: " + ", ".join(open_now) +
        "\nAn OPEN family alibis every registry entry under its glob. Either the loop "
        "iterable / call-site label stopped being statically resolvable, or the resolution "
        "in metrics_census (_local_literal_bindings, emitter_param_bindings) was narrowed.")

    censused = {m for t in tag_families for m in FAMILY_BY_TEMPLATE[t].members}
    assert len(censused) == 26, f"expected 26 concrete tag/* keys, censused {len(censused)}"
    declared = {m for t in tag_families for m in ms.REGISTRY[t].members}
    assert censused == declared
    assert not _matches_open_family("tag/not_a_real_key"), (
        "a `tag/...` key matches an OPEN registry template again — the seven families are "
        "closed, so nothing under tag/ should glob-match")


# Every construct Python has for binding a name, as a statement fragment binding
# `victim`. The two tests below splice each one into a function whose static
# resolution otherwise SUCCEEDS, and require the resolution to drop out.
#
# WHY this list and not a demonstration on one form: the two resolutions that
# close the `tag/*` families are only sound while the name they read holds one
# value, and the failure when it does not is SILENT — the census keeps reporting
# the pre-rebinding values while the emitter writes different keys, so the
# registry ends up documenting keys nothing emits while the real ones go
# unregistered, with every test in this file green. Counting one binding form
# (plain `=`, which `_local_literal_bindings` used to do) leaves the other
# sixteen invisible, and "invisible" here means wrong-and-quiet rather than
# unresolved-and-loud.
_REBINDING_FORMS = (
    "victim = 'live'",
    "victim += ('live',)",
    "victim: tuple = ('live',)",
    "for victim in (('live',),):\n    pass",
    "with open('/dev/null') as victim:\n    pass",
    "_ = (victim := 'live')",
    "victim, _other = ('live', 1)",
    "*victim, _other = ('live', 1)",
    "del victim",
    "def victim():\n    pass",
    "class victim:\n    pass",
    "import victim",
    "from x import y as victim",
    "try:\n    pass\nexcept ValueError as victim:\n    pass",
    "def _helper(victim):\n    return victim",
    "global victim",
    "nonlocal victim",
    "match ('a',):\n    case victim:\n        pass",
)

# A local bound once to a literal tuple and looped over to build keys — the exact
# shape of `tag_grad_cossim`'s `pg_group_names`, reduced to what the resolution
# needs. `{extra}` is where a rebinding fragment goes.
_LITERAL_EMITTER = '''
def emitter():
    pg_group_names = ("trunk", "policy_heads")
{extra}
    for g in pg_group_names:
        out[f"tag/gnorm_t/{{g}}"] = 0.0
'''

# The parameter half of the same shape: `mb_label` is a keyword-only argument
# `tag_grad_cossim` builds keys from and never assigns. Kept under the real
# function's name so `emitter_param_bindings` indexes the real call sites in src/.
_PARAM_EMITTER = '''
def tag_grad_cossim(policy, *, mb_label):
{extra}
    return {{f"tag/gnorm_t/{{mb_label}}": 0.0}}
'''


def _emitter_ast(template, extra):
    """Parse `template` with `extra` spliced in at body indentation."""
    import textwrap
    body = textwrap.indent(extra, "    ") if extra else "    pass"
    return ast.parse(template.format(extra=body)).body[0]


@pytest.mark.parametrize("rebind", ("", ) + _REBINDING_FORMS)
def test_a_rebound_local_stops_resolving_instead_of_keeping_its_first_value(rebind):
    """`_local_literal_bindings` must bind a name NOTHING else in the function binds.

    The empty-`rebind` case is the positive control and is not decoration: without
    it a helper that returned `{}` unconditionally would pass every other case
    here, which is the shape a "fix" takes when someone makes this test green by
    disabling the resolution instead of narrowing it.
    """
    fn = _emitter_ast(_LITERAL_EMITTER, rebind.replace("victim", "pg_group_names"))
    bound = census._local_literal_bindings(fn)
    if not rebind:
        assert bound.get("pg_group_names") == ("trunk", "policy_heads"), (
            "the one-hop literal resolution no longer binds an un-rebound local — the seven "
            "tag/* families lose their group axis and reopen")
        return
    assert "pg_group_names" not in bound, (
        f"`{rebind}` rebinds pg_group_names, but the census still resolves it to its first "
        "value. The family would keep a member list the emitter no longer writes, silently.")


@pytest.mark.parametrize("rebind", ("", ) + _REBINDING_FORMS)
def test_a_rebound_emitter_parameter_stops_resolving_instead_of_using_call_site_values(rebind):
    """`emitter_param_bindings` must drop a parameter the emitter body rebinds.

    Run against `tag_grad_cossim`'s REAL call sites — `emitter_param_bindings`
    takes the site (from which it derives the callee name it indexes `src/` by)
    and the function node separately, so substituting a same-named body with a
    rebinding in it is the mutation the real hole allows, with nothing else faked.

    The empty-`rebind` case is the positive control and carries it twice: the
    synthetic body must still bind (so the other cases cannot pass by the
    resolution having been disabled wholesale), and so must the UNMODIFIED
    emitter (so they cannot pass by the synthetic shape having drifted away from
    the real one).
    """
    site = next(s for s in census.EMITTER_SITES if s.qualname == "tag_grad_cossim")
    fn = _emitter_ast(_PARAM_EMITTER, rebind.replace("victim", "mb_label"))
    bound = census.emitter_param_bindings(site, fn)
    if not rebind:
        assert bound.get("mb_label") == ("mb0", "mbL"), (
            "a parameter that is NOT rebound must still bind from its call sites")
        real = census._find_qualname(census._module_ast(site.path), site.qualname)
        assert census.emitter_param_bindings(site, real).get("mb_label") == ("mb0", "mbL"), (
            "tag_grad_cossim's mb_label no longer resolves from its call sites — the seven "
            "tag/* families reopen and alibi any tag/* registry entry")
        return
    assert "mb_label" not in bound, (
        f"`{rebind}` rebinds the mb_label parameter, but the census still reports the "
        "call-site values. The emitter would write one key set while the registry "
        "documents another, with no test failing.")


def _open_family_templates():
    """Registered family templates whose members are UNENUMERABLE — and only those.

    A family is open when the registry declares no `members` for it: the concrete
    keys depend on a runtime value (a model's `named_parameters()`, a profiler
    section name, a minibatch label), so nothing can list them and a glob is the
    only honest coverage statement.

    PITFALL this signature exists to not repeat: this helper used to return EVERY
    family template, open or closed, and the caller globbed against it with
    `fnmatch`, whose `*` matches `/` as well. `game/*` — the largest emitted
    namespace in the registry — therefore acted as a blanket alibi for any
    `game/anything` entry, so the reverse-completeness half of this file was
    vacuous for every key under a closed template. A CLOSED family already knows
    its members; it must excuse those and nothing else.
    """
    return [k for k, s in ms.REGISTRY.items() if s.kind == "family" and "*" in k and not s.members]


def _declared_family_members():
    """Every concrete key a registered CLOSED family names in its `members`.

    This is the provenance that replaces the glob for closed templates, and it is
    not the registry vouching for itself: for the ten census-closed families
    `test_closed_family_members_match_the_census_exactly` pins `members` against
    the emitter's own literals, and for `losses/entropy/*` — closed by declaration
    because the census sees it as open —
    `test_losses_entropy_family_members_track_the_action_spec` pins it against
    `spec.action` plus a source pin on the loop the emitter iterates. That both
    checks exist for every members-declaring family is asserted in
    `test_closed_family_members_match_the_census_exactly`.
    """
    return {m for s in ms.REGISTRY.values() if s.kind == "family" for m in s.members}


def _matches_open_family(key):
    """True when `key` fits a registered OPEN family template (glob, `*` spans `/`)."""
    import fnmatch
    return any(fnmatch.fnmatchcase(key, t) for t in _open_family_templates())


def test_every_registered_emitted_key_is_actually_emitted():
    """Reverse direction: a registry entry nothing writes is a lie the registry tells.

    Accepted provenance for an `emitted` entry, in order: a concrete census key; a
    member of a census-closed family; a member a registered family DECLARES; one of
    `BaselineEvaluator.evaluate()`'s output keys (censused from that dict literal,
    since ScheduledEval merges the dict wholesale); a key
    `metrics_schema.PUFFERLIB_OWNED` declares as PufferLib's own; or a glob match
    against a template that is OPEN, i.e. one whose members are unenumerable.

    The last clause is the load-bearing restriction. Globbing against every
    template — including the closed ones — makes this test pass for any key at all
    under `game/`, `environment/action_*`, `split/*` or `losses/entropy/*`, which
    is most of the emitted surface. `_open_family_templates` carries the detail.
    """
    closed_members = {m for f in FAMILIES for m in f.members}
    declared_members = _declared_family_members()
    eval_keys = census.eval_output_keys() | set(ms.EVAL_EXTRA_KEYS)
    orphans = []
    for key in ms.keys_of_kind("emitted"):
        if key in EMITTED_BY_KEY or key in closed_members or key in declared_members:
            continue
        if key in eval_keys or key in ms.PUFFERLIB_OWNED or _matches_open_family(key):
            continue
        orphans.append(key)
    assert not orphans, ("registered as `emitted` but no emitter writes them — delete the "
                         "entry or fix the key:\n  " + "\n  ".join(orphans))


def test_every_registered_family_is_a_real_family_or_declared_foreign():
    foreign = set(ms.PUFFERLIB_OWNED)
    orphans = sorted(t for t in ms.keys_of_kind("family")
                     if t not in FAMILY_BY_TEMPLATE and t not in foreign)
    assert not orphans, ("`family` entries matching no f-string emitter:\n  " +
                         "\n  ".join(orphans))


def test_pufferlib_owned_keys_are_registered_and_not_ours():
    """The keys pufferl.mean_and_log writes itself, cross-checked against ITS source.

    `agent_steps` is the participating-step counter both frozen gate scripts key
    their evaluation window on, and no emitter of ours produces it. If a PufferLib
    upgrade renames it, both gates start reading a missing key and silently judge
    an empty window — so the rename has to fail here.
    """
    for key in ms.PUFFERLIB_OWNED:
        assert key in ms.REGISTRY, f"{key} is declared PufferLib-owned but is not registered"
        assert key not in EMITTED_BY_KEY, (
            f"{key} is declared PufferLib-owned but one of OUR emitters writes it "
            f"({EMITTED_BY_KEY[key].site}) — the declaration is wrong")

    import pufferlib.pufferl as pufferl
    tree = ast.parse(Path(pufferl.__file__).read_text())
    fn = census._find_qualname(tree, "mean_and_log")
    literals = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict):
            for k in node.value.keys:
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    literals.add(k.value)
    assert literals, ("no `logs = {...}` literal found in pufferl.mean_and_log — this check "
                      "has gone vacuous; re-derive it against the installed PufferLib")
    unregistered = sorted(k for k in literals if k not in ms.REGISTRY)
    assert not unregistered, (
        f"pufferl.mean_and_log writes {unregistered} into every row and metrics_schema does "
        "not carry them")


def test_derived_inputs_are_registered_keys():
    bad = []
    for key in ms.keys_of_kind("derived"):
        for src in ms.REGISTRY[key].inputs:
            if src not in ms.REGISTRY:
                bad.append(f"{key}: input {src!r} is not a registered key")
    assert not bad, "\n  ".join([""] + bad)


# ── The frozen readers ────────────────────────────────────────────────────


@pytest.mark.parametrize("script", census.FROZEN_READERS)
def test_every_frozen_reader_key_literal_is_registered(script):
    """The gate scripts never migrate to importing from here, so the registry chases
    them — parsed from source, because nothing enforces that they stay frozen."""
    literals = census.reader_key_literals(script)
    assert literals, f"extracted zero key literals from {script} — the extractor is vacuous"
    missing = sorted(k for k in literals if k not in ms.REGISTRY)
    assert not missing, (f"{script} reads keys the registry does not carry:\n  " +
                         "\n  ".join(f"{k}  (:{literals[k]})" for k in missing))


def test_rung1_gate_report_columns_are_registered():
    """GATES / REPORT_ONLY / REPORT_EXTRA column names — several of which are DERIVED
    (`kills_per_episode` the episode-weighted ratio, `losses/approx_kl_p90`) and are
    not emitted keys at all. Registering them is what keeps the extractor from being
    loosened until it demands an emitter for a column heading."""
    cols = census.reader_report_columns()
    assert cols, "extracted zero report columns from experiment/gate.py — the extractor is vacuous"
    missing = sorted(c for c in cols if c not in ms.REGISTRY)
    assert not missing, "unregistered rung1_gate report columns:\n  " + "\n  ".join(missing)
    assert ms.REGISTRY["losses/approx_kl_p90"].kind == "derived", (
        "losses/approx_kl_p90 is a REPORT_EXTRA column name, never an emitted key "
        "(its source is losses/approx_kl)")
    assert ms.REGISTRY["kills_per_episode"].kind == "derived"


def test_every_frozen_reader_key_is_registered_or_declared_out_of_surface():
    """The WHOLE frozen-reader surface — including what the extractor's exclusions hide.

    `test_every_frozen_reader_key_literal_is_registered` covers the literals the
    extractor SEES. This one covers the seam: `reader_key_literals` drops
    `.replace` / `.compile` / `Path` arguments by position, and a positional
    exclusion is the first thing that gets widened when this file goes red. Add
    `"get"` to `NON_KEY_CALLS` and every key both scripts read leaves the visible
    surface without a single test noticing.

    So the exclusion is enumerated rather than blanket: whatever it hides must be
    named in `READER_OUT_OF_SURFACE` with a reason, in both directions. Widening
    the exclusion moves real keys into the unexplained set (fails); a declaration
    that stops matching a hidden literal is stale (fails too).

    Today it hides exactly two things, neither a metrics key: a display prefix the
    gate's column headings are shortened with, and a default run directory.
    """
    unexplained, matched = [], {i: 0 for i in range(len(census.READER_OUT_OF_SURFACE))}
    for script in census.FROZEN_READERS:
        for literal, lineno in sorted(census.reader_hidden_call_literals(script).items()):
            hits = [
                i for i, d in enumerate(census.READER_OUT_OF_SURFACE)
                if d.script == script and d.literal == literal
            ]
            for i in hits:
                matched[i] += 1
            if not hits:
                unexplained.append(f"{script}:{lineno}  {literal!r}")
    assert not unexplained, (
        "key-shaped literals hidden from the frozen-reader census by a NON_KEY_CALLS "
        "exclusion, with no declared reason:\n  " + "\n  ".join(unexplained) +
        "\nEither the exclusion was widened over a real key — register it and take the call "
        "out of metrics_census.NON_KEY_CALLS — or add a READER_OUT_OF_SURFACE entry saying "
        "why it is not a metrics key.")

    stale = [
        f"{census.READER_OUT_OF_SURFACE[i].script}: {census.READER_OUT_OF_SURFACE[i].literal!r}"
        for i, n in matched.items() if n == 0
    ]
    assert not stale, ("READER_OUT_OF_SURFACE entries matching no hidden literal any more:\n  " +
                       "\n  ".join(stale) +
                       "\nA declaration kept past the literal it excuses is how an enumerated "
                       "exemption decays into a blanket one; delete it.")

    # Anti-vacuity. The visible surface is where the real keys are; if the
    # extractor stopped seeing them this whole file would go quiet, so assert it
    # still resolves both scripts' reads and that they are registered as a set.
    visible = {k for s in census.FROZEN_READERS for k in census.reader_key_literals(s)}
    assert len(visible) >= 25, (f"the frozen readers resolve to only {len(visible)} key "
                                "literals — the extractor has gone narrow")
    assert not sorted(k for k in visible if k not in ms.REGISTRY)


def test_derived_gate_columns_declare_the_source_keys_the_gate_actually_reads():
    """A derived column's `inputs` must be the keys rung1_gate reads to compute it.

    `test_rung1_gate_report_columns_are_registered` proves the NAME is registered.
    It does not look at `inputs`, which is the substantive half of a `derived`
    entry and pure documentation about a frozen script — swap
    `hit_per_on_target`'s numerator in the gate and the registry keeps describing
    the old ratio with every test green. `reader_derived_column_sources` re-derives
    the same fact from the script's source (following REPORT_EXTRA's argument
    tuples and, for GATES/REPORT_ONLY, the `m = {...}` literal in `seed_metrics`),
    so the claim is checked rather than asserted.

    This is also what makes the five bare `REPORT_ONLY` literals classified rather
    than merely tolerated: `shots_fired` is not the emitted `game/shots_fired` but
    is computed from it plus the episode weight; `rows` is `len(W)` and resolves to
    no key at all, which is the difference between "window bookkeeping" as prose
    and as a measurement.
    """
    sources = census.reader_derived_column_sources()
    assert sources, "extracted zero derived-column sources from experiment/gate.py — vacuous"

    # An UNREGISTERED column is test_rung1_gate_report_columns_are_registered's
    # failure to report, not this one's — skipped here so that test owns it and
    # this one does not raise a bare KeyError on top of a clear message.
    cols = census.reader_report_columns()
    derived_cols = sorted(c for c in cols if c in ms.REGISTRY and ms.REGISTRY[c].kind == "derived")
    assert len(derived_cols) >= 10, (
        f"only {len(derived_cols)} derived gate columns — the report tables shrank or the "
        "column extractor did")

    missing, wrong = [], []
    for col in derived_cols:
        if col not in sources:
            missing.append(col)
            continue
        declared, actual = set(ms.REGISTRY[col].inputs), set(sources[col])
        if declared != actual:
            wrong.append(f"{col}: registry declares inputs {sorted(declared)}, but rung1_gate "
                         f"computes it from {sorted(actual)}")
    assert not missing, ("derived rung1_gate columns whose computation this test could not "
                         "locate in the script:\n  " + "\n  ".join(missing) +
                         "\nThe column moved out of seed_metrics/REPORT_EXTRA, so its `inputs` "
                         "are now unchecked — re-point reader_derived_column_sources.")
    assert not wrong, "\n  ".join([""] + wrong)

    # `rows` is the deliberate empty case: window bookkeeping, no metrics key.
    assert sources["rows"] == frozenset(), (
        f"`rows` now resolves to {sorted(sources['rows'])}; it is len(W) and must resolve to "
        "nothing, which is the registry's claim that it has no inputs")
    assert sources["shots_fired"] == {
        "game/shots_fired", "environment/episodes"
    }, ("the bare `shots_fired` column must resolve to the emitted game/shots_fired it is "
        "computed from, episode-weighted — that resolution is its classification")


def test_rung1_gate_producer_column_matches_its_report_tables_both_ways():
    """`producer` is checked against rung1_gate's own GATES/REPORT_* tables.

    Both directions: a column the script defines must be a registered `derived`
    entry crediting it, and an entry crediting rung1_gate must be a column the
    script actually defines. Without the reverse half, `producer` is a free-text
    field that can name any script for any key.
    """
    cols = census.reader_report_columns()
    # Several columns ARE emitted keys used verbatim as headings
    # (`losses/entropy/shoot`, the two gated `eval/win_vs_random_as_*`); the
    # registry's own `emitted` set is the right surface to subtract, and
    # test_every_registered_emitted_key_is_actually_emitted is what keeps that set
    # honest rather than a place to hide a column.
    emitted_surface = set(ms.keys_of_kind("emitted"))
    unclaimed = sorted(c for c in cols
                       if c not in emitted_surface and ms.REGISTRY[c].producer != "rung1_gate")
    invented = sorted(k for k in ms.keys_of_kind("derived")
                      if ms.REGISTRY[k].producer == "rung1_gate" and k not in cols)
    assert not unclaimed, ("rung1_gate report columns that no emitter writes and no registry "
                           "entry credits to rung1_gate:\n  " + "\n  ".join(unclaimed))
    assert not invented, ("registered as computed by rung1_gate, but absent from its "
                          "GATES/REPORT_ONLY/REPORT_EXTRA tables:\n  " + "\n  ".join(invented))


# ── Consumers, both directions ────────────────────────────────────────────


def test_consumer_names_cover_every_read_and_no_read_is_invented():
    """Both directions at once. A `consumers` column checked in one direction only
    drifts in the other, and it is the first column of a registry to rot."""
    unregistered, missing_name, invented = [], [], []
    for consumer, keys in sorted(READS.items()):
        assert consumer in ms.CONSUMERS, f"census names consumer {consumer!r}, registry does not"
        for key, lineno in sorted(keys.items()):
            s = ms.REGISTRY.get(key)
            if s is None:
                unregistered.append(f"{consumer} reads unregistered {key} (:{lineno})")
            elif s.kind == "derived" and s.producer == consumer:
                # A script naming its own output column is definitional, not a row
                # read. `losses/approx_kl_p90` sits in rung1_gate's REPORT_EXTRA
                # tuple beside the real source keys, so the literal extractor
                # cannot tell them apart by position — the registry's `producer`
                # is what does, which is why the spec requires it to be classified
                # rather than have the extractor loosened to drop it.
                continue
            elif consumer not in s.consumers:
                missing_name.append(f"{key}: read by {consumer} (:{lineno}) but its "
                                    f"consumers are {list(s.consumers)}")
    for key, s in sorted(ms.REGISTRY.items()):
        for consumer in s.consumers:
            if key not in READS.get(consumer, {}):
                invented.append(f"{key}: names consumer {consumer}, which does not read it")
    assert not (unregistered or missing_name or invented), "\n  ".join([""] + unregistered +
                                                                       missing_name + invented)


# ── EVAL_KEYS ownership ───────────────────────────────────────────────────


def test_eval_keys_match_the_evaluate_output_contract():
    """EVAL_KEYS is `BaselineEvaluator.evaluate()`'s output contract. eval.baselines
    raises on a set mismatch at runtime, but only when an eval actually runs — and
    the §3 gate runs with eval off, so that guard never fires in the suite. Same
    contract, read from source."""
    assert set(ms.EVAL_KEYS) == census.eval_output_keys()


def test_eval_surface_is_eval_keys_plus_the_two_scheduler_stamps():
    """ScheduledEval adds eval/epoch and eval/wall_s ON TOP of evaluate()'s output —
    EVAL_KEYS alone under-declares the registry's eval/* surface by two."""
    assert set(ms.EVAL_SURFACE) == set(ms.EVAL_KEYS) | {"eval/epoch", "eval/wall_s"}
    registered = {k for k in ms.REGISTRY if k.startswith("eval/")}
    assert registered == set(ms.EVAL_SURFACE)


def test_eval_baselines_imports_eval_keys_from_here_and_not_the_reverse():
    """The direction is load-bearing, not stylistic: eval.baselines imports torch and
    env.c.cs2_env at module scope, so `from cs2rl.eval.baselines import EVAL_KEYS` would make
    a tuple of eight strings cost a torch import and break metrics_schema's
    import-lightness (tests/test_w1_modules.py). Checked from SOURCE — importing
    eval.baselines here to compare the objects would pull torch into this test."""
    tree = ast.parse((REPO_ROOT / "src" / "cs2rl" / "eval" / "baselines.py").read_text())
    imports_from_schema = any(
        isinstance(n, ast.ImportFrom) and n.module == "cs2rl.eval.metrics_schema" and any(
            a.name == "EVAL_KEYS" for a in n.names) for n in ast.walk(tree))
    assert imports_from_schema, (
        "src/cs2rl/eval/baselines.py must do `from cs2rl.eval.metrics_schema import "
        "EVAL_KEYS` — the registry is the single authority")
    assigns = [
        n for n in tree.body if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "EVAL_KEYS" for t in n.targets)
    ]
    assert not assigns, (
        "src/cs2rl/eval/baselines.py still assigns EVAL_KEYS — two authorities for "
        "the same contract is exactly what the move removed")

    schema_tree = ast.parse(
        (REPO_ROOT / "src" / "cs2rl" / "eval" / "metrics_schema.py").read_text())
    assert not [n for n in ast.walk(schema_tree) if is_back_edge(n)
                ], ("metrics_schema must never import cs2rl.eval.baselines (torch at its scope)")


# The module is_back_edge looks for, and the package metrics_schema's relative imports
# resolve against. A move of either file fails the test above loudly (its path is read).
_BASELINES = "cs2rl.eval.baselines"
_SCHEMA_PACKAGE = "cs2rl.eval"


def is_back_edge(n):
    """True if `n` imports `_BASELINES`, in any spelling a module in `_SCHEMA_PACKAGE` can use.

    Absolute: `import cs2rl.eval.baselines`, `from cs2rl.eval.baselines import ...`,
    `from cs2rl.eval import baselines`. Relative: `from . import baselines`,
    `from .baselines import ...`, resolved against `_SCHEMA_PACKAGE` with
    importlib.util.resolve_name first.

    PITFALL: comparing `n.module` alone is blind to every relative spelling: `from .
    import baselines` has `module=None, level=1`. The acyclic contract would still
    see such an edge, but this test would stay green on it.
    """
    if isinstance(n, ast.Import):
        return any(a.name == _BASELINES or a.name.startswith(_BASELINES + ".") for a in n.names)
    if not isinstance(n, ast.ImportFrom):
        return False
    module = n.module or ""
    if n.level:
        module = importlib.util.resolve_name("." * n.level + module, _SCHEMA_PACKAGE)
    return (module == _BASELINES or module.startswith(_BASELINES + ".")
            or any(f"{module}.{a.name}" == _BASELINES for a in n.names))


def test_is_back_edge_sees_every_spelling():
    """Positive and negative controls for is_back_edge, one parsed statement each.

    The real metrics_schema has no back-edge, so the test above is green whether or
    not is_back_edge works; these cases are what show that it does, relative spellings
    included (the #205 part 2a knock-out: a planted `from . import baselines` passed
    the old module-name comparison).
    """
    back = ("import cs2rl.eval.baselines", "import cs2rl.eval.baselines as b",
            "from cs2rl.eval.baselines import BaselineEvaluator",
            "from cs2rl.eval import baselines", "from . import baselines",
            "from .baselines import BaselineEvaluator", "from ..eval import baselines",
            "from ..eval.baselines import H_MOVE")
    fine = ("from cs2rl.eval import metrics_schema", "from . import metrics_schema",
            "from .metrics_schema import EVAL_KEYS", "import cs2rl.eval", "from cs2rl import eval",
            "from ..env import nav", "from cs2rl.eval.baselinesx import y")
    for code in back:
        assert any(is_back_edge(n) for n in ast.walk(ast.parse(code))), code
    for code in fine:
        assert not any(is_back_edge(n) for n in ast.walk(ast.parse(code))), code


def test_metrics_schema_is_in_the_import_lightness_test():
    """Creating a module and adding it to the subprocess guard is ONE task, by spec —
    a module that slips in unguarded is how the invariant dies."""
    from tests.test_w1_modules import W1_MODULES
    assert "cs2rl.eval.metrics_schema" in W1_MODULES
