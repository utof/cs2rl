"""scripts/trainer_equivalence.py's case hooks set only knobs the trainer has.

WHY: a hook that sets a config key or trainer attribute the trainer no longer has (a
renamed knob, a typo) is a silent no-op. The case stops reaching the branch it exists
for and still compares IDENTICAL on both sides of a refactor, so the oracle passes a
change it never exercised.

HOW: every hooked case runs both hook phases on a freshly built trainer and must add no
`trainer.config` key and no `vars(trainer)` attribute. One trainer is built per distinct
`build` kwargs: after the first, a build took under 0.1 s on the development VM
(2026-10-06), and the hooks only assign values, so the cases of one group can share it.
The positive control, a misspelt key and attribute, gets a trainer of its own.

PITFALL: on a shared trainer, a name is reported only for the first case that adds it;
a later case of the same group that sets it too passes until the first is fixed. The
same masking is why the control must not share a trainer with the cases.
"""
import importlib
import json


def _added_names(trainer, hook):
    """The config keys and trainer attributes ``hook`` adds over both of its phases.

    The after_eval phase runs as epoch 1, the epoch the collapse-watch hook acts on.
    """
    config, attrs = set(trainer.config), set(vars(trainer))
    hook(trainer, -1, "built")
    hook(trainer, 1, "after_eval")
    return sorted(set(trainer.config) - config), sorted(set(vars(trainer)) - attrs)


def _added_by_each(teq, build_kwargs, hooks):
    """{name: _added_names(...)} for ``hooks`` run in turn on one trainer built with
    ``build_kwargs``."""
    trainer, cleanup = teq.build(0, **build_kwargs)
    try:
        return {name: _added_names(trainer, hook) for name, hook in hooks.items()}
    finally:
        cleanup()


def test_every_case_hook_sets_only_existing_config_keys_and_attributes():
    teq = importlib.import_module("scripts.trainer_equivalence")
    control = teq._chain(teq._cfg(target_kll=1e-4), teq._attrs(total_minibatchs=7))
    assert _added_by_each(teq, {}, {"control": control}) == {
        "control": (["target_kll"], ["total_minibatchs"])
    }
    groups: dict[str, dict] = {}
    for name, case in teq.CASES.items():
        if case.get("hook"):
            groups.setdefault(json.dumps(case["build"], sort_keys=True), {})[name] = case["hook"]
    assert groups, "no case has a hook: the check below would pass on nothing"
    added = {}
    for build_kwargs, hooks in groups.items():
        for name, new in _added_by_each(teq, json.loads(build_kwargs), hooks).items():
            if new != ([], []):
                added[name] = new
    assert added == {}, ("these cases set (config keys, attributes) the trainer does not "
                         f"have, so they no longer reach their branch: {added}")
