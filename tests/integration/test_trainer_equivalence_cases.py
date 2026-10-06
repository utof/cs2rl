"""scripts/trainer_equivalence.py's case hooks set only knobs the trainer has.

WHY: a hook that sets a config key or trainer attribute the trainer no longer has (a
renamed knob, a typo) is a silent no-op. The case stops reaching the branch it exists
for and still compares IDENTICAL on both sides of a refactor, so the oracle passes a
change it never exercised.

HOW: every hooked case runs both hook phases on a freshly built trainer and must add no
`trainer.config` key and no `vars(trainer)` attribute. One trainer is built per distinct
`build` kwargs: after the first, a build took under 0.1 s on the development VM
(2026-10-06), and the hooks only assign values, so the cases of one group can share it. A
misspelt key and attribute on the same trainer is the positive control.
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


def test_every_case_hook_sets_only_existing_config_keys_and_attributes():
    teq = importlib.import_module("scripts.trainer_equivalence")
    groups: dict[str, list[str]] = {}
    for name, case in teq.CASES.items():
        if case.get("hook"):
            groups.setdefault(json.dumps(case["build"], sort_keys=True), []).append(name)
    assert groups, "no case has a hook: the check below would pass on nothing"
    added, control = {}, None
    for build_kwargs, names in groups.items():
        trainer, cleanup = teq.build(0, **json.loads(build_kwargs))
        try:
            for name in names:
                new = _added_names(trainer, teq.CASES[name]["hook"])
                if new != ([], []):
                    added[name] = new
            if control is None:
                control = _added_names(
                    trainer, teq._chain(teq._cfg(target_kll=1e-4), teq._attrs(total_minibatchs=7)))
        finally:
            cleanup()
    assert control == (["target_kll"], ["total_minibatchs"])
    assert added == {}, ("these cases set (config keys, attributes) the trainer does not "
                         f"have, so they no longer reach their branch: {added}")
