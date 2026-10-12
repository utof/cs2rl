"""The trainer's ``--opponent duel``: #152 L2's duel bot plays the opponent team (Rung 1c).

WHAT
----
``DuelOpponent`` drives every opponent row of the rollout with one ``cs2rl.eval.duel_bot``
``DuelBot``, which decides from each row's own observation and action mask, exactly as
cs2rl.experiment.oracle_duel's obs-only bots do. Each env plays one level of the run's mix
(``config["duel_mix"]``, parsed by cs2rl.train.config.parse_duel_mix) per episode: at
construction and at each of the env's episode ends it draws a new level, then configures
and resets the bot on all of that env's opponent rows. It also counts the hero's outcome
of every finished round per level (``OUTCOMES``); the trainer flushes the counts into each
logged metrics row as ``environment/duel/L{i}/{outcome}``, i the level's index in the mix.

WHY
---
Rung 1c fine-tunes against an opponent that shoots back, over a mix of levels rather than
one opponent (Rung 1c research A5: begin easy, keep a mix, hold a family out for the
gate). The per-level counts are what a later success-band sampler reads. It sits in its
own module, beside the trainer rather than in it, so it can be tested without torch: the
trainer only hands it numpy rows and writes its actions back.

PITFALLS
--------
* A round's outcome comes from the terminal info of its env (``winner``, cs2_types.h
  GameState). The vector env returns infos in env order and each env's round end adds
  exactly one info that carries ``winner`` (the harness also returns per-tick infos
  without it), so the k-th such info belongs to the k-th env, in row order, whose rows
  are done. ``step`` checks the two counts agree.
* The obs of a done row is already the new episode's first observation (Cs2Env auto-resets
  inside the step that ends the round), so the bot resets before it decides from it.
* One level per ENV, shared by its opponent rows: with one agent a side (the only roster
  cs2rl.train.config.assert_duel_env_supported allows) that is one row's level.
* The bot and the level draws use two Generators spawned from one seed, so the bot's
  stream is the one a ``DuelBot`` built with the first of them alone would draw.
* The bot's per-row state resets with the env's episode, never with the rollout.
* Not checkpointed: a resumed run rebuilds the bot, its levels and its Generators from
  the run seed, as the walker is rebuilt (cs2rl.train.loop._resume_full_state).
"""
from __future__ import annotations

import numpy as np

from cs2rl.eval.duel_bot import DuelBot, parse_params

OUTCOMES = ("win", "loss", "double", "timeout")
# cs2_types.h GameState.winner: 0 = T, 1 = CT, -1 = timed out, WINNER_DRAW = -2 (both
# teams died on one tick).
WINNER_TIMEOUT = -1
WINNER_DRAW = -2


def duel_rngs(seed: int) -> tuple[np.random.Generator, np.random.Generator]:
    """(the bot's Generator, the level draws' Generator), independent, from ``seed``."""
    bot, levels = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(bot), np.random.default_rng(levels)


class DuelOpponent:
    """The duel bot over the opponent rows of a rollout, with a per-episode level mix.

    ``bot`` is a DuelBot over all opponent rows, ``rows_per_env`` of them per env in env
    order; ``mix`` is ``[[spec, weight], ...]``; ``hero_team`` is the learner's team (0 =
    T, 1 = CT); ``rng`` draws the levels.
    """

    def __init__(self, bot: DuelBot, mix, rows_per_env: int, hero_team: int,
                 rng: np.random.Generator):
        if bot.n_rows % rows_per_env:
            raise ValueError(f"{bot.n_rows} rows are not whole envs of {rows_per_env}")
        self.bot, self.rows_per_env, self.hero, self.rng = bot, rows_per_env, hero_team, rng
        self.params = [parse_params(spec) for spec, _ in mix]
        self.weights = np.array([w for _, w in mix], dtype=np.float64)
        n_envs = bot.n_rows // rows_per_env
        self.level = np.zeros(n_envs, dtype=np.int64)  # each env's level this episode
        self.counts = np.zeros((len(self.params), len(OUTCOMES)), dtype=np.int64)
        self._new_episodes(np.ones(n_envs, dtype=bool))

    def _new_episodes(self, envs) -> None:
        """Draw a level for each env in the bool mask ``envs``; configure and reset its rows."""
        idx = np.flatnonzero(envs)
        if not len(idx):
            return
        self.level[idx] = self.rng.choice(len(self.params), size=len(idx), p=self.weights)
        rows = (idx[:, None] * self.rows_per_env + np.arange(self.rows_per_env)).ravel()
        self.bot.configure(rows,
                           [self.params[k] for k in np.repeat(self.level[idx], self.rows_per_env)])
        self.bot.reset(rows)

    def _outcome(self, winner: int) -> int:
        if winner == self.hero:
            return OUTCOMES.index("win")
        if winner == 1 - self.hero:
            return OUTCOMES.index("loss")
        if winner == WINNER_DRAW:
            return OUTCOMES.index("double")
        if winner == WINNER_TIMEOUT:
            return OUTCOMES.index("timeout")
        raise ValueError(f"unknown winner {winner!r}")

    def step(self, obs, mask, done, infos):
        """One tick: ``obs`` (n, OBS_DIM) and ``mask`` (n, ACTION_MASK_DIM) of the opponent
        rows, their bool ``done`` flags and the tick's info list. Counts the finished
        rounds, starts the done envs' new episodes, then returns the bot's int32 (n,
        ACTION_DIM) actions and float32 (n, AIM_DIM) aim, as DuelBot.step_obs does."""
        if len(obs) != self.bot.n_rows:
            raise ValueError(f"{len(obs)} opponent rows, the bot has {self.bot.n_rows}")
        ended = np.asarray(done, dtype=bool).reshape(-1, self.rows_per_env).any(axis=1)
        winners = [i["winner"] for i in infos if "winner" in i]
        if len(winners) != int(ended.sum()):
            raise AssertionError(f"{len(winners)} terminal infos for {int(ended.sum())} "
                                 "finished envs: cannot tell which level each round was")
        for level, winner in zip(self.level[ended], winners, strict=True):
            self.counts[level, self._outcome(int(winner))] += 1
        self._new_episodes(ended)
        return self.bot.step_obs(obs, mask)

    def pop_counts(self) -> np.ndarray:
        """The (levels, OUTCOMES) round counts since the last call; resets them."""
        out, self.counts = self.counts, np.zeros_like(self.counts)
        return out
