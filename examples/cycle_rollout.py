#!/usr/bin/env python3
"""cycle_rollout.py — autonomous fold / home / unfold / home cycles, recorded.

Runs ``online_rollout.py`` (FlashRT backend, same flags and env vars) with an
episodic strategy that, instead of one task with operator reset phases, steps
through a fixed task cycle and records every step as its own episode:

    1. "Fold the T-shirt properly"   up to 90 s
    2. "Return to home position"          15 s
    3. "Unfold the T-shirt"          up to 30 s
    4. "Return to home position"          15 s

The cycle repeats ``--cycles`` times (``--cycles=0`` repeats until Escape).
There are no reset phases: the home steps are the reset.  Each episode is
labelled with its own task, and the policy's prompt switches with it.

Usage:
    FLASHRT_CALIB_DATASET=videron/openarm_bimanual_shirt_folding_gen3_high_quality_rightfirst_recomputed_stats_ep1 \\
    python examples/cycle_rollout.py --cycles=5 \\
        --config_path=./helpers/rollout.yaml \\
        --strategy.type=episodic \\
        --policy.path=/Models/openarm_folding_gen3_level2_multitask_50k/ \\
        --task="Fold the T-shirt properly" \\
        --interpolation_multiplier=3 \\
        --inference.type=rtc \\
        --inference.rtc.execution_horizon=20 \\
        --inference.rtc.max_guidance_weight=5.0 \\
        --inference.rtc.prefix_attention_schedule=LINEAR \\
        --use_torch_compile=False \\
        --policy.device=cuda \\
        --dataset.repo_id=videron/rollout_gen3_level2_multitask \\
        --dataset.streaming_encoding=True \\
        --dataset.encoder_threads=2 \\
        --play_sounds=True

``--task`` is only used to calibrate and warm up FlashRT (it defaults to the
first step's task).  ``--dataset.num_episodes``, ``--dataset.episode_time_s``
and ``--dataset.reset_time_s`` are ignored: the cycle defines the episodes.

Keyboard (same as the episodic strategy):
    Right arrow  — end the current step early and move to the next one
    Left arrow   — discard the current step's episode and run the step again
    Escape       — stop; the in-progress episode is saved
"""

# NOTE: no ``from __future__ import annotations`` — see online_rollout.py.

import contextlib
import logging
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import online_rollout as _base  # noqa: E402  (FlashRT backend + rollout entry point)

from lerobot.datasets import VideoEncodingManager  # noqa: E402
from lerobot.rollout.strategies.episodic import EpisodicStrategy  # noqa: E402
from lerobot.utils.cycle_timer import CycleTimer  # noqa: E402
from lerobot.utils.utils import log_say  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Step:
    task: str          # must match a task string the checkpoint was trained on
    max_time_s: float  # the step ends here unless Right arrow ends it sooner


# Task strings match the training set of openarm_folding_gen3_level2_multitask
# (videron/openarm_bimanual_shirt_folding_gen3_345_episodes_fold_unfold_home_rightfirst,
# meta/tasks.parquet) exactly.
HOME = Step("Return to home position", 15.0)
CYCLE: tuple[Step, ...] = (
    Step("Fold the T-shirt properly", 90.0),
    HOME,
    Step("Unfold the T-shirt", 30.0),
    HOME,
)


def _pop_cycles_arg(argv: list[str]) -> int:
    """Remove ``--cycles N`` / ``--cycles=N`` from argv (draccus would reject it)."""
    cycles = 1
    for i, arg in enumerate(argv):
        if arg == "--cycles" and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
        elif arg.startswith("--cycles="):
            value = arg.split("=", 1)[1]
            del argv[i]
        else:
            continue
        cycles = int(value)
        if cycles < 0:
            raise SystemExit("--cycles must be >= 0 (0 = repeat until Escape)")
        break
    return cycles


class CycleStrategy(EpisodicStrategy):
    """Episodic recording that walks :data:`CYCLE`, one episode per step.

    Each step sets the engine's task and then resets it — ``reset()`` drops the
    previous step's queued actions and re-primes the task label — so the first
    chunk of a step is inferred with the new prompt from the current pose.
    ``_policy_loop`` (and the episode-end engine pause online_rollout installs
    on it) is inherited unchanged.
    """

    def __init__(self, config, cycles: int) -> None:
        super().__init__(config)
        self._cycles = cycles

    def run(self, ctx) -> None:
        cfg = ctx.runtime.cfg
        dataset = ctx.data.dataset
        if dataset is None:
            raise RuntimeError("cycle_rollout records every step: pass --dataset.repo_id=<user>/rollout_<name>")
        events = self._events
        play_sounds = cfg.play_sounds
        total_steps = self._cycles * len(CYCLE)  # 0 = until Escape

        logger.info(
            "Cycle plan: %s x %s",
            self._cycles or "until Escape",
            " -> ".join(f"{s.task!r} ({s.max_time_s:.0f} s)" for s in CYCLE),
        )

        timer = CycleTimer(cfg.fps, self._interpolator.multiplier)

        with VideoEncodingManager(dataset):
            try:
                step_index = 0
                while (total_steps == 0 or step_index < total_steps) and not events["stop_recording"]:
                    if ctx.runtime.shutdown_event.is_set():
                        break

                    cycle, position = divmod(step_index, len(CYCLE))
                    step = CYCLE[position]

                    # set_task before reset: reset discards the pending change
                    # along with the queue, so the new task applies from the
                    # first chunk rather than being blended over the old one.
                    self._engine.set_task(step.task)
                    self._engine.reset()
                    self._interpolator.reset()
                    timer.restart()
                    self._engine.resume()

                    cycle_label = f"{cycle + 1}/{self._cycles}" if self._cycles else f"{cycle + 1}"
                    logger.info(
                        "Cycle %s, step %d/%d: %r (max %.0f s) -> episode %d",
                        cycle_label, position + 1, len(CYCLE), step.task,
                        step.max_time_s, dataset.num_episodes,
                    )
                    log_say(step.task, play_sounds)

                    self._policy_loop(
                        ctx=ctx,
                        robot=ctx.hardware.robot_wrapper,
                        events=events,
                        features=ctx.data.dataset_features,
                        timer=timer,
                        control_time_s=step.max_time_s,
                        dataset=dataset,
                        single_task=step.task,
                    )

                    if events["rerecord_episode"]:
                        log_say("Re-record episode", play_sounds)
                        events["rerecord_episode"] = False
                        events["exit_early"] = False
                        dataset.clear_episode_buffer()
                        timer.log_episode_summary("discarded episode")
                        continue  # same step again, from wherever the arms are now

                    dataset.save_episode()
                    step_index += 1
                    timer.log_episode_summary(f"episode {dataset.num_episodes} ({step.task})")
            finally:
                # As in EpisodicStrategy.run: keep the frames of an episode cut
                # short by Escape, an exception or Ctrl-C.  save_episode raises
                # on an empty buffer, which is nothing to lose.
                logger.info("Cycle loop ended — saving any in-progress episode")
                timer.log_run_summary()
                with contextlib.suppress(Exception):
                    dataset.save_episode()


def main() -> None:
    cycles = _pop_cycles_arg(sys.argv)

    # FlashRT calibrates and warms up on --task; default it to the first step.
    if not any(a == "--task" or a.startswith("--task=") for a in sys.argv[1:]):
        sys.argv.append(f"--task={CYCLE[0].task}")

    def create_cycle_strategy(strategy_cfg):
        if strategy_cfg.type != "episodic":
            raise ValueError(
                f"cycle_rollout.py needs --strategy.type=episodic, got {strategy_cfg.type!r}"
            )
        return CycleStrategy(strategy_cfg, cycles)

    # online_rollout.rollout() builds its strategy through this module-level name.
    _base.create_strategy = create_cycle_strategy
    _base.main()


if __name__ == "__main__":
    main()
