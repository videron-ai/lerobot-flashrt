"""Hook that lets a teleoperator feed targets straight into the follower's control loop.

LeRobot only hands actions to the robot once per recorded frame. A teleoperator that
can produce targets faster (the OpenArm KER streams at 500 Hz) registers itself here,
and the streaming follower's 250 Hz loop samples it directly. Both live in the same
process (lerobot-record / lerobot-teleoperate / lerobot-rollout), so a module-level
registry is enough.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class StreamSample:
    # Keyed like LeRobot actions, e.g. "left_joint_1.pos". Degrees.
    positions: dict[str, float]
    # Same keys, deg/s. Used as MIT velocity feed-forward.
    velocities: dict[str, float]
    # True only while the data is fresh AND someone is actively teleoperating
    # (polling get_action). When False the follower falls back to send_action().
    active: bool
    # time.monotonic() of the data.
    stamp: float


class TargetSource(Protocol):
    def sample(self, now: float) -> StreamSample | None: ...

    # Optional: hold_until_ready(check). If present, the follower calls it at the end
    # of connect(), once its loops are running. It may block until the teleoperator
    # wants LeRobot's own loop to start (the KER uses it to keep park moves out of
    # recordings) and should call check() regularly, which raises if a loop has failed.


_lock = threading.Lock()
_source: TargetSource | None = None


def register_target_source(source: TargetSource) -> None:
    global _source
    with _lock:
        if _source is not None and _source is not source:
            raise RuntimeError("Another teleoperator is already streaming targets to the follower.")
        _source = source


def unregister_target_source(source: TargetSource) -> None:
    global _source
    with _lock:
        if _source is source:
            _source = None


def get_target_source() -> TargetSource | None:
    with _lock:
        return _source
