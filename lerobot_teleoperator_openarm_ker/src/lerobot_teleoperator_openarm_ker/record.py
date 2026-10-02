"""`lerobot-ker-record`: lerobot-record with the KER's guided park workflow.

Takes exactly the same flags as lerobot-record. It wraps LeRobot's record loop so the
KER teleoperator knows when an episode is being recorded (see guide.py). With
--teleop.park=false, or a different teleoperator, it behaves like plain lerobot-record.

    lerobot-ker-record --config_path=... --teleop.type=openarm_ker --teleop.park=true ...
    python -m lerobot_teleoperator_openarm_ker.record ...      # same thing
"""

import logging

from . import guide

logger = logging.getLogger(__name__)


def install() -> None:
    """Patch lerobot.scripts.lerobot_record.record_loop. Idempotent."""
    import lerobot.scripts.lerobot_record as rec

    from .openarm_ker import OpenArmKER

    if getattr(rec.record_loop, "_ker_guided", False):
        return
    original = rec.record_loop
    guide.launcher_active = True

    def record_loop(*args, **kwargs):
        teleop = kwargs.get("teleop")
        session = teleop.guide if isinstance(teleop, OpenArmKER) else None
        if session is None:
            return original(*args, **kwargs)

        robot, events = kwargs["robot"], kwargs["events"]
        if session.robot is None:
            # First loop of the session: also route robot.disconnect() through the
            # return to the rest pose.
            disconnect = robot.disconnect

            def disconnect_at_rest(*a, **k):
                try:
                    session.rest_before_disconnect()
                finally:
                    return disconnect(*a, **k)

            robot.disconnect = disconnect_at_rest
        session.attach(robot, events, kwargs["fps"])

        if kwargs.get("dataset") is None:
            session.start_reset()
        elif not session.wait_for_takeover():
            # Stopped while waiting: there is nothing to save, so have LeRobot drop the
            # (empty) episode instead of saving it.
            events["rerecord_episode"] = True
            return None
        try:
            return original(*args, **kwargs)
        finally:
            session.end_phase()

    record_loop._ker_guided = True
    rec.record_loop = record_loop


def main() -> None:
    import lerobot.scripts.lerobot_record as rec

    install()
    rec.main()


if __name__ == "__main__":
    main()
