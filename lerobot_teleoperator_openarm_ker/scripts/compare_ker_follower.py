"""Dry run: KER-mapped actions next to live follower positions. Nothing moves.

Only the followers' CAN buses are opened and no position command is ever sent.
The bus handshake does send each motor the Damiao enable frame, so support the
arms (or let them hang at rest) before running. Calibrate the followers with
`lerobot-calibrate` first so their zero matches what lerobot-record will use.

    python scripts/compare_ker_follower.py --left-port left_arm --right-port right_arm \
        --left-id my_bimanual_follower_left --right-id my_bimanual_follower_right \
        --no-can-fd [--mapping ker_mapping.json]

Without --mapping the plugin's defaults apply, including the software wrist remap.

For followers calibrated against their hard stops (openarm_videron, or the
lerobot_robot_openarm_hardstop types), add --hardstop and the folder holding their
calibration files, so the follower column is in the calibrated joint frame the
recording will use:

    python scripts/compare_ker_follower.py --left-port left_arm --right-port right_arm \
        --left-id my_bimanual_follower_left --right-id my_bimanual_follower_right \
        --no-can-fd --hardstop --calibration-dir ../calibration/openarm_follower_hardstop \
        [--mapping ker_mapping.json]

1. Put both followers and the KER in the same pose (arms hanging, grippers
   closed). The DELTA column is then the offset to put in the mapping file.
2. Move one KER joint and the same follower joint by hand in the same physical
   direction. If the two numbers move in opposite directions, flip that sign.
"""

import argparse
import sys
import time
from pathlib import Path

from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig

from lerobot_teleoperator_openarm_ker import OpenArmKER, OpenArmKERConfig

JOINTS = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--left-port", required=True)
    p.add_argument("--right-port", required=True)
    p.add_argument("--left-id", default=None)
    p.add_argument("--right-id", default=None)
    p.add_argument("--mapping", default=None, help="Mapping JSON to test (optional)")
    p.add_argument("--no-can-fd", action="store_true", help="Classic CAN (use_can_fd: False on the follower)")
    p.add_argument("--hardstop", action="store_true", help="Followers use hard-stop calibration (openarm_videron)")
    p.add_argument("--calibration-dir", type=Path, default=None, help="Folder with the followers' calibration files")
    args = p.parse_args()

    if args.hardstop:
        # Its motor bus applies the calibration's zero offsets to every position it reads.
        from lerobot_robot_openarm_hardstop import OpenArmFollowerHardstop as Follower
        from lerobot_robot_openarm_hardstop import OpenArmFollowerHardstopConfig as FollowerConfig
    else:
        Follower, FollowerConfig = OpenArmFollower, OpenArmFollowerConfig
    followers = {
        side: Follower(
            FollowerConfig(
                port=port, side=side, id=id_, use_can_fd=not args.no_can_fd, calibration_dir=args.calibration_dir
            )
        )
        for side, port, id_ in (
            ("left", args.left_port, args.left_id),
            ("right", args.right_port, args.right_id),
        )
    }
    if args.hardstop:
        missing = [str(f.calibration_fpath) for f in followers.values() if not f.is_calibrated]
        if missing:
            sys.exit(f"No hard-stop calibration at {', '.join(missing)}. Run lerobot-calibrate first.")
    for f in followers.values():
        f.bus.connect()  # handshake pings (and enables) each motor; no MIT command is sent

    ker = OpenArmKER(OpenArmKERConfig(mapping_path=args.mapping))
    ker.connect()

    try:
        while True:
            action = ker.get_action()
            lines = [f"{'joint':<18}{'KER':>9}{'follower':>10}{'DELTA':>9}"]
            for side, f in followers.items():
                states = f.bus.sync_read_all_states()
                for j in JOINTS:
                    k = action[f"{side}_{j}.pos"]
                    s = states.get(j)
                    fol = s["position"] if s is not None else float("nan")
                    lines.append(f"{side + '_' + j:<18}{k:>+9.1f}{fol:>+10.1f}{fol - k:>+9.1f}")
                lines.append("")
            sys.stdout.write("\033[H\033[J" + "\n".join(lines) + "\nCtrl+C to stop\n")
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        ker.disconnect()
        for f in followers.values():
            f.bus.disconnect(True)


if __name__ == "__main__":
    main()
