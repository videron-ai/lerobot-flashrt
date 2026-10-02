"""Read or set the CAN timeout (register 9) on one OpenArm's Damiao motors.

The timeout is the motor's own watchdog: if it receives no command for this long it
disables itself. It is counted in 50 us steps, and 0 turns it off. Run this with
nothing else using the CAN interface (no lerobot process connected to the arm).

    # read the current value on motors 1-8
    python scripts/damiao_can_timeout.py --port can0

    # set 100 ms (until the motors are powered off)
    python scripts/damiao_can_timeout.py --port can0 --set-ms 100

    # set 100 ms and save it to the motors' flash. Saving disables the motors first,
    # so the arm goes limp: support it before running this.
    python scripts/damiao_can_timeout.py --port can0 --set-ms 100 --save

Frame format from Damiao's reference library (DM_Control_Python): parameter frames go
to CAN ID 0x7FF as [id_lo, id_hi, cmd, register, value (4 bytes, little endian)] with
cmd 0x33 = read, 0x55 = write, 0xAA = save to flash.
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

import can

PARAM_ID = 0x7FF
RID_TIMEOUT = 9
CMD_READ, CMD_WRITE, CMD_SAVE = 0x33, 0x55, 0xAA
CMD_DISABLE = 0xFD
COUNT_S = 50e-6


def _param_frame(motor_id: int, cmd: int, rid: int = 0, value: int = 0, fd: bool = True) -> can.Message:
    data = bytes([motor_id & 0xFF, (motor_id >> 8) & 0xFF, cmd, rid]) + struct.pack("<I", value)
    return can.Message(arbitration_id=PARAM_ID, data=data, is_extended_id=False, is_fd=fd)


def _await_param(bus: can.BusABC, motor_id: int, cmd: int, rid: int, timeout: float = 0.1) -> int | None:
    end = time.monotonic() + timeout
    while (left := end - time.monotonic()) > 0:
        msg = bus.recv(timeout=left)
        if msg is None:
            break
        d = bytes(msg.data)
        if len(d) >= 8 and d[2] == cmd and d[3] == rid and (d[0] | (d[1] << 8)) == motor_id:
            return struct.unpack("<I", d[4:8])[0]
    return None


def _request(bus, motor_id, cmd, rid, value=0, fd=True, retries=3):
    for _ in range(retries):
        bus.send(_param_frame(motor_id, cmd, rid, value, fd))
        got = _await_param(bus, motor_id, cmd, rid)
        if got is not None:
            return got
    return None


def _fmt(counts: int | None) -> str:
    if counts is None:
        return "no reply"
    return "off (0)" if counts == 0 else f"{counts * COUNT_S * 1e3:g} ms ({counts} counts)"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", required=True, help="CAN interface, e.g. can0")
    p.add_argument("--interface", default="socketcan")
    p.add_argument("--no-fd", action="store_true", help="classic CAN frames (default: CAN FD, like LeRobot's OpenArm config)")
    p.add_argument("--ids", default="1-8", help="motor send IDs, e.g. 1-8 or 1,2,5")
    p.add_argument("--set-ms", type=float, help="new timeout in ms (0 = off)")
    p.add_argument("--save", action="store_true", help="also save to flash (disables the motors first)")
    p.add_argument("--yes", action="store_true", help="don't ask before disabling the motors for --save")
    args = p.parse_args(argv)

    ids: list[int] = []
    for part in args.ids.split(","):
        lo, _, hi = part.partition("-")
        ids.extend(range(int(lo, 0), int(hi or lo, 0) + 1))
    fd = not args.no_fd
    if args.save and args.set_ms is None:
        p.error("--save needs --set-ms")

    kwargs = {"channel": args.port, "interface": args.interface}
    if fd and args.interface == "socketcan":
        kwargs["fd"] = True
    bus = can.interface.Bus(**kwargs)
    try:
        failed = False
        print("current:")
        for i in ids:
            print(f"  motor 0x{i:02X}: {_fmt(_request(bus, i, CMD_READ, RID_TIMEOUT, fd=fd))}")
        if args.set_ms is None:
            return 0

        counts = round(args.set_ms * 1e-3 / COUNT_S)
        if args.save and not args.yes:
            input(f"Saving disables motors {args.ids} on {args.port}; the arm will go limp. "
                  "Support it, then press Enter (Ctrl-C to cancel)... ")
        print(f"writing {_fmt(counts)}:")
        for i in ids:
            if args.save:
                bus.send(can.Message(arbitration_id=i, data=[0xFF] * 7 + [CMD_DISABLE], is_extended_id=False, is_fd=fd))
                time.sleep(0.01)
            got = _request(bus, i, CMD_WRITE, RID_TIMEOUT, counts, fd=fd)
            ok = got == counts
            failed |= not ok
            print(f"  motor 0x{i:02X}: {_fmt(got)}{'' if ok else '  <-- FAILED'}")
            if args.save and ok:
                bus.send(_param_frame(i, CMD_SAVE, 0, 0, fd))
                time.sleep(0.1)
        if args.save:
            print("saved; re-enable the arm by reconnecting with lerobot")
        return 1 if failed else 0
    finally:
        bus.shutdown()


if __name__ == "__main__":
    sys.exit(main())
