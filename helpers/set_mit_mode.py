#!/usr/bin/env python3
"""
set_mit_mode.py — Temporarily switch OpenArm Damiao motors to a control mode (default MIT).

WHAT IT DOES (and does NOT do):
  * Writes the CTRL_MODE register over CAN using Damiao command 0x55 = write-to-RAM.
  * NEVER sends the flash-save command (0xAA). The change is volatile: active until
    the motors are powered off, then they revert to whatever is persisted in flash
    (e.g. Anvil's POS_VEL). Run once per power cycle, BEFORE launching lerobot.
  * Sends ONLY: motor-disable + a RAM register write (+ optional register reads).
    It never enables a motor and never sends a torque/position command, so it cannot
    itself move the arm or drive current. Motors are left DISABLED (limp); lerobot
    enables them in MIT mode when it connects.

PROTOCOL (verified against cmjang/DM_Motor_Control and the Seeed Damiao wiki):
  Param frame -> arbitration ID 0x7FF, 8 data bytes:
    [slave_id & 0xFF, slave_id >> 8, CMD, RID, val0, val1, val2, val3]
    CMD: 0x33 = read, 0x55 = write-to-RAM.   RID: CTRL_MODE = 10 (an integer register,
    so the value is a little-endian uint32: MIT=1, POS_VEL=2, VEL=3).
  Motor control frame -> arbitration ID = slave_id, data = [0xFF]*7 + [cmd]:
    0xFC enable, 0xFD disable, 0xFE set-zero, 0xFB clear-error.

SAFETY:
  * Run with NOTHING else controlling the bus (no lerobot, no Anvil stack running).
  * CAN interfaces must already be up (sudo ip link set canX up).
  * Do a --read-only pass first to confirm it talks to the motors and reads sane modes.
  * Cross-check one motor with Anvil's own tool:
        python -m openarm.damiao param get --motor-type DM8009 --iface follower_l 1 17 control_mode

Requires: python-can (already a lerobot dependency).
"""

import argparse
import struct
import sys
import time

import can  # python-can

# --- Damiao protocol constants (verified) -------------------------------------
PARAM_FRAME_ID = 0x7FF
CMD_READ  = 0x33          # read register
CMD_WRITE = 0x55          # write register to RAM only (volatile)
# CMD_SAVE = 0xAA         # save-to-flash — DELIBERATELY NEVER SENT
RID_CTRL_MODE = 10        # CTRL_MODE register index (integer register)
DISABLE_CMD = 0xFD

MODE_NAMES = {1: "MIT", 2: "POS_VEL", 3: "VEL", 4: "POS_FORCE"}
DEFAULT_MOTOR_IDS = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]
DEFAULT_INTERFACES = ["can0", "can1"]


def disable_motor(bus: can.BusABC, sid: int) -> None:
    bus.send(can.Message(
        arbitration_id=sid,
        data=[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, DISABLE_CMD],
        is_extended_id=False,
    ))


def _send_param(bus: can.BusABC, sid: int, cmd: int, rid: int, value: int = 0) -> None:
    val = struct.pack("<I", value & 0xFFFFFFFF)   # integer register -> uint32 LE
    data = [sid & 0xFF, (sid >> 8) & 0xFF, cmd, rid, val[0], val[1], val[2], val[3]]
    bus.send(can.Message(arbitration_id=PARAM_FRAME_ID, data=data, is_extended_id=False))


def read_control_mode(bus: can.BusABC, sid: int, timeout: float = 0.3):
    """Request CTRL_MODE and return the integer mode, or None if no response.
    Matches the param-response by frame *content* (slave id + RID), so it works
    regardless of the response arbitration ID and won't false-match motor feedback."""
    _send_param(bus, sid, CMD_READ, RID_CTRL_MODE)
    deadline = time.time() + timeout
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return None
        msg = bus.recv(timeout=remaining)
        if msg is None:
            return None
        d = msg.data
        if (len(d) >= 8 and d[0] == (sid & 0xFF) and d[1] == ((sid >> 8) & 0xFF)
                and d[3] == RID_CTRL_MODE and d[2] in (CMD_READ, CMD_WRITE)):
            return int.from_bytes(bytes(d[4:8]), "little")


def set_control_mode(bus: can.BusABC, sid: int, mode: int) -> None:
    _send_param(bus, sid, CMD_WRITE, RID_CTRL_MODE, mode)


def main() -> int:
    ap = argparse.ArgumentParser(description="Temporarily set Damiao control mode (RAM only, never saved).")
    ap.add_argument("--interfaces", nargs="+", default=DEFAULT_INTERFACES)
    ap.add_argument("--ids", nargs="+", type=lambda x: int(x, 0), default=DEFAULT_MOTOR_IDS)
    ap.add_argument("--mode", type=int, default=1, help="1=MIT (default), 2=POS_VEL, 3=VEL")
    ap.add_argument("--fd", action="store_true", help="Open bus in CAN-FD (default classic CAN 2.0)")
    ap.add_argument("--read-only", action="store_true",
                    help="Only read and print current modes. No writes, no disable. Safe first pass.")
    args = ap.parse_args()

    want = args.mode
    want_name = MODE_NAMES.get(want, str(want))
    failures = 0

    for iface in args.interfaces:
        try:
            bus = can.Bus(interface="socketcan", channel=iface, fd=args.fd)
        except Exception as e:
            print(f"[{iface}] ERROR opening bus: {e}")
            failures += 1
            continue

        action = "reading" if args.read_only else f"-> {want_name}"
        print(f"\n[{iface}] {len(args.ids)} motors {action}")
        try:
            for sid in args.ids:
                before = read_control_mode(bus, sid)
                before_s = MODE_NAMES.get(before, str(before)) if before is not None else "no-response"

                if args.read_only:
                    print(f"    0x{sid:02X}: current = {before_s}")
                    if before is None:
                        failures += 1
                    continue

                disable_motor(bus, sid)         # params must be changed while disabled
                time.sleep(0.02)
                set_control_mode(bus, sid, want)
                time.sleep(0.05)
                after = read_control_mode(bus, sid)

                if after == want:
                    print(f"    0x{sid:02X}: {before_s} -> {want_name}  OK")
                elif after is None:
                    print(f"    0x{sid:02X}: {before_s} -> wrote {want_name}, COULD NOT VERIFY "
                          f"(verify with Anvil 'param get')")
                    failures += 1
                else:
                    print(f"    0x{sid:02X}: {before_s} -> wanted {want_name} but reads "
                          f"{MODE_NAMES.get(after, after)}  *** FAIL ***")
                    failures += 1
        finally:
            bus.shutdown()

    if args.read_only:
        print("\nRead-only pass complete. No changes were made.")
    else:
        print("\nDone. Change is volatile (RAM only) and resets on power-off.")
        print("Motors left DISABLED; start lerobot to enable them in the new mode.")
    if failures:
        print(f"\n{failures} motor(s) could not be confirmed — investigate before commanding motion.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
