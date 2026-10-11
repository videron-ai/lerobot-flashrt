"""Wire format between the GUI server and the recorder, over one Unix socket.

Every message is a 5-byte header (payload length, kind) followed by the payload:

    JSON   a UTF-8 JSON object: status from the recorder, commands from the server
    FRAME  a camera preview: 1 byte name length, the camera name, then a JPEG
"""

import json
import struct

HEADER = struct.Struct("!IB")
JSON = 0x4A
FRAME = 0x46
MAX_PAYLOAD = 16 * 1024 * 1024


def pack_json(obj) -> bytes:
    payload = json.dumps(obj, separators=(",", ":")).encode()
    return HEADER.pack(len(payload), JSON) + payload


def pack_frame(name: str, jpeg: bytes) -> bytes:
    encoded = name.encode()
    return HEADER.pack(1 + len(encoded) + len(jpeg), FRAME) + bytes([len(encoded)]) + encoded + jpeg


def unpack_frame(payload: bytes) -> tuple[str, bytes]:
    n = payload[0]
    return payload[1 : 1 + n].decode(), payload[1 + n :]
