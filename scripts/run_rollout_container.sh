#!/usr/bin/env bash
# Start the lerobot + FlashRT rollout container for the bimanual OpenArm rig.
#
# Mounts this repo at /lerobot-flashrt (the lerobot submodule inside it
# shadows the copy baked into the image, see docker/Dockerfile) and the
# checkpoint directory at /Models, and maps the three cameras by USB port
# (/dev/v4l/by-path) to the stable names helpers/rollout.yaml expects:
# /dev/left_wrist, /dev/right_wrist, /dev/base.  Port-based paths survive
# reboots and re-plugging as long as each camera stays in the same USB port.
#
# Usage:
#     bash scripts/run_rollout_container.sh              # interactive shell
#     bash scripts/run_rollout_container.sh python examples/online_rollout.py ...
#
# Overrides (environment):
#     IMAGE         image tag                      (default lerobot_flashrt:1.0)
#     MODELS_DIR    host checkpoint directory      (default ~/Desktop/Models)
#     LEFT_WRIST_CAM / RIGHT_WRIST_CAM / BASE_CAM
#                   /dev/v4l/by-path entries for each camera

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lerobot_flashrt:1.0}"
MODELS_DIR="${MODELS_DIR:-$HOME/Desktop/Models}"
LEFT_WRIST_CAM="${LEFT_WRIST_CAM:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3:1.0-video-index0}"
RIGHT_WRIST_CAM="${RIGHT_WRIST_CAM:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:8:1.0-video-index0}"
BASE_CAM="${BASE_CAM:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:9:1.0-video-index0}"

# Resolve a by-path camera link to its /dev/videoN node, failing with the
# camera's name instead of docker's bare "no such file or directory".
resolve_cam() {
    local name="$1" link="$2"
    if [[ ! -e "$link" ]]; then
        echo "error: $name camera not found at $link" >&2
        echo "       connected cameras:" >&2
        ls /dev/v4l/by-path/ 2>/dev/null | grep 'video-index0$' | sed 's/^/         /' >&2 || true
        exit 1
    fi
    readlink -f "$link"
}

left_wrist="$(resolve_cam left_wrist "$LEFT_WRIST_CAM")"
right_wrist="$(resolve_cam right_wrist "$RIGHT_WRIST_CAM")"
base="$(resolve_cam base "$BASE_CAM")"

if [[ ! -e "$REPO_DIR/lerobot/pyproject.toml" ]]; then
    echo "error: $REPO_DIR/lerobot is empty — run 'git submodule update --init'" >&2
    exit 1
fi

if [[ $# -eq 0 ]]; then
    set -- /bin/bash
fi

exec docker run \
    -e DISPLAY="${DISPLAY:-}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$HOME/.Xauthority:/root/.Xauthority" \
    -v "$REPO_DIR:/lerobot-flashrt" \
    -v "$MODELS_DIR:/Models" \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --device="$left_wrist:/dev/left_wrist" \
    --device="$right_wrist:/dev/right_wrist" \
    --device="$base:/dev/base" \
    --gpus all -it --rm --network=host --cap-add=NET_ADMIN --ipc host \
    -w /lerobot-flashrt "$IMAGE" "$@"
