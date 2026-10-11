#!/usr/bin/env bash
# Open the recording GUI (record_gui/) for the bimanual OpenArm rig.
#
# The GUI runs inside the lerobot + FlashRT container and is used from a browser on
# the host (the container shares the host network).  This script finds a running
# container of the image, or starts one the same way scripts/run_rollout_container.sh
# does plus the KER's USB device, then starts the GUI in it and opens the page.
#
# Usage:
#     bash scripts/run_record_gui.sh            # the rig
#     bash scripts/run_record_gui.sh --demo     # simulated sessions, no robot needed
#     (further arguments go to `python -m record_gui`, e.g. --host 0.0.0.0)
#
# Overrides (environment):
#     IMAGE         image tag                        (default lerobot_flashrt:1.1)
#     CONTAINER     use this container, by name or id
#     PORT          port of the page                 (default 8642)
#     MODELS_DIR    host checkpoint directory        (default ~/Desktop/Models)
#     NAS_DIR       mounted at /nas when it exists   (default /mnt/nas)
#     LEFT_WRIST_CAM / RIGHT_WRIST_CAM / BASE_CAM
#                   /dev/v4l/by-path entries for each camera

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-lerobot_flashrt:1.1}"
PORT="${PORT:-8642}"
MODELS_DIR="${MODELS_DIR:-$HOME/Desktop/Models}"
NAS_DIR="${NAS_DIR:-/mnt/nas}"
LEFT_WRIST_CAM="${LEFT_WRIST_CAM:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:3:1.0-video-index0}"
RIGHT_WRIST_CAM="${RIGHT_WRIST_CAM:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:8:1.0-video-index0}"
BASE_CAM="${BASE_CAM:-/dev/v4l/by-path/pci-0000:00:14.0-usb-0:9:1.0-video-index0}"
KER_USB_ID="303a:4002"
URL="http://localhost:$PORT"

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

start_container() {
    local devices=() mounts=()
    devices+=(--device="$(resolve_cam left_wrist "$LEFT_WRIST_CAM"):/dev/left_wrist")
    devices+=(--device="$(resolve_cam right_wrist "$RIGHT_WRIST_CAM"):/dev/right_wrist")
    devices+=(--device="$(resolve_cam base "$BASE_CAM"):/dev/base")
    # The KER is passed by its current USB address, so re-plugging it needs a new container.
    local ker
    ker="$(lsusb -d "$KER_USB_ID" 2>/dev/null | sed -nE 's|^Bus ([0-9]+) Device ([0-9]+).*|/dev/bus/usb/\1/\2|p' | head -n1)"
    if [[ -n "$ker" ]]; then
        devices+=(--device "$ker")
    else
        echo "warning: no KER ($KER_USB_ID) on USB; starting the container without it" >&2
    fi
    [[ -d "$MODELS_DIR" ]] && mounts+=(-v "$MODELS_DIR:/Models")
    [[ -d "$NAS_DIR" ]] && mounts+=(-v "$NAS_DIR:/nas")
    docker run -d -it --rm \
        -e DISPLAY="${DISPLAY:-}" \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v "$HOME/.Xauthority:/root/.Xauthority" \
        -v "$REPO_DIR:/lerobot-flashrt" \
        "${mounts[@]}" \
        --ulimit memlock=-1 --ulimit stack=67108864 \
        "${devices[@]}" \
        --gpus all --network=host --cap-add=NET_ADMIN --ipc host \
        -w /lerobot-flashrt "$IMAGE" /bin/bash
}

container="${CONTAINER:-$(docker ps -q --filter "ancestor=$IMAGE" | head -n1)}"
if [[ -z "$container" ]]; then
    echo "No running $IMAGE container; starting one."
    container="$(start_container)"
fi
if ! docker exec "$container" test -d /lerobot-flashrt/record_gui; then
    echo "error: container $container does not have this repo at /lerobot-flashrt" >&2
    exit 1
fi

open_page() {
    command -v xdg-open >/dev/null && xdg-open "$URL" >/dev/null 2>&1 || echo "Open $URL"
}

if curl -fsS -o /dev/null "$URL" 2>/dev/null; then
    echo "The GUI is already running at $URL"
    open_page
    exit 0
fi

# Open the page once the server answers.
(
    for _ in $(seq 60); do
        if curl -fsS -o /dev/null "$URL" 2>/dev/null; then
            open_page
            exit 0
        fi
        sleep 0.5
    done
) &

tty=()
[[ -t 0 ]] && tty=(-t)
exec docker exec -i "${tty[@]}" -w /lerobot-flashrt "$container" python -m record_gui --port "$PORT" "$@"
