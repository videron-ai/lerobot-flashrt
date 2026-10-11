"""Browser GUI for guided KER recording sessions on the bimanual OpenArm rig.

    python -m record_gui              # inside the container, from the repo root
    bash scripts/run_record_gui.sh    # from the host: finds or starts the container

Two processes. The server (server.py) serves the page, runs the preflight checks and
starts one recorder at a time. The recorder (worker.py) is lerobot-ker-record with a
small bridge thread that reports its state and camera previews to the server and takes
the operator's commands back. Keeping them apart means the web server never competes
with the 250 Hz arm loops for the interpreter, and a recorder crash leaves the GUI up.

Needs nothing beyond what the lerobot_flashrt image already has (aiohttp, OpenCV).
"""
