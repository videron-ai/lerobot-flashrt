"""LeRobot robot plugin: bimanual OpenArm follower with a 250 Hz MIT control loop."""

from .bi_openarm_streaming_follower import BiOpenArmStreamingFollower
from .config_bi_openarm_streaming_follower import BiOpenArmStreamingFollowerConfig
from .streaming import StreamSample, get_target_source, register_target_source, unregister_target_source

__all__ = [
    "BiOpenArmStreamingFollower",
    "BiOpenArmStreamingFollowerConfig",
    "StreamSample",
    "get_target_source",
    "register_target_source",
    "unregister_target_source",
]
