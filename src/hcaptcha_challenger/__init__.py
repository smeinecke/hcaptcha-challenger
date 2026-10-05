# Time       : 2022/2/15 17:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
from __future__ import annotations

from pathlib import Path

from hcaptcha_challenger import models as types
from hcaptcha_challenger.agent.challenger import AgentConfig, AgentV
from hcaptcha_challenger.agent.collector import Collector, CollectorConfig
from hcaptcha_challenger.models import (
    CaptchaResponse,
    ChallengeTypeEnum,
    CoordinateGrid,
    FastShotModelType,
    RequestType,
    SCoTModelType,
)
from hcaptcha_challenger.tools import (
    ChallengeClassifier,
    ImageClassifier,
    SpatialBboxReasoner,
    SpatialPathReasoner,
    SpatialPointReasoner,
)
from hcaptcha_challenger.utils import init_log

__all__ = [
    "AgentConfig",
    "AgentV",
    "CaptchaResponse",
    'ChallengeClassifier',
    "ChallengeTypeEnum",
    "Collector",
    "CollectorConfig",
    "CoordinateGrid",
    "FastShotModelType",
    "ImageClassifier",
    "RequestType",
    "SCoTModelType",
    'SpatialBboxReasoner',
    'SpatialPathReasoner',
    'SpatialPointReasoner',
    "types",
]

LOG_DIR = Path(__file__).parent.joinpath("logs", "{time:YYYY-MM-DD}")

init_log(
    runtime=LOG_DIR.joinpath("runtime.log"),
    error=LOG_DIR.joinpath("error.log"),
    serialize=LOG_DIR.joinpath("serialize.log"),
)
