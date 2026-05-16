"""Factories that pick real or fake impls based on env."""
from __future__ import annotations

import os

from ._base import AnimationModel, PoseExtractor


def get_animation_model() -> AnimationModel:
    if os.getenv("ANIMATE_FAKE") == "1":
        from .fake import FakeAnimationModel

        return FakeAnimationModel()
    from .steadydancer import SteadyDancerModel

    return SteadyDancerModel()


def get_pose_extractor() -> PoseExtractor:
    if os.getenv("ANIMATE_FAKE") == "1":
        from .fake import FakePoseExtractor

        return FakePoseExtractor()
    from .dwpose import DwPoseExtractor

    return DwPoseExtractor()
