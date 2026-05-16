"""Every phase class implements the Phase Protocol."""
from __future__ import annotations

from insta_influencer.core.phase import Phase
from insta_influencer.ec2.phases.animate import AnimatePhase
from insta_influencer.ec2.phases.audio_attach import AudioAttachPhase
from insta_influencer.ec2.phases.interp import InterpPhase
from insta_influencer.ec2.phases.pose_extract import PoseExtractPhase
from insta_influencer.ec2.phases.reels_format import ReelsFormatPhase


def test_all_phases_satisfy_protocol() -> None:
    for cls in (
        PoseExtractPhase,
        AnimatePhase,
        InterpPhase,
        AudioAttachPhase,
        ReelsFormatPhase,
    ):
        instance = cls()
        assert isinstance(instance, Phase)
        assert isinstance(instance.name, str)
        assert isinstance(instance.timeout_s, int)
        assert instance.timeout_s > 0


def test_phase_names_are_unique() -> None:
    names = [
        cls().name
        for cls in (
            PoseExtractPhase,
            AnimatePhase,
            InterpPhase,
            AudioAttachPhase,
            ReelsFormatPhase,
        )
    ]
    assert len(names) == len(set(names))
