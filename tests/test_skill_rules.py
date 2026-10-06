"""Offline regression tests for skill-rule matching (no API keys needed)."""

from hcaptcha_challenger.models import ChallengeTypeEnum
from hcaptcha_challenger.skills import SkillManager
from hcaptcha_challenger.skills.schema import _normalize_text


def _skill_for(prompt: str, job=ChallengeTypeEnum.IMAGE_DRAG_SINGLE) -> str:
    return SkillManager().get_skill(prompt, job)


def test_screw_prompt_matches_dedicated_template():
    text = _skill_for("Please drag the screw to the empty joint")
    assert "empty joint" in text or "空" in text
    assert text != f"JobType: {ChallengeTypeEnum.IMAGE_DRAG_SINGLE.value}"


def test_pattern_prompt_matches_dedicated_template():
    text = _skill_for(
        "Complete the pattern by dragging the correct shape into the empty cell",
        job=ChallengeTypeEnum.IMAGE_DRAG_MULTI,
    )
    assert text != f"JobType: {ChallengeTypeEnum.IMAGE_DRAG_MULTI.value}"


def test_matching_outline_prompt_matches():
    text = _skill_for("Put the animal into its matching outline")
    assert text != f"JobType: {ChallengeTypeEnum.IMAGE_DRAG_SINGLE.value}"


def test_matching_silhouette_prompt_matches():
    text = _skill_for("Drag the shape into its matching silhouette")
    assert text != f"JobType: {ChallengeTypeEnum.IMAGE_DRAG_SINGLE.value}"


def test_normalize_text_strips_homoglyphs():
    assert _normalize_text("plеase") == "please"  # Cyrillic е
    assert _normalize_text("drąg") == "drag"
    assert _normalize_text("ｓcrew") == "screw"  # fullwidth


def test_homoglyph_prompt_still_matches():
    text = _skill_for("Please drag the sсrew to the empty jоint")  # Cyrillic с, о
    assert text != f"JobType: {ChallengeTypeEnum.IMAGE_DRAG_SINGLE.value}"


def test_unrelated_prompt_falls_back():
    text = _skill_for("Solve this challenge completely unrelated words xyzzy")
    assert "JobType:" in text
