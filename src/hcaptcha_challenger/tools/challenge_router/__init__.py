"""
ChallengeRouter - Challenge type classification tool.

This module provides tools to classify challenge screenshots into their
respective types (single/multi select, single/multi drag) and extract
the challenge prompt.
"""

from pathlib import Path

from hcaptcha_challenger.models import (
    DEFAULT_FAST_SHOT_MODEL,
    ChallengeRouterResult,
    FastShotModelType,
)
from hcaptcha_challenger.tools.internal.base import Reasoner
from hcaptcha_challenger.tools.internal.providers.protocol import ChatProvider
from hcaptcha_challenger.utils import load_desc


class ChallengeRouter(Reasoner[FastShotModelType, ChallengeRouterResult]):
    """
    Challenge router that classifies challenge type and extracts the prompt.

    This tool analyzes challenge screenshots and returns both the challenge
    type (single/multi select or drag) and the extracted challenge prompt.

    Attributes:
        description: The system prompt for the tool.
    """

    description: str = load_desc(Path(__file__).parent / "challenge_router.md")

    def __init__(
        self,
        gemini_api_key: str,
        model: FastShotModelType = DEFAULT_FAST_SHOT_MODEL,
        *,
        provider: ChatProvider | None = None,
        **kwargs,
    ):
        super().__init__(gemini_api_key, model, provider=provider, **kwargs)

    async def __call__(
        self, *, challenge_screenshot: str | Path, **kwargs
    ) -> ChallengeRouterResult:
        """
        Classify a challenge screenshot and extract its prompt.

        Args:
            challenge_screenshot: Path to the challenge image.
            **kwargs: Additional options passed to the provider.

        Returns:
            ChallengeRouterResult containing challenge_type and challenge_prompt.
        """
        return await self._provider.generate_with_images(
            images=[Path(challenge_screenshot)],
            user_prompt="提取 `challenge_prompt`，并返回正确的 `challenge_type`。",
            description=self.description,
            response_schema=ChallengeRouterResult,
            **kwargs,
        )


# Backward compatibility alias
ChallengeClassifier = ChallengeRouter
