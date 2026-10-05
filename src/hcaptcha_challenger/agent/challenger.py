# -*- coding: utf-8 -*-
# Time       : 2024/4/7 11:43
# Author     : QIN2DIM
# GitHub     : https://github.com/QIN2DIM
# Description:
import asyncio
import base64
import json
import math
import os
import random
import re
from asyncio import Queue
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple
from uuid import uuid4

import matplotlib.pyplot as plt
import msgpack
from loguru import logger
from PIL import Image, ImageDraw
from playwright.async_api import (
    Frame,
    FrameLocator,
    Locator,
    Page,
    Response,
    TimeoutError,
    expect,
)
from pydantic import Field, PrivateAttr, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.helper import create_coordinate_grid
from hcaptcha_challenger.models import (
    DEFAULT_FAST_SHOT_MODEL,
    DEFAULT_SCOT_MODEL,
    IGNORE_REQUEST_TYPE_LITERAL,
    INV,
    CaptchaPayload,
    CaptchaResponse,
    ChallengeSignal,
    ChallengeTypeEnum,
    CoordinateGrid,
    FastShotModelType,
    RequestType,
    SCoTModelType,
    SpatialPath,
)
from hcaptcha_challenger.skills import SkillManager
from hcaptcha_challenger.tools import (
    ChallengeRouter,
    ImageClassifier,
    SpatialPathReasoner,
    SpatialPointReasoner,
)
from hcaptcha_challenger.tools.internal.providers import ChatProvider, OpenAIProvider


def _generate_bezier_trajectory(
    start: Tuple[float, float], end: Tuple[float, float], steps: int
) -> List[Tuple[float, float]]:
    """
    Generates a quadratic bezier curve trajectory between start and end points.
    """
    points = []

    # Calculate distance between points
    distance = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)

    # Create control point(s) for the bezier curve
    # For longer distances, we use a higher control point offset
    offset_factor = min(0.3, max(0.1, distance / 1000))

    # Random control point that's offset from the midpoint
    mid_x = (start[0] + end[0]) / 2
    mid_y = (start[1] + end[1]) / 2

    # Create slight randomness in the control point
    control_x = mid_x + random.uniform(-1, 1) * distance * offset_factor
    control_y = mid_y + random.uniform(-1, 1) * distance * offset_factor

    # Generate points along the bezier curve
    for i in range(steps + 1):
        t = i / steps
        # Quadratic bezier formula
        x = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control_x + t**2 * end[0]
        y = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control_y + t**2 * end[1]
        points.append((x, y))

    return points


def _generate_dynamic_delays(steps: int, base_delay: int) -> List[float]:
    """
    Generates dynamic delays between mouse movements to simulate human-like acceleration/deceleration.
    """
    delays = []

    # Acceleration profile: slower at start and end, faster in the middle
    for i in range(steps + 1):
        progress = i / steps

        # Ease in-out function (slow start, fast middle, slow end)
        if progress < 0.5:
            factor = 2 * progress * progress  # Accelerate
        else:
            progress = progress - 1
            factor = 1 - (-2 * progress * progress)  # Decelerate

        # Adjust delay based on position in the curve (1.5x at ends, 0.6x in middle)
        delay_factor = 1.5 - 0.9 * factor

        # Add slight randomness to delays (±10%)
        random_factor = random.uniform(0.9, 1.1)

        delays.append(base_delay * delay_factor * random_factor)

    return delays


def _nearest_entity_center(
    pt: Tuple[int, int] | None,
    centers: List[Tuple[int, int]],
    radius: float,
) -> Tuple[int, int] | None:
    """Entity center nearest to pt, or None when beyond radius."""
    if not pt or not centers:
        return None
    nearest = min(centers, key=lambda c: (c[0] - pt[0]) ** 2 + (c[1] - pt[1]) ** 2)
    if math.hypot(nearest[0] - pt[0], nearest[1] - pt[1]) <= radius:
        return nearest
    return None


def _entity_centers_webpage(task, bbox: dict | None) -> List[Tuple[int, int]]:
    """
    Entity centers from the captcha payload converted to webpage coordinates.

    Entity `coords` are the image-relative top-left corner; `size` is w×h.
    """
    centers: List[Tuple[int, int]] = []
    if not (task and task.entities and bbox):
        return centers
    for ent in task.entities:
        if ent.coords and len(ent.coords) >= 2 and ent.size and len(ent.size) >= 2:
            centers.append(
                (
                    int(bbox["x"] + ent.coords[0] + ent.size[0] // 2),
                    int(bbox["y"] + ent.coords[1] + ent.size[1] // 2),
                )
            )
    return centers


SINGLE_IGNORE_TYPE = IGNORE_REQUEST_TYPE_LITERAL | RequestType | ChallengeTypeEnum
IGNORE_REQUEST_TYPE_LIST = List[SINGLE_IGNORE_TYPE]


class AgentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    # == Legacy Gemini Configuration (maintained for backward compatibility) == #
    GEMINI_API_KEY: SecretStr = Field(
        default_factory=lambda: SecretStr(os.environ.get("GEMINI_API_KEY", "")),
        description="Create API Key https://aistudio.google.com/app/apikey",
    )

    # == Modular LLM Provider Configuration == #
    LLM_PROVIDER: str = Field(
        default="gemini",
        description="LLM provider to use: 'gemini' or 'openai' (OpenAI-compatible)",
    )
    LLM_API_KEY: SecretStr | None = Field(
        default=None,
        description="API key for the selected LLM provider. If not set, uses GEMINI_API_KEY for Gemini provider.",
    )
    LLM_BASE_URL: str | None = Field(
        default=None,
        description="Base URL for OpenAI-compatible APIs (e.g., https://openrouter.ai/api/v1)",
    )
    LLM_MODEL: str | None = Field(
        default=None,
        description="Model name to use. If not set, uses provider-specific defaults.",
    )
    OPENAI_ENABLE_LOGGING: bool = Field(
        default=False,
        description="Enable request logging for OpenAI provider (logs full request with base64 truncated).",
    )

    cache_dir: Path = Path("tmp/.cache")
    challenge_dir: Path = Path("tmp/.challenge")
    captcha_response_dir: Path = Path("tmp/.captcha")
    ignore_request_types: IGNORE_REQUEST_TYPE_LIST | None = Field(default_factory=list)
    ignore_request_questions: List[str] | None = Field(default_factory=list)

    DISABLE_BEZIER_TRAJECTORY: bool = Field(
        default=False,
        description="If you use Camoufox, it is recommended to turn off "
        "the custom Bessel track generator of hcaptcha-challenger "
        "and use Camoufox(humanize=True)",
    )

    DISABLE_HSW_REVERSE: bool = Field(
        default=False,
        description="Force disable HSW reverse engineering and fallback to visual recognition. "
        "Useful for testing the fallback branch when HSW decoding fails.",
    )

    MAX_CRUMB_COUNT: int = Field(
        default=2,
        description="""
        CRUMB_COUNT: The number of challenge rounds you need to solve once the challenge starts.
        In the vast majority of cases this value will be 2, some specialized sites will set this value to 3.
        In most cases you don't need to change this value, the `_review_challenge_type` task determines the exact value of `CRUMB_COUNT` based on the information of the assigned task.
        Only manually change this value if you are working on a very specific task that prevents the `_review_challenge_type` from hijacking the task information and the maximum number of tasks > 2.
        """,
    )

    EXECUTION_TIMEOUT: float = Field(
        default=120,
        description="When your local network is poor, increase this value appropriately [unit: second]",
    )
    RESPONSE_TIMEOUT: float = Field(
        default=30,
        description="When your local network is poor, increase this value appropriately [unit: second]",
    )
    RETRY_ON_FAILURE: bool = Field(
        default=True, description="Re-execute the challenge when it fails"
    )
    WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS: int = Field(
        default=1500,
        description="When your local network is poor, increase this value appropriately [unit: millisecond]",
    )

    MAX_CHALLENGES: int = Field(
        default=10,
        description="Maximum number of challenges to attempt before aborting (credit protection)",
    )

    ENTITY_SNAP_RADIUS: int = Field(
        default=80,
        description="Max pixel distance for snapping LLM drag endpoints to known entity "
        "centers from the captcha payload [unit: pixel]",
    )

    LLM_REQUEST_TIMEOUT: float = Field(
        default=90.0,
        description="Per-request timeout for LLM API calls [unit: second]",
    )

    VERIFY_DRAG_PATHS: bool = Field(
        default=True,
        description="Run a self-verification pass on drag predictions: render proposed "
        "paths onto the challenge image and let the model confirm or correct them",
    )

    USE_HTML5_DRAG: bool = Field(
        default=False,
        description="Use HTML5 Drag and Drop API instead of mouse simulation for drag challenges",
    )

    USE_POINTER_EVENTS: bool = Field(
        default=False,
        description="Use Pointer Events API (pointerdown/move/up) instead of mouse simulation for drag challenges",
    )

    CHALLENGE_CLASSIFIER_MODEL: FastShotModelType = Field(
        default=DEFAULT_FAST_SHOT_MODEL,
        description="For the challenge classification task \n"
        "Used as last resort when HSW decoding fails.",
    )
    IMAGE_CLASSIFIER_MODEL: SCoTModelType = Field(
        default=DEFAULT_SCOT_MODEL, description="For the challenge type: `image_label_binary`"
    )
    SPATIAL_POINT_REASONER_MODEL: SCoTModelType = Field(
        default=DEFAULT_SCOT_MODEL,
        description="For the challenge type: `image_label_area_select` (single/multi)",
    )
    SPATIAL_PATH_REASONER_MODEL: SCoTModelType = Field(
        default=DEFAULT_SCOT_MODEL,
        description="For the challenge type: `image_drag_drop` (single/multi)",
    )

    coordinate_grid: CoordinateGrid | None = Field(default_factory=CoordinateGrid)

    enable_challenger_debug: bool | None = Field(default=False, description="Enable debug mode")

    enable_model_debug_caching: bool = Field(
        default=False, description="Enable caching of model requests and responses to disk"
    )

    # == Skills Configuration == #
    custom_skills_path: Path | None = Field(
        default=None, description="Path to custom skills rules.yaml"
    )
    enable_skills_update: bool = Field(
        default=False, description="Enable auto-update of skills from GitHub"
    )
    skills_update_repo: str = Field(
        default="QIN2DIM/hcaptcha-challenger", description="GitHub repo for skills update"
    )
    skills_update_branch: str = Field(default="main", description="GitHub branch for skills update")

    _last_cache_key: Path | None = PrivateAttr(default=None)

    @field_validator('GEMINI_API_KEY', mode="before")
    @classmethod
    def validate_api_key(cls, v: Any) -> str:
        """
        Validates that the GEMINI_API_KEY is not empty.

        Args:
            v: The API key value to validate

        Returns:
            The validated API key

        Raises:
            ValueError: If the API key is empty
        """
        # Emptiness is checked at the model level — LLM_API_KEY is an
        # accepted substitute when LLM_PROVIDER=openai.
        return v

    @model_validator(mode="after")
    def _require_any_api_key(self) -> "AgentConfig":
        gemini_key = self.GEMINI_API_KEY.get_secret_value()
        llm_key = self.LLM_API_KEY.get_secret_value() if self.LLM_API_KEY else ""
        if not gemini_key and not llm_key:
            raise ValueError(
                "An LLM API key is required. Set GEMINI_API_KEY or LLM_API_KEY. "
                "Create API Key -> https://aistudio.google.com/app/apikey"
            )
        if self.LLM_PROVIDER.lower() == "gemini" and not gemini_key:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_PROVIDER=gemini. "
                "Create API Key -> https://aistudio.google.com/app/apikey"
            )
        return self

    @property
    def spatial_grid_cache(self):
        return self.cache_dir.joinpath("spatial_grid")

    def create_cache_key(
        self,
        captcha_payload: CaptchaPayload | None = None,
        request_type: str = "type",
        prompt: str = "unknown",
    ) -> Path:
        """

        Args:
            captcha_payload:
            request_type:
            prompt:

        Returns: ./.challenge / require_type / prompt / current_time

        """
        current_datetime = datetime.now()
        current_time = current_datetime.strftime("%Y%m%d/%Y%m%d%H%M%S%f")

        prompt = prompt.translate(str.maketrans("", "", "".join(INV)))

        if not captcha_payload:
            _cache_key_temp = self.challenge_dir.joinpath(request_type, prompt, current_time)
            if self.enable_challenger_debug:
                logger.debug(f"Create cache-key [NotCaptchaPayload] - {_cache_key_temp.resolve()}")
            self._last_cache_key = _cache_key_temp
            return _cache_key_temp

        cache_key = self.challenge_dir.joinpath(
            captcha_payload.request_type.value,
            captcha_payload.get_requester_question(),
            current_time,
        )
        self._last_cache_key = cache_key

        try:
            _cache_path_captcha = cache_key.joinpath(f"{cache_key.name}_captcha.json")
            _cache_path_captcha.parent.mkdir(parents=True, exist_ok=True)

            _unpacked_data = captcha_payload.model_dump(mode="json")
            _cache_path_captcha.write_text(
                json.dumps(_unpacked_data, indent=2, ensure_ascii=False), encoding="utf8"
            )
        except Exception as e:
            logger.error(f"Failed to write captcha payload to cache: {e}")

        if self.enable_challenger_debug:
            logger.debug(f"Create cache-key [Direct] - {cache_key.resolve()}")

        return cache_key


class RoboticArm:

    def __init__(self, page: Page, config: AgentConfig):
        self.page = page
        self.config = config
        self._debug = config.enable_challenger_debug

        self._challenge_router = ChallengeRouter(
            gemini_api_key=self._get_api_key(),
            model=self._get_model(self.config.CHALLENGE_CLASSIFIER_MODEL),
            provider=self._create_provider(self._get_model(self.config.CHALLENGE_CLASSIFIER_MODEL)),
        )
        self._image_classifier = ImageClassifier(
            gemini_api_key=self._get_api_key(),
            model=self._get_model(self.config.IMAGE_CLASSIFIER_MODEL),
            provider=self._create_provider(self._get_model(self.config.IMAGE_CLASSIFIER_MODEL)),
        )
        self._spatial_path_reasoner = SpatialPathReasoner(
            gemini_api_key=self._get_api_key(),
            model=self._get_model(self.config.SPATIAL_PATH_REASONER_MODEL),
            provider=self._create_provider(self._get_model(self.config.SPATIAL_PATH_REASONER_MODEL)),
        )
        self._spatial_point_reasoner = SpatialPointReasoner(
            gemini_api_key=self._get_api_key(),
            model=self._get_model(self.config.SPATIAL_POINT_REASONER_MODEL),
            provider=self._create_provider(self._get_model(self.config.SPATIAL_POINT_REASONER_MODEL)),
        )
        self._skill_manager = SkillManager(agent_config=config)
        self.signal_crumb_count: int | None = None
        self.captcha_payload: CaptchaPayload | None = None
        self._challenge_prompt: str | None = None

        self._checkbox_selector = "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=checkbox')]"
        self._challenge_selector = "//iframe[starts-with(@src,'https://newassets.hcaptcha.com/captcha/v1/') and contains(@src, 'frame=challenge')]"

    def _get_api_key(self) -> str:
        """Get the API key based on provider configuration."""
        if self.config.LLM_PROVIDER.lower() == "openai":
            return (self.config.LLM_API_KEY or self.config.GEMINI_API_KEY).get_secret_value()
        return self.config.GEMINI_API_KEY.get_secret_value()

    def _get_model(self, default_model: str) -> str:
        """Get the model name, using override if set."""
        return self.config.LLM_MODEL or default_model

    def _create_provider(self, model: str) -> ChatProvider | None:
        """Create the LLM provider for one reasoner role."""
        provider_type = self.config.LLM_PROVIDER.lower()

        if provider_type == "openai":
            api_key = self._get_api_key()
            if not api_key:
                raise ValueError("API key is required. Set LLM_API_KEY or GEMINI_API_KEY.")
            return OpenAIProvider(
                api_key=api_key,
                model=model,
                base_url=self.config.LLM_BASE_URL,
                enable_logging=self.config.OPENAI_ENABLE_LOGGING,
                request_timeout=self.config.LLM_REQUEST_TIMEOUT,
            )
        elif provider_type == "gemini":
            # Return None to use default Gemini provider created by Reasoner
            return None
        else:
            logger.warning(f"Unknown provider '{provider_type}', falling back to Gemini")
            return None

    @property
    def checkbox_selector(self) -> str:
        return self._checkbox_selector

    @property
    def challenge_selector(self) -> str:
        return self._challenge_selector

    async def get_challenge_frame_locator(self) -> Frame | None:
        candidate_frame = self._find_challenge_frame_recursive(self.page.main_frame, max_depth=4)

        if candidate_frame:
            with suppress(Exception):
                challenge_view = candidate_frame.locator("//div[@class='challenge-view']")
                is_visible = await challenge_view.is_visible(timeout=1000)

                if is_visible:
                    return candidate_frame

        try:
            challenge_frames = []
            all_frames = self.page.frames
            for frame in all_frames:
                if (
                    frame.url.startswith("https://newassets.hcaptcha.com/captcha/v1/")
                    and "frame=challenge" in frame.url
                ):
                    challenge_frames.append(frame)

            for frame in challenge_frames:
                with suppress(Exception):
                    challenge_view = frame.locator("//div[@class='challenge-view']")
                    if await challenge_view.is_visible():
                        return frame
        except Exception as e:
            logger.error(f"Error finding all iframes: {e}")

        logger.error("Cannot find a valid challenge frame")
        return None

    def _find_challenge_frame_recursive(
        self, frame: Frame, current_depth=0, max_depth=4
    ) -> Frame | None:
        if current_depth >= max_depth:
            return None

        candidate_frames = []

        for child_frame in frame.child_frames:
            if (
                not child_frame.child_frames
                and child_frame.url.startswith("https://newassets.hcaptcha.com/captcha/v1/")
                and "frame=challenge" in child_frame.url
            ):
                candidate_frames.append(child_frame)
            else:
                found_in_child = self._find_challenge_frame_recursive(
                    child_frame, current_depth + 1, max_depth
                )
                if found_in_child:
                    return found_in_child

        if candidate_frames:
            return candidate_frames[0]

        return None

    def _match_user_prompt(self, job_type: ChallengeTypeEnum) -> str:
        try:
            challenge_prompt = (
                self.captcha_payload.get_requester_question()
                if self.captcha_payload
                else self._challenge_prompt
            )
            if challenge_prompt and isinstance(challenge_prompt, str):
                return self._skill_manager.get_skill(challenge_prompt, job_type)
        except Exception as e:
            logger.warning(f"Error while processing captcha payload: {e}")

        return f"Please note that the current task type is: {job_type.value}"

    async def click_by_mouse(self, locator: Locator):
        bbox = await locator.bounding_box()
        if bbox is None:
            raise ValueError("Element is not visible or does not exist")

        x: float = bbox['x']
        y: float = bbox['y']
        width: float = bbox['width']
        height: float = bbox['height']

        center_x = x + width / 2
        center_y = y + height / 2

        await self.page.mouse.move(center_x, center_y)

        await self.page.mouse.click(center_x, center_y, delay=150)

    async def click_checkbox(self):
        checkbox_frame = self.page.frame_locator(self.checkbox_selector)
        checkbox_element = checkbox_frame.locator("//div[@id='checkbox']")
        await self.click_by_mouse(checkbox_element)

    async def refresh_challenge(self):
        try:
            refresh_frame = await self.get_challenge_frame_locator()
            refresh_element = refresh_frame.locator("//div[@class='refresh button']")
            await self.click_by_mouse(refresh_element)
        except TimeoutError as err:
            logger.warning(f"Failed to click refresh button - {err=}")

    async def check_crumb_count(self):
        """Page turn in tasks"""
        # Determine the number of tasks based on hsw
        if isinstance(self.signal_crumb_count, int) and self.signal_crumb_count >= 1:
            return self.signal_crumb_count

        # Determine the number of tasks based on DOM
        await self.page.wait_for_timeout(500)
        frame_challenge = await self.get_challenge_frame_locator()
        crumbs = frame_challenge.locator("//div[@class='Crumb']")
        with suppress(Exception):
            crumbs_count = await crumbs.count()
            return crumbs_count if crumbs_count else 1
        return self.config.MAX_CRUMB_COUNT if await crumbs.first.is_visible() else 1

    async def check_challenge_type(self) -> RequestType | ChallengeTypeEnum | None:
        # fixme
        with suppress(Exception):
            await self.page.wait_for_selector(self.challenge_selector, timeout=1000)

        frame_challenge = await self.get_challenge_frame_locator()

        samples = frame_challenge.locator("//div[@class='task-image']")
        count = await samples.count()
        if isinstance(count, int) and count == 9:
            return RequestType.IMAGE_LABEL_BINARY
        if isinstance(count, int) and count == 0:
            tms = self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS * 1.5
            await self.page.wait_for_timeout(tms)
            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            cache_path = self.config.cache_dir.joinpath(f"challenge_view/_artifacts/{uuid4()}.png")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            await challenge_view.screenshot(type="png", path=cache_path)
            router_result = await self._challenge_router(
                challenge_screenshot=cache_path,
                request_debug_path=cache_path.with_name(f"{cache_path.stem}_model_request.json") if self.config.enable_model_debug_caching else None,
            )
            self._challenge_prompt = router_result.challenge_prompt
            return router_result.challenge_type
        return None

    async def _wait_for_all_loaders_complete(self):
        """Wait for all loading indicators to complete (become invisible)"""
        frame_challenge = await self.get_challenge_frame_locator()

        await self.page.wait_for_timeout(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS)

        loading_indicators = frame_challenge.locator("//div[@class='loading-indicator']")
        count = await loading_indicators.count()

        if count == 0:
            logger.info("No load indicator found in the page")
            return True

        for i in range(count):
            loader = loading_indicators.nth(i)
            try:
                await expect(loader).to_have_attribute(
                    "style", re.compile(r"opacity:\s*0"), timeout=30000
                )
                await loading_indicators.nth(i).get_attribute("style")  # It cannot be removed
            except TimeoutError:
                logger.warning(f"The load indicator {i + 1}/{count} waits for a timeout")
            except ValueError:
                # todo requires smarter waiting methods
                await self.page.wait_for_timeout(130)

        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(1),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/2) - Wait 1 second - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def _capture_spatial_mapping(
        self, frame_challenge: FrameLocator | Frame, cache_key: Path, crumb_id: int | str
    ):
        # Extract challenge image directly from browser canvas/img element (more reliable than screenshot)
        challenge_screenshot = cache_key.joinpath(f"{cache_key.name}_{crumb_id}_challenge_view.png")
        challenge_screenshot.parent.mkdir(parents=True, exist_ok=True)

        image_data = await frame_challenge.evaluate("""() => {
            const challengeView = document.querySelector('.challenge-view');
            if (!challengeView) return null;

            // Try canvas first (hCaptcha often renders puzzle as canvas)
            const canvas = challengeView.querySelector('canvas');
            if (canvas) {
                return canvas.toDataURL('image/png').split(',')[1];
            }

            // Try img element
            const img = challengeView.querySelector('img');
            if (img) {
                const tempCanvas = document.createElement('canvas');
                tempCanvas.width = img.naturalWidth || img.width;
                tempCanvas.height = img.naturalHeight || img.height;
                const ctx = tempCanvas.getContext('2d');
                ctx.drawImage(img, 0, 0);
                return tempCanvas.toDataURL('image/png').split(',')[1];
            }

            // Fallback: render challenge-view div contents to canvas
            const rect = challengeView.getBoundingClientRect();
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = rect.width;
            tempCanvas.height = rect.height;
            const ctx = tempCanvas.getContext('2d');

            // Try to capture background image if set
            const style = window.getComputedStyle(challengeView);
            const bgImage = style.backgroundImage;
            if (bgImage && bgImage !== 'none') {
                const bgUrl = bgImage.slice(4, -1).replace(/"/g, "");
                return new Promise((resolve, reject) => {
                    const bgImg = new Image();
                    bgImg.crossOrigin = 'anonymous';
                    bgImg.onload = () => {
                        ctx.drawImage(bgImg, 0, 0, rect.width, rect.height);
                        resolve(tempCanvas.toDataURL('image/png').split(',')[1]);
                    };
                    bgImg.onerror = () => resolve(null);
                    bgImg.src = bgUrl;
                });
            }

            return null;
        }""")

        if image_data:
            challenge_screenshot.write_bytes(base64.b64decode(image_data))
            logger.debug(f"Extracted challenge image from browser canvas: {challenge_screenshot}")
        else:
            # Fallback to Playwright screenshot if canvas extraction fails
            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            await challenge_view.screenshot(type="png", path=challenge_screenshot)
            logger.debug(f"Fell back to Playwright screenshot: {challenge_screenshot}")

        challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
        bbox = await challenge_view.bounding_box()

        # Save grid field
        result = create_coordinate_grid(
            challenge_screenshot,
            bbox,
            x_line_space_num=self.config.coordinate_grid.x_line_space_num,
            y_line_space_num=self.config.coordinate_grid.y_line_space_num,
            color=self.config.coordinate_grid.color,
            adaptive_contrast=self.config.coordinate_grid.adaptive_contrast,
        )

        grid_divisions = cache_key.joinpath(f"{cache_key.name}_{crumb_id}_spatial_helper.png")
        grid_divisions.parent.mkdir(parents=True, exist_ok=True)
        plt.imsave(str(grid_divisions.resolve()), result)

        return challenge_screenshot, grid_divisions

    def _save_drag_debug_overlay(
        self,
        screenshot_path: Path,
        output_path: Path,
        bbox: dict,
        entity_center: Tuple[int, int] | None,
        llm_start: Tuple[int, int] | None,
        llm_end: Tuple[int, int] | None,
        corrected_start: Tuple[int, int] | None,
    ):
        """
        Draw debug markers on a challenge screenshot to visualize coordinate accuracy.
        All coordinates are in webpage (absolute) space.
        """
        try:
            img = Image.open(screenshot_path).convert("RGBA")
            draw = ImageDraw.Draw(img)
            # Convert bbox-relative image coordinates to screenshot-relative
            ox, oy = int(bbox["x"]), int(bbox["y"])

            def rel_to_screenshot(px: int, py: int) -> Tuple[int, int]:
                return (px - ox, py - oy)

            # Entity center (corrected start origin) - green circle
            if entity_center:
                ex, ey = rel_to_screenshot(*entity_center)
                r = 8
                draw.ellipse([ex - r, ey - r, ex + r, ey + r], outline="lime", width=3)
                draw.text((ex + 10, ey - 10), "ENTITY", fill="lime")

            # LLM predicted start - red X
            if llm_start:
                sx, sy = rel_to_screenshot(*llm_start)
                r = 10
                draw.line([(sx - r, sy - r), (sx + r, sy + r)], fill="red", width=3)
                draw.line([(sx + r, sy - r), (sx - r, sy + r)], fill="red", width=3)
                draw.text((sx + 10, sy + 10), "LLM-START", fill="red")

            # Corrected start - green X
            if corrected_start:
                cx, cy = rel_to_screenshot(*corrected_start)
                r = 10
                draw.line([(cx - r, cy - r), (cx + r, cy + r)], fill="lime", width=3)
                draw.line([(cx + r, cy - r), (cx - r, cy + r)], fill="lime", width=3)
                draw.text((cx + 10, cy + 10), "CORRECTED-START", fill="lime")

            # LLM predicted end - blue circle
            if llm_end:
                edx, edy = rel_to_screenshot(*llm_end)
                r = 10
                draw.ellipse([edx - r, edy - r, edx + r, edy + r], outline="cyan", width=3)
                draw.text((edx + 10, edy + 10), "LLM-END", fill="cyan")

            # Connect corrected start to LLM end with a line
            if corrected_start and llm_end:
                cx, cy = rel_to_screenshot(*corrected_start)
                edx, edy = rel_to_screenshot(*llm_end)
                draw.line([(cx, cy), (edx, edy)], fill="yellow", width=2)

            img.save(output_path)
            logger.debug(f"Drag debug overlay saved to: {output_path}")
        except Exception as err:
            logger.warning(f"Failed to save drag debug overlay: {err}")

    async def _perform_drag_drop(self, path: SpatialPath, steps: int = 25, delay_ms: int = 15):
        """
        Performs a human-like drag and drop operation using bezier curve trajectory.

        Args:
            path: The SpatialPath containing start and end coordinates
            steps: Number of intermediate steps for the mouse movement
            delay_ms: Base delay between steps in milliseconds
        """
        start_x, start_y = path.start_point.x, path.start_point.y
        end_x, end_y = path.end_point.x, path.end_point.y

        if self.config.DISABLE_BEZIER_TRAJECTORY:
            await self.page.mouse.move(start_x, start_y)
            await self.page.mouse.down()
            await self.page.mouse.move(end_x, end_y)
            await self.page.mouse.up()
            return

        # Move to the starting position
        await self.page.mouse.move(start_x, start_y)

        # Small random delay before pressing down (human reaction time)
        await asyncio.sleep(random.uniform(0.05, 0.15))

        # Press the mouse button down
        await self.page.mouse.down()

        # Generate a bezier curve path with a control point
        points = _generate_bezier_trajectory((start_x, start_y), (end_x, end_y), steps)

        # Add velocity variation (slow start, fast middle, slow end)
        delays = _generate_dynamic_delays(steps, base_delay=delay_ms)

        # Perform the drag with human-like movement
        for i, ((current_x, current_y), delay) in enumerate(zip(points, delays)):
            # Add slight "noise" to the path (more pronounced near the end)
            if i > steps * 0.7:  # In the last 30% of the movement
                # More micro-adjustments near the end
                noise_factor = 0.5 if i > steps * 0.9 else 0.2
                current_x += random.uniform(-noise_factor, noise_factor)
                current_y += random.uniform(-noise_factor, noise_factor)

            await self.page.mouse.move(current_x, current_y)
            await asyncio.sleep(delay / 1000)

        # Ensure we end exactly at the target position
        await self.page.mouse.move(end_x, end_y)

        # Small pause before releasing (human precision adjustment)
        await asyncio.sleep(random.uniform(0.05, 0.1))

        # Release the mouse button at the destination
        await self.page.mouse.up()

        # Small pause between drag operations
        await asyncio.sleep(random.uniform(0.08, 0.12))

    async def _get_frame_relative_drag_points(
        self, frame_challenge: Frame, path: SpatialPath
    ) -> tuple[float, float, float, float]:
        frame_element = await frame_challenge.frame_element()
        frame_bbox = await frame_element.bounding_box()
        if not frame_bbox:
            raise ValueError("Cannot determine challenge frame bounding box for drag dispatch")

        return (
            path.start_point.x - frame_bbox["x"],
            path.start_point.y - frame_bbox["y"],
            path.end_point.x - frame_bbox["x"],
            path.end_point.y - frame_bbox["y"],
        )

    async def _perform_drag_drop_html5(self, frame_challenge: Frame, path: SpatialPath):
        """
        Performs drag using HTML5 Drag and Drop API via JavaScript event dispatching.
        This is more reliable for web apps using native HTML5 DnD.
        """
        start_x, start_y, end_x, end_y = await self._get_frame_relative_drag_points(
            frame_challenge, path
        )

        # Use JavaScript to dispatch HTML5 DnD events
        await frame_challenge.evaluate(
            """
            ([startX, startY, endX, endY]) => {
                // Find element at start position
                const startEl = document.elementFromPoint(startX, startY);
                const endEl = document.elementFromPoint(endX, endY);

                if (!startEl || !endEl) {
                    console.error('Could not find elements at drag positions');
                    return false;
                }

                // Create and dispatch drag events
                const dragStartEvent = new DragEvent('dragstart', {
                    bubbles: true,
                    cancelable: true,
                    clientX: startX,
                    clientY: startY,
                    dataTransfer: new DataTransfer()
                });
                startEl.dispatchEvent(dragStartEvent);

                const dragOverEvent = new DragEvent('dragover', {
                    bubbles: true,
                    cancelable: true,
                    clientX: endX,
                    clientY: endY,
                    dataTransfer: dragStartEvent.dataTransfer
                });
                endEl.dispatchEvent(dragOverEvent);

                const dropEvent = new DragEvent('drop', {
                    bubbles: true,
                    cancelable: true,
                    clientX: endX,
                    clientY: endY,
                    dataTransfer: dragStartEvent.dataTransfer
                });
                endEl.dispatchEvent(dropEvent);

                const dragEndEvent = new DragEvent('dragend', {
                    bubbles: true,
                    cancelable: true,
                    clientX: endX,
                    clientY: endY,
                    dataTransfer: dragStartEvent.dataTransfer
                });
                startEl.dispatchEvent(dragEndEvent);

                return true;
            }
            """,
            [start_x, start_y, end_x, end_y]
        )
        await asyncio.sleep(0.2)

    def _drag_mechanisms(self) -> List[str]:
        """Drag mechanisms in preference order; the configured one runs first."""
        if self.config.USE_POINTER_EVENTS:
            preferred = "pointer"
        elif self.config.USE_HTML5_DRAG:
            preferred = "html5"
        else:
            preferred = "mouse"
        order = ["mouse", "pointer", "html5"]
        order.remove(preferred)
        return [preferred, *order]

    async def _drag_accepted(self, frame_challenge: Frame) -> bool:
        """hCaptcha keeps showing 'Skip' until a drag actually registers."""
        try:
            submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
            btn_text = await submit_btn.text_content()
            logger.debug(f"Submit button text after drag: '{btn_text}'")
            return not (btn_text and "skip" in btn_text.lower())
        except Exception:
            return True

    @staticmethod
    def _render_paths_preview(image_path, output_path, bbox, paths) -> None:
        """Draw proposed drag paths on the challenge image for model verification."""
        img = Image.open(image_path).convert("RGBA")
        draw = ImageDraw.Draw(img)
        ox, oy = int(bbox["x"]), int(bbox["y"])
        for path in paths:
            sx, sy = int(path.start_point.x) - ox, int(path.start_point.y) - oy
            ex, ey = int(path.end_point.x) - ox, int(path.end_point.y) - oy
            draw.line([(sx, sy), (ex, ey)], fill="yellow", width=3)
            r = 8
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], outline="lime", width=3)
            draw.ellipse([ex - r, ey - r, ex + r, ey + r], outline="blue", width=3)
        img.save(output_path)

    async def _perform_drag_drop_pointer(self, frame_challenge: Frame, path: SpatialPath):
        """
        Performs drag using Pointer Events API (unified mouse/touch/pen events).
        This is the modern alternative to HTML5 DnD for better browser compatibility.
        """
        start_x, start_y, end_x, end_y = await self._get_frame_relative_drag_points(
            frame_challenge, path
        )

        await frame_challenge.evaluate(
            """
            ([startX, startY, endX, endY]) => {
                const startEl = document.elementFromPoint(startX, startY);
                const endEl = document.elementFromPoint(endX, endY);

                if (!startEl || !endEl) {
                    console.error('Could not find elements at drag positions');
                    return false;
                }

                // Pointer events sequence
                const pointerDownEvent = new PointerEvent('pointerdown', {
                    bubbles: true,
                    cancelable: true,
                    pointerId: 1,
                    clientX: startX,
                    clientY: startY,
                    button: 0,
                    buttons: 1
                });
                startEl.dispatchEvent(pointerDownEvent);

                // Simulate pointer move through intermediate points
                const steps = 10;
                for (let i = 1; i <= steps; i++) {
                    const progress = i / steps;
                    const curX = startX + (endX - startX) * progress;
                    const curY = startY + (endY - startY) * progress;

                    const pointerMoveEvent = new PointerEvent('pointermove', {
                        bubbles: true,
                        cancelable: true,
                        pointerId: 1,
                        clientX: curX,
                        clientY: curY,
                        button: 0,
                        buttons: 1
                    });
                    document.dispatchEvent(pointerMoveEvent);
                }

                const pointerUpEvent = new PointerEvent('pointerup', {
                    bubbles: true,
                    cancelable: true,
                    pointerId: 1,
                    clientX: endX,
                    clientY: endY,
                    button: 0,
                    buttons: 0
                });
                document.dispatchEvent(pointerUpEvent);

                return true;
            }
            """,
            [start_x, start_y, end_x, end_y]
        )
        await asyncio.sleep(0.2)

    async def _inspect_hcaptcha_drag_elements(self, frame):
        """Inspect hCaptcha's draggable elements and their event listeners"""
        logger.debug("Inspecting hCaptcha drag elements...")

        inspection_result = await frame.evaluate("""
            () => {
                const results = {
                    draggables: [],
                    eventListeners: {},
                    canvasElements: [],
                    touchAction: []
                };

                // Find all elements with drag-related styles or attributes
                const allElements = document.querySelectorAll('*');
                allElements.forEach(el => {
                    const computedStyle = window.getComputedStyle(el);
                    const hasDrag = computedStyle.cursor === 'grab' ||
                                     computedStyle.cursor === 'move' ||
                                     computedStyle.cursor === 'pointer' ||
                                     el.getAttribute('draggable') === 'true';

                    if (hasDrag) {
                        results.draggables.push({
                            tag: el.tagName,
                            id: el.id,
                            class: el.className,
                            cursor: computedStyle.cursor,
                            draggable: el.getAttribute('draggable'),
                            position: {
                                x: el.offsetLeft,
                                y: el.offsetTop,
                                width: el.offsetWidth,
                                height: el.offsetHeight
                            },
                            dataAttrs: Array.from(el.attributes)
                                .filter(a => a.name.startsWith('data-'))
                                .map(a => `${a.name}=${a.value}`)
                        });
                    }

                    // Check for touch-action
                    if (computedStyle.touchAction !== 'auto') {
                        results.touchAction.push({
                            tag: el.tagName,
                            class: el.className,
                            touchAction: computedStyle.touchAction
                        });
                    }

                    // Check for canvas elements
                    if (el.tagName === 'CANVAS') {
                        results.canvasElements.push({
                            id: el.id,
                            class: el.className,
                            width: el.width,
                            height: el.height
                        });
                    }
                });

                // Try to get event listeners if available (Chrome devtools)
                if (typeof getEventListeners === 'function') {
                    results.draggables.forEach(el => {
                        const domEl = document.querySelector(
                            el.id ? `#${el.id}` :
                            el.class ? `.${el.class.split(' ')[0]}` :
                            el.tag
                        );
                        if (domEl) {
                            results.eventListeners[el.id || el.class] = getEventListeners(domEl);
                        }
                    });
                }

                return results;
            }
        """)

        logger.debug(f"Drag elements inspection: {inspection_result}")
        return inspection_result

    async def challenge_image_label_binary(self):
        frame_challenge = await self.get_challenge_frame_locator()
        crumb_count = await self.check_crumb_count()
        cache_key = self.config.create_cache_key(self.captcha_payload)

        for cid in range(crumb_count):
            await self._wait_for_all_loaders_complete()

            # Get challenge-view
            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            challenge_screenshot = cache_key.joinpath(f"{cache_key.name}_{cid}_challenge_view.png")
            await challenge_view.screenshot(type="png", path=challenge_screenshot)

            # Image classification
            response = await self._image_classifier(
                challenge_screenshot=challenge_screenshot,
                request_debug_path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_request.json") if self.config.enable_model_debug_caching else None,
            )
            boolean_matrix = response.convert_box_to_boolean_matrix()

            logger.debug(f'[{cid+1}/{crumb_count}]ToolInvokeMessage: {response.log_message}')
            if self.config.enable_model_debug_caching:
                self._image_classifier.cache_response(
                    path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_answer.json")
                )

            # drive the browser to work on the challenge
            positive_cases = 0
            xpath_task_image = "//div[@class='task' and contains(@aria-label, '{index}')]"
            for i, should_be_clicked in enumerate(boolean_matrix):
                if should_be_clicked:
                    task_image = frame_challenge.locator(xpath_task_image.format(index=i + 1))
                    await self.click_by_mouse(task_image)
                    positive_cases += 1
                elif positive_cases == 0 and i == len(boolean_matrix) - 1:
                    task_image = frame_challenge.locator(xpath_task_image.format(index=1))
                    await self.click_by_mouse(task_image)

            # {{< Verify >}}
            with suppress(TimeoutError):
                submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                await self.click_by_mouse(submit_btn)

    async def challenge_image_drag_drop(self, job_type: ChallengeTypeEnum):
        frame_challenge = await self.get_challenge_frame_locator()
        crumb_count = await self.check_crumb_count()
        cache_key = self.config.create_cache_key(self.captcha_payload)

        for cid in range(crumb_count):
            await self.page.wait_for_timeout(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS)

            raw, projection = await self._capture_spatial_mapping(frame_challenge, cache_key, cid)

            user_prompt = self._match_user_prompt(job_type)
            extra_images: List[Path] | None = None
            cropped_raw = raw  # Default to uncropped image

            # Get task and check if it's a compare/matching puzzle
            task = None
            is_compare_puzzle = False
            is_tube_puzzle = False
            if (
                self.captcha_payload
                and cid < len(self.captcha_payload.tasklist)
                and (task := self.captcha_payload.tasklist[cid])
                and task.entities
            ):
                # Heuristic: if entities have entity_uri, it's likely a compare/matching puzzle
                is_compare_puzzle = any(ent.entity_uri for ent in task.entities)

                # Detect tube/pipe challenges: entities on the right side are draggable SOURCE pieces
                # Compare puzzles have targets on the left, tube puzzles have sources on the right
                question = self.captcha_payload.get_requester_question()
                if is_compare_puzzle and question:
                    # Tube challenge keywords in the question
                    is_tube_by_question = any(
                        kw in question.lower()
                        for kw in ["pipe", "tube", "reach the other side"]
                    )
                    # All entities in the right-side panel → they are sources, not targets.
                    # Threshold is relative to image width (~right 30%).
                    entity_coords = [
                        ent.coords for ent in task.entities
                        if ent.coords and len(ent.coords) >= 1
                    ]
                    try:
                        img_w = Image.open(raw).size[0]
                    except Exception:
                        img_w = 500
                    # all([]) is True → require a non-empty list, otherwise every compare
                    # puzzle whose entities lack coords would be misclassified as tube
                    is_tube_by_position = bool(entity_coords) and all(
                        coords[0] > img_w * 0.7 for coords in entity_coords
                    )
                    is_tube_puzzle = is_tube_by_question or is_tube_by_position
                    if is_tube_puzzle:
                        is_compare_puzzle = False
                        logger.debug(f"Detected tube/pipe puzzle: question='{question}', entities on right side")

            # Enhance prompt when challenge has target entities with reference icons
            # Only do this for compare/matching puzzles (where entity_uri provides reference icons)
            # For connection/assembly puzzles (like pipe connection), entity images are not needed
            if is_compare_puzzle:
                entity_count = len(task.entities)
                entity_positions = []
                for i, ent in enumerate(task.entities):
                    if ent.coords and len(ent.coords) >= 2:
                        entity_positions.append(
                            f"目标 {i+1}: 位置({ent.coords[0]},{ent.coords[1]}), 显示参考图标"
                        )
                if entity_positions:
                    if entity_count > 1:
                        user_prompt += (
                            f"\n\n注意：共有{entity_count}个目标位置显示参考图标，"
                            f"请返回所有匹配的拖放操作，每个匹配对应一个path。"
                            f"不匹配的物体不要拖动。目标位置：\n"
                        )
                    else:
                        user_prompt += (
                            f"\n\n注意：{entity_positions[0]}，"
                            f"请拖放匹配的物体到该位置。"
                        )
                    user_prompt += "\n".join(entity_positions)
                logger.debug(f"Enhanced drag prompt for {entity_count} target entity(ies)")

                # Crop the image to exclude the right side (draggable elements) to fix LLM source/target swap
                # Only do this for single-entity compare puzzles where we have the entity center to map coordinates
                # For connection/assembly puzzles or multi-entity, keep full image
                cropped_raw = raw
                should_crop = len(task.entities) == 1  # Only crop single-entity challenges
                if should_crop:
                    try:
                        img = Image.open(raw)
                        width, height = img.size
                        # Crop just past the rightmost target edge — entities are the
                        # targets; fall back to the left ~70% when coords are missing.
                        target_edges = [
                            ent.coords[0] + ent.size[0]
                            for ent in task.entities
                            if ent.coords and len(ent.coords) >= 2
                            and ent.size and len(ent.size) >= 1
                        ]
                        crop_x = (
                            min(width, max(target_edges) + 24)
                            if target_edges
                            else int(width * 0.7)
                        )
                        if width > crop_x:
                            cropped_img = img.crop((0, 0, crop_x, height))
                            cropped_raw = cache_key.joinpath(f"{cache_key.name}_{cid}_cropped.png")
                            cropped_img.save(cropped_raw)
                            logger.debug(f"Cropped image from {width}x{height} to {crop_x}x{height}, saved to {cropped_raw}")
                    except Exception as e:
                        logger.warning(f"Failed to crop image: {e}, using original")
                        cropped_raw = raw

                # Extract entity images directly from the browser (already loaded, no HTTP download needed)
                try:
                    extra_images_dir = cache_key.joinpath("entity_images")
                    extra_images_dir.mkdir(exist_ok=True)
                    extra_images = []
                    for i, ent in enumerate(task.entities):
                        if ent.entity_uri:
                            entity_path = extra_images_dir / f"entity_{i}.png"

                            # Extract raw image data from the browser via canvas-based base64 encoding
                            # This avoids URL expiry issues and captures exactly what the user sees
                            image_data = await frame_challenge.evaluate(
                                """(imageUrl) => {
                                    return new Promise((resolve, reject) => {
                                        const img = document.querySelector(`img[src*="${imageUrl}"]`);
                                        if (!img) {
                                            reject(new Error('Image not found in DOM: ' + imageUrl));
                                            return;
                                        }
                                        const canvas = document.createElement('canvas');
                                        canvas.width = img.naturalWidth;
                                        canvas.height = img.naturalHeight;
                                        const ctx = canvas.getContext('2d');
                                        ctx.drawImage(img, 0, 0);
                                        resolve(canvas.toDataURL('image/png').split(',')[1]);
                                    });
                                }""",
                                ent.entity_uri,
                            )
                            entity_path.write_bytes(base64.b64decode(image_data))
                            extra_images.append(entity_path)
                            logger.debug(f"Extracted entity image {i} from browser: {entity_path}")
                except Exception as e:
                    logger.warning(f"Failed to extract entity images from browser: {e}")
                    extra_images = None

            response = await self._spatial_path_reasoner(
                challenge_screenshot=cropped_raw if 'cropped_raw' in locals() else raw,
                grid_divisions=projection,
                auxiliary_information=user_prompt,
                extra_images=extra_images,
                request_debug_path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_request.json") if self.config.enable_model_debug_caching else None,
            )
            logger.debug(f'[{cid+1}/{crumb_count}]ToolInvokeMessage: {response.log_message}')
            if self.config.enable_model_debug_caching:
                self._spatial_path_reasoner.cache_response(
                    path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_answer.json")
                )

            # Get challenge bbox for coordinate translation (needed for all drag types)
            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            bbox = await challenge_view.bounding_box()

            # Snap LLM drag endpoints to known entity geometry (ground truth from the
            # captcha payload). Entity coords are image-relative → convert to webpage.
            # - non-compare puzzles (tube, single-entity): entities are drag SOURCES
            #   → snap start_point to the nearest entity center
            # - compare puzzles (plates/destinations): entities are drop TARGETS
            #   → snap end_point to the nearest entity center
            entity_centers = _entity_centers_webpage(task, bbox)
            if entity_centers:
                logger.debug(f"Entity centers (webpage): {entity_centers}")

            # Single-entity non-compare challenges ("drag the shape to the center"):
            # the sole entity IS the draggable → unconditional start correction.
            corrected_start = (
                entity_centers[0]
                if len(entity_centers) == 1 and not is_compare_puzzle
                else None
            )
            if corrected_start:
                logger.debug(f"Entity correction: webpage_start={corrected_start}")

            snap_radius = self.config.ENTITY_SNAP_RADIUS

            def _snap_path(path: SpatialPath) -> None:
                """Snap a path's endpoints to known entity geometry, in place."""
                llm_start = (path.start_point.x, path.start_point.y)
                llm_end = (path.end_point.x, path.end_point.y)
                snapped_start = corrected_start or (
                    _nearest_entity_center(llm_start, entity_centers, snap_radius)
                    if not is_compare_puzzle
                    else None
                )
                if snapped_start and snapped_start != llm_start:
                    logger.debug(f"Snapped start {llm_start} -> {snapped_start}")
                if snapped_start:
                    path.start_point.x, path.start_point.y = snapped_start
                if is_compare_puzzle:
                    snapped_end = _nearest_entity_center(
                        llm_end, entity_centers, snap_radius
                    )
                    if snapped_end and snapped_end != llm_end:
                        logger.debug(f"Snapped end {llm_end} -> {snapped_end}")
                        path.end_point.x, path.end_point.y = snapped_end

            # Original (pre-snap) endpoints, kept for the debug overlay
            originals = [
                (
                    (int(p.start_point.x), int(p.start_point.y)),
                    (int(p.end_point.x), int(p.end_point.y)),
                )
                for p in (response.paths or [])
            ]
            for path in response.paths or []:
                _snap_path(path)

            # Self-verification pass: render the proposed drags onto the challenge
            # image and let the model confirm or correct them before executing.
            if self.config.VERIFY_DRAG_PATHS and response.paths and bbox:
                try:
                    preview = cache_key.joinpath(f"{cache_key.name}_{cid}_verify_preview.png")
                    self._render_paths_preview(
                        image_path=raw,
                        output_path=preview,
                        bbox=bbox,
                        paths=response.paths,
                    )
                    verified = await self._spatial_path_reasoner(
                        challenge_screenshot=preview,
                        grid_divisions=projection,
                        auxiliary_information=(
                            user_prompt
                            + "\n\n图中已标出当前预测：绿色圆圈=起点，蓝色圆圈=终点，"
                            "黄色连线=拖拽路径。请核对每条路径是否正确完成题目要求；"
                            "若正确请原样返回，若有偏差请返回修正后的 paths。"
                        ),
                        extra_images=extra_images,
                    )
                    if verified and verified.paths:
                        logger.debug(f"Verification pass: {verified.log_message}")
                        response.paths = verified.paths
                        originals = [
                            (
                                (int(p.start_point.x), int(p.start_point.y)),
                                (int(p.end_point.x), int(p.end_point.y)),
                            )
                            for p in response.paths
                        ]
                        for path in response.paths:
                            _snap_path(path)
                except Exception as err:
                    logger.warning(f"Drag verification failed, keeping original: {err}")

            paths_count = len(response.paths) if response.paths else 0
            logger.debug(f"LLM returned {paths_count} drag paths for task {cid+1}/{crumb_count}")

            # Debug overlays (original vs final endpoint per path)
            if bbox:
                for idx, path in enumerate(response.paths):
                    llm_start, llm_end = originals[idx] if idx < len(originals) else (None, None)
                    final_start = (int(path.start_point.x), int(path.start_point.y))
                    entity_center_abs = _nearest_entity_center(
                        llm_end if is_compare_puzzle else llm_start,
                        entity_centers,
                        float("inf"),
                    )
                    self._save_drag_debug_overlay(
                        screenshot_path=raw,
                        output_path=cache_key.joinpath(
                            f"{cache_key.name}_{cid}_drag_debug_overlay_{idx}.png"
                        ),
                        bbox=bbox,
                        entity_center=entity_center_abs,
                        llm_start=llm_start,
                        llm_end=llm_end,
                        corrected_start=final_start if not is_compare_puzzle else None,
                    )

            # Execute drags, falling back to alternate mechanisms when hCaptcha
            # doesn't register them (submit button stays on "Skip")
            drag_accepted = False
            for mech_name in self._drag_mechanisms():
                for idx, path in enumerate(response.paths):
                    logger.debug(f"Executing drag path {idx+1}/{paths_count} [{mech_name}]: {path}")
                    if mech_name == "pointer":
                        await self._perform_drag_drop_pointer(frame_challenge, path)
                    elif mech_name == "html5":
                        await self._perform_drag_drop_html5(frame_challenge, path)
                    else:
                        await self._perform_drag_drop(path)

                    if paths_count > 1 and idx < paths_count - 1:
                        await asyncio.sleep(0.3)

                if not response.paths or await self._drag_accepted(frame_challenge):
                    drag_accepted = bool(response.paths)
                    break
                logger.warning(
                    f"'{mech_name}' drag not registered (button still 'Skip'); "
                    "trying next mechanism"
                )

            # {{< Verify >}}
            with suppress(TimeoutError):
                submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                if drag_accepted:
                    await self.click_by_mouse(submit_btn)
                else:
                    logger.warning("No drag mechanism was accepted; not clicking Skip.")

    async def challenge_image_label_select(self, job_type: ChallengeTypeEnum):
        frame_challenge = await self.get_challenge_frame_locator()
        crumb_count = await self.check_crumb_count()
        cache_key = self.config.create_cache_key(self.captcha_payload)

        for cid in range(crumb_count):
            await self.page.wait_for_timeout(self.config.WAIT_FOR_CHALLENGE_VIEW_TO_RENDER_MS)

            raw, projection = await self._capture_spatial_mapping(frame_challenge, cache_key, cid)

            user_prompt = self._match_user_prompt(job_type)

            response = await self._spatial_point_reasoner(
                challenge_screenshot=raw,
                grid_divisions=projection,
                auxiliary_information=user_prompt,
                request_debug_path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_request.json") if self.config.enable_model_debug_caching else None,
            )
            logger.debug(f'[{cid+1}/{crumb_count}]ToolInvokeMessage: {response.log_message}')
            if self.config.enable_model_debug_caching:
                self._spatial_point_reasoner.cache_response(
                    path=cache_key.joinpath(f"{cache_key.name}_{cid}_model_answer.json")
                )

            # Convert webpage-absolute coordinates (from LLM/grid) to challenge-view-relative
            challenge_view = frame_challenge.locator("//div[@class='challenge-view']")
            bbox = await challenge_view.bounding_box()
            if bbox:
                for point in response.points:
                    rel_x = int(point.x - bbox["x"])
                    rel_y = int(point.y - bbox["y"])
                    await challenge_view.click(position={"x": rel_x, "y": rel_y}, delay=180)
                    await self.page.wait_for_timeout(500)
            else:
                # Fallback to absolute coordinates if bbox unavailable
                for point in response.points:
                    await self.page.mouse.click(point.x, point.y, delay=180)
                    await self.page.wait_for_timeout(500)

            # {{< Verify >}}
            with suppress(TimeoutError):
                submit_btn = frame_challenge.locator("//div[@class='button-submit button']")
                await self.click_by_mouse(submit_btn)


class AgentV:

    def __init__(self, page: Page, agent_config: AgentConfig):
        self.page = page
        self.config = agent_config

        self.robotic_arm = RoboticArm(page=page, config=agent_config)

        self._captcha_payload: CaptchaPayload | None = None
        self._captcha_payload_queue: Queue[CaptchaPayload | None] = Queue()
        self._captcha_response_queue: Queue[CaptchaResponse] = Queue()
        self.cr_list: List[CaptchaResponse] = []
        self._challenge_count: int = 0  # Track challenges for MAX_CHALLENGES limit

        self.page.on("response", self._task_handler)

    def _cache_validated_captcha_response(self, cr: CaptchaResponse):
        if not cr.is_pass:
            return

        self.cr_list.append(cr)

        try:
            captcha_response = cr.model_dump(mode="json", by_alias=True)
            current_time = datetime.now().strftime("%Y%m%d/%Y%m%d%H%M%S%f")
            cache_path = self.config.captcha_response_dir.joinpath(f"{current_time}.json")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            t = json.dumps(captcha_response, indent=2, ensure_ascii=False)
            cache_path.write_text(t, encoding="utf-8")
        except Exception as err:
            logger.error(f"Saving captcha response failed - {err}")

    @logger.catch
    async def _task_handler(self, response: Response):
        if response.url.endswith("/hsw.js"):
            try:
                hsw_text = await response.text()
                await self.page.evaluate(hsw_text)
                await self.page.evaluate(
                    """
                    () => {
                        return typeof hsw === 'function' ? true : 'hsw不是函数';
                    }
                    """
                )
            except Exception as err:
                logger.error(f"An error occurred while injecting hsw script: {err}")
        elif "/getcaptcha/" in response.url:
            self._captcha_payload = None

            # Content-Type: application/json
            if response.headers.get("content-type", "") == "application/json":
                data = await response.json()
                if data.get("pass"):
                    while not self._captcha_response_queue.empty():
                        self._captcha_response_queue.get_nowait()
                    cr = CaptchaResponse(**data)
                    self._captcha_response_queue.put_nowait(cr)
                    return
                if data.get("request_config"):
                    captcha_payload = CaptchaPayload(**data)
                    self._captcha_payload_queue.put_nowait(captcha_payload)
                    return

            # Content-Type: stream
            try:
                raw_data = await response.body()

                # [DEBUG] Force fallback to visual recognition for testing
                if self.config.DISABLE_HSW_REVERSE:
                    logger.warning("HSW reverse disabled by config, fallback to regular processing")
                    self._captcha_payload_queue.put_nowait(None)
                    return

                has_hsw = await self.page.evaluate(
                    """
                    () => {
                        return typeof hsw === 'function' ? true : false;
                    }
                    """
                )

                if has_hsw:
                    result = await self.page.evaluate(
                        f"""
                        async () => {{
                            const byteArray = new Uint8Array({list(raw_data)});
                            console.log('Data has been converted to Uint8Array, length:', byteArray.length);

                            try {{
                                const hswResult = await hsw(0, byteArray);
                                return Array.from(hswResult);
                            }} catch (e) {{
                                return {{error: e.toString()}};
                            }}
                        }}
                        """
                    )

                    if isinstance(result, list) and not any(
                        isinstance(x, dict) and "error" in x for x in result
                    ):
                        unpacked_data = msgpack.unpackb(bytes(result))
                        captcha_payload = CaptchaPayload(**unpacked_data)
                        self._captcha_payload_queue.put_nowait(captcha_payload)

                        return
                # If the reverse fails, fall back to the original process
                else:
                    logger.warning("HSW reverse failed, fallback to regular processing")
                    self._captcha_payload_queue.put_nowait(None)
            except Exception as err:
                logger.error(f"Reverse processing getcaptcha failed: {err}")
                self._captcha_payload_queue.put_nowait(None)
        elif "/checkcaptcha/" in response.url:
            try:
                metadata = await response.json()
                self._captcha_response_queue.put_nowait(CaptchaResponse(**metadata))
            except Exception as err:
                logger.exception(err)

    async def _review_challenge_type(self) -> RequestType | ChallengeTypeEnum:
        try:
            self._captcha_payload = await asyncio.wait_for(
                self._captcha_payload_queue.get(), timeout=30.0
            )
            await self.page.wait_for_timeout(500)
        except asyncio.TimeoutError:
            logger.error("Wait for captcha payload to timeout")
            self._captcha_payload = None

        self.robotic_arm.signal_crumb_count = None
        self.robotic_arm.captcha_payload = None
        if not self._captcha_payload:
            return await self.robotic_arm.check_challenge_type()

        try:
            request_type = self._captcha_payload.request_type
            tasklist = self._captcha_payload.tasklist
            tasklist_length = len(tasklist)
            self.robotic_arm.captcha_payload = self._captcha_payload
            match request_type:
                case RequestType.IMAGE_LABEL_BINARY:
                    self.robotic_arm.signal_crumb_count = int(tasklist_length / 9)
                    return RequestType.IMAGE_LABEL_BINARY
                case RequestType.IMAGE_LABEL_AREA_SELECT:
                    self.robotic_arm.signal_crumb_count = tasklist_length
                    max_shapes = self._captcha_payload.request_config.max_shapes_per_image
                    if not isinstance(max_shapes, int):
                        return await self.robotic_arm.check_challenge_type()
                    return (
                        ChallengeTypeEnum.IMAGE_LABEL_SINGLE_SELECT
                        if max_shapes == 1
                        else ChallengeTypeEnum.IMAGE_LABEL_MULTI_SELECT
                    )
                case RequestType.IMAGE_DRAG_DROP:
                    self.robotic_arm.signal_crumb_count = tasklist_length
                    if not tasklist or len(tasklist) == 0:
                        logger.error("IMAGE_DRAG_DROP: empty tasklist")
                        raise ValueError("Empty tasklist for drag challenge")
                    entities = tasklist[0].entities or []
                    entity_count = len(entities)
                    logger.debug(f"IMAGE_DRAG_DROP: tasklist_len={tasklist_length}, entities={entity_count}")
                    return (
                        ChallengeTypeEnum.IMAGE_DRAG_SINGLE
                        if entity_count == 1
                        else ChallengeTypeEnum.IMAGE_DRAG_MULTI
                    )

            logger.warning(f"Unknown request_type: {request_type=}")
        except Exception as err:
            logger.exception(f"Error parsing challenge type: {err}")

        # Fallback to visual recognition solution
        return await self.robotic_arm.check_challenge_type()

    async def _solve_captcha(self):
        self._challenge_count += 1
        if self._challenge_count > self.config.MAX_CHALLENGES:
            logger.warning(
                f"MAX_CHALLENGES limit ({self.config.MAX_CHALLENGES}) exceeded. "
                "Aborting to protect API credits."
            )
            raise RuntimeError(f"MAX_CHALLENGES limit ({self.config.MAX_CHALLENGES}) reached")

        challenge_type = await self._review_challenge_type()
        logger.debug(
            f"Start Challenge - type={challenge_type.value} count={self.robotic_arm.signal_crumb_count}"
        )

        try:
            # {{< Skip specific challenge questions >}}
            with suppress(Exception):
                if self.config.ignore_request_questions and self._captcha_payload:
                    for q in self.config.ignore_request_questions:
                        if q in self._captcha_payload.get_requester_question():
                            await self.page.wait_for_timeout(2000)
                            await self.robotic_arm.refresh_challenge()
                            return await self._solve_captcha()

            # {{< challenge start >}}
            match challenge_type:
                case RequestType.IMAGE_LABEL_BINARY:
                    if RequestType.IMAGE_LABEL_BINARY not in self.config.ignore_request_types:
                        return await self.robotic_arm.challenge_image_label_binary()
                case challenge_type.IMAGE_LABEL_SINGLE_SELECT:
                    if (
                        RequestType.IMAGE_LABEL_AREA_SELECT not in self.config.ignore_request_types
                        and challenge_type.IMAGE_LABEL_SINGLE_SELECT
                        not in self.config.ignore_request_types
                    ):
                        return await self.robotic_arm.challenge_image_label_select(challenge_type)
                case challenge_type.IMAGE_LABEL_MULTI_SELECT:
                    if (
                        RequestType.IMAGE_LABEL_AREA_SELECT not in self.config.ignore_request_types
                        and challenge_type.IMAGE_LABEL_MULTI_SELECT
                        not in self.config.ignore_request_types
                    ):
                        return await self.robotic_arm.challenge_image_label_select(challenge_type)
                case challenge_type.IMAGE_DRAG_SINGLE:
                    if (
                        RequestType.IMAGE_DRAG_DROP not in self.config.ignore_request_types
                        and ChallengeTypeEnum.IMAGE_DRAG_SINGLE
                        not in self.config.ignore_request_types
                    ):
                        return await self.robotic_arm.challenge_image_drag_drop(challenge_type)
                case challenge_type.IMAGE_DRAG_MULTI:
                    if (
                        RequestType.IMAGE_DRAG_DROP not in self.config.ignore_request_types
                        and ChallengeTypeEnum.IMAGE_DRAG_MULTI
                        not in self.config.ignore_request_types
                    ):
                        return await self.robotic_arm.challenge_image_drag_drop(challenge_type)
                # {{< HCI >}}
                case _:
                    # todo Agentic Workflow | zero-shot challenge
                    logger.warning(f"Unknown types of challenges: {challenge_type}")
            # {{< challenge end >}}

            await self.page.wait_for_timeout(2000)
            await self.robotic_arm.refresh_challenge()
            return await self._solve_captcha()
        except Exception as err:
            # This is an execution error inside the challenge,
            # hcaptcha challenge does not automatically refresh
            logger.exception(f"ChallengeException - type={challenge_type.value} {err=}")
            await self.page.wait_for_timeout(5000)
            await self.robotic_arm.refresh_challenge()
            return await self._solve_captcha()

    def _log_challenge_outcome(self, cr: CaptchaResponse | None, error: str | None = None) -> None:
        """
        Write `_result.json` into the current challenge cache dir, joining cached
        model request/answer artifacts to the real hCaptcha outcome.
        """
        try:
            cache_key = self.config._last_cache_key
            if not cache_key:
                return
            payload = self._captcha_payload
            outcome = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "is_pass": cr.is_pass if cr else None,
                "error": error or (cr.error if cr else "response_timeout"),
                "request_type": (
                    payload.request_type.value if payload and payload.request_type else None
                ),
                "prompt": payload.get_requester_question() if payload else None,
                "models": {
                    "challenge_classifier": self.config.CHALLENGE_CLASSIFIER_MODEL,
                    "image_classifier": self.config.IMAGE_CLASSIFIER_MODEL,
                    "spatial_point": self.config.SPATIAL_POINT_REASONER_MODEL,
                    "spatial_path": self.config.SPATIAL_PATH_REASONER_MODEL,
                },
            }
            cache_key.mkdir(parents=True, exist_ok=True)
            cache_key.joinpath("_result.json").write_text(
                json.dumps(outcome, indent=2, ensure_ascii=False), encoding="utf8"
            )
        except Exception as err:
            logger.warning(f"Failed to write challenge outcome: {err}")

    async def wait_for_challenge(self) -> ChallengeSignal:
        # Assigning human-computer challenge tasks to the main thread coroutine.
        # ----------------------------------------------------------------------
        try:
            if self._captcha_response_queue.empty():
                await asyncio.wait_for(self._solve_captcha(), timeout=self.config.EXECUTION_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("Challenge execution timed out", timeout=self.config.EXECUTION_TIMEOUT)
            self._log_challenge_outcome(None, error="execution_timeout")
            return ChallengeSignal.EXECUTION_TIMEOUT

        # Waiting for hCAPTCHA response processing result
        # -----------------------------------------------
        # After the completion of the human-machine challenge workflow,
        # it is expected to obtain a signal indicating whether the challenge was successful in the cr_queue.
        logger.debug("Start checking captcha response")
        try:
            cr = await asyncio.wait_for(
                self._captcha_response_queue.get(), timeout=self.config.RESPONSE_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.error(f"Wait for captcha response timeout {self.config.RESPONSE_TIMEOUT}s")

            # Debug dump: capture page state to investigate why no response was received
            try:
                debug_ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                debug_dir = self.config.challenge_dir.joinpath("_timeout_debug", debug_ts)
                debug_dir.mkdir(parents=True, exist_ok=True)

                # Save full page screenshot
                screenshot_path = debug_dir.joinpath("page_screenshot.png")
                await self.page.screenshot(path=str(screenshot_path), full_page=False)

                # Save page HTML
                html_path = debug_dir.joinpath("page_content.html")
                html_content = await self.page.content()
                html_path.write_text(html_content, encoding="utf-8")

                # Save challenge frame HTML if available
                try:
                    frame_challenge = await self.get_challenge_frame_locator()
                    frame_html_path = debug_dir.joinpath("challenge_frame.html")
                    frame_html = await frame_challenge.page.content()
                    frame_html_path.write_text(frame_html, encoding="utf-8")
                except Exception:
                    pass

                logger.debug(f"Timeout debug state saved to: {debug_dir}")
            except Exception as dump_err:
                logger.warning(f"Failed to save timeout debug state: {dump_err}")

            # Treat timeout-as-no-response as failure and retry if enabled
            # (hCaptcha may still be showing a challenge that wasn't solved)
            self._log_challenge_outcome(None, error="response_timeout")
            if self.config.RETRY_ON_FAILURE:
                logger.warning("Challenge response timeout, treating as failure and retrying")
                await self.page.wait_for_timeout(2000)
                return await self.wait_for_challenge()
            return ChallengeSignal.EXECUTION_TIMEOUT
        else:
            self._log_challenge_outcome(cr)
            # Match: Timeout / Loss
            if not cr or not cr.is_pass:
                if self.config.RETRY_ON_FAILURE:
                    logger.warning("Failed to challenge, try to retry the strategy")
                    await self.page.wait_for_timeout(2000)
                    return await self.wait_for_challenge()
                return ChallengeSignal.FAILURE
            # Match: Success
            if cr.is_pass:
                logger.success("Challenge success")
                self._cache_validated_captcha_response(cr)
                return ChallengeSignal.SUCCESS

        return ChallengeSignal.FAILURE
