# -*- coding: utf-8 -*-
"""
GeminiProvider - Google Gemini API implementation.

This provider wraps the google-genai SDK to provide image-based content generation.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import ClassVar, List, Type, TypeVar, cast

from google import genai
from google.genai import types
from loguru import logger
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_fixed

from hcaptcha_challenger.models import THINKING_LEVEL_MODELS

ResponseT = TypeVar("ResponseT", bound=BaseModel)


def extract_first_json_block(text: str) -> dict | None:
    """Extract the first JSON code block from text."""
    import re

    pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return json.loads(matches[0])
    return None


class GeminiProvider:
    """
    Gemini-based chat provider implementation.

    This class encapsulates all Gemini-specific logic, making it easy to
    swap out for other providers in the future.
    """

    _rate_limit_lock: ClassVar[asyncio.Lock | None] = None
    _next_request_after: ClassVar[float] = 0.0

    def __init__(self, api_key: str, model: str):
        """
        Initialize the Gemini provider.

        Args:
            api_key: Gemini API key.
            model: Model name to use (e.g., "gemini-2.5-pro").
        """
        self._api_key = api_key
        self._model = model
        self._client: genai.Client | None = None
        self._response: types.GenerateContentResponse | None = None
        self._last_request_payload: dict | None = None
        self._max_requests_per_minute = max(
            1, int(os.getenv("HCAPTCHA_GEMINI_MAX_REQUESTS_PER_MINUTE", "5"))
        )

    @classmethod
    def _get_rate_limit_lock(cls) -> asyncio.Lock:
        if cls._rate_limit_lock is None:
            cls._rate_limit_lock = asyncio.Lock()
        return cls._rate_limit_lock

    async def _wait_for_rate_limit_slot(self) -> None:
        """
        Throttle generate_content calls to avoid free-tier quota bursts.

        Gemini free-tier projects are often limited to 5 requests/minute for
        specific models, so we serialize requests and keep a minimum interval.
        """
        min_interval = 60 / self._max_requests_per_minute

        async with self._get_rate_limit_lock():
            now = time.monotonic()
            wait_seconds = self._next_request_after - now

            if wait_seconds > 0:
                logger.debug(
                    "Gemini rate limit active - wait {:.2f}s before next request", wait_seconds
                )
                await asyncio.sleep(wait_seconds)
                now = time.monotonic()

            self.__class__._next_request_after = now + min_interval

    @property
    def client(self) -> genai.Client:
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    @property
    def last_response(self) -> types.GenerateContentResponse | None:
        """Get the last response for debugging/caching purposes."""
        return self._response

    @property
    def last_request_payload(self) -> dict | None:
        """Get the last prepared request payload for debugging/caching purposes."""
        return self._last_request_payload

    async def _upload_files(self, files: List[Path]) -> list[types.File]:
        """Upload multiple files concurrently."""
        valid_files = [f for f in files if f and Path(f).exists()]
        if not valid_files:
            return []
        upload_tasks = [self.client.aio.files.upload(file=f) for f in valid_files]
        return list(await asyncio.gather(*upload_tasks))

    @staticmethod
    def _files_to_parts(files: List[types.File]) -> List[types.Part]:
        """Convert uploaded files to parts."""
        return [types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type) for f in files]

    def _set_thinking_config(self, config: types.GenerateContentConfig) -> None:
        """Configure thinking settings based on model capabilities."""
        # Do not request thoughts in solve flows: they add latency/cost without
        # improving our structured output parsing.
        config.thinking_config = types.ThinkingConfig(include_thoughts=False)

        if self._model in THINKING_LEVEL_MODELS:
            thinking_level = types.ThinkingLevel.HIGH

            config.thinking_config = types.ThinkingConfig(
                include_thoughts=False, thinking_level=thinking_level
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - "
            f"Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def generate_with_images(
        self,
        *,
        images: List[Path],
        response_schema: Type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        request_debug_path: Path | None = None,
        **kwargs,
    ) -> ResponseT:
        """
        Generate content with image inputs.

        Args:
            images: List of image file paths to include in the request.
            user_prompt: User-provided prompt/instructions.
            description: System instruction/description for the model.
            response_schema: Pydantic model class for structured output.
            **kwargs: Additional options passed to the API.

        Returns:
            Parsed response matching the response_schema type.
        """
        # Upload files
        uploaded_files = await self._upload_files(images)
        parts = self._files_to_parts(uploaded_files)

        # Add user prompt if provided
        if user_prompt and isinstance(user_prompt, str):
            parts.append(types.Part.from_text(text=user_prompt))

        contents = [types.Content(role="user", parts=parts)]

        self._last_request_payload = {
            "provider": "gemini",
            "model": self._model,
            "prepared_at": time.time(),
            "image_paths": [str(Path(image).resolve()) for image in images],
            "user_prompt": user_prompt,
            "system_instruction": description,
            "response_mime_type": "application/json",
            "response_schema_name": response_schema.__name__,
            "response_schema_json": response_schema.model_json_schema(),
        }

        if request_debug_path is not None:
            self.cache_request(request_debug_path)

        # Build config
        config = types.GenerateContentConfig(
            system_instruction=description,
            media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        # Set thinking config if applicable
        self._set_thinking_config(config=config)

        # Respect provider-side request pacing before calling Gemini.
        await self._wait_for_rate_limit_slot()

        # Generate response
        self._response: types.GenerateContentResponse = (
            await self.client.aio.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        )

        # Parse response
        if self._response.parsed:
            parsed = self._response.parsed
            if isinstance(parsed, BaseModel):
                return response_schema(**parsed.model_dump())
            if isinstance(parsed, dict):
                return response_schema(**cast(dict[str, object], parsed))

        # Fallback to JSON extraction
        if response_text := self._response.text:
            json_data = extract_first_json_block(response_text)
            if json_data:
                return response_schema(**json_data)

        raise ValueError(f"Failed to parse response: {response_text}")

    def cache_response(self, path: Path) -> None:
        """Cache the last response to a file."""
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._response.model_dump(mode="json"), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to cache response: {e}")

    def cache_request(self, path: Path) -> None:
        """Cache the last prepared request payload to a file."""
        if not self._last_request_payload:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self._last_request_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to cache request: {e}")
