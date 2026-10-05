"""
OpenAIProvider - OpenAI-compatible API implementation.

This provider works with any OpenAI-compatible API including:
- OpenAI (api.openai.com)
- OpenRouter (openrouter.ai)
- Together AI
- Local LLMs (Ollama, llama-cpp-python, etc.)
- Any other OpenAI-compatible service

Requires: pip install openai
"""

import base64
import json
import re
from pathlib import Path
from typing import TypeVar

from loguru import logger
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed

ResponseT = TypeVar("ResponseT", bound=BaseModel)


def _is_retryable(exc: BaseException) -> bool:
    """
    Decide whether a failed request is worth retrying.

    Retry: rate limits (429), server errors (>=500), connection/timeout errors,
    and transient model-output issues (unparseable/empty JSON).
    Do NOT retry: auth/permission errors (401-404, 410) — they never succeed.
    """
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    if isinstance(exc, (APIConnectionError, APITimeoutError, TimeoutError)):
        return True
    if isinstance(exc, ValueError):
        # Transient model-output issues (bad/empty JSON, schema mismatch);
        # input validation errors (e.g. "No valid images provided") fail fast.
        msg = str(exc).lower()
        return "json" in msg or "empty response" in msg or "validation error" in msg
    return False


def extract_first_json_block(text: str) -> dict | None:
    """Extract the first JSON code block from text."""
    pattern = r"```json\s*([\s\S]*?)```"
    matches = re.findall(pattern, text)
    if matches:
        return json.loads(matches[0])
    return None


def encode_image_to_base64(image_path: Path) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_image_mime_type(image_path: Path) -> str:
    """Determine MIME type from file extension."""
    ext = image_path.suffix.lower()
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return mime_types.get(ext, "image/jpeg")


class OpenAIProvider:
    """
    OpenAI-compatible chat provider implementation.

    This class encapsulates all OpenAI-specific logic and works with any
    OpenAI-compatible API endpoint.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        enable_logging: bool = False,
        request_timeout: float = 90.0,
    ):
        """
        Initialize the OpenAI provider.

        Args:
            api_key: API key for the service.
            model: Model name to use (e.g., "gpt-4o", "anthropic/claude-3.5-sonnet").
            base_url: Optional base URL for OpenAI-compatible APIs.
                     If not provided, uses default OpenAI endpoint.
            enable_logging: If True, logs full request details with base64 truncated.
            request_timeout: Per-request timeout in seconds (default 90).
        """
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._enable_logging = enable_logging
        self._request_timeout = request_timeout
        self._client: AsyncOpenAI | None = None
        self._response = None

    @property
    def client(self) -> AsyncOpenAI:
        """Lazy-initialize the OpenAI client."""
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client

    def _build_image_content(self, images: list[Path]) -> list[dict]:
        """Build image content blocks for the API."""
        content = []
        for image_path in images:
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            base64_image = encode_image_to_base64(image_path)
            mime_type = get_image_mime_type(image_path)

            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{base64_image}",
                        "detail": "high",
                    },
                }
            )
        return content

    def _truncate_base64(self, data: str, max_chars: int = 100) -> str:
        """Truncate base64 data for logging."""
        if len(data) <= max_chars:
            return data
        return f"{data[:max_chars]}... (truncated, {len(data)} chars total)"

    def _log_request(
        self,
        messages: list[dict],
        response_schema: dict,
        temperature: float,
    ) -> None:
        """Log the full request with base64 images truncated."""
        log_data = {
            "model": self._model,
            "base_url": self._base_url,
            "messages": [],
            "response_format": response_schema,
            "temperature": temperature,
        }

        for msg in messages:
            msg_copy = {"role": msg["role"]}
            if isinstance(msg["content"], str):
                msg_copy["content"] = msg["content"]
            elif isinstance(msg["content"], list):
                msg_copy["content"] = []
                for item in msg["content"]:
                    if item.get("type") == "text":
                        msg_copy["content"].append(item)
                    elif item.get("type") == "image_url":
                        url = item["image_url"]["url"]
                        if url.startswith("data:"):
                            # Truncate base64 data
                            mime, b64 = url.split(",", 1)
                            truncated_b64 = self._truncate_base64(b64)
                            msg_copy["content"].append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"{mime},{truncated_b64}", "detail": item["image_url"].get("detail", "high")},
                                }
                            )
                        else:
                            msg_copy["content"].append(item)
            log_data["messages"].append(msg_copy)

        logger.info(f"OpenAI API Request:\n{json.dumps(log_data, indent=2, ensure_ascii=False)}")

    def _log_response(self, response) -> None:
        """Log the API response for debugging."""
        log_data = {
            "id": response.id,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "finish_reason": choice.finish_reason,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                        "refusal": choice.message.refusal,
                    },
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            }
            if response.usage
            else None,
        }
        logger.info(f"OpenAI API Response:\n{json.dumps(log_data, indent=2, ensure_ascii=False)}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        retry=retry_if_exception(_is_retryable),
        before_sleep=lambda retry_state: logger.warning(
            f"Retry request ({retry_state.attempt_number}/3) - "
            f"Wait 3 seconds - Exception: {retry_state.outcome.exception()}"
        ),
    )
    async def generate_with_images(
        self,
        *,
        images: list[Path],
        response_schema: type[ResponseT],
        user_prompt: str | None = None,
        description: str | None = None,
        **kwargs,
    ) -> ResponseT:
        """
        Generate content with image inputs.

        Args:
            images: List of image file paths to include in the request.
            response_schema: Pydantic model class for structured output.
            user_prompt: User-provided prompt/instructions.
            description: System instruction/description for the model.
            **kwargs: Additional options (temperature, etc.).

        Returns:
            Parsed response matching the response_schema type.
        """
        # Build image content
        image_content = self._build_image_content(images)
        if not image_content:
            raise ValueError("No valid images provided")

        # Build message content
        content = image_content
        if user_prompt:
            content.insert(0, {"type": "text", "text": user_prompt})

        messages = [
            {
                "role": "system",
                "content": description or "You are a helpful assistant.",
            },
            {"role": "user", "content": content},
        ]

        # Get schema for structured output
        schema = response_schema.model_json_schema()

        # Log request if enabled
        if self._enable_logging:
            self._log_request(
                messages=messages,
                response_schema={"name": response_schema.__name__, "schema": schema},
                temperature=kwargs.get("temperature", 0.1),
            )

        # Make the API call
        response = await self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": response_schema.__name__,
                    "schema": schema,
                },
            },
            temperature=kwargs.get("temperature", 0.1),
            timeout=self._request_timeout,
        )

        self._response = response

        # Log response if enabled
        if self._enable_logging:
            self._log_response(response)

        # Parse the response
        if response.choices and response.choices[0].message.content:
            content_text = response.choices[0].message.content

            # Try direct JSON parse first
            try:
                json_data = json.loads(content_text)
                return response_schema(**json_data)
            except json.JSONDecodeError:
                # Fallback: extract JSON from markdown code blocks
                json_data = extract_first_json_block(content_text)
                if json_data:
                    return response_schema(**json_data)
                raise ValueError("Failed to parse JSON response: content was not valid JSON or markdown-wrapped JSON")

        raise ValueError("Empty response from API")

    def cache_response(self, path: Path) -> None:
        """Cache the last response to a file."""
        if not self._response:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            cache_data = {
                "model": self._response.model,
                "choices": [
                    {
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content,
                        }
                    }
                    for choice in self._response.choices
                ],
                "usage": self._response.usage.model_dump()
                if self._response.usage
                else None,
            }
            path.write_text(
                json.dumps(cache_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to cache response: {e}")
