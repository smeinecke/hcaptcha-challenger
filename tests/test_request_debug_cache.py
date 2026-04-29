import json
from pathlib import Path

from hcaptcha_challenger.tools.internal.providers.gemini import GeminiProvider


def test_gemini_provider_can_cache_prepared_request(tmp_path: Path):
    provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
    provider._last_request_payload = {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "image_paths": ["/tmp/example.png"],
        "user_prompt": "user",
        "system_instruction": "system",
        "response_schema_name": "ExampleSchema",
    }

    path = tmp_path / "request.json"
    provider.cache_request(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["model"] == "gemini-2.5-flash"
    assert data["user_prompt"] == "user"
    assert data["system_instruction"] == "system"
