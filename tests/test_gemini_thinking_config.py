from google.genai import types

from hcaptcha_challenger.tools.internal.providers.gemini import GeminiProvider


def test_gemini_provider_disables_thought_output():
    provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
    config = types.GenerateContentConfig()

    provider._set_thinking_config(config)

    assert config.thinking_config is not None
    assert config.thinking_config.include_thoughts is False
