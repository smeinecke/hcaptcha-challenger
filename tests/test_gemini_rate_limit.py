import pytest

from hcaptcha_challenger.tools.internal.providers.gemini import GeminiProvider


@pytest.mark.asyncio
async def test_gemini_provider_waits_for_rate_limit_slot(monkeypatch):
    provider = GeminiProvider(api_key="test-key", model="gemini-2.5-flash")
    provider._max_requests_per_minute = 5

    GeminiProvider._next_request_after = 110.0
    GeminiProvider._rate_limit_lock = None

    slept = []
    now_values = iter([100.0, 110.0])

    def fake_monotonic():
        try:
            return next(now_values)
        except StopIteration:
            return 110.0

    monkeypatch.setattr(
        "hcaptcha_challenger.tools.internal.providers.gemini.time.monotonic",
        fake_monotonic,
    )

    async def fake_sleep(seconds: float):
        slept.append(seconds)

    monkeypatch.setattr(
        "hcaptcha_challenger.tools.internal.providers.gemini.asyncio.sleep",
        fake_sleep,
    )

    await provider._wait_for_rate_limit_slot()

    assert slept == [10.0]
    assert GeminiProvider._next_request_after == pytest.approx(122.0)
