# -*- coding: utf-8 -*-
"""Offline unit tests for drag-drop accuracy helpers (no API calls)."""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError

from hcaptcha_challenger.agent.challenger import (
    AgentV,
    RoboticArm,
    _entity_centers_webpage,
    _nearest_entity_center,
)
from hcaptcha_challenger.tools.internal.providers.openai import _is_retryable


class _Entity:
    def __init__(self, coords, size):
        self.coords = coords
        self.size = size


class _Task:
    def __init__(self, entities):
        self.entities = entities


CENTERS = [(470, 100), (470, 191), (470, 282)]


class TestNearestEntityCenter:
    def test_snaps_within_radius(self):
        # Model returned piece corner instead of center → snaps to nearest entity
        assert _nearest_entity_center((436, 250), CENTERS, 80) == (470, 282)

    def test_exact_hit(self):
        assert _nearest_entity_center((470, 191), CENTERS, 80) == (470, 191)

    def test_beyond_radius_returns_none(self):
        assert _nearest_entity_center((150, 150), CENTERS, 80) is None

    def test_empty_inputs(self):
        assert _nearest_entity_center(None, CENTERS, 80) is None
        assert _nearest_entity_center((470, 282), [], 80) is None

    def test_nearest_not_first(self):
        # Closer to a non-first center
        assert _nearest_entity_center((460, 95), CENTERS, 80) == (470, 100)


class TestEntityCentersWebpage:
    BBOX = {"x": 50.0, "y": 100.0}

    def test_top_left_to_center_conversion(self):
        task = _Task([_Entity([436, 74], [64, 52])])
        assert _entity_centers_webpage(task, self.BBOX) == [(518, 200)]

    def test_multiple_entities(self):
        task = _Task(
            [_Entity([436, 74], [64, 52]), _Entity([436, 256], [64, 52])]
        )
        assert _entity_centers_webpage(task, self.BBOX) == [(518, 200), (518, 382)]

    def test_missing_geometry_skipped(self):
        task = _Task([_Entity(None, [64, 52]), _Entity([436, 74], None)])
        assert _entity_centers_webpage(task, self.BBOX) == []

    def test_no_bbox_or_task(self):
        task = _Task([_Entity([436, 74], [64, 52])])
        assert _entity_centers_webpage(task, None) == []
        assert _entity_centers_webpage(None, self.BBOX) == []


def _status_err(code: int) -> APIStatusError:
    req = httpx.Request("POST", "https://example.test/v1/chat")
    return APIStatusError("err", response=httpx.Response(code, request=req), body=None)


class TestRetryPredicate:
    @pytest.mark.parametrize("code", [429, 500, 502, 503])
    def test_retryable_status(self, code):
        assert _is_retryable(_status_err(code)) is True

    @pytest.mark.parametrize("code", [400, 401, 402, 403, 404, 410])
    def test_permanent_status_not_retried(self, code):
        # 402 = model not enabled, 403 = pool unavailable, 410 = deprecated
        assert _is_retryable(_status_err(code)) is False

    def test_connection_and_timeout(self):
        req = httpx.Request("POST", "https://example.test")
        assert _is_retryable(APIConnectionError(request=req)) is True
        assert _is_retryable(APITimeoutError(req)) is True
        assert _is_retryable(TimeoutError()) is True

    def test_transient_output_errors(self):
        assert _is_retryable(ValueError("Failed to parse JSON content")) is True
        assert _is_retryable(ValueError("Empty response")) is True

    def test_input_errors_not_retried(self):
        assert _is_retryable(ValueError("No valid images provided")) is False
        assert _is_retryable(KeyError("x")) is False


class TestDragMechanismOrder:
    def _arm(self, pointer=False, html5=False):
        arm = object.__new__(RoboticArm)
        arm.config = SimpleNamespace(USE_POINTER_EVENTS=pointer, USE_HTML5_DRAG=html5)
        return arm

    def test_default_mouse_first(self):
        assert self._arm()._drag_mechanisms() == ["mouse", "pointer", "html5"]

    def test_configured_mechanism_first(self):
        assert self._arm(pointer=True)._drag_mechanisms() == ["pointer", "mouse", "html5"]
        assert self._arm(html5=True)._drag_mechanisms() == ["html5", "mouse", "pointer"]

    def test_all_mechanisms_present(self):
        for arm in (self._arm(), self._arm(pointer=True), self._arm(html5=True)):
            assert sorted(arm._drag_mechanisms()) == ["html5", "mouse", "pointer"]


class TestChallengeOutcomeLogging:
    def _agent(self, cache_key):
        agent = object.__new__(AgentV)
        agent.config = SimpleNamespace(
            _last_cache_key=cache_key,
            CHALLENGE_CLASSIFIER_MODEL="m1",
            IMAGE_CLASSIFIER_MODEL="m2",
            SPATIAL_POINT_REASONER_MODEL="m3",
            SPATIAL_PATH_REASONER_MODEL="m4",
        )
        agent._captcha_payload = None
        return agent

    def test_pass_result_written(self, tmp_path: Path):
        ck = tmp_path / "challenge"
        self._agent(ck)._log_challenge_outcome(SimpleNamespace(is_pass=True, error=""))
        data = json.loads((ck / "_result.json").read_text())
        assert data["is_pass"] is True
        assert data["models"]["spatial_path"] == "m4"
        assert "ts" in data

    def test_timeout_result_written(self, tmp_path: Path):
        ck = tmp_path / "challenge"
        self._agent(ck)._log_challenge_outcome(None, error="response_timeout")
        data = json.loads((ck / "_result.json").read_text())
        assert data["is_pass"] is None
        assert data["error"] == "response_timeout"

    def test_no_cache_key_is_noop(self):
        self._agent(None)._log_challenge_outcome(None)  # must not raise
