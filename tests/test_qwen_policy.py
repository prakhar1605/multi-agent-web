"""Qwen adapter tests. No network, no key, no browser.

The malformed replies quoted here are REAL -- they are what the model actually
produced when the adapter asked for absolute pixel coordinates, captured by the
grounding gate. They are kept as regression cases: if someone reverts the
coordinate convention, these show exactly how it fails.
"""

from __future__ import annotations

import pytest

from multi_agent_web.actions import Click, Done, KeyPress, Navigate, Scroll, Type
from multi_agent_web.browser import PageInfo
from multi_agent_web.config import QwenConfig
from multi_agent_web.policy.qwen import (
    CallBudget,
    CallBudgetExceeded,
    QwenProtocolError,
    build_system_prompt,
    build_user_message,
    parse_generation,
    strip_code_fence,
)

VIEWPORT = (1280, 720)


class TestCoordinateConversion:
    """0-1000 normalized -> viewport pixels, against the screenshot's own size."""

    def test_click_point_converts(self) -> None:
        raw = '{"thought": "t", "action": {"type": "click", "point": [500, 500]}}'
        thought, action = parse_generation(raw, VIEWPORT)
        assert isinstance(action, Click)
        assert action.x == pytest.approx(640.0)
        assert action.y == pytest.approx(360.0)
        assert thought == "t"

    def test_real_grounding_values_land_on_target(self) -> None:
        """The values the model gave for 'Documentation', whose rect is
        x=60..228, y=16..52. The centre is (144, 34)."""
        raw = '{"thought": "t", "action": {"type": "click", "point": [112, 47]}}'
        _, action = parse_generation(raw, VIEWPORT)
        assert isinstance(action, Click)
        assert 60 <= action.x <= 228
        assert 16 <= action.y <= 52

    def test_uses_actual_screenshot_size_not_a_constant(self) -> None:
        raw = '{"thought": "t", "action": {"type": "click", "point": [500, 500]}}'
        _, action = parse_generation(raw, (800, 600))
        assert isinstance(action, Click)
        assert action.x == pytest.approx(400.0)
        assert action.y == pytest.approx(300.0)

    def test_edge_prediction_is_clamped_inside_the_viewport(self) -> None:
        raw = '{"thought": "t", "action": {"type": "click", "point": [1000, 1000]}}'
        _, action = parse_generation(raw, VIEWPORT)
        assert isinstance(action, Click)
        assert action.x == pytest.approx(1279.0)
        assert action.y == pytest.approx(719.0)

    def test_scroll_delta_1000_is_one_viewport_and_is_not_clamped(self) -> None:
        raw = '{"thought": "t", "action": {"type": "scroll", "delta": [0, 2000]}}'
        _, action = parse_generation(raw, VIEWPORT)
        assert isinstance(action, Scroll)
        assert action.delta_x == pytest.approx(0.0)
        # Two full viewports: distances are legitimately unbounded.
        assert action.delta_y == pytest.approx(1440.0)

    def test_negative_scroll_goes_up(self) -> None:
        raw = '{"thought": "t", "action": {"type": "scroll", "delta": [0, -500]}}'
        _, action = parse_generation(raw, VIEWPORT)
        assert isinstance(action, Scroll)
        assert action.delta_y == pytest.approx(-360.0)


class TestNonSpatialActionsPassThrough:
    """Actions with no coordinates are validated by our own ACTION_ADAPTER."""

    def test_done(self) -> None:
        raw = '{"thought": "t", "action": {"type": "done", "answer": "42"}}'
        _, action = parse_generation(raw, VIEWPORT)
        assert isinstance(action, Done)
        assert action.answer == "42"

    def test_type_with_enter(self) -> None:
        raw = (
            '{"thought": "t", "action": {"type": "type", "text": "hello", '
            '"press_enter": true}}'
        )
        _, action = parse_generation(raw, VIEWPORT)
        assert isinstance(action, Type)
        assert action.text == "hello" and action.press_enter is True

    def test_navigate_and_key_press(self) -> None:
        _, nav = parse_generation(
            '{"thought": "t", "action": {"type": "navigate", '
            '"url": "https://example.com"}}',
            VIEWPORT,
        )
        assert isinstance(nav, Navigate)
        _, key = parse_generation(
            '{"thought": "t", "action": {"type": "key_press", "key": "Enter"}}',
            VIEWPORT,
        )
        assert isinstance(key, KeyPress)


class TestCodeFenceTolerance:
    """General models fence JSON regardless of instructions. Strip, then log."""

    def test_json_fence_is_stripped(self) -> None:
        raw = (
            '```json\n{"thought": "t", "action": {"type": "click", '
            '"point": [500, 500]}}\n```'
        )
        _, action = parse_generation(raw, VIEWPORT)
        assert isinstance(action, Click)

    def test_bare_fence_is_stripped(self) -> None:
        raw = '```\n{"thought": "t", "action": {"type": "done", "answer": "x"}}\n```'
        _, action = parse_generation(raw, VIEWPORT)
        assert isinstance(action, Done)

    def test_strip_reports_whether_it_fired(self) -> None:
        assert strip_code_fence('```json\n{"a": 1}\n```') == ('{"a": 1}', True)
        assert strip_code_fence('{"a": 1}') == ('{"a": 1}', False)

    def test_fence_logged_at_info(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level("INFO"):
            parse_generation(
                '```json\n{"thought": "t", "action": {"type": "done", "answer": ""}}\n```',
                VIEWPORT,
            )
        assert any("code fence" in r.message for r in caplog.records)


class TestFailsLoudly:
    """Every failure names the problem and carries the raw generation."""

    def test_pixel_era_regression_x_as_a_pair(self) -> None:
        """Real output from when the prompt asked for absolute pixels."""
        raw = '{"thought": "t", "action": {"type": "click", "x": [112, 47]}}'
        with pytest.raises(QwenProtocolError) as exc:
            parse_generation(raw, VIEWPORT)
        assert "unexpected key" in str(exc.value)
        assert raw in str(exc.value)

    def test_pixel_era_regression_broken_json(self) -> None:
        """Real output: the model broke JSON to emit a pair."""
        with pytest.raises(QwenProtocolError, match="not valid JSON"):
            parse_generation('{"type": "click", "x": 904, 45}', VIEWPORT)

    def test_pixel_era_regression_hoisted_key(self) -> None:
        raw = '{"thought": "t", "action": "click", "x": [501, 456]}'
        with pytest.raises(QwenProtocolError, match="unexpected top-level key"):
            parse_generation(raw, VIEWPORT)

    def test_point_must_be_a_two_element_array(self) -> None:
        for bad in ("[1]", "[1, 2, 3]", '"500,500"', "500", '["a", "b"]'):
            raw = '{"thought": "t", "action": {"type": "click", "point": %s}}' % bad
            with pytest.raises(QwenProtocolError, match="two-element array"):
                parse_generation(raw, VIEWPORT)

    def test_missing_point(self) -> None:
        with pytest.raises(QwenProtocolError, match="missing required key 'point'"):
            parse_generation('{"thought": "t", "action": {"type": "click"}}', VIEWPORT)

    def test_unknown_action_type(self) -> None:
        raw = '{"thought": "t", "action": {"type": "teleport", "point": [1, 2]}}'
        with pytest.raises(QwenProtocolError) as exc:
            parse_generation(raw, VIEWPORT)
        assert raw in str(exc.value)

    def test_unexpected_key_on_a_passthrough_action(self) -> None:
        raw = (
            '{"thought": "t", "action": {"type": "done", "answer": "x", '
            '"confidence": 0.9}}'
        )
        with pytest.raises(QwenProtocolError):
            parse_generation(raw, VIEWPORT)

    def test_empty_reply(self) -> None:
        with pytest.raises(QwenProtocolError, match="empty reply"):
            parse_generation("   ", VIEWPORT)

    def test_prose_instead_of_json(self) -> None:
        with pytest.raises(QwenProtocolError, match="not valid JSON"):
            parse_generation("I'll click the Documentation button.", VIEWPORT)

    def test_no_action_key(self) -> None:
        with pytest.raises(QwenProtocolError, match="no 'action' key"):
            parse_generation('{"thought": "t"}', VIEWPORT)


class TestPrompt:
    def test_system_prompt_states_the_normalized_convention(self) -> None:
        prompt = build_system_prompt(1280, 720)
        assert "0-1000" in prompt
        assert '"point": [x, y]' in prompt
        # Must not imply pixels -- that is what caused the original failure.
        assert "1280x720 pixels" not in prompt

    def test_user_message_has_no_history_on_the_first_step(self) -> None:
        msg = build_user_message("find it", [], PageInfo(url="u", title="t"))
        assert "# GOAL\nfind it" in msg
        assert "(none yet" in msg

    def test_history_renders_thought_action_and_errors(self) -> None:
        history = [
            {
                "index": 1,
                "thought": "clicking search",
                "action": {"type": "click", "x": 10.0, "y": 20.0},
                "error": "click (10, 20) is outside the viewport",
            }
        ]
        msg = build_user_message("t", history, PageInfo(url="u", title="ti"))
        assert "## Step 1" in msg
        assert "THOUGHT: clicking search" in msg
        assert "RESULT: click (10, 20) is outside the viewport" in msg


class TestCallBudget:
    def test_reserve_raises_when_exhausted(self) -> None:
        budget = CallBudget(limit=2)
        budget.reserve()
        budget.reserve()
        with pytest.raises(CallBudgetExceeded, match="2/2 calls"):
            budget.reserve()

    def test_tracks_calls_per_agent(self) -> None:
        budget = CallBudget(limit=10)
        budget.reserve(0)
        budget.reserve(1)
        budget.reserve(0)
        assert budget.by_agent == {0: 2, 1: 1}
        assert budget.calls == 3

    def test_accumulates_usage_including_reasoning_and_image_tokens(self) -> None:
        budget = CallBudget(limit=10)
        budget.record_usage(
            {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "completion_tokens_details": {"reasoning_tokens": 40},
                "prompt_tokens_details": {"image_tokens": 79},
            }
        )
        budget.record_usage({"prompt_tokens": 10, "total_tokens": 10})
        assert budget.prompt_tokens == 110
        assert budget.total_tokens == 160
        assert budget.reasoning_tokens == 40
        assert budget.image_tokens == 79

    def test_missing_usage_is_tolerated(self) -> None:
        budget = CallBudget(limit=1)
        budget.record_usage(None)
        budget.record_usage({})
        assert budget.total_tokens == 0

    def test_ledger_is_safe_to_serialize(self) -> None:
        budget = CallBudget(limit=5)
        budget.reserve(0)
        payload = budget.as_dict()
        assert set(payload) == {
            "calls", "limit", "retries", "prompt_tokens", "completion_tokens",
            "reasoning_tokens", "image_tokens", "total_tokens", "calls_by_agent",
        }


class TestCredentialSafety:
    def test_api_key_is_not_exposed_by_repr_or_dump(self) -> None:
        config = QwenConfig(api_key="sk-super-secret-value", base_url="https://x")
        assert "sk-super-secret-value" not in repr(config)
        assert "sk-super-secret-value" not in str(config)
        assert "sk-super-secret-value" not in str(config.model_dump())
        assert "sk-super-secret-value" not in config.model_dump_json()
        # Reaching the real value takes an explicit, greppable call.
        assert config.api_key.get_secret_value() == "sk-super-secret-value"

    def test_chat_url_requires_v1(self) -> None:
        config = QwenConfig(api_key="k", base_url="https://host")
        assert config.chat_url == "https://host/v1/chat/completions"
