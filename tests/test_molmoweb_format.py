"""Wire-format tests for the MolmoWeb adapter. No GPU, no network, no server.

The generations quoted here are REAL: they were derived from the public
``allenai/MolmoWeb-SyntheticTrajs`` training data by tracing rows through the
reference serializer. The training target string *is* the generation format, so
these are the strings the model is trained to emit. See docs/molmoweb_format.md.

If one of these tests fails, the format spec and the adapter have diverged --
which is exactly the failure we want to catch here rather than mid-run.
"""

from __future__ import annotations

import pytest

from multi_agent_web.actions import Click, Done, Navigate, Scroll, Type
from multi_agent_web.browser import PageInfo
from multi_agent_web.policy.molmoweb import (
    MolmoWebProtocolError,
    build_user_message,
    parse_generation,
)

VIEWPORT = (1280, 720)

# --- verbatim training targets --------------------------------------------
CLICK_GEN = (
    '{"thought": "The goal is to find information about Apple AirTag\'s pricing, '
    'availability, and bulk options. Since AirTag is an accessory, I will navigate '
    'to the Accessories section of the Apple Store to locate it.", '
    '"action": {"name": "click", "x": 71.1, "y": 3.1, "button": "left", '
    '"click_type": "single"}}'
)
TYPE_GEN = (
    '{"thought": "I will type \'AirTag\' into the search field to find its pricing '
    'and availability.", "action": {"name": "keyboard_type", "text": "AirTag"}}'
)
ANSWER_GEN = (
    '{"thought": "The pricing for the Apple AirTag is $29.00 for a 1-pack.", '
    '"action": {"name": "send_msg_to_user", "msg": "[ANSWER] The Apple AirTag is '
    'priced at $29.00 for a single pack and $99.00 for a 4-pack."}}'
)
EXIT_GEN = (
    '{"thought": "The goal is now fully satisfied.", '
    '"action": {"name": "send_msg_to_user", "msg": "[EXIT]"}}'
)
GOTO_GEN = '{"thought": "", "action": {"name": "goto", "url": "https://247sports.com"}}'
SCROLL_GEN = '{"thought": "", "action": {"name": "scroll", "delta_x": 0.0, "delta_y": 100.0}}'
# A click whose element was below the fold: normalize clipped y to the int 100.
CLIPPED_GEN = (
    '{"thought": "", "action": {"name": "click", "x": 14.1, "y": 100, '
    '"button": "left", "click_type": "single"}}'
)


class TestParseRealGenerations:
    def test_click_converts_percent_to_pixels(self) -> None:
        thought, action, raw_dict = parse_generation(CLICK_GEN, VIEWPORT)
        assert isinstance(action, Click)
        # 71.1% of 1280 = 910.08 -> 910.1 ; 3.1% of 720 = 22.32 -> 22.3
        assert action.x == pytest.approx(910.1)
        assert action.y == pytest.approx(22.3)
        assert thought.startswith("The goal is to find information")
        # The raw dict is kept in the model's own percent space, untouched.
        assert raw_dict == {
            "name": "click",
            "x": 71.1,
            "y": 3.1,
            "button": "left",
            "click_type": "single",
        }

    def test_keyboard_type_never_submits(self) -> None:
        _, action, _ = parse_generation(TYPE_GEN, VIEWPORT)
        assert isinstance(action, Type)
        assert action.text == "AirTag"
        # MolmoWeb has no press_enter; Enter arrives as a separate keyboard_press.
        assert action.press_enter is False

    def test_answer_sentinel_strips_prefix(self) -> None:
        _, action, _ = parse_generation(ANSWER_GEN, VIEWPORT)
        assert isinstance(action, Done)
        assert action.sentinel == "[ANSWER]"
        assert action.answer.startswith("The Apple AirTag is priced at $29.00")
        assert "[ANSWER]" not in action.answer

    def test_exit_sentinel_yields_empty_answer(self) -> None:
        _, action, _ = parse_generation(EXIT_GEN, VIEWPORT)
        assert isinstance(action, Done)
        assert action.sentinel == "[EXIT]"
        assert action.answer == ""

    def test_goto(self) -> None:
        thought, action, _ = parse_generation(GOTO_GEN, VIEWPORT)
        assert isinstance(action, Navigate)
        assert action.url == "https://247sports.com"
        assert thought == ""

    def test_scroll_delta_100_is_one_viewport(self) -> None:
        _, action, _ = parse_generation(SCROLL_GEN, VIEWPORT)
        assert isinstance(action, Scroll)
        assert action.delta_x == pytest.approx(0.0)
        assert action.delta_y == pytest.approx(720.0)

    def test_clipped_int_coordinate_is_accepted_and_clamped(self) -> None:
        """Clipped values serialize as int 100, and 100% would land off-viewport."""
        _, action, _ = parse_generation(CLIPPED_GEN, VIEWPORT)
        assert isinstance(action, Click)
        # 100% of 720 = 720.0, clamped to dim-2 so the click stays inside.
        assert action.y == pytest.approx(718.0)
        assert action.x == pytest.approx(180.5)

    def test_base_style_bare_action_has_no_thought(self) -> None:
        thought, action, _ = parse_generation(
            '{"name": "goto", "url": "https://example.com"}', VIEWPORT
        )
        assert isinstance(action, Navigate)
        assert thought == ""

    def test_percentages_use_actual_screenshot_size_not_a_constant(self) -> None:
        _, action, _ = parse_generation(CLICK_GEN, (800, 600))
        assert isinstance(action, Click)
        assert action.x == pytest.approx(568.8)  # 71.1% of 800
        assert action.y == pytest.approx(18.6)  # 3.1% of 600


class TestParserFailsLoudly:
    """Every failure must name the problem and carry the raw generation."""

    def test_unknown_action_name(self) -> None:
        raw = '{"thought": "t", "action": {"name": "teleport", "x": 1}}'
        with pytest.raises(MolmoWebProtocolError) as exc:
            parse_generation(raw, VIEWPORT)
        assert "teleport" in str(exc.value)
        assert raw in str(exc.value)

    def test_unexpected_key_is_not_ignored(self) -> None:
        raw = (
            '{"thought": "t", "action": {"name": "click", "x": 1.0, "y": 2.0, '
            '"selector": "#login"}}'
        )
        with pytest.raises(MolmoWebProtocolError) as exc:
            parse_generation(raw, VIEWPORT)
        assert "selector" in str(exc.value)

    def test_missing_required_key(self) -> None:
        with pytest.raises(MolmoWebProtocolError, match="missing required key 'y'"):
            parse_generation('{"action": {"name": "click", "x": 1.0}}', VIEWPORT)

    def test_invalid_json(self) -> None:
        with pytest.raises(MolmoWebProtocolError, match="not valid JSON"):
            parse_generation("click at 50, 50 please", VIEWPORT)

    def test_server_error_string(self) -> None:
        with pytest.raises(MolmoWebProtocolError, match="model server reported"):
            parse_generation("Predictor error: All predictors are busy", VIEWPORT)

    def test_empty_generation(self) -> None:
        with pytest.raises(MolmoWebProtocolError, match="empty generation"):
            parse_generation("   ", VIEWPORT)

    def test_message_without_sentinel_is_an_error_not_a_silent_done(self) -> None:
        raw = '{"action": {"name": "send_msg_to_user", "msg": "working on it"}}'
        with pytest.raises(MolmoWebProtocolError, match=r"\[ANSWER\] or \[EXIT\]"):
            parse_generation(raw, VIEWPORT)

    @pytest.mark.parametrize(
        "name", ["hover_at", "scroll_at", "browser_nav", "report_infeasible", "dblclick"]
    )
    def test_unmodelled_actions_raise_by_name(self, name: str) -> None:
        raw = '{"action": {"name": "%s"}}' % name
        with pytest.raises(NotImplementedError) as exc:
            parse_generation(raw, VIEWPORT)
        assert name in str(exc.value)

    def test_non_left_button_is_not_silently_downgraded(self) -> None:
        raw = (
            '{"action": {"name": "click", "x": 1.0, "y": 2.0, "button": "right", '
            '"click_type": "single"}}'
        )
        with pytest.raises(NotImplementedError, match="right"):
            parse_generation(raw, VIEWPORT)


class TestPromptFormat:
    """The template's whitespace is load-bearing -- it is what training used."""

    def test_matches_reference_template_byte_for_byte(self) -> None:
        past = [
            {
                "index": 1,
                "thought": "I will start by navigating to the Apple homepage.",
                "action": {"name": "goto", "url": "https://www.apple.com/"},
            },
            {
                "index": 2,
                "thought": "I will navigate to the Accessories section.",
                "action": {
                    "name": "click",
                    "x": 71.1,
                    "y": 3.1,
                    "button": "left",
                    "click_type": "single",
                },
            },
        ]
        got = build_user_message(
            "Check the pricing of Apple AirTag.",
            past,
            PageInfo(url="https://www.apple.com/shop", title="Buy AirTag - Apple"),
        )
        expected = (
            "\n# GOAL\nCheck the pricing of Apple AirTag.\n"
            "\n# PREVIOUS STEPS\n"
            "## Step 1\n"
            "THOUGHT: I will start by navigating to the Apple homepage.\n"
            "ACTION: {'name': 'goto', 'url': 'https://www.apple.com/'}\n"
            "## Step 2\n"
            "THOUGHT: I will navigate to the Accessories section.\n"
            "ACTION: {'name': 'click', 'x': 71.1, 'y': 3.1, 'button': 'left', "
            "'click_type': 'single'}\n"
            "\n# CURRENTLY ACTIVE PAGE\n"
            "Page 0: Buy AirTag - Apple | https://www.apple.com/shop\n"
            "\n# NEXT STEP\n"
        )
        assert got == expected

    def test_history_uses_python_repr_not_json(self) -> None:
        """Single quotes. Training rendered dicts with str(); do not 'fix' it."""
        got = build_user_message(
            "t",
            [{"index": 1, "thought": "th", "action": {"name": "goto", "url": "u"}}],
            PageInfo(url="u", title="t"),
        )
        assert "ACTION: {'name': 'goto', 'url': 'u'}" in got
        assert '{"name": "goto"' not in got

    def test_empty_history_still_emits_the_section(self) -> None:
        got = build_user_message("t", [], PageInfo(url="u", title="ti"))
        assert "# PREVIOUS STEPS\n\n# CURRENTLY ACTIVE PAGE" in got

    def test_long_url_and_title_are_truncated_to_100(self) -> None:
        long_url = "https://example.com/" + "x" * 300
        got = build_user_message("t", [], PageInfo(url=long_url, title="y" * 300))
        line = next(ln for ln in got.splitlines() if ln.startswith("Page 0:"))
        title, _, url = line[len("Page 0: "):].partition(" | ")
        assert len(title) == 100 and title.endswith("... (truncated)")
        assert len(url) == 100 and url.endswith("... (truncated)")
