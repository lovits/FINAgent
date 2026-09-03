import pytest

from tradingagents.scheduler.actions import (
    ACTION_BY_NODE,
    NODE_BY_ACTION,
    SchedulerAction,
    node_for_action,
    parse_action,
)


def test_action_vocabulary_has_thirteen_unique_tokens() -> None:
    assert len(SchedulerAction) == 13
    assert len({action.value for action in SchedulerAction}) == 13


def test_non_terminal_actions_map_to_unique_original_nodes() -> None:
    assert len(NODE_BY_ACTION) == 12
    assert len(ACTION_BY_NODE) == 12
    assert ACTION_BY_NODE["Sentiment Analyst"] is SchedulerAction.SENTIMENT


def test_parse_action_normalizes_strings() -> None:
    assert parse_action("  <ACT_NEWS>\n") is SchedulerAction.NEWS
    with pytest.raises(ValueError, match="unknown scheduler action"):
        parse_action("<ACT_TOOL>")


def test_stop_does_not_map_to_expert_node() -> None:
    with pytest.raises(ValueError, match="STOP"):
        node_for_action(SchedulerAction.STOP)
