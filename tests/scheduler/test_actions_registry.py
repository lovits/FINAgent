import pytest

from tradingagents.scheduler.actions import (
    ACTION_BY_NODE,
    NODE_BY_ACTION,
    SchedulerAction,
    node_for_action,
    parse_action,
)
from tradingagents.scheduler.agent_registry import AGENT_REGISTRY, registry_for_analysts


def test_action_tokens_and_nodes_are_unique():
    assert len({action.value for action in SchedulerAction}) == len(SchedulerAction)
    assert len(NODE_BY_ACTION) == 12
    assert len(ACTION_BY_NODE) == 12
    assert set(AGENT_REGISTRY) == set(NODE_BY_ACTION)


def test_registry_filters_unselected_analysts_but_keeps_fixed_experts():
    registry = registry_for_analysts(("market", "news"))

    assert SchedulerAction.MARKET in registry
    assert SchedulerAction.NEWS in registry
    assert SchedulerAction.SENTIMENT not in registry
    assert SchedulerAction.FUNDAMENTALS not in registry
    assert SchedulerAction.TRADER in registry
    assert SchedulerAction.PORTFOLIO_MANAGER in registry


def test_parse_action_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown scheduler action"):
        parse_action("<ACT_DOES_NOT_EXIST>")


def test_stop_has_no_expert_node():
    with pytest.raises(ValueError, match="STOP does not map"):
        node_for_action(SchedulerAction.STOP)
