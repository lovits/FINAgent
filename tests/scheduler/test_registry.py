import pytest

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.registry import AGENT_REGISTRY, registry_for_analysts


def test_registry_covers_every_non_terminal_action() -> None:
    expected = set(SchedulerAction) - {SchedulerAction.STOP}
    assert set(AGENT_REGISTRY) == expected
    assert len({spec.node_name for spec in AGENT_REGISTRY.values()}) == 12


def test_registry_filters_only_optional_analysts() -> None:
    specs = registry_for_analysts(("market", "news"))
    actions = {spec.action for spec in specs}

    assert SchedulerAction.MARKET in actions
    assert SchedulerAction.NEWS in actions
    assert SchedulerAction.SENTIMENT not in actions
    assert SchedulerAction.FUNDAMENTALS not in actions
    assert SchedulerAction.TRADER in actions
    assert SchedulerAction.PORTFOLIO_MANAGER in actions


def test_prompt_dict_uses_string_action_token() -> None:
    spec = AGENT_REGISTRY[SchedulerAction.NEWS]
    assert spec.to_prompt_dict()["action"] == "<ACT_NEWS>"
    assert spec.to_prompt_dict()["tool_policy_owner"] == "expert_agent"
    assert spec.to_prompt_dict()["internal_tools"] == (
        "get_news",
        "get_global_news",
        "get_insider_transactions",
        "get_macro_indicators",
        "get_prediction_markets",
    )


def test_researchers_require_all_task_selected_reports() -> None:
    for action in (SchedulerAction.BULL, SchedulerAction.BEAR):
        assert AGENT_REGISTRY[action].prerequisites == (
            "all task-selected analyst reports are complete",
        )


@pytest.mark.parametrize("selected", [(), ("unknown",), ("market", "market")])
def test_registry_rejects_invalid_analyst_selection(selected: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        registry_for_analysts(selected)
