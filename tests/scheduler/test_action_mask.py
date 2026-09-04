from tradingagents.scheduler.action_mask import compute_action_mask
from tradingagents.scheduler.actions import SchedulerAction


def _state() -> dict:
    return {
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {"history": "", "count": 0},
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {"history": "", "count": 0},
        "final_trade_decision": "",
    }


def test_initial_mask_contains_only_selected_missing_analysts() -> None:
    mask = compute_action_mask(_state(), ("market", "news"))
    assert mask.valid_actions == (SchedulerAction.MARKET, SchedulerAction.NEWS)


def test_analysis_unlocks_researchers_without_forcing_fixed_order() -> None:
    state = _state()
    state["news_report"] = "complete news evidence"
    actions = compute_action_mask(state).valid_actions

    assert SchedulerAction.MARKET in actions
    assert SchedulerAction.NEWS not in actions
    assert SchedulerAction.BULL in actions
    assert SchedulerAction.BEAR in actions


def test_debate_unlocks_research_manager() -> None:
    state = _state()
    state["market_report"] = "market evidence"
    state["investment_debate_state"] = {"history": "Bull: evidence", "count": 1}
    actions = compute_action_mask(state).valid_actions
    assert SchedulerAction.RESEARCH_MANAGER in actions


def test_shallow_intent_does_not_hard_code_debate_or_risk_turn_counts() -> None:
    state = _state()
    state["market_report"] = "market evidence"
    state["investment_debate_state"] = {"history": "long debate", "count": 99}
    debate_actions = compute_action_mask(state).valid_actions

    assert SchedulerAction.BULL in debate_actions
    assert SchedulerAction.BEAR in debate_actions
    assert SchedulerAction.RESEARCH_MANAGER in debate_actions

    state["investment_plan"] = "Hold with risk controls"
    state["trader_investment_plan"] = "FINAL TRANSACTION PROPOSAL: HOLD"
    state["risk_debate_state"] = {"history": "long risk debate", "count": 99}
    risk_actions = compute_action_mask(state).valid_actions

    assert SchedulerAction.AGGRESSIVE in risk_actions
    assert SchedulerAction.CONSERVATIVE in risk_actions
    assert SchedulerAction.NEUTRAL in risk_actions
    assert SchedulerAction.PORTFOLIO_MANAGER in risk_actions


def test_plan_trader_and_risk_dependencies() -> None:
    state = _state()
    state["investment_plan"] = "Hold with risk controls"
    assert compute_action_mask(state).valid_actions == (SchedulerAction.TRADER,)

    state["trader_investment_plan"] = "FINAL TRANSACTION PROPOSAL: HOLD"
    actions = compute_action_mask(state).valid_actions
    assert actions == (
        SchedulerAction.AGGRESSIVE,
        SchedulerAction.CONSERVATIVE,
        SchedulerAction.NEUTRAL,
    )

    state["risk_debate_state"] = {"history": "Aggressive: upside", "count": 1}
    assert SchedulerAction.PORTFOLIO_MANAGER in compute_action_mask(state).valid_actions


def test_final_decision_only_allows_stop() -> None:
    state = _state()
    state["final_trade_decision"] = "**Rating**: Hold"
    assert compute_action_mask(state).valid_actions == (SchedulerAction.STOP,)


def test_budget_exhaustion_has_explicit_reason() -> None:
    mask = compute_action_mask(_state(), step=16, max_steps=16)
    assert mask.valid_actions == ()
    assert mask.reason == "scheduler_max_steps_exceeded"


def test_no_progress_masks_immediate_repeat() -> None:
    mask = compute_action_mask(
        _state(),
        ("market", "news"),
        last_action=SchedulerAction.MARKET,
        no_progress_count=1,
    )
    assert mask.valid_actions == (SchedulerAction.NEWS,)
