from tradingagents.scheduler.action_mask import compute_action_mask
from tradingagents.scheduler.actions import SchedulerAction


def _state(**updates):
    state = {
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
    state.update(updates)
    return state


def test_initial_mask_only_contains_selected_analysts():
    mask = compute_action_mask(_state(), ("market", "news"))

    assert mask.valid_actions == (SchedulerAction.MARKET, SchedulerAction.NEWS)


def test_research_and_trade_prerequisites_are_enforced():
    with_report = _state(market_report="bullish")
    mask = compute_action_mask(with_report, ("market",))
    assert SchedulerAction.BULL in mask.valid_actions
    assert SchedulerAction.BEAR in mask.valid_actions
    assert SchedulerAction.RESEARCH_MANAGER not in mask.valid_actions
    assert SchedulerAction.TRADER not in mask.valid_actions

    with_debate = _state(
        market_report="bullish",
        investment_debate_state={"history": "Bull: case", "count": 1},
    )
    mask = compute_action_mask(with_debate, ("market",))
    assert SchedulerAction.RESEARCH_MANAGER in mask.valid_actions


def test_risk_and_stop_prerequisites_are_enforced():
    state = _state(
        market_report="done",
        investment_debate_state={"history": "debate", "count": 2},
        investment_plan="plan",
        trader_investment_plan="Buy",
        risk_debate_state={"history": "risk", "count": 1},
    )
    mask = compute_action_mask(state, ("market",))

    assert SchedulerAction.AGGRESSIVE in mask.valid_actions
    assert SchedulerAction.PORTFOLIO_MANAGER in mask.valid_actions
    assert SchedulerAction.STOP not in mask.valid_actions

    state["final_trade_decision"] = "Overweight"
    assert compute_action_mask(state, ("market",)).valid_actions == (
        SchedulerAction.STOP,
    )


def test_max_steps_and_no_progress_produce_safe_mask():
    exhausted = compute_action_mask(_state(), step=16, max_steps=16)
    assert exhausted.valid_actions == ()
    assert exhausted.reason == "scheduler_max_steps_exceeded"

    repeated = compute_action_mask(
        _state(),
        ("market", "news"),
        last_action=SchedulerAction.MARKET,
        no_progress_count=1,
    )
    assert repeated.valid_actions == (SchedulerAction.NEWS,)
