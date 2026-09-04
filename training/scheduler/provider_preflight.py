"""One-decision OpenRouter preflight that never prints the API key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.contracts import SchedulerContext
from tradingagents.scheduler.prompt import build_scheduler_input
from tradingagents.scheduler.teacher_policy import (
    OpenRouterTeacherGateway,
    TeacherGatewayError,
    TeacherSchedulerPolicy,
)


def run_preflight(model: str) -> dict[str, object]:
    state = {
        "company_of_interest": "AAPL",
        "asset_type": "stock",
        "instrument_context": "Apple Inc. (AAPL)",
        "trade_date": "2026-08-28",
        "past_context": "",
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
    valid = (SchedulerAction.MARKET, SchedulerAction.NEWS)
    serialized = build_scheduler_input(
        task_id="teacher-provider-preflight",
        state=state,
        valid_actions=valid,
        selected_analysts=("market", "news"),
    )
    context = SchedulerContext(
        "teacher-provider-preflight",
        state,
        serialized,
        valid,
        ("market", "news"),
    )
    policy = TeacherSchedulerPolicy(OpenRouterTeacherGateway(model=model))
    try:
        decision = policy.select_action(context)
    except TeacherGatewayError as exc:
        return {"status": "failed", "model": model, "error": str(exc)}
    return {
        "status": "passed",
        "model": model,
        "action": decision.action.value,
        "decision_attempts": decision.decision_attempts,
    }


def main() -> None:
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="z-ai/glm-5.3-flash")
    args = parser.parse_args()
    result = run_preflight(args.model)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
