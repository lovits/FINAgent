"""Canonical prompt construction shared by Teacher and learned policies."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .actions import SchedulerAction, parse_action
from .contracts import SchedulerContext
from .registry import AGENT_CATALOG_VERSION, registry_for_analysts

PROMPT_VERSION = "scheduler-prompt-v3"
TEACHER_PROMPT_VERSION = "teacher-scheduler-v3"
STATE_SCHEMA_VERSION = "scheduler-state-v1"
COMPLETION_CONTRACT_VERSION = "completion-v1"
ORCHESTRATION_PROFILE_VERSION = "multi-analyst-shallow-v1"

_BUSINESS_STATE_FIELDS = (
    "company_of_interest",
    "asset_type",
    "instrument_context",
    "trade_date",
    "past_context",
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "investment_debate_state",
    "investment_plan",
    "trader_investment_plan",
    "risk_debate_state",
    "final_trade_decision",
)

_SYSTEM_PROMPT = """You are the central Agent Scheduler for TradingAgents.
Your only job is to choose exactly one next Expert Agent action.
The selected Expert Agent executes before you receive the next state.
Choose only from VALID_ACTIONS. Do not call tools, write financial analysis,
produce a trading decision, or output a future route. Choose STOP only when
the completion contract is satisfied. Return one JSON object and no other text."""

_SHALLOW_DYNAMIC_OBJECTIVE = (
    "Execute every analyst selected by the task input exactly once, choosing their "
    "order dynamically, and never call an unselected analyst. After all selected "
    "analyst reports are complete, follow a shallow route: avoid redundant debate "
    "and move promptly through synthesis, trading, risk review, and final decision."
)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def business_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Select complete business fields while excluding raw LangChain messages."""

    return {field: state.get(field) for field in _BUSINESS_STATE_FIELDS}


def completion_contract(max_steps: int) -> dict[str, object]:
    return {
        "max_steps": max_steps,
        "stop_requires": ["final_trade_decision"],
        "research_manager_requires": ["investment_debate_state.history"],
        "trader_requires": ["investment_plan"],
        "risk_agents_require": ["trader_investment_plan"],
        "portfolio_manager_requires": ["risk_debate_state.history"],
        "one_action_per_decision": True,
        "tools_owned_by": "expert_agent",
    }


def agent_catalog(selected_analysts: tuple[str, ...]) -> list[dict[str, object]]:
    return [spec.to_prompt_dict() for spec in registry_for_analysts(selected_analysts)]


def orchestration_profile(
    selected_analysts: tuple[str, ...],
) -> dict[str, object]:
    return {
        "profile_id": ORCHESTRATION_PROFILE_VERSION,
        "selected_analysts": list(selected_analysts),
        "selected_analyst_count": len(selected_analysts),
        "selected_analysts_are_required": True,
        "research_style": "shallow",
        "routing_policy": "dynamic",
        "objective": _SHALLOW_DYNAMIC_OBJECTIVE,
    }


def build_scheduler_input(
    *,
    task_id: str,
    state: Mapping[str, Any],
    valid_actions: Sequence[SchedulerAction | str],
    selected_analysts: tuple[str, ...],
    history: Sequence[SchedulerAction | str] = (),
    step: int = 0,
    max_steps: int = 16,
    no_progress_count: int = 0,
) -> str:
    """Serialize the exact semantic input used by Teacher and local policies."""

    parsed_valid = tuple(parse_action(action) for action in valid_actions)
    parsed_history = tuple(parse_action(action) for action in history)
    if not task_id or not parsed_valid:
        raise ValueError("scheduler input requires task_id and valid actions")
    if step < 0 or max_steps <= 0 or step > max_steps:
        raise ValueError("invalid scheduler prompt step budget")
    sections = (
        ("SCHEDULER_ROLE", _SYSTEM_PROMPT),
        (
            f'ORCHESTRATION_PROFILE version="{ORCHESTRATION_PROFILE_VERSION}"',
            _json(orchestration_profile(selected_analysts)),
        ),
        (
            f'AGENT_CATALOG version="{AGENT_CATALOG_VERSION}"',
            _json(agent_catalog(selected_analysts)),
        ),
        (
            f'COMPLETION_CONTRACT version="{COMPLETION_CONTRACT_VERSION}"',
            _json(completion_contract(max_steps)),
        ),
        (
            "TASK",
            _json(
                {
                    "task_id": task_id,
                    "ticker": state.get("company_of_interest"),
                    "trade_date": state.get("trade_date"),
                    "asset_type": state.get("asset_type"),
                }
            ),
        ),
        (
            f'CURRENT_STATE version="{STATE_SCHEMA_VERSION}"',
            _json(business_state(state)),
        ),
        (
            "EXECUTION",
            _json(
                {
                    "history": [action.value for action in parsed_history],
                    "step": step,
                    "max_steps": max_steps,
                    "remaining_steps": max_steps - step,
                    "no_progress_count": no_progress_count,
                }
            ),
        ),
        ("VALID_ACTIONS", _json([action.value for action in parsed_valid])),
        ("SCHEDULER_ACTION", ""),
    )
    return "\n\n".join(f"<{title}>\n{body}" for title, body in sections)


def teacher_response_schema(
    valid_actions: Sequence[SchedulerAction | str],
) -> dict[str, object]:
    parsed = tuple(parse_action(action) for action in valid_actions)
    if not parsed:
        raise ValueError("teacher response schema requires valid actions")
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [action.value for action in parsed],
            }
        },
        "required": ["action"],
        "additionalProperties": False,
    }


def build_teacher_messages(
    context: SchedulerContext,
    *,
    positive_examples: Sequence[Mapping[str, object]] = (),
    failure_examples: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, str]]:
    system = "\n".join(
        (
            _SYSTEM_PROMPT,
            f"PROMPT_VERSION={TEACHER_PROMPT_VERSION}",
            "POSITIVE_EXAMPLES=" + _json(list(positive_examples)),
            "CORRECTED_FAILURE_EXAMPLES=" + _json(list(failure_examples)),
        )
    )
    user = "\n".join(
        (
            context.serialized_state,
            "OUTPUT_SCHEMA=" + _json(teacher_response_schema(context.valid_actions)),
        )
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_teacher_correction(
    *,
    error_type: str,
    previous_output: object,
    valid_actions: Sequence[SchedulerAction | str],
) -> str:
    parsed = tuple(parse_action(action) for action in valid_actions)
    if not parsed:
        raise ValueError("teacher correction requires valid actions")
    return "\n".join(
        (
            "Your previous response was invalid.",
            f"ERROR_TYPE={error_type}",
            "PREVIOUS_OUTPUT=" + _json(previous_output),
            "No Expert Agent was executed and CURRENT_STATE is unchanged.",
            "VALID_ACTIONS=" + _json([action.value for action in parsed]),
            'Return only {"action":"<ACT_...>"}.',
            "This is the final correction attempt.",
        )
    )
