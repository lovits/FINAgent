"""Versioned Strong Teacher prompt construction."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .actions import SchedulerAction, parse_action
from .agent_registry import AgentSpec

TEACHER_PROMPT_VERSION = "v1"


@dataclass(frozen=True)
class TeacherPromptInput:
    serialized_state: str
    valid_actions: tuple[SchedulerAction, ...]
    agent_specs: tuple[AgentSpec, ...]
    hard_rules: tuple[str, ...]
    positive_examples: tuple[Mapping[str, Any], ...] = ()
    failure_examples: tuple[Mapping[str, Any], ...] = ()


def _agent_catalog(specs: Sequence[AgentSpec]) -> list[dict[str, Any]]:
    return [
        {
            "id": spec.key,
            "action_token": spec.action.value,
            "purpose": spec.purpose,
            "reads": list(spec.reads),
            "writes": list(spec.writes),
            "prerequisites": list(spec.prerequisites),
            "internal_tools": list(spec.internal_tools),
            "tool_policy_owner": spec.tool_policy_owner,
        }
        for spec in specs
    ]


def build_teacher_messages(request: TeacherPromptInput) -> list[dict[str, str]]:
    """Build a grounded prompt that asks for exactly one valid agent action."""

    valid = [parse_action(action).value for action in request.valid_actions]
    if not valid:
        raise ValueError("Teacher cannot choose from an empty action set")

    system = "\n".join(
        (
            "You are the frozen Strong Route Teacher for TradingAgents.",
            "Choose exactly one next Expert Agent action from VALID_ACTIONS.",
            "Do not call tools, invent agents, produce a trade decision, or plan the whole path.",
            "The chosen Expert Agent will execute before you observe the next state.",
            "Return only JSON matching OUTPUT_SCHEMA.",
            f"TEACHER_PROMPT_VERSION={TEACHER_PROMPT_VERSION}",
            "AGENT_CATALOG=" + json.dumps(_agent_catalog(request.agent_specs), sort_keys=True),
            "HARD_RULES=" + json.dumps(list(request.hard_rules), ensure_ascii=False),
            "POSITIVE_EXAMPLES="
            + json.dumps(list(request.positive_examples), ensure_ascii=False, sort_keys=True),
            "CORRECTED_FAILURE_EXAMPLES="
            + json.dumps(list(request.failure_examples), ensure_ascii=False, sort_keys=True),
        )
    )
    user = "\n".join(
        (
            "STATE=" + request.serialized_state,
            "VALID_ACTIONS=" + json.dumps(valid),
            "OUTPUT_SCHEMA={\"next_agent\":\"<ACT_...>\","
            "\"reason_code\":\"UPPER_SNAKE_CASE\",\"reason\":\"short audit reason\"}",
        )
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
