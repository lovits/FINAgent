"""Scheduler-only state extension; the original AgentState remains unchanged."""

from typing import Annotated

from tradingagents.agents.utils.agent_states import AgentState


class SchedulerAgentState(AgentState):
    scheduler_task_id: Annotated[str, "Stable task identifier for scheduler traces"]
    scheduler_action: Annotated[str, "Most recent scheduler action token"]
    scheduler_step: Annotated[int, "Number of scheduler decisions made"]
    scheduler_history: Annotated[list[str], "Ordered scheduler action tokens"]
    scheduler_valid_actions: Annotated[list[str], "Actions valid at the latest decision"]
    scheduler_policy_id: Annotated[str, "Policy that made the latest decision"]
    scheduler_decision_metadata: Annotated[
        dict, "Usage and diagnostics for the latest scheduler decision"
    ]
    scheduler_no_progress_count: Annotated[int, "Consecutive decisions without state progress"]
    scheduler_last_state_signature: Annotated[
        str, "Business-state signature before the latest action"
    ]
    scheduler_agent_calls: Annotated[int, "Expert Agent calls selected by the scheduler"]
    scheduler_requested_mode: Annotated[str, "Requested orchestration mode"]
    scheduler_fallback_reason: Annotated[str, "Reason a dynamic run fell back to Static"]
