"""SchedulerPolicy adapter around the frozen Strong Teacher API."""

from __future__ import annotations

from pathlib import Path

from .policy import PolicyDecision, SchedulerContext
from .teacher_context import make_teacher_prompt_input
from .teacher_gateway import OpenRouterTeacherGateway
from .teacher_prompt import TEACHER_PROMPT_VERSION, build_teacher_messages


class StrongTeacherPolicy:
    def __init__(
        self,
        gateway: OpenRouterTeacherGateway,
        *,
        context_dir: str | Path | None = None,
    ):
        self.gateway = gateway
        self.context_dir = context_dir
        self.policy_id = f"strong-teacher:{gateway.model}:{TEACHER_PROMPT_VERSION}"

    def select_action(self, context: SchedulerContext) -> PolicyDecision:
        prompt_input = make_teacher_prompt_input(
            context.serialized_state,
            context.valid_actions,
            context.selected_analysts,
            path=self.context_dir,
        )
        messages = build_teacher_messages(prompt_input)
        return self.gateway.select_action(messages, context.valid_actions)
