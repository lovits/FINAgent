import json

import pytest

from tradingagents.scheduler.actions import SchedulerAction
from tradingagents.scheduler.trajectory import SchedulerTrajectory, TrajectoryStep
from training.scheduler.build_sft_dataset import examples_from_trajectory, write_sft_jsonl


def _accepted(status="accepted"):
    trajectory = SchedulerTrajectory(
        trajectory_id="trajectory-1",
        task_id="task",
        run_id="run",
        mode="learned",
        policy_id="teacher",
        ticker="NVDA",
        trade_date="2025-01-02",
        teacher_model="google/gemini-3.8-flash",
        teacher_prompt_version="v1",
        verifier_version="v1",
        status=status,
        metadata={"source": "strong_teacher_verified"},
    )
    trajectory.steps.append(
        TrajectoryStep(
            step_id=0,
            serialized_state='{"state":0}',
            valid_actions=[SchedulerAction.MARKET.value],
            selected_action=SchedulerAction.MARKET.value,
            policy_id="teacher",
        )
    )
    return trajectory


def test_only_accepted_trajectory_becomes_action_only_sft_data(tmp_path):
    examples = examples_from_trajectory(_accepted())
    assert len(examples) == 1
    assert examples[0].input_text == '{"state":0}'
    assert examples[0].target_action == "<ACT_MARKET>"
    assert examples[0].metadata["teacher_model"] == "google/gemini-3.8-flash"

    path = tmp_path / "sft.jsonl"
    write_sft_jsonl(examples, path)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["target_action"] == "<ACT_MARKET>"


def test_rejected_trajectory_cannot_become_sft_positive():
    with pytest.raises(ValueError, match="only accepted"):
        examples_from_trajectory(_accepted(status="rejected"))
