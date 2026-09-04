from tradingagents.scheduler.teacher_context import load_teacher_examples


def test_default_teacher_examples_are_valid_and_cover_stop_and_failures() -> None:
    positive, failures = load_teacher_examples()
    assert any(example["output"]["action"] == "<ACT_STOP>" for example in positive)
    assert any(
        example.get("profile_id") == "multi-analyst-shallow-v1"
        for example in positive
    )
    assert any(example["error"] == "scheduler_must_not_call_tools" for example in failures)
