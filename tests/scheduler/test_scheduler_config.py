from tradingagents.default_config import DEFAULT_CONFIG


def test_scheduler_defaults_preserve_static_mode() -> None:
    assert DEFAULT_CONFIG["orchestration_mode"] == "static"
    assert DEFAULT_CONFIG["scheduler_max_steps"] == 16
    assert DEFAULT_CONFIG["scheduler_max_context_tokens"] == 32768
    assert DEFAULT_CONFIG["teacher_model"] == "google/gemini-3.8-flash"
    assert DEFAULT_CONFIG["scheduler_base_model"] == "Qwen/Qwen3-1.7B"
