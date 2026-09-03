from training.scheduler.runtime_config import scheduler_runtime_config


def test_runtime_config_reapplies_environment_after_dotenv_load(monkeypatch) -> None:
    monkeypatch.setenv("TRADINGAGENTS_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("TRADINGAGENTS_QUICK_THINK_LLM", "expert-quick")
    monkeypatch.setenv("TRADINGAGENTS_TEACHER_MODEL", "google/gemini-3.8-flash")
    monkeypatch.setenv("TRADINGAGENTS_TEMPERATURE", "0.0")
    monkeypatch.setenv("TRADINGAGENTS_MAX_DEBATE_ROUNDS", "2")

    config = scheduler_runtime_config({"scheduler_action_temperature": 0.8})

    assert config["llm_provider"] == "openrouter"
    assert config["quick_think_llm"] == "expert-quick"
    assert config["teacher_model"] == "google/gemini-3.8-flash"
    assert float(config["temperature"]) == 0.0
    assert config["max_debate_rounds"] == 2
    assert config["scheduler_action_temperature"] == 0.8
