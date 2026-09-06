import os

from tradingagents.web.schemas import WebSettingsUpdate
from tradingagents.web.settings import WebSettingsStore


def test_settings_mask_and_persist_secret(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-example-secret-1234")
    path = tmp_path / ".env"
    store = WebSettingsStore(path)

    public = store.update(
        WebSettingsUpdate(
            openrouter_api_key="sk-or-new-secret-5678",
            expert_model="z-ai/glm-5.3-flash",
            teacher_model="google/gemini-3.8-flash",
        )
    )

    assert public["masked_key"] == "sk-or-…5678"
    assert "sk-or-new-secret-5678" not in str(public)
    assert "OPENROUTER_API_KEY='sk-or-new-secret-5678'" in path.read_text()
    assert os.stat(path).st_mode & 0o777 == 0o600
