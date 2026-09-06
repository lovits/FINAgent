from __future__ import annotations

import os
import threading
from pathlib import Path

from dotenv import find_dotenv, set_key

from tradingagents.default_config import DEFAULT_CONFIG

from .schemas import WebSettingsUpdate

OPENROUTER_URL = "https://openrouter.ai/api/v1"


class WebSettingsStore:
    def __init__(self, env_path: str | Path | None = None):
        discovered = find_dotenv(usecwd=True)
        project_env = Path(__file__).resolve().parents[2] / ".env"
        self.env_path = Path(env_path or discovered or project_env)
        self._lock = threading.RLock()
        self.expert_model = str(DEFAULT_CONFIG["quick_think_llm"])
        self.teacher_model = str(DEFAULT_CONFIG["teacher_model"])

    def public(self) -> dict[str, object]:
        with self._lock:
            key = os.environ.get("OPENROUTER_API_KEY", "").strip()
            adapter = str(DEFAULT_CONFIG.get("scheduler_adapter_path") or "")
            return {
                "provider": "openrouter",
                "key_configured": bool(key),
                "masked_key": _mask_key(key),
                "expert_model": self.expert_model,
                "teacher_model": self.teacher_model,
                "scheduler_base_model": str(DEFAULT_CONFIG["scheduler_base_model"]),
                "scheduler_adapter": Path(adapter).name if adapter else None,
            }

    def update(self, value: WebSettingsUpdate) -> dict[str, object]:
        with self._lock:
            if value.openrouter_api_key:
                key = value.openrouter_api_key.strip()
                os.environ["OPENROUTER_API_KEY"] = key
                self._persist("OPENROUTER_API_KEY", key)
            self.expert_model = value.expert_model
            self.teacher_model = value.teacher_model
            self._persist("TRADINGAGENTS_QUICK_THINK_LLM", self.expert_model)
            self._persist("TRADINGAGENTS_DEEP_THINK_LLM", self.expert_model)
            self._persist("TRADINGAGENTS_TEACHER_MODEL", self.teacher_model)
        return self.public()

    def runtime_overrides(self) -> dict[str, object]:
        with self._lock:
            return {
                "llm_provider": "openrouter",
                "backend_url": OPENROUTER_URL,
                "quick_think_llm": self.expert_model,
                "deep_think_llm": self.expert_model,
                "teacher_model": self.teacher_model,
            }

    def _persist(self, name: str, value: str) -> None:
        self.env_path.touch(exist_ok=True)
        self.env_path.chmod(0o600)
        set_key(str(self.env_path), name, value)


def _mask_key(value: str) -> str | None:
    if not value:
        return None
    if len(value) <= 10:
        return "••••••••"
    return f"{value[:6]}…{value[-4:]}"
