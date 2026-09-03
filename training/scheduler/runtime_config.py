"""Load scheduler runtime settings after an optional .env file is available."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tradingagents.default_config import DEFAULT_CONFIG, _apply_env_overrides


def scheduler_runtime_config(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = _apply_env_overrides(dict(DEFAULT_CONFIG))
    if overrides:
        config.update(overrides)
    return config
