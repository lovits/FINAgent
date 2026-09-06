from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Analyst = Literal["market", "social", "news", "fundamentals"]
OrchestrationMode = Literal["static", "teacher", "learned"]
ResearchDepth = Literal["shallow", "medium", "deep"]


class CreateRunRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    analysis_date: date
    output_language: str = Field(default="Chinese", min_length=1, max_length=40)
    analysts: list[Analyst] = Field(min_length=1, max_length=4)
    research_depth: ResearchDepth = "shallow"
    orchestration_mode: OrchestrationMode = "static"

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        ticker = value.strip().upper()
        if not re.fullmatch(r"[A-Z0-9._\-^=]+", ticker):
            raise ValueError("ticker contains unsupported characters")
        return ticker

    @field_validator("analysts")
    @classmethod
    def unique_analysts(cls, values: list[Analyst]) -> list[Analyst]:
        if len(values) != len(set(values)):
            raise ValueError("analysts must be unique")
        return values

    @model_validator(mode="after")
    def validate_mode_constraints(self) -> CreateRunRequest:
        if self.orchestration_mode == "learned" and self.research_depth != "shallow":
            raise ValueError("local learned scheduler only supports shallow research")
        return self


class WebSettingsUpdate(BaseModel):
    openrouter_api_key: str | None = Field(default=None, min_length=8, max_length=512)
    expert_model: str = Field(min_length=3, max_length=160)
    teacher_model: str = Field(min_length=3, max_length=160)

    @field_validator("expert_model", "teacher_model")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        model = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9._:/-]+", model):
            raise ValueError("model ID contains unsupported characters")
        return model
