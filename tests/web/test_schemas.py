import pytest
from pydantic import ValidationError

from tradingagents.web.schemas import CreateRunRequest


def _payload(**overrides):
    return {
        "ticker": "NVDA",
        "analysis_date": "2026-08-20",
        "output_language": "Chinese",
        "analysts": ["market", "news"],
        "research_depth": "shallow",
        "orchestration_mode": "static",
        **overrides,
    }


def test_learned_scheduler_accepts_shallow() -> None:
    request = CreateRunRequest(**_payload(orchestration_mode="learned"))
    assert request.research_depth == "shallow"


@pytest.mark.parametrize("depth", ["medium", "deep"])
def test_learned_scheduler_rejects_other_depths(depth: str) -> None:
    with pytest.raises(ValidationError, match="only supports shallow"):
        CreateRunRequest(
            **_payload(orchestration_mode="learned", research_depth=depth)
        )


def test_request_requires_at_least_one_unique_analyst() -> None:
    with pytest.raises(ValidationError):
        CreateRunRequest(**_payload(analysts=[]))
    with pytest.raises(ValidationError, match="must be unique"):
        CreateRunRequest(**_payload(analysts=["market", "market"]))
