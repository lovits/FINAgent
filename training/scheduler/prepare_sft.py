"""Build one split-aware SFT JSONL from accepted Static and verified Teacher traces."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable

from tradingagents.scheduler.store import TrajectoryStore
from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .build_sft import (
    SFTExample,
    build_sft_examples,
    verified_teacher_ids,
    write_sft_dataset,
)
from .pair_dataset import load_comparisons


def prepare_sft_dataset(
    static: Iterable[SchedulerTrajectory],
    teacher: Iterable[SchedulerTrajectory],
    *,
    verified_ids: set[str],
    token_counter: Callable[[str], int],
    max_tokens: int = 32768,
) -> tuple[list[SFTExample], int]:
    static_examples, static_overflow = build_sft_examples(
        static,
        source="static",
        token_counter=token_counter,
        max_tokens=max_tokens,
    )
    teacher_examples, teacher_overflow = build_sft_examples(
        teacher,
        source="teacher_verified",
        token_counter=token_counter,
        max_tokens=max_tokens,
        verified_teacher_ids=verified_ids,
    )
    return _deduplicate([*static_examples, *teacher_examples]), (
        static_overflow + teacher_overflow
    )


def _deduplicate(examples: Iterable[SFTExample]) -> list[SFTExample]:
    values = []
    seen: set[tuple[str, str]] = set()
    for example in examples:
        key = (example.input_text, example.target_action)
        if key in seen:
            continue
        seen.add(key)
        values.append(example)
    return values


def qwen_token_counter(model_id: str, revision: str | None = None) -> Callable[[str], int]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def count(input_text: str) -> int:
        token_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return len(token_ids)

    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", required=True)
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--comparisons", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--revision")
    parser.add_argument("--max-tokens", type=int, default=32768)
    args = parser.parse_args()

    static = TrajectoryStore(args.static).load()
    teacher = TrajectoryStore(args.teacher).load()
    comparisons = load_comparisons(args.comparisons)
    examples, overflow = prepare_sft_dataset(
        static,
        teacher,
        verified_ids=verified_teacher_ids(comparisons),
        token_counter=qwen_token_counter(args.model_id, args.revision),
        max_tokens=args.max_tokens,
    )
    run_ids = sorted({trajectory.run_id for trajectory in [*static, *teacher]})
    manifest = write_sft_dataset(
        examples,
        args.output,
        overflow_count=overflow,
        source_run_ids=run_ids,
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
