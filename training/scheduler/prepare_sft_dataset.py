"""Audit accepted trajectories and materialize split-aware Scheduler SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradingagents.scheduler.trajectory import SchedulerTrajectory

from .build_sft_dataset import (
    SFT_SCHEMA_VERSION,
    SFTExample,
    examples_from_static_record,
    examples_from_trajectory,
)
from .trajectory_audit import AUDIT_VERSION, build_audit_report, load_and_audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _examples_for(record: dict[str, Any]) -> list[SFTExample]:
    if record.get("record_type") == "static_langgraph":
        return examples_from_static_record(record)
    return examples_from_trajectory(SchedulerTrajectory.from_dict(record))


def _write_examples(examples: Sequence[SFTExample], path: Path) -> str:
    content = "".join(
        json.dumps(asdict(example), ensure_ascii=False, sort_keys=True) + "\n"
        for example in examples
    )
    path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def prepare_sft_dataset(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    dataset_id: str,
    overwrite: bool = False,
    created_at: str | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"SFT dataset already exists: {manifest_path}")
    output.mkdir(parents=True, exist_ok=True)

    audited = load_and_audit(input_paths)
    audit_report = build_audit_report(audited)
    (output / "audit_report.json").write_text(
        json.dumps(audit_report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if audit_report["blocking_records"]:
        raise ValueError(
            f"{audit_report['blocking_records']} accepted trajectories failed audit; "
            f"see {output / 'audit_report.json'}"
        )

    by_split: dict[str, list[SFTExample]] = defaultdict(list)
    trajectories_by_split: Counter[str] = Counter()
    ticker_splits: dict[str, set[str]] = defaultdict(set)
    duplicate_examples = 0
    seen_examples: set[tuple[str, str, str]] = set()
    for record, result in audited:
        if not result.eligible_for_sft:
            continue
        examples = _examples_for(record)
        task_split = str(examples[0].metadata.get("task_split") or "") if examples else ""
        if task_split not in {"train", "validation", "test", "pilot"}:
            raise ValueError(f"unsupported or missing task split: {task_split!r}")
        trajectories_by_split[task_split] += 1
        ticker = str(examples[0].metadata.get("ticker") or "")
        if ticker:
            ticker_splits[ticker].add(task_split)
        for example in examples:
            key = (task_split, example.input_text, example.target_action)
            if key in seen_examples:
                duplicate_examples += 1
                continue
            seen_examples.add(key)
            by_split[task_split].append(example)
    if not by_split:
        raise ValueError("no audited trajectories are eligible for SFT")
    overlap = {ticker: splits for ticker, splits in ticker_splits.items() if len(splits) > 1}
    if overlap:
        raise ValueError(f"ticker-disjoint SFT split violation: {sorted(overlap)}")

    split_manifest = {}
    for task_split, examples in sorted(by_split.items()):
        path = output / f"{task_split}.jsonl"
        digest = _write_examples(examples, path)
        split_manifest[task_split] = {
            "file": path.name,
            "sha256": digest,
            "trajectories": trajectories_by_split[task_split],
            "examples": len(examples),
            "action_counts": dict(Counter(example.target_action for example in examples)),
        }

    source_manifest = [
        {
            "path": str(Path(path)),
            "sha256": _sha256(Path(path)),
        }
        for path in input_paths
    ]
    manifest = {
        "dataset_id": dataset_id,
        "schema_version": SFT_SCHEMA_VERSION,
        "created_at_utc": created_at
        or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "method": "audited_action_only_sft",
        "policy_target": "next_agent_action_token",
        "observation_loss_masked": True,
        "audit_version": AUDIT_VERSION,
        "source_files": source_manifest,
        "input_trajectories": audit_report["records"],
        "eligible_trajectories": audit_report["eligible_for_sft"],
        "duplicate_examples_removed": duplicate_examples,
        "ticker_counts": {
            split: sum(split in splits for splits in ticker_splits.values())
            for split in by_split
        },
        "splits": split_manifest,
        "usage_boundary": (
            "Only audited accepted actions are positive SFT labels. Rejected actions and "
            "Agent/Tool observations are never trained as target tokens."
        ),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = prepare_sft_dataset(
        args.input,
        args.output_dir,
        dataset_id=args.dataset_id,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "dataset_id": manifest["dataset_id"],
                "eligible_trajectories": manifest["eligible_trajectories"],
                "splits": {
                    split: details["examples"]
                    for split, details in manifest["splits"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
