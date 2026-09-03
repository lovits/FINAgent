"""Download and record the exact public Qwen3-1.7B revision on AutoDL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from tradingagents.scheduler.store import write_json_atomic

DEFAULT_MODEL_ID = "Qwen/Qwen3-1.7B"
DEFAULT_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"


def download_model(
    destination: str | Path,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    revision: str = DEFAULT_REVISION,
) -> dict[str, object]:
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    resolved = snapshot_download(
        model_id,
        revision=revision,
        local_dir=target,
        token=False,
    )
    required = ("config.json", "tokenizer.json", "model.safetensors.index.json")
    missing = [name for name in required if not (target / name).is_file()]
    shards = sorted(target.glob("model-*.safetensors"))
    if missing or not shards:
        raise FileNotFoundError(
            f"incomplete model download; missing={missing}, shards={len(shards)}"
        )
    total_bytes = sum(
        int(getattr(sibling, "size", 0) or 0) for sibling in info.siblings or ()
    )
    manifest = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "local_path": str(Path(resolved).resolve()),
        "repository_bytes": total_bytes,
        "safetensor_shards": [path.name for path in shards],
    }
    write_json_atomic(target / "model_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        default="/root/autodl-tmp/models/Qwen3-1.7B",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    args = parser.parse_args()
    print(
        json.dumps(
            download_model(
                args.destination,
                model_id=args.model_id,
                revision=args.revision,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
