#!/usr/bin/env bash
set -euo pipefail

repo="lovits/TradingAgents-RL"
tag="training-data-2026-09-06"
stage_dir="$(mktemp -d /tmp/tradingagents-data-release.XXXXXX)"
trap 'rm -rf "$stage_dir"' EXIT

if find artifacts data -name '.env*' -print -quit | grep -q .; then
  echo "Refusing to publish environment files." >&2
  exit 1
fi
if rg -l --hidden \
  -g '!*.safetensors' -g '!*.pt' -g '!*.bin' -g '!*.tar' \
  'sk-or-v1-[A-Za-z0-9_-]{20,}|-----BEGIN (OPENSSH |RSA |EC )?PRIVATE KEY-----' \
  artifacts data > "$stage_dir/secret-scan-files.txt"; then
  echo "Refusing to publish files matching credential patterns." >&2
  exit 1
fi

COPYFILE_DISABLE=1 tar -cf - --exclude='.DS_Store' data \
  | zstd -T0 -6 -o "$stage_dir/training-data.tar.zst"

COPYFILE_DISABLE=1 tar -cf - \
  --exclude='.DS_Store' \
  --exclude='*.safetensors' \
  --exclude='*.pt' \
  --exclude='*.bin' \
  --exclude='*.tar' \
  --exclude='*/checkpoint-*' \
  --exclude='*/checkpoint' \
  --exclude='*/test-adapter' \
  artifacts \
  | zstd -T0 -6 -o "$stage_dir/training-artifacts-no-models.tar.zst"

for archive in "$stage_dir"/*.tar.zst; do
  size="$(stat -f '%z' "$archive")"
  if (( size >= 2000000000 )); then
    split -b 1900m "$archive" "$archive.part-"
    rm "$archive"
  fi
done

(cd "$stage_dir" && find . -maxdepth 1 -type f \
  \( -name '*.tar.zst' -o -name '*.part-*' \) -print \
  | sort | sed 's#^./##' | xargs shasum -a 256 > SHA256SUMS)

if ! gh release view "$tag" --repo "$repo" >/dev/null 2>&1; then
  gh release create "$tag" --repo "$repo" --target main \
    --title "Training data and evaluation snapshot (2026-09-06)" \
    --notes "Public training-data snapshot for the Scheduler project. Includes datasets, trajectories, logs, metrics, and evaluation reports. Model weights, optimizer states, .env files, API keys, virtual environments, and machine-local caches are intentionally excluded. See SHA256SUMS for integrity."
fi
gh release upload "$tag" "$stage_dir"/training-* "$stage_dir"/SHA256SUMS \
  --repo "$repo" --clobber

gh release view "$tag" --repo "$repo" --json url,assets
