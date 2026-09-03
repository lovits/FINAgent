from types import SimpleNamespace

import pytest

from training.scheduler.download_model import download_model
from training.scheduler.smoke_qwen import QwenSmokeConfig, run_smoke


def test_download_records_exact_revision_and_required_files(monkeypatch, tmp_path) -> None:
    def fake_download(*args, **kwargs):
        destination = kwargs["local_dir"]
        (destination / "config.json").write_text("{}")
        (destination / "tokenizer.json").write_text("{}")
        (destination / "model.safetensors.index.json").write_text("{}")
        (destination / "model-00001-of-00001.safetensors").write_bytes(b"weights")
        return str(destination)

    monkeypatch.setattr("training.scheduler.download_model.snapshot_download", fake_download)
    monkeypatch.setattr(
        "training.scheduler.download_model.HfApi.model_info",
        lambda *args, **kwargs: SimpleNamespace(
            sha="resolved-sha",
            siblings=(SimpleNamespace(size=7),),
        ),
    )
    manifest = download_model(tmp_path, model_id="Qwen/test", revision="requested")
    assert manifest["requested_revision"] == "requested"
    assert manifest["resolved_revision"] == "resolved-sha"
    assert manifest["repository_bytes"] == 7


def test_qwen_smoke_requires_cuda(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        run_smoke(QwenSmokeConfig(output_path=str(tmp_path / "report.json")))
