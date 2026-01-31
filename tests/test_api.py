from __future__ import annotations

from io import BytesIO
from pathlib import Path

import torch
from fastapi.testclient import TestClient
from PIL import Image

from cv200.api import app
from cv200.utils import save_json


class _TinyModel(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x).flatten(1)
        return self.fc(x)


def _make_artifact(tmp_path: Path) -> Path:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    num_classes = 3
    model = _TinyModel(num_classes=num_classes).eval()
    example = torch.randn(1, 3, 224, 224)
    ts = torch.jit.trace(model, example)
    ts.save(str(artifact_dir / "model.ts"))

    save_json(artifact_dir / "labels.json", {"0": "0", "1": "1", "2": "2"})
    save_json(
        artifact_dir / "preprocess.json",
        {
            "task": "task2",
            "image_size": 224,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    )
    return artifact_dir


def _img_bytes(size: int = 224) -> bytes:
    img = Image.new("RGB", (size, size), color=(100, 20, 30))
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_health_endpoint() -> None:
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_endpoint(tmp_path: Path) -> None:
    client = TestClient(app)
    artifact_dir = _make_artifact(tmp_path)
    import os

    os.environ["MODEL_ARTIFACT_DIR"] = str(artifact_dir)

    r = client.post(
        "/predict",
        data={"top_k": "2"},
        files={"file": ("img.jpg", _img_bytes(), "image/jpeg")},
    )
    assert r.status_code == 200
    payload = r.json()
    assert payload["top_k"] == 2
    assert len(payload["predictions"]) == 2


def test_predict_batch_endpoint(tmp_path: Path) -> None:
    client = TestClient(app)
    artifact_dir = _make_artifact(tmp_path)
    import os

    os.environ["MODEL_ARTIFACT_DIR"] = str(artifact_dir)

    r = client.post(
        "/predict_batch",
        data={"top_k": "2"},
        files=[
            ("files", ("img1.jpg", _img_bytes(), "image/jpeg")),
            ("files", ("img2.jpg", _img_bytes(), "image/jpeg")),
        ],
    )
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, list)
    assert len(payload) == 2
    assert payload[0]["top_k"] == 2
    assert len(payload[0]["predictions"]) == 2
