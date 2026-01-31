from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image
from PIL import Image as PILImage
from PIL import UnidentifiedImageError
from pydantic import BaseModel, Field
from torchvision.transforms import v2 as T

from cv200.models import topk_from_logits
from cv200.utils import load_json

app = FastAPI(title="cv200-inference")


def _parse_cors_allow_origins() -> list[str]:
    """
    Comma-separated list of allowed browser origins for CORS.

    Example:
      CORS_ALLOW_ORIGINS=https://your-ui.vercel.app,https://your-custom-domain.com

    Note: This is only needed for browser-based clients (e.g. Vercel UI).
    """
    raw = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
    if not raw:
        return []
    parts = [p.strip().rstrip("/") for p in raw.split(",")]
    return [p for p in parts if p]


_cors_origins = _parse_cors_allow_origins()
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_DIM = 4096
# Protect against decompression bombs (roughly matches MAX_IMAGE_DIM^2 ~= 16.8MP).
PILImage.MAX_IMAGE_PIXELS = MAX_IMAGE_DIM * MAX_IMAGE_DIM


class HealthResponse(BaseModel):
    status: str
    artifact_dir: str | None = None
    artifact_ok: bool = False


class PredictionResponse(BaseModel):
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    class_id: int = Field(ge=0)


class PredictRequest(BaseModel):
    top_k: int = Field(default=5, ge=1, le=50)


class PredictResponse(BaseModel):
    top_k: int
    predictions: list[PredictionResponse]


def predict_request_form(
    top_k: int = Form(5),
) -> PredictRequest:
    # Pydantic validation happens here (e.g., top_k range).
    return PredictRequest(top_k=top_k)


def _artifact_dir_from_env() -> str:
    """
    Server-side artifact location.

    IMPORTANT: Clients must not be able to provide filesystem paths (public repo safety).
    """
    p = os.environ.get("MODEL_ARTIFACT_DIR", "").strip()
    if not p:
        raise RuntimeError("MODEL_ARTIFACT_DIR is not set on the server.")
    return p


def validate_image(file: UploadFile) -> tuple[bool, str]:
    """
    Validate input image file.

    Returns: (is_valid, error_message)
    """
    ct = (file.content_type or "").lower()
    if not ct.startswith("image/"):
        return False, "Invalid file type: expected an image (content-type image/*)."

    # Best-effort size check without loading into memory.
    try:
        f = file.file
        pos = f.tell()
        f.seek(0, 2)  # end
        size = f.tell()
        f.seek(pos, 0)
    except Exception:
        size = None

    if size is not None and size > MAX_IMAGE_BYTES:
        return False, f"Image too large: max {MAX_IMAGE_BYTES} bytes."

    # Dimension check (PIL open can raise for invalid images).
    try:
        f = file.file
        pos = f.tell()
        f.seek(0)
        img = Image.open(f)
        img.verify()  # validate headers without decoding full image
        f.seek(0)
        img = Image.open(f)
        w, h = img.size
        f.seek(pos, 0)
    except (UnidentifiedImageError, OSError, PILImage.DecompressionBombError):
        return False, "Invalid image file: failed to decode."
    except Exception:
        return False, "Invalid image file."

    if w > MAX_IMAGE_DIM or h > MAX_IMAGE_DIM:
        return False, f"Image dimensions too large: max {MAX_IMAGE_DIM}x{MAX_IMAGE_DIM}."

    return True, ""


def _build_preprocess_from_artifact(artifact_dir: Path) -> torch.nn.Module:
    pp_path = artifact_dir / "preprocess.json"
    if not pp_path.exists():
        raise FileNotFoundError(f"Missing preprocess.json in artifact: {pp_path}")
    pp = load_json(pp_path)
    image_size = int(pp["image_size"])
    mean = pp["mean"]
    std = pp["std"]

    task = pp.get("task", "task2")
    if task == "task1":
        # Mirror the deterministic val pipeline (pad + center crop), no resize.
        return T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Pad(padding=image_size // 2, fill=0),
                T.CenterCrop(size=(image_size, image_size)),
                T.Normalize(mean=mean, std=std),
            ]
        )
    # task2 default: resize
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Resize(size=(image_size, image_size), antialias=True),
            T.Normalize(mean=mean, std=std),
        ]
    )


@lru_cache(maxsize=8)
def _load_artifact_cached(artifact_dir_str: str) -> dict[str, Any]:
    artifact_dir = Path(artifact_dir_str).resolve()
    model_path = artifact_dir / "model.ts"
    labels_path = artifact_dir / "labels.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.ts in artifact: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.json in artifact: {labels_path}")

    model = torch.jit.load(str(model_path), map_location="cpu")
    model.eval()
    labels = load_json(labels_path)  # keys are strings
    preprocess = _build_preprocess_from_artifact(artifact_dir)
    return {"model": model, "labels": labels, "preprocess": preprocess}


@app.get("/healthz")
def healthz() -> HealthResponse:
    artifact_dir = os.environ.get("MODEL_ARTIFACT_DIR")
    ok = False
    if artifact_dir:
        p = Path(artifact_dir)
        ok = (
            (p / "model.ts").exists()
            and (p / "labels.json").exists()
            and (p / "preprocess.json").exists()
        )
    return HealthResponse(status="ok", artifact_dir=artifact_dir, artifact_ok=ok)


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    req: PredictRequest = Depends(predict_request_form),
) -> PredictResponse:
    ok, err = validate_image(file)
    if not ok:
        raise HTTPException(status_code=400, detail=err)

    try:
        artifact = _load_artifact_cached(_artifact_dir_from_env())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        file.file.seek(0)
        img = Image.open(file.file).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid image file: failed to decode.") from e

    x = artifact["preprocess"](img).unsqueeze(0)  # [1, 3, H, W]
    with torch.inference_mode():
        logits = artifact["model"](x)
        probs, idx = topk_from_logits(logits, k=req.top_k)

    probs = probs.squeeze(0).tolist()
    idx = idx.squeeze(0).tolist()
    labels = artifact["labels"]

    preds: list[PredictionResponse] = []
    for p, i in zip(probs, idx):
        preds.append(
            PredictionResponse(
                class_id=int(i),
                class_name=labels.get(str(i), str(i)),
                confidence=float(p),
            )
        )

    return PredictResponse(top_k=req.top_k, predictions=preds)


@app.post("/predict_batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    req: PredictRequest = Depends(predict_request_form),
) -> list[PredictResponse]:
    """
    Batch inference endpoint.

    Accepts multiple images via multipart/form-data and returns a list of prediction
    responses in the same order as the input files.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    try:
        artifact = _load_artifact_cached(_artifact_dir_from_env())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    labels = artifact["labels"]
    preprocess = artifact["preprocess"]

    images_t: list[torch.Tensor] = []
    for i, f in enumerate(files):
        ok, err = validate_image(f)
        if not ok:
            raise HTTPException(status_code=400, detail=f"file[{i}]: {err}")
        try:
            f.file.seek(0)
            img = Image.open(f.file).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"file[{i}]: Invalid image file: failed to decode."
            ) from e
        images_t.append(preprocess(img))

    x = torch.stack(images_t, dim=0)  # [B, 3, H, W]
    with torch.inference_mode():
        logits = artifact["model"](x)
        probs, idx = topk_from_logits(logits, k=req.top_k)  # [B, K], [B, K]

    probs_list = probs.tolist()
    idx_list = idx.tolist()

    out: list[PredictResponse] = []
    # NOTE: keep Python 3.9 compatibility for HF Spaces (zip(strict=...) is 3.10+).
    for row_probs, row_idx in zip(probs_list, idx_list):
        preds: list[PredictionResponse] = []
        for p, cls_i in zip(row_probs, row_idx):
            preds.append(
                PredictionResponse(
                    class_id=int(cls_i),
                    class_name=labels.get(str(int(cls_i)), str(int(cls_i))),
                    confidence=float(p),
                )
            )
        out.append(PredictResponse(top_k=req.top_k, predictions=preds))
    return out
