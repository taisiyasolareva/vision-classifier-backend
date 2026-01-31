from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cv200.checkpointing import load_checkpoint
from cv200.models import ModelConfig, build_model
from cv200.utils import ensure_dir, load_json, save_json


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output", type=str, required=True, help="Directory to write artifact files")
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    out = ensure_dir(args.output)
    ckpt = load_checkpoint(args.ckpt, map_location="cpu")
    meta = ckpt.get("meta", {})

    arch = meta.get("arch", "resnet18")
    num_classes = int(meta.get("num_classes", 200))

    model = build_model(ModelConfig(arch=arch, num_classes=num_classes, pretrained=False))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    device = torch.device(args.device)
    model = model.to(device)

    # Script with an example input matching common image_size; artifact should include preprocess.json for real inputs.
    image_size = int(meta.get("image_size", 224))
    example = torch.randn(1, 3, image_size, image_size, device=device)
    scripted = torch.jit.trace(model, example)
    scripted.save(str(out / "model.ts"))

    # Copy label mapping / preprocess if present next to checkpoint
    ckpt_dir = Path(args.ckpt).resolve().parent
    labels_src = ckpt_dir / "labels.json"
    preprocess_src = ckpt_dir / "preprocess.json"
    run_meta_src = ckpt_dir / "run_meta.json"
    if labels_src.exists():
        save_json(out / "labels.json", load_json(labels_src))
    if preprocess_src.exists():
        save_json(out / "preprocess.json", load_json(preprocess_src))
    if run_meta_src.exists():
        save_json(out / "run_meta.json", load_json(run_meta_src))

    # Always write export meta for traceability
    save_json(
        out / "export_meta.json",
        {
            "source_checkpoint": str(Path(args.ckpt).resolve()),
            "arch": arch,
            "num_classes": num_classes,
            "image_size": image_size,
        },
    )
    print(f"Wrote artifact to: {out}")


if __name__ == "__main__":
    main()
