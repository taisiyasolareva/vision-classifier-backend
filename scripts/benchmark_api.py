from __future__ import annotations

"""
Benchmark script for the FastAPI inference service.

Usage:
  python scripts/benchmark_api.py \
    --url http://localhost:8000 \
    --num-requests 100 \
    --batch-size 1

Notes:
- Uses stdlib HTTP (no extra dependencies).
- Generates a small in-memory RGB image by default (no dataset required).
"""

import argparse
import base64
import hashlib
import json
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

def _make_test_image_bytes() -> bytes:
    """
    A tiny valid JPEG (1x1) embedded to keep benchmark stdlib-only.

    Note: this is intentionally small; for end-to-end production profiling,
    benchmark with a representative image distribution.
    """
    b64 = (
        # 1x1 JPEG
        "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
        "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/"
        "2wCEAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
        "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/"
        "wAARCABkAGQDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAf/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwC4AA//2Q=="
    )
    return base64.b64decode(b64)


def _encode_multipart_formdata(
    *,
    fields: dict[str, str],
    files: list[tuple[str, str, str, bytes]],
) -> tuple[bytes, str]:
    """
    Build multipart/form-data body.

    files: list of (field_name, filename, content_type, content_bytes)
    Returns: (body, content_type_header_value)
    """
    boundary = f"----cv200boundary{uuid.uuid4().hex}"
    crlf = "\r\n"
    chunks: list[bytes] = []

    for name, value in fields.items():
        chunks.append(f"--{boundary}{crlf}".encode("utf-8"))
        chunks.append(f'Content-Disposition: form-data; name="{name}"{crlf}{crlf}'.encode("utf-8"))
        chunks.append(value.encode("utf-8"))
        chunks.append(crlf.encode("utf-8"))

    for field_name, filename, content_type, content in files:
        chunks.append(f"--{boundary}{crlf}".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{filename}"{crlf}'.encode(
                "utf-8"
            )
        )
        chunks.append(f"Content-Type: {content_type}{crlf}{crlf}".encode("utf-8"))
        chunks.append(content)
        chunks.append(crlf.encode("utf-8"))

    chunks.append(f"--{boundary}--{crlf}".encode("utf-8"))
    body = b"".join(chunks)
    return body, f"multipart/form-data; boundary={boundary}"


def _percentile(xs_ms: list[float], p: float) -> float:
    if not xs_ms:
        return float("nan")
    if p <= 0:
        return float(min(xs_ms))
    if p >= 100:
        return float(max(xs_ms))
    xs = sorted(xs_ms)
    k = (len(xs) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


def _safe_get_json(url: str, *, timeout_s: float) -> tuple[dict[str, object] | None, str | None]:
    """
    Best-effort GET and parse JSON.

    Returns: (payload, error_string)
    """
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            payload = resp.read()
        return json.loads(payload.decode("utf-8")), None
    except Exception as e:
        return None, str(e)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_fingerprint(artifact_dir: Path) -> dict[str, object]:
    """
    Fingerprint a *local* artifact dir (useful for local benchmarking).
    """
    files = ["model.ts", "labels.json", "preprocess.json"]
    out: dict[str, object] = {"artifact_dir": str(artifact_dir)}
    for name in files:
        fp = artifact_dir / name
        if not fp.exists():
            out[name] = None
            continue
        out[name] = {
            "sha256": _sha256_file(fp),
            "bytes": fp.stat().st_size,
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", type=str, required=True, help="Base URL, e.g. http://localhost:8000")
    p.add_argument(
        "--artifact-dir",
        type=str,
        default="",
        help="Optional: for reporting only (the server should be configured via MODEL_ARTIFACT_DIR).",
    )
    p.add_argument("--num-requests", type=int, default=100, help="Number of requests to send")
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Images per request (1 uses /predict, >1 uses /predict_batch)",
    )
    p.add_argument("--top-k", type=int, default=5, help="Top-k predictions per image")
    p.add_argument("--timeout-seconds", type=float, default=30.0, help="Per-request timeout")
    p.add_argument(
        "--out", type=str, default="reports/serving_benchmark.json", help="Output JSON path"
    )
    args = p.parse_args()

    if args.num_requests <= 0:
        raise SystemExit("--num-requests must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")

    base = args.url.rstrip("/")
    endpoint = "/predict" if args.batch_size == 1 else "/predict_batch"
    full_url = base + endpoint
    healthz_url = base + "/healthz"

    # Use an embedded image to keep the benchmark self-contained.
    img_bytes = _make_test_image_bytes()
    img_ct = "image/jpeg"

    fields = {"top_k": str(args.top_k)}
    file_field = "file" if args.batch_size == 1 else "files"

    # Warmup request (not included in stats)
    warmup_body, warmup_ct = _encode_multipart_formdata(
        fields=fields,
        files=[(file_field, "warmup.jpg", img_ct, img_bytes)] * args.batch_size,
    )
    req = Request(full_url, data=warmup_body, method="POST")
    req.add_header("Content-Type", warmup_ct)
    try:
        with urlopen(req, timeout=args.timeout_seconds) as resp:
            _ = resp.read()
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise SystemExit(f"Warmup request failed: HTTPError {e.code}: {body}")
    except Exception as e:
        raise SystemExit(f"Warmup request failed: {e}")

    healthz_payload, healthz_err = _safe_get_json(healthz_url, timeout_s=args.timeout_seconds)

    lat_ms: list[float] = []
    errors: list[str] = []
    t0 = time.perf_counter()

    for i in range(args.num_requests):
        body, content_type = _encode_multipart_formdata(
            fields=fields,
            files=[(file_field, f"img_{j}.jpg", img_ct, img_bytes) for j in range(args.batch_size)],
        )
        req = Request(full_url, data=body, method="POST")
        req.add_header("Content-Type", content_type)

        start = time.perf_counter()
        try:
            with urlopen(req, timeout=args.timeout_seconds) as resp:
                payload = resp.read()
                # Validate JSON response shape lightly
                _ = json.loads(payload.decode("utf-8"))
        except HTTPError as e:
            errors.append(f"HTTPError {e.code}: {e.read().decode('utf-8', errors='ignore')}")
            continue
        except URLError as e:
            errors.append(f"URLError: {e}")
            continue
        except Exception as e:
            errors.append(f"Error: {e}")
            continue
        end = time.perf_counter()
        lat_ms.append((end - start) * 1000.0)

        # Light progress for long runs
        if (i + 1) % max(1, args.num_requests // 10) == 0:
            print(f"{i+1}/{args.num_requests} requests complete")

    t1 = time.perf_counter()
    wall_s = t1 - t0

    total_reqs = len(lat_ms)
    total_images = total_reqs * args.batch_size
    reqs_per_s = float(total_reqs) / float(max(1e-9, wall_s))
    imgs_per_s = float(total_images) / float(max(1e-9, wall_s))

    summary = {
        "url": base,
        "endpoint": endpoint,
        "artifact_dir_hint": args.artifact_dir or None,
        "healthz": {
            "url": healthz_url,
            "payload": healthz_payload,
            "error": healthz_err,
        },
        "num_requests_target": args.num_requests,
        "num_requests_completed": total_reqs,
        "batch_size": args.batch_size,
        "total_images": total_images,
        "latency_ms": {
            "p50": _percentile(lat_ms, 50),
            "p95": _percentile(lat_ms, 95),
            "p99": _percentile(lat_ms, 99),
            "mean": (sum(lat_ms) / float(len(lat_ms))) if lat_ms else float("nan"),
            "min": float(min(lat_ms)) if lat_ms else float("nan"),
            "max": float(max(lat_ms)) if lat_ms else float("nan"),
        },
        "throughput": {"requests_per_s": reqs_per_s, "images_per_s": imgs_per_s},
        "errors": errors[:20],  # cap to keep JSON small
    }

    if args.artifact_dir:
        p_art = Path(args.artifact_dir).expanduser()
        if p_art.exists():
            summary["artifact_fingerprint"] = _artifact_fingerprint(p_art)
        else:
            summary["artifact_fingerprint"] = {"artifact_dir": args.artifact_dir, "error": "path does not exist"}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Print summary table
    print("")
    print("=== Serving benchmark summary ===")
    print(f"URL: {summary['url']}{summary['endpoint']}")
    print(
        f"Completed requests: {total_reqs}/{args.num_requests} | batch_size={args.batch_size} | total_images={total_images}"
    )
    print(f"Throughput: {reqs_per_s:.2f} req/s | {imgs_per_s:.2f} img/s")
    print(
        "Latency (ms): "
        f"p50={summary['latency_ms']['p50']:.2f} "
        f"p95={summary['latency_ms']['p95']:.2f} "
        f"p99={summary['latency_ms']['p99']:.2f} "
        f"mean={summary['latency_ms']['mean']:.2f}"
    )
    if errors:
        print(f"Errors: {len(errors)} (first: {errors[0]})")
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
