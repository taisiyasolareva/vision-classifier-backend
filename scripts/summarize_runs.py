from __future__ import annotations

"""
Summarize experiment results into review-friendly JSON + Markdown.

Outputs (written to --out-dir):
  - results_task1.json
  - results_task2.json
  - results_summary.md

Inputs (merged if both exist):
  1) Local runs directory (default: runs/)
     - uses runs/<RUN_ID>/run_meta.json (cli args)
     - uses runs/<RUN_ID>/checkpoint_best.pt (meta metrics) when available
     - uses runs/<RUN_ID>/metrics.jsonl (last line) as a fallback

  2) reports/experiment_summary.json (Drive/Colab index)
     - dict keyed by run_id with val_top1/val_top5, epochs_completed, expected_epochs, is_complete
     - may not include arch/config/etc (those remain null)

Behavior:
  - Dedupe by run_id across sources; select the "best record" per run_id.
  - Label each run as complete vs incomplete.
  - "Best-so-far" in markdown is allowed to be incomplete, but is clearly marked as such.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

# Allow running this script without `pip install -e .`
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


TASK1_PASS_TOP1 = 0.44
TASK2_PASS_TOP1 = 0.84


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _infer_task(run_id: str, *, fallback: Optional[str] = None) -> Optional[str]:
    rid = run_id.strip()
    if rid.startswith("T1-") or "T1" in rid:
        return "task1"
    if rid.startswith("T2-") or "T2" in rid:
        return "task2"
    if fallback in ("task1", "task2"):
        return fallback
    return None


def _is_complete(
    *, epochs_completed: Optional[int], expected_epochs: Optional[int], ratio: float
) -> bool:
    if epochs_completed is None or expected_epochs is None or expected_epochs <= 0:
        return False
    return (epochs_completed / expected_epochs) >= ratio


def _load_checkpoint_meta(checkpoint_path: Path) -> dict[str, Any]:
    """
    Try to read checkpoint meta without crashing if torch isn't available.
    Expected format: torch.save({...,"meta": {...}}).
    """
    try:
        import torch  # type: ignore
    except Exception:
        return {}

    try:
        ckpt = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(ckpt, dict):
            meta = ckpt.get("meta")
            if isinstance(meta, dict):
                return meta
        return {}
    except Exception:
        return {}


def _last_metrics_line(metrics_jsonl: Path) -> dict[str, Any]:
    """
    Read last JSON line. Returns {} on failure.
    """
    try:
        with metrics_jsonl.open("r", encoding="utf-8") as f:
            last = None
            for line in f:
                line = line.strip()
                if not line:
                    continue
                last = line
        if not last:
            return {}
        obj = json.loads(last)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _run_record_from_local_run(
    run_dir: Path, *, completion_ratio: float
) -> Optional[dict[str, Any]]:
    run_id = run_dir.name
    run_meta_path = run_dir / "run_meta.json"
    if not run_meta_path.exists():
        return None

    run_meta = _read_json(run_meta_path)
    cli_args = run_meta.get("cli_args", {}) if isinstance(run_meta, dict) else {}
    meta_in_run_meta = run_meta.get("meta", {}) if isinstance(run_meta, dict) else {}

    task = _infer_task(
        run_id, fallback=cli_args.get("task") if isinstance(cli_args, dict) else None
    )
    expected_epochs = None
    if isinstance(cli_args, dict):
        expected_epochs = cli_args.get("epochs")
    expected_epochs = int(expected_epochs) if isinstance(expected_epochs, int) else None

    # Primary: checkpoint_best.pt meta
    ckpt_best_path = run_dir / "checkpoint_best.pt"
    ckpt_meta: dict[str, Any] = (
        _load_checkpoint_meta(ckpt_best_path) if ckpt_best_path.exists() else {}
    )

    # Fallback: metrics.jsonl last line
    metrics_line: dict[str, Any] = {}
    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_line = _last_metrics_line(metrics_path)

    # Extract metrics with fallback priority: checkpoint meta -> metrics.jsonl -> run_meta.json meta
    epoch = ckpt_meta.get("epoch")
    if epoch is None:
        epoch = metrics_line.get("epoch")
    epoch = int(epoch) if isinstance(epoch, int) else None

    val_top1 = _safe_float(ckpt_meta.get("val_top1"))
    if val_top1 is None:
        val_top1 = _safe_float(metrics_line.get("val_top1"))
    if val_top1 is None:
        val_top1 = _safe_float(meta_in_run_meta.get("val_top1"))

    val_top5 = _safe_float(ckpt_meta.get("val_top5"))
    if val_top5 is None:
        val_top5 = _safe_float(metrics_line.get("val_top5"))
    if val_top5 is None:
        val_top5 = _safe_float(meta_in_run_meta.get("val_top5"))

    train_top1 = _safe_float(ckpt_meta.get("train_top1"))
    if train_top1 is None:
        train_top1 = _safe_float(metrics_line.get("train_top1"))
    if train_top1 is None:
        train_top1 = _safe_float(meta_in_run_meta.get("train_top1"))

    # Determine completeness. If expected_epochs missing, treat as incomplete unless explicitly 1-epoch smoke.
    is_complete = _is_complete(
        epochs_completed=epoch, expected_epochs=expected_epochs, ratio=completion_ratio
    )
    if expected_epochs == 1 and epoch == 1:
        is_complete = True

    # Build record compatible with existing results_task*.json schema (plus extra fields).
    rec: dict[str, Any] = {
        "run_id": run_id,
        "task": task,
        "arch": (cli_args.get("arch") if isinstance(cli_args, dict) else None)
        or meta_in_run_meta.get("arch"),
        "config": cli_args.get("config") if isinstance(cli_args, dict) else None,
        "fine_tune_strategy": (
            cli_args.get("fine_tune_strategy") if isinstance(cli_args, dict) else None
        )
        or meta_in_run_meta.get("fine_tune_strategy"),
        "pretrained": (
            (cli_args.get("pretrained") if isinstance(cli_args, dict) else None)
            if isinstance(cli_args, dict)
            else None
        ),
        "image_size": (cli_args.get("image_size") if isinstance(cli_args, dict) else None)
        or meta_in_run_meta.get("image_size"),
        "unfreeze_last_n": (cli_args.get("unfreeze_last_n") if isinstance(cli_args, dict) else None)
        or meta_in_run_meta.get("unfreeze_last_n"),
        "trainable_params_m": meta_in_run_meta.get("trainable_params_m"),
        "notes": cli_args.get("notes") if isinstance(cli_args, dict) else None,
        "wandb_project": cli_args.get("wandb_project") if isinstance(cli_args, dict) else None,
        "wandb_run_name": cli_args.get("wandb_run_name") if isinstance(cli_args, dict) else None,
        "path": str(run_dir.resolve()),
        "epoch": epoch,
        "expected_epochs": expected_epochs,
        "is_complete": is_complete,
        "val_top1": val_top1 if val_top1 is not None else 0.0,
        "val_top5": val_top5 if val_top5 is not None else 0.0,
        "train_top1": train_top1,
        "source": "local_runs",
    }
    return rec


def _run_records_from_experiment_summary(summary_path: Path) -> list[dict[str, Any]]:
    """
    Convert reports/experiment_summary.json (dict keyed by run_id) into run records.
    """
    if not summary_path.exists():
        return []
    obj = _read_json(summary_path)
    if not isinstance(obj, dict):
        return []

    out: list[dict[str, Any]] = []
    for run_id, v in obj.items():
        if v is None:
            continue
        if not isinstance(v, dict):
            continue

        task = _infer_task(run_id)
        rec: dict[str, Any] = {
            "run_id": run_id,
            "task": task,
            "arch": None,
            "config": None,
            "fine_tune_strategy": None,
            "pretrained": None,
            "image_size": None,
            "unfreeze_last_n": None,
            "trainable_params_m": None,
            "notes": None,
            "wandb_project": None,
            "wandb_run_name": None,
            "path": v.get("path"),
            "epoch": v.get("epochs_completed"),
            "expected_epochs": v.get("expected_epochs"),
            "is_complete": bool(v.get("is_complete", False)),
            "val_top1": _safe_float(v.get("final_val_top1")) or 0.0,
            "val_top5": _safe_float(v.get("final_val_top5")) or 0.0,
            "train_top1": _safe_float(v.get("final_train_top1")),
            "source": "experiment_summary",
        }
        out.append(rec)

    return out


def _pick_better_record(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """
    Choose the better record for the same run_id.
    We want the record that best represents the run's final/most complete state.
    """

    def key(r: dict[str, Any]) -> tuple[int, int, float, int]:
        is_complete = 1 if r.get("is_complete") else 0
        epoch = r.get("epoch") or 0
        try:
            epoch_i = int(epoch)
        except Exception:
            epoch_i = 0
        val_top1 = r.get("val_top1") or 0.0
        try:
            val_f = float(val_top1)
        except Exception:
            val_f = 0.0
        has_arch = 1 if r.get("arch") else 0
        return (is_complete, epoch_i, val_f, has_arch)

    return a if key(a) >= key(b) else b


def _merge_by_run_id(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for r in records:
        run_id = r.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            continue
        if run_id not in merged:
            merged[run_id] = r
        else:
            merged[run_id] = _pick_better_record(merged[run_id], r)
    return list(merged.values())


def _sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(r: dict[str, Any]) -> tuple[float, int, int]:
        val = r.get("val_top1") or 0.0
        try:
            val_f = float(val)
        except Exception:
            val_f = 0.0
        complete = 1 if r.get("is_complete") else 0
        epoch = r.get("epoch") or 0
        try:
            epoch_i = int(epoch)
        except Exception:
            epoch_i = 0
        return (val_f, complete, epoch_i)

    return sorted(records, key=sort_key, reverse=True)


def _best_records(
    records: list[dict[str, Any]]
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """
    Returns (best_so_far, best_complete). best_so_far may be incomplete.
    """

    def is_smoke(r: dict[str, Any]) -> bool:
        rid = (r.get("run_id") or "").strip().lower()
        # Treat explicit SMOKE runs as non-candidates for "best" selection.
        return rid.startswith("smoke")

    if not records:
        return None, None

    # Prefer selecting best runs from non-smoke experiments. If none exist, fall back to all.
    non_smoke = [r for r in records if not is_smoke(r)]
    pool = non_smoke if non_smoke else records

    sorted_all = _sort_records(pool)
    best_so_far = sorted_all[0] if sorted_all else None
    complete_only = [r for r in sorted_all if r.get("is_complete")]
    best_complete = complete_only[0] if complete_only else None
    return best_so_far, best_complete


def _fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x * 100:.2f}%"


def _task_pass(task: str, val_top1: float) -> bool:
    if task == "task1":
        return val_top1 >= TASK1_PASS_TOP1
    if task == "task2":
        return val_top1 >= TASK2_PASS_TOP1
    return False


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def _write_markdown_summary(
    path: Path,
    *,
    task1: list[dict[str, Any]],
    task2: list[dict[str, Any]],
    sources_used: list[str],
) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    t1_best, t1_best_complete = _best_records(task1)
    t2_best, t2_best_complete = _best_records(task2)

    def format_best(task: str, r: Optional[dict[str, Any]]) -> str:
        if not r:
            return "- None"
        run_id = r.get("run_id")
        val_top1 = float(r.get("val_top1") or 0.0)
        val_top5 = float(r.get("val_top5") or 0.0)
        complete = bool(r.get("is_complete"))
        epoch = r.get("epoch")
        expected = r.get("expected_epochs")
        suffix = " (INCOMPLETE)" if not complete else ""
        gate = "PASS" if _task_pass(task, val_top1) else "FAIL"
        ep = (
            f"{epoch}/{expected}"
            if epoch is not None and expected is not None
            else str(epoch or "N/A")
        )
        return f"- `{run_id}` val_top1={val_top1:.4f} ({_fmt_pct(val_top1)}) val_top5={val_top5:.4f} ({_fmt_pct(val_top5)}) epochs={ep} {gate}{suffix}"

    lines: list[str] = []
    lines.append("# Results Summary (auto-generated)")
    lines.append("")
    lines.append(f"- Generated: {ts}")
    lines.append(f"- Sources: {', '.join(sources_used) if sources_used else 'none'}")
    lines.append("")
    lines.append("## Task 1")
    lines.append("")
    lines.append("### Best-so-far (may be incomplete)")
    lines.append(format_best("task1", t1_best))
    lines.append("")
    lines.append("### Best complete")
    lines.append(format_best("task1", t1_best_complete))
    lines.append("")
    lines.append("## Task 2")
    lines.append("")
    lines.append("### Best-so-far (may be incomplete)")
    lines.append(format_best("task2", t2_best))
    lines.append("")
    lines.append("### Best complete")
    lines.append(format_best("task2", t2_best_complete))
    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- Complete/incomplete is determined by epochs_completed vs expected_epochs (default threshold: 95%)."
    )
    lines.append("- Incomplete runs can still be shown as best-so-far, but are explicitly marked.")
    lines.append(
        "- `SMOKE-*` runs are excluded from best-of selection by default (but still appear in results JSON)."
    )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", type=str, default="runs", help="Local runs directory (optional).")
    p.add_argument(
        "--experiment-summary",
        type=str,
        default="reports/experiment_summary.json",
        help="Path to reports/experiment_summary.json (optional).",
    )
    p.add_argument(
        "--out-dir", type=str, default="reports", help="Output directory for results files."
    )
    p.add_argument(
        "--completion-ratio",
        type=float,
        default=0.95,
        help="Completion threshold: epochs_completed/expected_epochs >= ratio.",
    )
    args = p.parse_args()

    runs_dir = (REPO_ROOT / args.runs_dir).resolve()
    exp_summary_path = (REPO_ROOT / args.experiment_summary).resolve()
    out_dir = (REPO_ROOT / args.out_dir).resolve()

    sources_used: list[str] = []
    records: list[dict[str, Any]] = []

    # Local runs
    if runs_dir.exists() and runs_dir.is_dir():
        local: list[dict[str, Any]] = []
        for child in sorted(runs_dir.iterdir()):
            if not child.is_dir():
                continue
            rec = _run_record_from_local_run(child, completion_ratio=float(args.completion_ratio))
            if rec is not None:
                local.append(rec)
        if local:
            sources_used.append(f"local:{runs_dir}")
            records.extend(local)

    # experiment_summary.json
    exp_records = _run_records_from_experiment_summary(exp_summary_path)
    if exp_records:
        sources_used.append(f"experiment_summary:{exp_summary_path}")
        records.extend(exp_records)

    merged = _merge_by_run_id(records)

    task1 = _sort_records([r for r in merged if r.get("task") == "task1"])
    task2 = _sort_records([r for r in merged if r.get("task") == "task2"])

    # Write outputs
    _write_json(out_dir / "results_task1.json", task1)
    _write_json(out_dir / "results_task2.json", task2)
    _write_markdown_summary(
        out_dir / "results_summary.md", task1=task1, task2=task2, sources_used=sources_used
    )

    print(f"Wrote: {out_dir / 'results_task1.json'}")
    print(f"Wrote: {out_dir / 'results_task2.json'}")
    print(f"Wrote: {out_dir / 'results_summary.md'}")


if __name__ == "__main__":
    main()
