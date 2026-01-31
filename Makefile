PYTHON ?= python3
VENV ?= .venv

DATA_ROOT ?= /path/to/data_root
ARTIFACT_DIR ?= runs/task2_resnet50/artifact
# IMPORTANT: macOS often exports TMPDIR to /var/folders/... which can fail on low disk.
# Default to a repo-local temp dir for reproducible runs. Override explicitly via:
#   make TMPDIR=/some/other/path <target>
TMPDIR := $(CURDIR)/.tmp

.PHONY: help venv install train-task1 train-task2 eval export serve eda profile validate test lint type-check \
	smoke-task1 smoke-task2-frozen smoke-task2-partial smoke-task2-full summarize stage1-fast stage1

help:
	@echo "Common tasks:"
	@echo "  make venv           # create virtualenv"
	@echo "  make install        # install dependencies + package"
	@echo "  make train-task1    # train Task 1 (no pretrain, no resize)"
	@echo "  make train-task2    # train Task 2 (pretrain + resize)"
	@echo "  make eval           # evaluate a checkpoint"
	@echo "  make export         # export TorchScript artifact"
	@echo "  make serve          # start FastAPI server"
	@echo "  make eda            # run basic EDA"
	@echo "  make profile        # compute dataset profile (mean/std, imbalance)"
	@echo "  make validate       # run basic dataset validation"
	@echo "  make test           # run pytest"
	@echo "  make lint           # run black/isort/ruff"
	@echo "  make type-check     # run mypy"
	@echo "  make smoke-task2-frozen  # Task 2 smoke fine-tune (frozen backbone)"
	@echo "  make smoke-task2-partial # Task 2 smoke fine-tune (partial unfreeze)"
	@echo "  make smoke-task2-full    # Task 2 smoke fine-tune (full fine-tune)"
	@echo "  make smoke-task1         # Task 1 smoke run (tiny subset, CPU OK)"
	@echo "  make summarize           # summarize runs/* into reports/results_*.json"
	@echo "  make stage1-fast         # (internal: script in archive; prints message)"
	@echo "  make stage1              # (internal: script in archive; prints message)"

venv:
	$(PYTHON) -m venv $(VENV)

install:
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install -e .

train-task1:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python -m cv200.train \
		--data-root "$(DATA_ROOT)" \
		--task task1 \
		--arch resnet18 \
		--epochs 15 \
		--batch-size 128 \
		--lr 1e-3 \
		--output-dir "runs/task1_resnet18"

train-task2:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python -m cv200.train \
		--data-root "$(DATA_ROOT)" \
		--task task2 \
		--arch resnet50 \
		--epochs 10 \
		--batch-size 128 \
		--lr 1e-4 \
		--output-dir "runs/task2_resnet50"

smoke-task1:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python -m cv200.train \
		--data-root "$(DATA_ROOT)" \
		--task task1 \
		--arch resnet18 \
		--epochs 1 \
		--batch-size 16 \
		--num-workers 0 \
		--limit-train-samples 32 \
		--limit-val-samples 32 \
		--device cpu \
		--notes "smoke: task1 pipeline" \
		--output-dir "runs/SMOKE-T1"

smoke-task2-frozen:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python -m cv200.train \
		--data-root "$(DATA_ROOT)" \
		--task task2 \
		--config "configs/smoke_task2_resnet50_frozen.yaml" \
		--notes "smoke: task2 fine-tuning mechanics (frozen, offline-safe)" \
		--output-dir "runs/SMOKE-T2-FROZEN"

smoke-task2-partial:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python -m cv200.train \
		--data-root "$(DATA_ROOT)" \
		--task task2 \
		--config "configs/smoke_task2_resnet50_partial1.yaml" \
		--notes "smoke: task2 fine-tuning mechanics (partial1, offline-safe)" \
		--output-dir "runs/SMOKE-T2-PARTIAL1"

smoke-task2-full:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python -m cv200.train \
		--data-root "$(DATA_ROOT)" \
		--task task2 \
		--config "configs/smoke_task2_resnet50_full.yaml" \
		--notes "smoke: task2 fine-tuning mechanics (full, offline-safe)" \
		--output-dir "runs/SMOKE-T2-FULL"

eval:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python -m cv200.eval \
		--data-root "$(DATA_ROOT)" \
		--ckpt "runs/task2_resnet50/checkpoint_best.pt"

export:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python -m cv200.export \
		--ckpt "runs/task2_resnet50/checkpoint_best.pt" \
		--output "runs/task2_resnet50/artifact"

serve:
	MODEL_ARTIFACT_DIR="$(ARTIFACT_DIR)" $(VENV)/bin/uvicorn cv200.api:app --host 0.0.0.0 --port 8000

eda:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python scripts/eda.py \
		--data-root "$(DATA_ROOT)" \
		--output-dir reports/eda

profile:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python scripts/profile_data.py \
		--data-root "$(DATA_ROOT)" \
		--output reports/data_profile.json

validate:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python scripts/validate_data.py \
		--data-root "$(DATA_ROOT)"

summarize:
	mkdir -p "$(TMPDIR)"
	TMPDIR="$(TMPDIR)" $(VENV)/bin/python scripts/summarize_runs.py --runs-dir runs --out-dir reports

# Stage 1 fast ladder (Task 1) — internal script moved to archive; see internal notes.
stage1-fast:
	@echo "Internal script moved to archive/internal/scripts/; see internal notes (not part of public repo)."

# Stage 1 full ladder (Task 1) — internal script moved to archive; see internal notes.
stage1:
	@echo "Internal script moved to archive/internal/scripts/; see internal notes (not part of public repo)."

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m black src tests scripts
	$(PYTHON) -m isort src tests scripts
	$(PYTHON) -m ruff check src tests scripts

type-check:
	$(PYTHON) -m mypy src


