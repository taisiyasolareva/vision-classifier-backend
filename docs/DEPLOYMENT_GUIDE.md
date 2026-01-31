# Deployment Guide — End-to-End Vision Classification System

This guide documents the intended public deployment topology for **Production Computer Vision: 200‑Class Image Classification**:

- **Backend (vision-classifier-api)**: FastAPI inference API deployed on Hugging Face Spaces (repo: **vision-classifier-backend**)
- **Frontend (vision-classifier-ui)**: React + Vite UI (repo: **vision-classifier-frontend**) deployed on a static host (e.g., Vercel)

## Backend: Hugging Face Spaces (FastAPI)

### Canonical public URL (current)

- API base: `https://solarevat-cv200.hf.space`
- Docs: `https://solarevat-cv200.hf.space/docs`
- Health: `https://solarevat-cv200.hf.space/healthz`

### Important: endpoint paths are root paths

The backend routes are:
- `POST /predict`
- `POST /predict_batch`
- `GET /healthz`
- `GET /docs`

If you see `/api/*` in the frontend, that is a **dev proxy prefix** (Vite) and **not** a backend route prefix.

### Model artifact contract

The API loads an exported TorchScript artifact directory via `MODEL_ARTIFACT_DIR` (server-side only). The artifact directory must include:
- `model.ts`
- `labels.json`
- `preprocess.json`

## Frontend: React + Vite (UI)

The UI calls the backend API and supports **uploading an image** for inference.

### Note on out-of-domain uploads (expected behavior)

The model was trained on a specific dataset distribution. The UI allows arbitrary image uploads for demonstration, but:
- out-of-domain images may produce confident but meaningless predictions
- this is expected for a closed-set classifier without explicit OOD detection

## CORS (production safety)

If the UI and API are on different domains (e.g., Vercel UI → HF Spaces API), configure the API’s CORS allowlist to include the deployed UI origin.

Do **not** use `allow_origins=["*"]` for a public deployment unless you are intentionally making the API open to any website.

## Quick end-to-end smoke check (no dataset required)

1) Verify the deployed API:
- Open `https://solarevat-cv200.hf.space/docs`
- Call `GET /healthz`

2) Verify local backend + benchmark (uses generated in-memory images):

```bash
MODEL_ARTIFACT_DIR=/path/to/artifact uvicorn cv200.api:app --host 0.0.0.0 --port 8000
python scripts/benchmark_api.py --url http://localhost:8000 --num-requests 100 --batch-size 1
```

