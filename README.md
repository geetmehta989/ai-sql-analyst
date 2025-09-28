# AI SQL Analyst Monorepo

- Frontend: `/frontend` (React + Vite + TS)
- Backend: `/backend` (FastAPI on Vercel Python runtime)
- Agent: `/agent` (LiteLLM via OpenAI SDK)
- Database: SQLite for dev, `DATABASE_URL` (e.g., Neon) for prod

## Local Dev

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=... # LiteLLM proxy key
export OPENAI_BASE_URL=https://proxyllm.ximplify.id
export DATABASE_URL=sqlite:///./dev.db
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Configure `VITE_REACT_APP_BACKEND_URL` to point to backend.

## Deploy

- Vercel project per app. Backend uses `api/index.py` entry.
- Set env vars on Vercel:
  - `OPENAI_API_KEY`
  - `OPENAI_BASE_URL` (https://proxyllm.ximplify.id)
  - `DATABASE_URL` (Neon Postgres recommended)
  - Frontend env: `VITE_REACT_APP_BACKEND_URL` pointing to deployed backend URL

## CI/CD

- On push to `main`, GitHub Actions deploys both apps to Vercel using repo secrets:
  - `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID_FRONTEND`, `VERCEL_PROJECT_ID_BACKEND`
  - `OPENAI_API_KEY`, `OPENAI_BASE_URL`, optional `DATABASE_URL`

## Endpoints

- `POST /upload` (multipart/form-data: file): loads Excel sheets as SQL tables with cleaned columns
- `POST /ask` (json: { question, top_k?, table_hint? }): LLM → SQL → execute → returns answer, data, chart spec
