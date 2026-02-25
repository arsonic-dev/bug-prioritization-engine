# 🐛 Bug Prioritization Engine

An ML-powered REST API that predicts bug severity and business impact from Jira data — so engineering teams always fix the highest-priority issues first.

## ✨ Features

- **Severity Classification** — Predicts Critical / High / Medium / Low using XGBoost
- **Impact Scoring** — Composite 0–100 business impact score per bug
- **Ranked Queue** — GET endpoint returns all open Jira issues sorted by priority
- **SHAP Explanations** — Every prediction explains *why* a bug was ranked that way
- **Auto-retraining** — APScheduler retrains models nightly from fresh Jira data
- **Hot-swap** — New models replace old ones without restarting the API
- **Degradation Guard** — Rejects new models if F1-macro drops more than 5 points

---

## 🗂 Project Structure

```
bug-prioritization-engine/
│
├── api/                        # FastAPI application
│   ├── __init__.py
│   ├── main.py                 # App factory, lifespan, CORS, exception handler
│   ├── routes.py               # All route handlers
│   └── schemas.py              # Pydantic v2 request/response models
│
├── data/                       # Data ingestion & feature engineering
│   ├── __init__.py
│   ├── jira_fetcher.py         # Jira REST API client (pagination + retry)
│   └── preprocessor.py         # BugFeatureEngineer (fit/transform pipeline)
│
├── models/                     # ML pipeline
│   ├── __init__.py
│   ├── artifacts/              # Saved model files (gitignored, persisted in Docker volume)
│   │   └── .gitkeep
│   ├── trainer.py              # XGBoost training + Optuna HPO
│   ├── predictor.py            # Inference engine
│   └── explainer.py            # SHAP-based explanations
│
├── scheduler/                  # Background jobs
│   ├── __init__.py
│   └── retrain_job.py          # APScheduler daily retraining
│
├── tests/                      # Test suite
│   ├── __init__.py
│   └── test_preprocessor.py    # Feature engineering unit tests
│
├── .env.example                # Environment variable template
├── .gitignore
├── config.py                   # Centralised config via pydantic-settings
├── docker-compose.yml
├── Dockerfile
├── Makefile                    # Dev shortcuts
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI 0.111 |
| ML Models | XGBoost 2.0 (classifier + regressor) |
| HPO | Optuna 3.6 |
| Explainability | SHAP 0.45 |
| Text Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Feature Engineering | scikit-learn 1.5 |
| Data Source | Jira REST API v3 (via `jira` SDK) |
| Scheduling | APScheduler 3.10 |
| Config | pydantic-settings + python-dotenv |
| Runtime | Python 3.11 |
| Container | Docker + Docker Compose |

---

## 🚀 Quick Start (Local)

### 1. Clone & enter the project

```bash
git clone https://github.com/yourorg/bug-prioritization-engine.git
cd bug-prioritization-engine
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
# or
make install
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
JIRA_URL=https://yourorg.atlassian.net
JIRA_EMAIL=your-email@example.com
JIRA_API_TOKEN=your-jira-api-token   # Generate at: id.atlassian.com → Security → API tokens
JIRA_PROJECT_KEY=MYPROJ
RETRAIN_API_KEY=your-long-random-secret
MODEL_PATH=models/artifacts
```

### 5. Start the API

```bash
make run
# or
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Bootstrap your first model

The API starts in **degraded mode** until models are trained. Trigger the first training run:

```bash
curl -X POST http://localhost:8000/api/v1/retrain \
     -H "X-Retrain-API-Key: your-long-random-secret"
```

This fetches the last 90 days of resolved Jira bugs and trains both models (~5–15 min depending on dataset size).

### 7. Check health

```bash
curl http://localhost:8000/api/v1/health
```

```json
{
  "status": "ok",
  "model_version": "1718000000",
  "trained_at": "2024-06-10T02:00:00+00:00",
  "feature_count": 234,
  "classifier_f1_macro": 0.847
}
```

---

## 🐳 Docker Deployment

### Build and run with Docker Compose

```bash
# Copy and configure env
cp .env.example .env
# (edit .env with your Jira credentials)

# Build image and start container
docker-compose up -d

# Watch logs
docker-compose logs -f api

# Trigger first training run
curl -X POST http://localhost:8000/api/v1/retrain \
     -H "X-Retrain-API-Key: your-key"
```

Model artefacts are stored in a named Docker volume (`model_artifacts`) so they survive container restarts and re-deploys.

### Stop

```bash
docker-compose down          # keeps volume
docker-compose down -v       # also deletes saved models
```

---

## 📡 API Reference

Interactive docs available at **http://localhost:8000/docs** (Swagger UI) and **http://localhost:8000/redoc**.

### `POST /api/v1/predict`

Predict severity and impact for a single bug.

**Request body:**
```json
{
  "summary": "Login page throws 500 on empty password",
  "description": "Steps: 1. Go to /login  2. Submit with empty password.",
  "priority_name": "critical",
  "priority_ordinal": 4,
  "issuetype": "Bug",
  "components": "Auth|Frontend",
  "labels": "regression|customer-reported",
  "reporter": "Jane Smith",
  "comment_count": 3,
  "vote_count": 5,
  "watch_count": 12,
  "story_points": 3.0,
  "age_days": 2.0
}
```

**Response:**
```json
{
  "severity_label": "Critical",
  "severity_confidence": 0.91,
  "severity_probabilities": {
    "Critical": 0.91, "High": 0.06, "Medium": 0.02, "Low": 0.01
  },
  "impact_score": 87.4,
  "severity_explanation": [
    {"feature": "Component: Auth", "impact": "+4.21", "direction": "increases severity"},
    {"feature": "Watch count",     "impact": "+3.18", "direction": "increases severity"},
    {"feature": "Reporter bug rate","impact": "+2.05","direction": "increases severity"}
  ],
  "impact_explanation": [
    {"feature": "Component failure rate", "impact": "+9.33", "direction": "increases impact score"}
  ]
}
```

---

### `GET /api/v1/ranked-queue?project=MYPROJ&limit=50`

Returns all open Jira issues ranked by predicted impact (highest first). Cached for 5 minutes.

**Response:**
```json
{
  "project": "MYPROJ",
  "total_issues": 142,
  "issues": [
    {
      "jira_key": "MYPROJ-88",
      "summary": "Checkout crashes on Safari iOS",
      "predicted_severity": "Critical",
      "severity_confidence": 0.89,
      "impact_score": 91.2,
      "rank": 1
    }
  ],
  "generated_at": "2024-06-10T09:15:00Z",
  "cached": false
}
```

---

### `GET /api/v1/explain/{jira_key}`

Fetches a Jira issue by key and returns a full SHAP explanation.

```bash
curl http://localhost:8000/api/v1/explain/MYPROJ-88
```

---

### `POST /api/v1/retrain`

Triggers manual model retraining. Protected by API key header.

```bash
curl -X POST http://localhost:8000/api/v1/retrain \
     -H "X-Retrain-API-Key: your-secret-key"
```

---

### `GET /api/v1/health`

Returns current model version, last training timestamp, and system status.

---

## 🧪 Running Tests

```bash
make test
# or
pytest tests/ -v --tb=short
```

The test suite covers:
- Feature matrix shape and null-safety
- Target variable construction (severity labels, impact scores)
- Historical lookup tables (reporter bug rate, component failure rate)
- Inference column alignment
- Edge cases (empty text, unknown components)
- Joblib serialisation round-trip

---

## ⏰ Scheduled Retraining

The API automatically retrains models every night at **2:00 AM UTC** using the last 90 days of resolved Jira issues.

**Performance guard:** if the new model's F1-macro is more than 0.05 below the current model, the old model is kept and a warning is logged.

Configure the schedule via `.env`:
```env
RETRAIN_CRON_HOUR=2          # 0-23 UTC
RETRAIN_LOOKBACK_DAYS=90
```

---

## 🔑 Getting a Jira API Token

1. Log in to [id.atlassian.com](https://id.atlassian.com)
2. Go to **Security** → **API tokens**
3. Click **Create API token**
4. Copy the token into `JIRA_API_TOKEN` in your `.env`

---

## 📦 Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `JIRA_URL` | ✅ | — | Jira instance base URL |
| `JIRA_EMAIL` | ✅ | — | Authenticating user email |
| `JIRA_API_TOKEN` | ✅ | — | Jira API token |
| `JIRA_PROJECT_KEY` | ✅ | — | Project key (e.g. `MYPROJ`) |
| `RETRAIN_API_KEY` | ✅ | — | Secret for `POST /retrain` |
| `MODEL_PATH` | ❌ | `models/artifacts` | Where to save/load models |
| `CACHE_TTL_SECONDS` | ❌ | `300` | Ranked-queue cache TTL |
| `EMBEDDING_MODEL` | ❌ | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `RETRAIN_CRON_HOUR` | ❌ | `2` | UTC hour for nightly retrain |
| `RETRAIN_LOOKBACK_DAYS` | ❌ | `90` | Days of history for retraining |
| `LOG_LEVEL` | ❌ | `INFO` | `DEBUG / INFO / WARNING / ERROR` |
| `MLFLOW_TRACKING_URI` | ❌ | — | Enable MLflow (e.g. `http://localhost:5000`) |

---

## 🏗 Deploying to a VPS / Cloud VM

```bash
# On your server
git clone https://github.com/yourorg/bug-prioritization-engine.git
cd bug-prioritization-engine
cp .env.example .env
# Fill in .env ...

# Install Docker + Docker Compose if not present
curl -fsSL https://get.docker.com | sh

# Start
docker-compose up -d

# Bootstrap models
curl -X POST http://YOUR_SERVER_IP:8000/api/v1/retrain \
     -H "X-Retrain-API-Key: your-secret"
```

For production, put **nginx** in front of the API and terminate TLS there.

---

## 📄 License

MIT
