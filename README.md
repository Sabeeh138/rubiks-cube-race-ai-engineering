# Rubik's Cube Race

AI Engineering Assignment 02: a browser-based benchmark for comparing four assigned OpenRouter models on Rubik's Cube move-selection tasks.

## Features

- Deterministic Rubik's Cube engine using canonical 54-facelet `URFDLB` states and standard face turns.
- Exact short-state reference solver (up to 5 HTM moves).
- FastAPI backend, responsive dashboard, recorded cube-state playback, benchmark metrics, and JSON result storage.
- Four assigned model integrations, offline mocks, a scripted Claude demonstration mode, and a separate state-aware exploratory condition.
- Official experiment configuration is kept separate from exploratory diagnostics and simulated results.

## Models

| Model | OpenRouter ID | API format |
|---|---|---|
| Jev 1.13 | `typesafe/jev-1.13` | Decisions API |
| GPT-6 Luna Decisions | `openai/gpt-6-luna-decisions` | Decisions API |
| Cloudflare Clef | `cloudflare/clef` | Decisions API |
| Claude Opus 5.5 | `anthropic/claude-opus-5.5` | Chat Completions |

In recording, diagnostic, and full state-aware exploratory modes, Claude uses the existing scripted fallback and is labelled `SIMULATED — NOT A LIVE CLAUDE RESPONSE`. It is not included in live model metrics. The other three models use real OpenRouter requests and may incur charges.

## Requirements and setup (Windows PowerShell)

Requirements: Python 3.11+ and pip. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Add your key to the local `.env` file:

```dotenv
OPENROUTER_API_KEY=your_key_here
```

Keep `.env` private; it must not be committed. `.env.example` contains placeholders only. Never put API keys in source code, screenshots, or README text.

## Run the application

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000/>. API documentation is at <http://127.0.0.1:8000/docs>; health check is <http://127.0.0.1:8000/api/health>.

The dashboard supports mock, live, recording, state-aware diagnostic, and full state-aware exploratory modes. Mock mode is the default and does not contact OpenRouter. Configure `STUDENT_NAME` and `STUDENT_ROLL_NUMBER` in the ignored `.env` file if those labels are required in the recording.

## Experiment modes and result integrity

- **Official live experiment:** uses the frozen `rubiks-move-objective-v1` instruction and the pinned five-puzzle protocol. The official primary run remains pending.
- **State-aware diagnostic:** uses `rubiks-move-objective-v2-state-aware` on the one-move `R` scramble, optionally with engine-generated one-step action previews. The diagnostic run solved the puzzle with all three live models (3/3); this result is exploratory.
- **Full State-Aware Experiment — Exploratory:** applies v2 with action previews to all five pinned puzzles. The completed run recorded **2 of 15 live trials solved**; the five Claude entries were scripted and excluded from live metrics. Run ID: `4849d1fc-45d2-4ac4-a777-9c0155654248`.
- **Verified reference solution:** uses the exact solver to calculate and verify a short solution. It is an algorithmic baseline, not an LLM result, and creates no benchmark trial.

The v2 exploratory results must not be reported as results from the frozen official v1 experiment. A successful live solve is counted only when the cube engine verifies the final cube state as solved. Mock and scripted outcomes are not evidence of live model performance.

The full exploratory experiment can make up to **450 live requests** (3 models × 5 puzzles × 30 moves). Candidate previews increase request context; actual provider cost varies. Review the dashboard warning and usage limits before running it.

## Tests

Run the Python suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Run the JavaScript frontend tests (Node.js 18+):

```powershell
node --test tests/frontend/*.test.mjs
node --check app/static/app.js
```

Tests use mocked HTTP transports and do not contact OpenRouter. A passing test suite verifies software behavior, not model performance.

## API overview

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/benchmark` | Start a benchmark run |
| `GET` | `/api/benchmark/{run_id}` | Run status and progress |
| `GET` | `/api/benchmark/{run_id}/results` | Trial results and aggregate metrics |
| `GET` | `/api/benchmark/{run_id}/trials/{trial_id}` | Single trial details |
| `GET` | `/api/results?limit=20` | Stored run summaries |
| `POST` | `/api/cube/preview` | Preview a scramble and cube state |
| `POST` | `/api/cube/reference-solution` | Calculate and verify an exact short solution |

## Project layout

```text
app/
  api/          # FastAPI routes and schemas
  benchmark/    # Runner, metrics, storage, dataset, reporting
  cube/         # Cube engine, scrambles, short-state solver
  models/       # Adapter contract, mocks, OpenRouter adapters, prompts
  services/     # Benchmark orchestration
  static/       # Dashboard and cube playback
  config.py     # Environment configuration
docs/           # API compatibility and experiment protocol
tests/          # Python and frontend tests
.env.example
.gitignore
pyproject.toml
README.md
```

### Results and screen recording
<img width="911" height="380" alt="final result 2" src="https://github.com/user-attachments/assets/73b90966-311a-4578-a05c-67467e8d313e" />
<img width="944" height="322" alt="final result" src="https://github.com/user-attachments/assets/ee55f3f1-7458-4485-86fd-6e398b0681f2" />
