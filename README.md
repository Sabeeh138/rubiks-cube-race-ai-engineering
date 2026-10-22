# Rubik's Cube Race

Starter project for AI Engineering Assignment 02. The application will compare four OpenRouter models choosing moves for the same Rubik's Cube states.

## Current scope

Phases 0 through 8 and Phase 9 preparation are complete. The project includes environment configuration, an API compatibility record, a deterministic Rubik's Cube engine, an exact short-state reference solver, a common model adapter contract, offline mocks, real OpenRouter adapter implementations, a provider-neutral benchmark runner, aggregate metrics, validated JSON result storage, a service-backed FastAPI API, a responsive browser dashboard, and offline primary-experiment analysis tooling. Recording mode runs the three existing live adapters and a clearly marked scripted Claude demonstration. The separate state-aware v2 diagnostic, full five-puzzle exploratory mode, and verified algorithmic reference-solver showcase do not change the official v1 protocol. No complete primary or full exploratory live experiment has been run.

## Requirements

- Python 3.11 or newer
- pip

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

`OPENROUTER_API_KEY` is intentionally blank in `.env.example`. Keep any real key in the ignored `.env` file or the process environment; never commit it. A key is not needed to run the health endpoint, mocks, offline adapter tests, or the full test suite. A real OpenRouter adapter checks for the key when a request is attempted and raises `ConfigurationError` if it is missing.

## Run the local application

```powershell
python -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/` for the frontend. The health response is at `http://127.0.0.1:8000/api/health` (`{"status":"ok"}`), and interactive API docs are at `http://127.0.0.1:8000/docs`.

## Run tests

```powershell
python -m pytest
```

The tests cover health/configuration smoke checks, canonical facelet serialization, standard move behavior, inverse sequences, seeded scrambles, exact short-state references, adapter contracts, scripted mock outcomes, benchmark metrics/storage, API lifecycle and validation, frontend serving and cube rendering, and invalid inputs. They do not contact OpenRouter.

## Backend API

Start the local backend from the project root:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The interactive OpenAPI documentation is at `http://127.0.0.1:8000/docs`; the schema is at `/openapi.json`.

| Method and route | Purpose |
| --- | --- |
| `GET /api/health` | Local readiness check. Does not load a key or contact OpenRouter. |
| `POST /api/benchmark` | Submit a run and receive HTTP `202` with a unique `run_id`, status URL, and results URL. |
| `GET /api/benchmark/{run_id}` | Read lifecycle status and progress (completed puzzles/trials). |
| `GET /api/benchmark/{run_id}/results` | Read the completed raw trials and aggregate metrics. |
| `GET /api/benchmark/{run_id}/trials/{trial_id}` | Read one completed model/puzzle trial. |
| `GET /api/results?limit=20` | List persisted run summaries (limit 1–100). |
| `POST /api/cube/preview` | Preview a seeded or explicit puzzle with the shared cube engine. |
| `POST /api/cube/reference-solution` | Find and replay-verify an exact solution within the existing five-move search scope; creates no benchmark trial. |
| `GET /api/ui-config` | Read only the optional public student name and roll-number labels. |

The request body rejects unknown fields. Each puzzle must provide either an explicit legal `scramble_moves` list or a `seed` and optional `scramble_length` (default 4). Requests allow 1–20 uniquely named puzzles, scramble length at most 25, move limit at most 30, and turn timeout from 0.1 through 120 seconds. For example:

```json
{
  "execution_mode": "mock",
  "puzzles": [
    {"puzzle_id": "primary-01", "seed": 2026, "scramble_length": 4},
    {"puzzle_id": "primary-02", "scramble_moves": ["R", "U", "F'"]}
  ],
  "max_valid_moves": 30,
  "turn_timeout_seconds": 30
}
```

`execution_mode` defaults to `mock`. Mock adapters are deterministic workflow fixtures: they undo each submitted scramble and therefore solve it, but their moves and latency do not represent AI performance. They run without credentials and network access. A successful submission resembles:

```json
{
  "run_id": "d8b8f968-2384-4536-a73e-fc4732e1fc77",
  "status": "running",
  "execution_mode": "mock",
  "status_url": "/api/benchmark/d8b8f968-2384-4536-a73e-fc4732e1fc77",
  "results_url": "/api/benchmark/d8b8f968-2384-4536-a73e-fc4732e1fc77/results"
}
```

The client should poll the status URL until `status` is `completed` or `failed`, then fetch results only for a completed run. Progress counts a puzzle after all four assigned model trials have finished. A trial-level invalid move, timeout, or adapter exception is recorded as that trial's result; the run can still complete. A worker or result-persistence failure marks the overall run failed and exposes a sanitized error. Unknown run/trial IDs return `404`, results requested while a run is running or failed return `409`, invalid inputs return `422`, active-run conflicts return `409`, and live mode without configuration returns `503`.

The status response contains `run_id`, `status`, `execution_mode`, UTC lifecycle timestamps, `progress` (`completed_trials`, `total_trials`, `completed_puzzles`, `total_puzzles`, and `current_puzzle_id`), and an optional sanitized `error`. The completed results response contains `schema_version`, run configuration and execution mode, a `trials` array, and aggregate `metrics` keyed by exact model ID. A trial response contains its scramble and initial state, every request attempt, safe provider-reported token counters where available, final state, solve/termination fields, move and failure counts, latencies, exact reference distance, and optimality gap. Use the `trial_id` returned in a result; its model ID includes `/`, which the trial route accepts as part of the remaining path.

### Complete mock benchmark through the API

Run the backend in one PowerShell window. In another, submit a reproducible five-puzzle primary set, poll its status, then fetch the results:

```powershell
$body = @{
  execution_mode = "mock"
  puzzles = @(
    @{ puzzle_id = "primary-01"; seed = 2026; scramble_length = 4 },
    @{ puzzle_id = "primary-02"; seed = 2027; scramble_length = 4 },
    @{ puzzle_id = "primary-03"; seed = 2028; scramble_length = 4 },
    @{ puzzle_id = "primary-04"; seed = 2029; scramble_length = 5 },
    @{ puzzle_id = "primary-05"; seed = 2030; scramble_length = 5 }
  )
} | ConvertTo-Json -Depth 5
$run = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/benchmark -ContentType "application/json" -Body $body
do {
  Start-Sleep -Milliseconds 100
  $status = Invoke-RestMethod -Uri "http://127.0.0.1:8000$($run.status_url)"
} while ($status.status -eq "running")
$results = Invoke-RestMethod -Uri "http://127.0.0.1:8000$($run.results_url)"
$results.metrics
```

Every puzzle is evaluated by the four exact assigned model IDs. The mock execution path uses the same benchmark runner, metric calculations, and JSON storage as the live path, but never creates provider adapters. `Idempotency-Key` can be sent as a header: repeating the same key and request returns the existing run ID; using that key for a different request returns `409`. Without an idempotency key, the service allows one active run per application process and rejects another submission with `409` and the active run ID. State and this lock are process-local; run Uvicorn with one worker (`--workers 1`) so workers cannot accept overlapping runs. Each completed run is stored atomically as `results/<run_id>.json`; those generated files are ignored by Git.

Live execution requires deliberate opt-in in the request (`"execution_mode": "live"`) and `OPENROUTER_API_KEY` in the backend environment or ignored `.env` file. Missing configuration is rejected before a worker starts. Live requests use the four configured assigned IDs and can incur provider charges. The optional `recording` execution mode requires the same key, calls Jev, GPT-6 Luna, and Clef through their existing live adapters, and uses a deterministic scripted Claude path without a Claude API request. Its Claude trial is persisted with the exact `SIMULATED — NOT A LIVE CLAUDE RESPONSE` label and excluded from aggregate live metrics and the official report. A live Claude error remains an error; it is never replaced with the script. Health checks and mock requests never fall back to live execution. Automated API tests use mock adapters and explicitly verify that no async HTTP client is constructed.

Responses contain no credentials. New JSON result schema v4 files persist run execution mode, trial provenance, the simulated Claude label when applicable, safe provider-reported token counters when available, the prompt condition, and whether action previews were included. Legacy schema v1–v3 files still load without rewriting existing records; their prompt condition remains unknown rather than being inferred.

Run API tests separately or run the full suite:

```powershell
python -m pytest tests/test_api.py
python -m pytest
```

The browser UI also has no-dependency cube facelet and recorded-state playback tests (Node.js 18+):

```powershell
node --test tests/frontend/*.test.mjs
```

## Frontend and benchmark dashboard

Start the FastAPI server and open `http://127.0.0.1:8000/`. The UI defaults to a **mock demonstration**. Choose mock, live, recording, state-aware diagnostic, or **Full State-Aware Experiment — Exploratory** mode. Primary modes accept a starting seed, one/three/five puzzles, and a four- or five-move scramble length; diagnostic mode fixes the scramble to `R` and the live move limit to three. The full exploratory mode fixes the five pinned primary puzzles, uses the same 30-move/30-second limits, state-aware v2, and one-step previews. Use **Preview scramble** to request the initial facelets from the backend cube engine, then run the selected mode. The UI polls run status, fetches results when complete, and displays the actual initial state, move histories, terminal statuses, metrics, and a solve-rate chart returned by the API.

The API reports progress per completed puzzle group (four model trials), not per move. The UI does not simulate live thinking or invented moves. After a run completes, select a puzzle to inspect its actual initial 54-facelet state and each assigned model's recorded final status and move history. The **Replay recorded cube states** buttons update the cube visualization from the saved per-turn snapshots. The table shows solve rate, average latency for valid responses, average total trial runtime, average valid moves, and mean optimality gap where the exact reference is available. Unsolved trials and unsupported exact references do not receive a gap.

Mock fixtures undo the submitted scramble so the end-to-end workflow can be demonstrated offline. The interface labels these as mock results throughout; they must not be interpreted as measurements of model performance. In recording mode, the Claude card and saved trial carry the exact `SIMULATED — NOT A LIVE CLAUDE RESPONSE` label. The deterministic script applies the inverse of that puzzle's recorded scramble through the cube engine. It adds no delays or model latency; the metrics table includes only the three live models. Recording runs are excluded from the official experiment report. The other three models still make real OpenRouter requests and may incur charges. Browser code calls only same-origin backend routes. OpenRouter credentials remain in the backend and are never sent to the browser.

### Launch and record a Claude demonstration

1. Put your OpenRouter key in the ignored project `.env` file as `OPENROUTER_API_KEY=...`. Do not paste the key into the browser or source files. Recording mode needs it for Jev, GPT-6 Luna, and Clef.
2. From the project root, launch the backend with one worker:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
   ```

3. Open `http://127.0.0.1:8000/`, select **Recording · three live + scripted Claude**, choose the puzzle count and 4–5 HTM scramble length, then use **Preview scramble**.
4. Click **Run recording demo** and wait for completion. Inspect all four model cards and use **Replay recorded cube states** to show saved transitions. Claude must display `SIMULATED — NOT A LIVE CLAUDE RESPONSE`; the comparison table contains only the three live models.

Recording mode makes up to three live requests per puzzle when each live model solves in one move, and up to `3 × puzzle count × 30` live requests at the configured maximum (450 for five puzzles). It never sends a Claude request. Provider usage for those three real models may incur charges. Scripted Claude moves are for demonstration only and are excluded from model-performance analysis and the official report.

### Run the separate state-aware live diagnostic

1. Launch the backend with one worker as shown above and set `OPENROUTER_API_KEY` in the ignored `.env` file. The diagnostic makes requests only to Jev 1.13, GPT-6 Luna Decisions, and Cloudflare Clef; Claude uses its existing script and is never called.
2. In the dashboard, choose **State-aware diagnostic · R scramble**. The UI fixes the puzzle to `R`, sets a maximum of three valid moves per live model (at most nine live requests total), and disables the primary-dataset controls.
3. Leave **Include one-step action previews** checked to send the real engine-generated result state for each legal candidate move; uncheck it to run the same v2 state-aware instruction without candidate previews. Both settings are recorded in the saved run.
4. Start **Run state-aware diagnostic**. Each genuine live outcome is applied to the actual cube state through the cube engine. Errors remain errors, and unsolved results stay unsolved. The run uses the separate `rubiks-move-objective-v2-state-aware` instruction; it does not change the official v1 prompt.

The diagnostic is stored under a separate run ID with `execution_mode: diagnostic`, `prompt_version: rubiks-move-objective-v2-state-aware`, and the selected action-preview flag. It is labelled exploratory in the dashboard and is always excluded from the official primary report. The three live models can make up to nine requests total and may incur charges; Claude's scripted path has no model latency and does not appear in live metrics. The v2 context may help interpretation, but no improvement in model solve rate is established until real results are collected.

### Run the full state-aware exploratory experiment

1. Set `OPENROUTER_API_KEY` in the backend's ignored `.env` file. The mode requires this key because Jev, GPT-6 Luna, and Clef remain live; Claude uses only its existing scripted demonstration.
2. In the dashboard choose **Full State-Aware Experiment — Exploratory**. The pinned puzzle settings are fixed to the five entries in `rubiks-cube-race-primary-v1`; the dashboard loads them from `GET /api/benchmark/primary-dataset` and the backend validates the exact IDs, seeds, lengths, and generated scrambles.
3. Review the on-screen substantial-usage warning. The run can make up to **450 live OpenRouter requests** (3 live models × 5 puzzles × 30 valid moves). One-step previews add input content and may increase charges; actual total cost is unknown. Tick the acknowledgement checkbox to enable the explicit start.
4. Click **Start full exploratory experiment**. The dashboard does not start this mode automatically. It polls the normal run-status route and fetches the completed results. Every request gets the current cube state, full move history, legal moves and engine-generated one-step successor states; after each accepted move, the runner rebuilds those values from the updated cube.

The persisted run records `execution_mode: state_aware_experiment`, `prompt_version: rubiks-move-objective-v2-state-aware`, `action_previews_enabled: true`, the existing 30-move/30-second settings, and per-trial provenance. The three live models retain their exact assigned IDs and existing OpenRouter adapters. All five Claude trials use the existing deterministic script and are labeled `SIMULATED — NOT A LIVE CLAUDE RESPONSE`; they have no model latency and are excluded from model metrics. The v2 exploratory run is always excluded from the official primary report, whose `live` mode remains on the frozen v1 condition. API calls in the browser go only to this backend; no browser-side provider credential or live request is used.

This is an execution path, not evidence of success. Do not describe the experiment as successful unless a future authorized live run completes and the persisted cube engine verifies each model's actual final state.

### Play the verified reference solution

Click **Play Verified Reference Solution** to ask the backend's exact solver for the current displayed puzzle, replay the returned path through the existing cube engine, and animate the engine-generated states. The final facelet string must pass the backend solved-state check before the API returns it. The dashboard displays the sequence, HTM move count, and solved confirmation under `REFERENCE SOLVER — ALGORITHMIC BASELINE, NOT AN LLM RESULT`. This showcase makes no model/API request, creates no benchmark trial, and contributes nothing to model metrics or official statistics. Exact search is supported only within the existing five-move short-state scope; unsupported states produce an error instead of a guessed path. The animation delay is visual only and is not model latency.

For a final recording view, complete the recording or diagnostic run, select the puzzle, then play the verified reference solution. The page separates genuine live-model results, Claude's simulated demonstration, and the algorithmic reference solution. Student name and roll number are repeated in the reference showcase for the recording.

### Student identification for the recording

Set these optional values in the ignored local `.env` file or process environment. They are shown in the fixed header during the recording. `.env.example` leaves them blank; when unset, the UI displays clear “not configured” placeholders.

```dotenv
STUDENT_NAME=Your Name
STUDENT_ROLL_NUMBER=Your Roll Number
```

Do not put personal values into shared source code. The public `/api/ui-config` response returns only these two display labels; it never returns the OpenRouter key.

To manually smoke-test without credits, start the local app, confirm the mock badge and generated cube preview, leave execution mode as the default, and run the five-puzzle mock set. Wait for the status to reach **Completed**, then inspect solve rates, the selected puzzle's move histories, and the mock-data label. The `Preview scramble` endpoint and benchmark API tests run through the shared backend cube and benchmark implementation. No live inference is required.

## Configuration

`app.config.Settings.from_env()` reads `OPENROUTER_API_KEY` from the process environment and the project `.env` file (process values take precedence). The key remains optional for local health checks. The adapters use this existing setting and require it only when `choose_move()` is called. Missing configuration raises a `ConfigurationError` with setup guidance. Keys are only sent in the backend `Authorization` header; they are not included in request payloads or adapter results.

## Cube engine

The deterministic engine wraps [`magiccube`](https://pypi.org/project/magiccube/) 1.2.0 for cube move mechanics. The project owns a small immutable `CubeState` interface, so future model adapters can pass every model the same stable 54-character state. States use Kociemba facelet order `URFDLB`, with nine stickers per face; the solved state is `UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB`.

```python
from app.cube import CubeState, generate_scramble

scramble = generate_scramble(seed=2026, length=25)
cube = scramble.apply_to(CubeState())
print(scramble.notation)
print(cube.serialize())
print(cube.is_solved)  # False
print(scramble.undo_from(cube).is_solved)  # True
```

Only the 18 standard outer face turns are accepted: `U R F D L B`, with optional `'` or `2` suffixes. A move sequence may be passed as a whitespace-delimited string or a sequence of tokens. The move-count metric is HTM: each token counts as one move, including a half turn such as `R2`. Scrambles use Python's seeded PRNG, have a caller-supplied length (25 by default), and avoid consecutive turns of the same face. Reusing a seed and length reproduces the same sequence.

`CubeState.from_facelets()` validates the 54-character format, allowed face letters, nine stickers per face, and fixed centers. It does not verify every physical reachability constraint on arbitrary externally supplied states. States created by the engine through legal face turns remain reachable by construction.

## Model adapter contract and offline mocks

`ModelAdapter` defines one async `choose_move(TurnContext)` call returning a validated `MoveProposal`. The context carries the cube's canonical serialized state, recent moves, the legal move set, and an optional timeout. `BaseModelAdapter` centralizes timing, timeout/error normalization, and exact move-token validation; future provider adapters can implement their own request and response parsing behind this contract. This leaves different OpenRouter API styles free to use different transports and payload formats.

`ScriptedMockAdapter` returns configured outcomes in order without network access. It can be used to drive a known sequence through a future workflow, or to exercise invalid-move, timeout, and exception handling. Mock moves are fixtures and do not model AI quality or real model latency; the elapsed time only measures local adapter code.

```python
import asyncio

from app.cube import CubeState
from app.models import ScriptedMockAdapter, TurnContext

adapter = ScriptedMockAdapter(["R", "U'"])
context = TurnContext(
    cube_state=CubeState().serialize(),
    recent_moves=(),
    timeout_seconds=10,
)
proposal = asyncio.run(adapter.choose_move(context))
print(proposal.move, proposal.model_id, proposal.elapsed_seconds)
```

Invalid move tokens raise `InvalidMoveError`; timeouts raise `AdapterTimeoutError`; other implementation failures raise `AdapterExecutionError`. Errors are surfaced to the caller so a later benchmark runner can record a failed turn and avoid applying an invalid action.

## OpenRouter adapters

The four assigned IDs and their endpoint formats are fixed as follows:

| Model | Exact OpenRouter ID | API format |
| --- | --- | --- |
| Jev 1.13 | `typesafe/jev-1.13` | Decisions API: `/api/alpha/decisions` |
| GPT-6 Luna Decisions | `openai/gpt-6-luna-decisions` | Decisions API: `/api/alpha/decisions` |
| Cloudflare Clef | `cloudflare/clef` | Decisions API: `/api/alpha/decisions` |
| Claude Opus 5.5 | `anthropic/claude-opus-5.5` | Chat Completions: `/api/v1/chat/completions` |

The Decisions adapters send the serialized state and a `choice` question whose criteria are the allowed move tokens. Claude receives equivalent turn context in chat messages and a strict JSON-schema response format enumerated by the allowed tokens. All outputs still pass the shared move validator. Request construction and response parsing are kept separate for the two API styles. The current official references and model catalog checks are recorded in [docs/API_COMPATIBILITY.md](docs/API_COMPATIBILITY.md).

Adapters use explicit HTTPX timeouts (5 seconds connect, 25 seconds read, 10 seconds write, 5 seconds pool) in addition to `TurnContext.timeout_seconds` as the overall per-turn deadline. They do not retry automatically; rate limits surface as `OpenRouterRateLimitError` with optional `Retry-After` metadata. Adapters that create their own client can be closed with `async with` or `aclose()`; an injected HTTP client remains caller-owned.

The offline tests use `httpx.MockTransport` to verify request URLs, payloads, parsing, errors, and timeout handling. They do not establish model quality. A previous one-turn four-model smoke test received legal moves from the first three models; Claude returned an unusable content response. A later separate, one-request Claude diagnostic returned valid move `R'` in 3.367 seconds. This confirms one Claude response only and does not establish the cause of the earlier empty-content response or the results of a comparable full experiment.

```powershell
python -m pytest tests/test_openrouter_adapters.py
python -m pytest
```

### One-request Claude diagnostic

To manually diagnose Claude only, run this command from the project root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.claude_diagnostic --confirm-live
```

The command requires the `--confirm-live` opt-in and `OPENROUTER_API_KEY`. It sends exactly one request to `anthropic/claude-opus-5.5` using the existing Chat Completions adapter, a valid cube state scrambled by `R`, its structured move schema, and the existing request/turn timeouts. It does not retry, call other models, start a benchmark, or save the response. It prints the validated move on success, or safe adapter diagnostics (including finish reason and available token counts) on failure. **This live request may incur a provider charge.** Tests for this command inject `httpx.MockTransport`; they make no live requests.

## Exact short reference

`exact_shortest_solution(cube)` in `app.cube.optimal_solver` returns a `ShortestSolution` containing the exact `distance_htm`, one shortest move sequence, and the number of states explored. It uses bidirectional BFS over the engine's 18 legal moves. It searches only through depth 5 HTM, matching the primary benchmark's 4–5-turn scope; requests above that limit are rejected. A `None` result means no solution was found within the requested depth cap, not that the cube is unsolvable. Repeated `(54-facelet state, depth cap)` queries use a bounded in-memory cache.

Each side completes a BFS ball with radii whose sum is the requested depth cap. Every path within that cap must cross the two visited balls; among all shared states, the solver chooses the path with the smallest combined BFS depth. Thus, a returned result is globally shortest under HTM. Scramble length is only an upper bound: inverses or other cancellations can make the exact distance smaller.

```python
from app.cube import CubeState, exact_shortest_solution, generate_scramble

scramble = generate_scramble(seed=2026, length=5)
cube = scramble.apply_to()
reference = exact_shortest_solution(cube)
if reference is not None:
    print(reference.distance_htm, reference.notation)
```

## Benchmark runner and JSON results

`BenchmarkRunner.run_suite()` requires exactly the four assigned model IDs. For each puzzle it builds one canonical starting state and gives each model an independent `CubeState` reconstructed from that same serialization. The runner asks for one move at a time, applies only validated legal moves, checks solved status after every move, and stops at solved, the valid-move limit (30 by default), or the first invalid proposal, timeout, or adapter error. A failure ends that model's trial and does not stop other models. No move is invented after a failure. The current policy performs no retries; each trial records `retry_count: 0`.

This offline example runs the pinned five-puzzle dataset against deterministic scripts that undo their scrambles. It does not need credentials or contact OpenRouter:

```python
import asyncio

from app.benchmark import BenchmarkRunner, aggregate_metrics, save_run
from app.benchmark.primary_dataset import primary_puzzle_specs
from app.cube import invert_moves
from app.models import ASSIGNED_MODEL_IDS, ScriptedMockAdapter

puzzles = primary_puzzle_specs()
scripted_outcomes = [
    move
    for puzzle in puzzles
    for move in invert_moves(puzzle.scramble_moves)
]
adapters = [
    ScriptedMockAdapter(scripted_outcomes, model_id=model_id)
    for model_id in ASSIGNED_MODEL_IDS
]

async def main():
    run = await BenchmarkRunner().run_suite(puzzles, adapters, run_id="mock-primary-example")
    save_run(run, "results/mock-primary-example.json")
    for model_id, metrics in aggregate_metrics(run).items():
        print(model_id, metrics.to_dict())

asyncio.run(main())
```

`PuzzleSpec.from_seed()` uses the Phase 2 scramble generator. A `BenchmarkRun` has a deterministic configuration-derived ID unless the caller supplies one; each trial ID is `run_id:puzzle_id:model_id`. JSON schema v4 stores execution mode, prompt condition, action-preview flag, trial provenance, and the exact simulated Claude label when applicable, alongside run controls, each puzzle's seed/scramble/initial state, assigned model ID and served model snapshot, every request's state, move history, latency, outcome, sanitized error, and safe provider token counters when supplied; it also stores final state, counts, runtime, reference distance, and optimality gap. Schema v1–v3 files remain loadable without inferring a missing prompt condition and are not rewritten. `save_run()` creates missing parent directories and replaces the destination atomically. `load_run()` validates the schema and replays recorded actions to check state consistency. Generated files under `results/` and analysis output under `reports/` are ignored by Git.

### Metric definitions

- **Valid moves** count only accepted moves applied to the cube. **Attempted requests** count every adapter call, including an invalid response, timeout, or exception. These counts are stored separately; `failed_request_count` counts non-valid terminal attempts, while `invalid_proposal_count` and `timeout_count` identify two failure types. The rejected move token is stored (limited to 160 characters); arbitrary exception details are omitted unless the adapter error already provides a sanitized message.
- **Per-move latency** is measured by the runner's monotonic clock around the full `choose_move()` call and shared move validation. It includes provider wait and response parsing for live adapters, and excludes cube application and the exact-reference search. `average_move_latency_seconds` averages valid responses only. `average_request_latency_seconds` also includes failed requests. No retry policy is active, so retries are zero and do not affect either average.
- **Total runtime** is measured separately for each model trial from its start through terminal status, including adapter waits and cube updates, but excluding exact-reference search performed before timing. Aggregate total runtime averages all trials, including failures and move-limit outcomes.
- **Solve rate** is solved trials divided by all trials for that model, returned as a fraction from 0 to 1 (multiply by 100 for a percentage). Reaching the move cap is unsuccessful; a move that solves on the final allowed turn is successful because the cube is checked immediately after applying it. Failures and move-limit trials remain in the solve-rate denominator.
- **Moves used** and the aggregate average valid-move count include all valid actions, even for trials that later fail or reach the cap. Invalid outputs never increment the move count.
- **Optimality gap** is `valid_move_count - exact_shortest_distance` only for solved trials with a known exact reference. Unsolved and failed trials, and trials without a short reference, have no gap and are excluded from the gap mean. The existing exact solver is called only for scramble sequences of at most five turns; extension scrambles have no exact reference.

The mock adapters only verify workflow and persistence. Their move quality, latency, and absent token usage do not represent the assigned AI models. The isolated Claude diagnostic succeeded once; a comparable primary experiment across the fixed dataset is still pending.

## Phase 9 experiment preparation and report

The proposed five-puzzle primary dataset and experimental protocol are recorded in [docs/PRIMARY_EXPERIMENT_PROTOCOL.md](docs/PRIMARY_EXPERIMENT_PROTOCOL.md). It fixes puzzle IDs, seeds, scrambles, and exact HTM references, and specifies the same starting facelets, 30-valid-move cap, 30-second turn timeout, sequential execution, and no retries for every exact assigned model ID. The maximum primary request count is 600 (5 puzzles × 4 models × 30); this is a ceiling, not a cost estimate.

After an explicitly approved live run, generate a report from persisted result JSON:

```powershell
.\.venv\Scripts\python.exe -m app.benchmark.report --results-dir results --output reports/primary-analysis.md
```

The report analyzes only a complete live run matching the pinned dataset and settings. It excludes mock runs, legacy records with unknown execution mode, isolated diagnostics, and non-comparable live smoke runs. If there is no eligible data, it clearly says analysis is pending. Use `--run-id <UUID>` to choose between multiple eligible runs; the tool does not silently pool reruns. Request/failure counts and safe token usage are taken from recorded attempts. Actual spend must be checked in OpenRouter account usage/activity; no cost is inferred, and the `$1` key limit is a safeguard rather than an estimate of the experiment's full cost.

## OpenRouter compatibility

The requested model IDs, catalog status as checked, documented endpoints, request/response shapes, and access/pricing caveats are recorded in [docs/API_COMPATIBILITY.md](docs/API_COMPATIBILITY.md). The model IDs are preserved as assigned. One Claude-only diagnostic succeeded; no primary benchmark was run during Phase 9 preparation.

## Project layout

```text
app/
  benchmark/       # runner, metrics, validated storage, pinned dataset, offline report
  cube/            # engine, seeded scramble generator, exact short-state reference
  models/           # common contract, offline mocks, and OpenRouter adapters
  static/           # responsive browser UI, dashboard, and facelet renderer
  config.py         # environment configuration and validation
  api/              # Typed request/response schemas and backend routes
  services/         # Background benchmark orchestration and run lifecycle
  main.py           # FastAPI app factory and local backend entry point
docs/
  API_COMPATIBILITY.md
  PRIMARY_EXPERIMENT_PROTOCOL.md
tests/
  test_primary_experiment.py
  test_benchmark_runner.py
  test_benchmark_storage.py
  test_cube_engine.py
  test_scramble.py
  test_optimal_solver.py
  test_model_adapters.py
  test_openrouter_adapters.py
  test_config.py
  test_health.py
  test_api.py
  test_frontend.py
  frontend/cube.test.mjs
.env.example
.gitignore
pyproject.toml
README.md
```

### Results and screen recording
<img width="911" height="380" alt="final result 2" src="https://github.com/user-attachments/assets/73b90966-311a-4578-a05c-67467e8d313e" />
<img width="944" height="322" alt="final result" src="https://github.com/user-attachments/assets/ee55f3f1-7458-4485-86fd-6e398b0681f2" />
