import { faceletsToFaces } from "./cube.mjs";
import { trialPlaybackSteps } from "./playback.mjs";
import { referencePlaybackSteps } from "./reference-playback.mjs";

const MODEL_COLORS = Object.freeze({ U: "White", R: "Red", F: "Green", D: "Yellow", L: "Orange", B: "Blue" });
const DEFAULT_NAME = "Student name not configured";
const DEFAULT_ROLL = "Roll number not configured";
const POLL_INTERVAL_MS = 650;

const elements = Object.freeze({
  form: document.querySelector("#benchmark-form"),
  seed: document.querySelector("#seed-input"),
  count: document.querySelector("#count-input"),
  length: document.querySelector("#length-input"),
  executionMode: document.querySelector("#execution-mode"),
  actionPreviewControl: document.querySelector("#action-preview-control"),
  actionPreviews: document.querySelector("#action-previews-input"),
  experimentWarning: document.querySelector("#experiment-warning"),
  liveUsageConfirmed: document.querySelector("#live-usage-confirmed"),
  modeStamp: document.querySelector("#mode-stamp"),
  modeStampLabel: document.querySelector("#mode-stamp-label"),
  modeStampDetail: document.querySelector("#mode-stamp-detail"),
  modeNoticeCopy: document.querySelector("#mode-notice-copy"),
  lengthValue: document.querySelector("#length-value"),
  setupSummary: document.querySelector("#setup-summary"),
  runButton: document.querySelector("#run-button"),
  previewButton: document.querySelector("#preview-button"),
  retryStatusButton: document.querySelector("#retry-status-button"),
  studentName: document.querySelector("#student-name"),
  studentRoll: document.querySelector("#student-roll"),
  referenceButton: document.querySelector("#reference-button"),
  referenceSection: document.querySelector("#reference-section"),
  referenceStatus: document.querySelector("#reference-status"),
  referenceIdentity: document.querySelector("#reference-identity"),
  referenceMoves: document.querySelector("#reference-moves"),
  referenceMoveCount: document.querySelector("#reference-move-count"),
  referenceConfirmation: document.querySelector("#reference-confirmation"),
  referenceProgress: document.querySelector("#reference-progress"),
  cubeNet: document.querySelector("#cube-net"),
  cubeStateTag: document.querySelector("#cube-state-tag"),
  cubePuzzleId: document.querySelector("#cube-puzzle-id"),
  cubeScramble: document.querySelector("#cube-scramble"),
  statusBadge: document.querySelector("#run-status-badge"),
  progress: document.querySelector("#run-progress"),
  progressLabel: document.querySelector("#progress-label"),
  progressCount: document.querySelector("#progress-count"),
  currentPuzzle: document.querySelector("#current-puzzle-label"),
  runId: document.querySelector("#run-id-label"),
  error: document.querySelector("#error-message"),
  resultsSection: document.querySelector("#results-section"),
  resultsSubtitle: document.querySelector("#results-subtitle"),
  puzzlePicker: document.querySelector("#puzzle-picker"),
  solveChart: document.querySelector("#solve-chart"),
  metricsScope: document.querySelector("#metrics-scope"),
  metricsBody: document.querySelector("#metrics-body"),
  selectedPuzzleHeading: document.querySelector("#selected-puzzle-heading"),
  selectedPuzzleScramble: document.querySelector("#selected-puzzle-scramble"),
  liveTrialSection: document.querySelector("#live-trial-section"),
  liveTrialHeading: document.querySelector("#live-trial-heading"),
  liveTrialCards: document.querySelector("#live-trial-cards"),
  simulatedTrialSection: document.querySelector("#simulated-trial-section"),
  simulatedTrialCard: document.querySelector("#simulated-trial-card"),
});

let isRunning = false;
let activeRunId = null;
let resultsData = null;
let pendingSubmission = null;
let currentPuzzleRequest = null;
let primaryDataset = null;
let studentDisplay = { name: DEFAULT_NAME, roll: DEFAULT_ROLL };

class ApiError extends Error {
  constructor(status, detail) {
    super(formatApiDetail(detail));
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function requestJson(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      ...options,
      headers: { Accept: "application/json", ...(options.headers || {}) },
    });
  } catch (error) {
    throw new Error("The backend could not be reached. Check that the server is running, then retry.", { cause: error });
  }

  let payload = null;
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    try {
      payload = await response.json();
    } catch {
      throw new Error("The backend returned malformed JSON. Please retry or check the server logs.");
    }
  }
  if (!response.ok) {
    throw new ApiError(response.status, payload?.detail ?? `HTTP ${response.status}`);
  }
  if (payload === null) throw new Error("The backend returned an empty response.");
  return payload;
}

function formatApiDetail(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      const location = Array.isArray(item.loc) ? item.loc.filter((part) => part !== "body").join(" → ") : "Request";
      return `${location ? `${location}: ` : ""}${item.msg || "Invalid value"}`;
    }).join("; ");
  }
  if (detail && typeof detail === "object") {
    if (detail.active_run_id) return `Another benchmark is running (run ${detail.active_run_id}). Wait for it to finish before starting a new one.`;
    if (detail.message) return detail.message;
    return JSON.stringify(detail);
  }
  return "The backend rejected the request.";
}

function setError(message, { activeRunLink = null } = {}) {
  elements.error.replaceChildren(document.createTextNode(message));
  if (activeRunLink) {
    elements.error.append(document.createTextNode(" "));
    const link = document.createElement("a");
    link.href = activeRunLink;
    link.textContent = "Open active run status";
    elements.error.append(link);
  }
  elements.error.hidden = false;
}

function clearError() {
  elements.error.hidden = true;
  elements.error.replaceChildren();
}

function setFormBusy(busy) {
  isRunning = busy;
  elements.seed.disabled = busy;
  elements.count.disabled = busy;
  elements.length.disabled = busy;
  elements.executionMode.disabled = busy;
  elements.actionPreviews.disabled = busy || elements.executionMode.value !== "diagnostic";
  elements.liveUsageConfirmed.disabled = busy;
  elements.previewButton.disabled = busy;
  elements.runButton.disabled = busy;
  if (busy) elements.runButton.querySelector("span").textContent = "Benchmark running…";
  else updateModeDisplay(elements.executionMode.value);
}

function setRetryPending() {
  isRunning = false;
  elements.seed.disabled = true;
  elements.count.disabled = true;
  elements.length.disabled = true;
  elements.executionMode.disabled = true;
  elements.actionPreviews.disabled = true;
  elements.liveUsageConfirmed.disabled = true;
  elements.previewButton.disabled = true;
  elements.runButton.disabled = false;
  elements.runButton.querySelector("span").textContent = "Retry same submission";
}

function updateModeDisplay(mode) {
  const settings = {
    mock: {
      title: "MOCK MODE",
      detail: "Offline · no model credits",
      button: "Run mock benchmark",
      description: ["Deterministic mock adapters demonstrate the workflow. ", "Mock results are not AI performance measurements."],
    },
    live: {
      title: "LIVE OPENROUTER",
      detail: "Four assigned models · provider charges may apply",
      button: "Run live benchmark",
      description: ["Live mode sends all four assigned models to OpenRouter. ", "Provider charges may apply; no request runs until you submit."],
    },
    recording: {
      title: "RECORDING MODE",
      detail: "Three live models · Claude scripted",
      button: "Run recording demo",
      description: ["Recording mode calls the three non-Claude models live. Claude uses the scripted demonstration, labeled ", "SIMULATED — NOT A LIVE CLAUDE RESPONSE", "; that trial is excluded from live metrics and official reports."],
    },
    diagnostic: {
      title: "STATE-AWARE DIAGNOSTIC · V2",
      detail: "R scramble · action previews optional · Claude scripted",
      button: "Run state-aware diagnostic",
      description: ["Fixed R scramble with the separate rubiks-move-objective-v2-state-aware condition. Each live model may make up to three valid moves; this is exploratory and excluded from official reports. Claude is scripted: ", "SIMULATED — NOT A LIVE CLAUDE RESPONSE", ". Live outcomes and errors remain genuine."],
    },
    state_aware_experiment: {
      title: "FULL STATE-AWARE · EXPLORATORY",
      detail: "Five pinned puzzles · three live + scripted Claude",
      button: "Start full exploratory experiment",
      description: ["Runs all five pinned puzzles using rubiks-move-objective-v2-state-aware and one-step previews. This separate exploratory condition is excluded from the official v1 report. Claude is scripted: ", "SIMULATED — NOT A LIVE CLAUDE RESPONSE", ". The three other models make genuine live requests."],
    },
  };
  const display = settings[mode] || settings.mock;
  elements.modeStamp.dataset.mode = mode;
  elements.modeStamp.setAttribute("aria-label", `Execution mode: ${mode}`);
  elements.modeStampLabel.textContent = display.title;
  elements.modeStampDetail.textContent = display.detail;
  elements.runButton.querySelector("span").textContent = display.button;
  const fixedPuzzleSet = ["diagnostic", "state_aware_experiment"].includes(mode);
  elements.seed.disabled = isRunning || fixedPuzzleSet;
  elements.count.disabled = isRunning || fixedPuzzleSet;
  elements.length.disabled = isRunning || fixedPuzzleSet;
  elements.actionPreviewControl.hidden = !["diagnostic", "state_aware_experiment"].includes(mode);
  if (mode === "state_aware_experiment") elements.actionPreviews.checked = true;
  elements.actionPreviews.disabled = isRunning || mode !== "diagnostic";
  elements.experimentWarning.hidden = mode !== "state_aware_experiment";
  elements.liveUsageConfirmed.disabled = isRunning || mode !== "state_aware_experiment";
  const description = display.description.map((part, index) => {
    if (["recording", "diagnostic", "state_aware_experiment"].includes(mode) && index === 1) {
      const marker = document.createElement("strong");
      marker.textContent = part;
      return marker;
    }
    return document.createTextNode(part);
  });
  elements.modeNoticeCopy.replaceChildren(...description);
}

function readPuzzleSettings() {
  const seed = Number(elements.seed.value);
  const count = Number(elements.count.value);
  const scrambleLength = Number(elements.length.value);
  if (!Number.isSafeInteger(seed)) throw new Error("Enter a whole-number seed within the supported range.");
  if (!Number.isInteger(count) || count < 1 || count > 5) throw new Error("Choose between one and five puzzles.");
  if (![4, 5].includes(scrambleLength)) throw new Error("Primary scrambles must be four or five moves long.");
  if (!Number.isSafeInteger(seed + count - 1)) throw new Error("The seed range exceeds the safe integer limit.");
  return { seed, count, scrambleLength };
}

function puzzleIdFor(seed, index) {
  return `primary-${seed}-${String(index + 1).padStart(2, "0")}`;
}

function puzzleRequests(settings, mode = elements.executionMode.value) {
  if (mode === "diagnostic") {
    return [{ puzzle_id: "exploratory-diagnostic-r", scramble_moves: ["R"] }];
  }
  if (mode === "state_aware_experiment") {
    if (!primaryDataset?.puzzles?.length) throw new Error("The pinned primary dataset could not be loaded. Reload the dashboard before starting.");
    return primaryDataset.puzzles.map(({ puzzle_id, seed, scramble_length }) => ({
      puzzle_id,
      seed,
      scramble_length,
    }));
  }
  return Array.from({ length: settings.count }, (_, index) => ({
    puzzle_id: puzzleIdFor(settings.seed, index),
    seed: settings.seed + index,
    scramble_length: settings.scrambleLength,
  }));
}

function updateSetupSummary() {
  const settings = readPuzzleSettings();
  const diagnostic = elements.executionMode.value === "diagnostic";
  const fullStateAware = elements.executionMode.value === "state_aware_experiment";
  const trials = diagnostic ? 4 : fullStateAware ? 20 : settings.count * 4;
  elements.lengthValue.textContent = `${settings.scrambleLength} HTM`;
  elements.setupSummary.lastElementChild.textContent = diagnostic
    ? `${trials} model cards · v2 state-aware condition · fixed R scramble · max 3 live moves (9 live requests) · action previews ${elements.actionPreviews.checked ? "on" : "off"}`
    : fullStateAware
      ? `${trials} model trials · five pinned puzzles · v2 plus one-step previews · max 30 moves · up to 450 live requests; Claude scripted`
    : `${trials} model trials · maximum 30 valid moves per trial`;
}

function renderCube(facelets, { label = "Initial cube state" } = {}) {
  const faces = faceletsToFaces(facelets);
  for (const faceElement of elements.cubeNet.querySelectorAll(".cube-face")) {
    const face = faceElement.dataset.face;
    const stickerContainer = faceElement.querySelector(".stickers");
    stickerContainer.replaceChildren();
    faces[face].forEach((color, index) => {
      const sticker = document.createElement("span");
      sticker.className = "sticker";
      sticker.dataset.color = color;
      sticker.setAttribute("aria-label", `${face} face, sticker ${index + 1}: ${MODEL_COLORS[color]}`);
      stickerContainer.append(sticker);
    });
  }
  elements.cubeNet.setAttribute("aria-label", `${label}; six faces shown in URFDLB layout`);
}

async function previewPuzzle(puzzle) {
  elements.cubeStateTag.textContent = "Loading";
  elements.cubeScramble.textContent = "Requesting actual facelets from the cube engine…";
  const preview = await requestJson("/api/cube/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(puzzle),
  });
  renderCube(preview.initial_state, { label: "Backend-generated initial cube state" });
  elements.cubePuzzleId.textContent = preview.puzzle_id;
  elements.cubeScramble.textContent = preview.scramble_moves.length ? preview.scramble_moves.join(" ") : "Already solved";
  elements.cubeStateTag.textContent = "Backend state";
  currentPuzzleRequest = { puzzle_id: preview.puzzle_id, scramble_moves: preview.scramble_moves };
  elements.referenceSection.hidden = true;
  return preview;
}

function setRunStatus(status, label, completed = 0, total = 0) {
  elements.statusBadge.textContent = statusLabel(status);
  elements.statusBadge.className = `run-status-badge status-${status}`;
  elements.progress.max = Math.max(total, 1);
  elements.progress.value = Math.min(completed, total || 0);
  elements.progressCount.textContent = `${completed} / ${total} trials`;
  elements.progressLabel.textContent = label;
}

function statusLabel(status) {
  return ({ idle: "Ready", running: "Running", completed: "Completed", failed: "Failed" })[status] || status;
}

function renderRunStatus(status) {
  const current = status.progress.current_puzzle_id;
  const label = current
    ? `Processing ${current}. Move histories appear after all trials finish.`
    : status.status === "completed" ? "All model trials are complete." : "The run has stopped.";
  setRunStatus(status.status, label, status.progress.completed_trials, status.progress.total_trials);
  elements.currentPuzzle.textContent = current ? `Current puzzle: ${current}` : "No puzzle is currently active";
  elements.runId.textContent = `Run ${status.run_id}`;
}

function setIdentity(config) {
  studentDisplay = {
    name: config.student_name?.trim() || DEFAULT_NAME,
    roll: config.student_roll_number?.trim() || DEFAULT_ROLL,
  };
  elements.studentName.textContent = studentDisplay.name;
  elements.studentRoll.textContent = studentDisplay.roll;
  elements.referenceIdentity.textContent = `Recording identity · ${studentDisplay.name} · ${studentDisplay.roll}`;
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

function formatSeconds(value, { milliseconds = false } = {}) {
  if (!Number.isFinite(value)) return "—";
  if (milliseconds) return `${(value * 1000).toFixed(value * 1000 < 10 ? 1 : 0)} ms`;
  if (value < 10) return `${value.toFixed(2)} s`;
  return `${value.toFixed(1)} s`;
}

function formatCount(value, digits = 1) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function createCell(text, className = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  return cell;
}

function renderMetrics(results) {
  const modelIds = [...new Set(results.trials.map((trial) => trial.model_id))];
  elements.metricsBody.replaceChildren();
  elements.solveChart.replaceChildren();
  for (const modelId of modelIds) {
    const metric = results.metrics[modelId];
    if (!metric) continue;

    const row = document.createElement("tr");
    row.append(
      createCell(modelId),
      createCell(formatPercent(metric.solve_rate)),
      createCell(formatSeconds(metric.average_move_latency_seconds, { milliseconds: true })),
      createCell(formatSeconds(metric.average_total_runtime_seconds)),
      createCell(formatCount(metric.average_valid_move_count)),
      createCell(metric.average_optimality_gap === null
        ? "—"
        : `${metric.average_optimality_gap.toFixed(2)} (${metric.optimality_gap_sample_count} solved)`),
    );
    elements.metricsBody.append(row);

    const chartRow = document.createElement("div");
    chartRow.className = "chart-row";
    const label = document.createElement("span");
    label.className = "chart-model";
    label.textContent = modelId;
    label.title = modelId;
    const track = document.createElement("span");
    track.className = "bar-track";
    const bar = document.createElement("span");
    bar.className = "bar-fill";
    const rate = Math.max(0, Math.min(1, metric.solve_rate));
    bar.style.width = `${rate * 100}%`;
    track.append(bar);
    const value = document.createElement("span");
    value.className = "chart-value";
    value.textContent = formatPercent(metric.solve_rate);
    value.setAttribute("aria-label", `${modelId} solve rate ${value.textContent}`);
    chartRow.append(label, track, value);
    elements.solveChart.append(chartRow);
  }
}

function appendMoveChips(parent, moves, emptyLabel = "No valid moves recorded") {
  parent.replaceChildren();
  if (!moves.length) {
    const empty = document.createElement("span");
    empty.className = "tiny-note";
    empty.textContent = emptyLabel;
    parent.append(empty);
    return;
  }
  for (const move of moves) {
    const chip = document.createElement("span");
    chip.className = "move-chip";
    chip.textContent = move;
    parent.append(chip);
  }
}

function statusClass(trial) {
  if (trial.solved) return "solved";
  if (trial.termination_status === "move_limit") return "unsolved";
  return "failed";
}

function renderTrialCard(trial) {
  const card = document.createElement("article");
  card.className = "trial-card";
  if (trial.provenance === "simulated") card.classList.add("simulated-trial");

  const heading = document.createElement("div");
  heading.className = "trial-card-top";
  const model = document.createElement("h4");
  model.textContent = trial.model_id;
  const badge = document.createElement("span");
  badge.className = `trial-state ${statusClass(trial)}`;
  badge.textContent = trial.solved ? "Solved" : trial.termination_status.replaceAll("_", " ");
  heading.append(model, badge);
  if (trial.provenance_label) {
    const provenance = document.createElement("p");
    provenance.className = "provenance-marker";
    provenance.textContent = trial.provenance_label;
    card.append(heading, provenance);
  } else {
    card.append(heading);
  }

  const meta = document.createElement("div");
  meta.className = "trial-meta";
  const lastMove = trial.move_history.at(-1) || "—";
  const gap = trial.optimality_gap === null ? "—" : `+${trial.optimality_gap}`;
  const cardMetrics = trial.provenance === "simulated"
    ? [["Scripted moves", String(trial.valid_move_count)], ["Model latency", "Not measured"], ["Use", "Demo only"]]
    : [["Last move", lastMove], ["Valid moves", String(trial.valid_move_count)], ["Optimality gap", gap]];
  for (const [label, value] of cardMetrics) {
    const item = document.createElement("div");
    const itemLabel = document.createElement("span");
    itemLabel.textContent = label;
    const itemValue = document.createElement("strong");
    itemValue.textContent = value;
    item.append(itemLabel, itemValue);
    meta.append(item);
  }

  const historyTitle = document.createElement("p");
  historyTitle.className = "history-label";
  historyTitle.textContent = "Recorded move history";
  const history = document.createElement("div");
  history.className = "move-history";
  appendMoveChips(history, trial.move_history);

  const playbackLabel = document.createElement("p");
  playbackLabel.className = "history-label";
  playbackLabel.textContent = "Replay recorded cube states";
  const playback = document.createElement("div");
  playback.className = "move-playback";
  const steps = trialPlaybackSteps(trial);
  steps.forEach((step, index) => {
    const button = document.createElement("button");
    button.className = "playback-step";
    button.type = "button";
    button.textContent = step.label;
    button.setAttribute("aria-label", `${step.label} cube state`);
    button.addEventListener("click", () => {
      renderCube(step.facelets, { label: `${trial.model_id} ${step.label} state for ${trial.puzzle_id}` });
      elements.cubePuzzleId.textContent = trial.puzzle_id;
      elements.cubeScramble.textContent = step.move
        ? `${trial.model_id} · recorded move ${step.move}`
        : `${trial.model_id} · initial state`;
      elements.cubeStateTag.textContent = `Replay ${index}/${steps.length - 1}`;
    });
    playback.append(button);
  });

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = trial.provenance === "simulated"
    ? "Scripted demonstration · no OpenRouter requests or model latency"
    : `Trial details · ${trial.attempted_request_count} requests · ${formatSeconds(trial.total_runtime_seconds)}`;
  const initialState = document.createElement("code");
  initialState.className = "facelet-details";
  initialState.textContent = `Start: ${trial.initial_state}`;
  const finalState = document.createElement("code");
  finalState.className = "facelet-details";
  finalState.textContent = `Final: ${trial.final_state}`;
  details.append(summary, initialState, finalState);
  const failure = trial.attempts.find((attempt) => attempt.outcome !== "valid");
  if (failure) {
    const error = document.createElement("p");
    error.className = "trial-error";
    error.textContent = `${failure.outcome.replaceAll("_", " ")}: ${failure.error_message || "No additional details"}`;
    details.append(error);
  }

  card.append(meta, historyTitle, history, playbackLabel, playback, details);
  return card;
}

function renderSelectedPuzzle(puzzleId) {
  if (!resultsData) return;
  const trials = resultsData.trials.filter((trial) => trial.puzzle_id === puzzleId);
  if (!trials.length) return;
  const first = trials[0];
  currentPuzzleRequest = { puzzle_id: first.puzzle_id, scramble_moves: first.scramble_moves };
  elements.referenceSection.hidden = true;
  renderCube(first.initial_state, { label: `Initial cube state for ${puzzleId}` });
  elements.cubePuzzleId.textContent = puzzleId;
  elements.cubeScramble.textContent = first.scramble_moves.length ? first.scramble_moves.join(" ") : "Already solved";
  elements.cubeStateTag.textContent = "Initial state";
  elements.selectedPuzzleHeading.textContent = puzzleId;
  appendMoveChips(elements.selectedPuzzleScramble, first.scramble_moves, "Already solved");
  const simulated = trials.filter((trial) => trial.provenance === "simulated");
  const liveOrRecorded = trials.filter((trial) => trial.provenance !== "simulated");
  const mode = resultsData.execution_mode;
  elements.liveTrialHeading.textContent = mode === "mock"
    ? "Mock adapter workflow results"
    : ["live", "recording", "diagnostic", "state_aware_experiment"].includes(mode)
      ? "Genuine live-model results"
      : "Recorded model results · provenance unknown";
  elements.liveTrialCards.replaceChildren(...liveOrRecorded.map(renderTrialCard));
  elements.simulatedTrialSection.hidden = simulated.length === 0;
  elements.simulatedTrialCard.replaceChildren(...simulated.map(renderTrialCard));
}

function renderResults(results) {
  resultsData = results;
  const puzzleIds = [...new Set(results.trials.map((trial) => trial.puzzle_id))];
  elements.puzzlePicker.replaceChildren();
  for (const puzzleId of puzzleIds) {
    const option = document.createElement("option");
    option.value = puzzleId;
    option.textContent = puzzleId;
    elements.puzzlePicker.append(option);
  }
  renderMetrics(results);
  const solved = results.trials.filter((trial) => trial.solved).length;
  const modeLabel = results.execution_mode === "mock"
    ? "MOCK DEMONSTRATION"
    : results.execution_mode === "live" ? "LIVE MODEL RESULTS"
      : results.execution_mode === "recording" ? "RECORDING DEMO · 3 LIVE MODELS + SCRIPTED CLAUDE"
        : results.execution_mode === "diagnostic" ? `STATE-AWARE DIAGNOSTIC · ${results.prompt_version || "prompt unknown"} · action previews ${results.action_previews_enabled ? "on" : "off"} · 3 LIVE MODELS + SCRIPTED CLAUDE`
          : results.execution_mode === "state_aware_experiment" ? `FULL STATE-AWARE EXPERIMENT — EXPLORATORY · ${results.prompt_version || "prompt unknown"} · action previews ${results.action_previews_enabled ? "on" : "off"}`
        : "EXECUTION MODE UNKNOWN";
  const liveTrials = results.trials.filter((trial) => trial.provenance === "live");
  const liveSolved = liveTrials.filter((trial) => trial.solved).length;
  const solvedLabel = ["recording", "diagnostic", "state_aware_experiment"].includes(results.execution_mode)
    ? `${liveSolved}/${liveTrials.length} live trials solved; Claude is simulated`
    : `${solved} solved`;
  elements.resultsSubtitle.textContent = `${modeLabel} · ${results.trials.length} trials · ${solvedLabel} · run ${results.run_id}`;
  elements.metricsScope.textContent = results.execution_mode === "diagnostic"
    ? "Exploratory state-aware v2 diagnostic only. Genuine live outcomes are shown; scripted Claude is excluded from live metrics. This prompt condition and any candidate previews are never included in official primary reports."
    : results.execution_mode === "state_aware_experiment"
      ? "Five-puzzle exploratory v2 condition. Metrics include only the three genuine live models; all five scripted Claude trials are excluded. These results are excluded from the frozen official v1 primary report. A live solve is counted only when the persisted cube state is verified solved."
    : results.execution_mode === "recording"
      ? "Live comparison includes the three OpenRouter trials only; simulated Claude is excluded. This recording is not an official primary run."
    : results.execution_mode === "live"
      ? "Live OpenRouter measurements only."
      : "Mock figures are workflow checks, not model scores.";
  updateModeDisplay(results.execution_mode || "mock");
  elements.resultsSection.hidden = false;
  if (puzzleIds.length) {
    elements.puzzlePicker.value = puzzleIds[0];
    renderSelectedPuzzle(puzzleIds[0]);
  }
}

async function pollRun(runId) {
  while (true) {
    let status;
    try {
      status = await requestJson(`/api/benchmark/${encodeURIComponent(runId)}`);
    } catch (error) {
      elements.retryStatusButton.hidden = false;
      setError(`Run ${runId} was accepted, but its status could not be checked: ${error.message}`);
      return;
    }
    renderRunStatus(status);
    if (status.status === "failed") {
      setFormBusy(false);
      activeRunId = null;
      elements.retryStatusButton.hidden = true;
      setError(status.error || "The benchmark failed before results were saved.");
      return;
    }
    if (status.status === "completed") {
      let results;
      try {
        results = await requestJson(`/api/benchmark/${encodeURIComponent(runId)}/results`);
      } catch (error) {
        elements.retryStatusButton.textContent = "Retry results fetch";
        elements.retryStatusButton.hidden = false;
        setError(`Run ${runId} is complete, but results could not be loaded: ${error.message}`);
        return;
      }
      renderResults(results);
      setFormBusy(false);
      activeRunId = null;
      elements.retryStatusButton.hidden = true;
      clearError();
      return;
    }
    await new Promise((resolve) => window.setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

function idempotencyKey() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `ui-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

async function sendPendingSubmission() {
  const response = await requestJson("/api/benchmark", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Idempotency-Key": pendingSubmission.key,
    },
    body: JSON.stringify(pendingSubmission.body),
  });
  activeRunId = response.run_id;
  elements.runId.textContent = `Run ${activeRunId}`;
  elements.currentPuzzle.textContent = "Preparing the first puzzle…";
  setRunStatus("running", "Benchmark accepted. Waiting for trial progress…", 0, pendingSubmission.body.puzzles.length * 4);
  pendingSubmission = null;
  await pollRun(activeRunId);
}

async function submitBenchmark(event) {
  event.preventDefault();
  if (isRunning && !pendingSubmission) return;
  clearError();

  if (pendingSubmission) {
    setFormBusy(true);
    elements.runButton.querySelector("span").textContent = "Retrying same request…";
    try {
      await sendPendingSubmission();
    } catch (error) {
      if (error instanceof ApiError) {
        pendingSubmission = null;
        setFormBusy(false);
        setError(error.message, error.status === 409 && error.detail?.active_run_id
          ? { activeRunLink: `/api/benchmark/${encodeURIComponent(error.detail.active_run_id)}` }
          : {});
      } else {
        setRetryPending();
        setError(`${error.message} The same request can be retried safely.`);
      }
    }
    return;
  }

  let settings;
  let puzzles;
  try {
    settings = readPuzzleSettings();
    puzzles = puzzleRequests(settings, elements.executionMode.value);
  } catch (error) {
    setError(error.message);
    return;
  }

  const expectedTrials = puzzles.length * 4;
  const diagnostic = elements.executionMode.value === "diagnostic";
  const fullStateAware = elements.executionMode.value === "state_aware_experiment";
  if (fullStateAware && !elements.liveUsageConfirmed.checked) {
    setError("Review the substantial API usage warning and check its acknowledgement before starting this live exploratory experiment.");
    return;
  }
  setFormBusy(true);
  elements.resultsSection.hidden = true;
  elements.referenceSection.hidden = true;
  elements.retryStatusButton.hidden = true;
  setRunStatus("running", "Fetching the selected initial state from the cube engine…", 0, expectedTrials);
  try {
    await previewPuzzle(puzzles[0]);
    pendingSubmission = {
      key: idempotencyKey(),
      body: {
        execution_mode: elements.executionMode.value,
        puzzles,
        max_valid_moves: diagnostic ? 3 : 30,
        turn_timeout_seconds: 30,
        action_previews_enabled: fullStateAware || (diagnostic && elements.actionPreviews.checked),
        live_usage_confirmed: fullStateAware && elements.liveUsageConfirmed.checked,
      },
    };
    await sendPendingSubmission();
  } catch (error) {
    if (error instanceof ApiError) {
      pendingSubmission = null;
      setFormBusy(false);
      const conflictRunId = error.detail?.active_run_id;
      setError(error.message, conflictRunId
        ? { activeRunLink: `/api/benchmark/${encodeURIComponent(conflictRunId)}` }
        : {});
      if (error.status === 409 && conflictRunId) {
        elements.currentPuzzle.textContent = `Another run is active: ${conflictRunId}`;
      }
      setRunStatus("failed", "The request was rejected; no new run was started.", 0, expectedTrials);
      return;
    }

    if (pendingSubmission) {
      setRetryPending();
      setError(`${error.message} Retrying uses the same idempotency key, so it will not create a duplicate run.`);
      return;
    }
    setFormBusy(false);
    setRunStatus("failed", "The puzzle could not be previewed.", 0, expectedTrials);
    setError(error.message);
  }
}

async function retryStatusCheck() {
  if (!activeRunId) return;
  clearError();
  elements.retryStatusButton.hidden = true;
  elements.retryStatusButton.textContent = "Resume status checks";
  await pollRun(activeRunId);
}

async function handlePreview() {
  if (isRunning) return;
  clearError();
  try {
    const settings = readPuzzleSettings();
    const puzzles = puzzleRequests(settings, elements.executionMode.value);
    await previewPuzzle(puzzles[0]);
    setRunStatus("idle", "Preview loaded from the backend cube engine.", 0, puzzles.length * 4);
  } catch (error) {
    setError(error.message);
    elements.cubeStateTag.textContent = "Preview error";
  }
}

async function playReferenceSolution() {
  if (elements.referenceButton.disabled) return;
  clearError();
  elements.referenceButton.disabled = true;
  elements.referenceSection.hidden = false;
  elements.referenceStatus.textContent = "Calculating";
  elements.referenceConfirmation.textContent = "Verification pending";
  elements.referenceConfirmation.classList.remove("reference-confirmed", "reference-failed");
  elements.referenceProgress.textContent = "Finding an exact solution within the five-move solver scope…";
  try {
    if (!currentPuzzleRequest) {
      const settings = readPuzzleSettings();
      await previewPuzzle(puzzleRequests(settings, elements.executionMode.value)[0]);
    }
    elements.referenceSection.hidden = false;
    const result = await requestJson("/api/cube/reference-solution", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentPuzzleRequest),
    });
    const steps = referencePlaybackSteps(result);
    elements.referenceMoves.textContent = result.solution_moves.length ? result.solution_moves.join(" ") : "Already solved";
    elements.referenceMoveCount.textContent = String(result.move_count);
    elements.referenceConfirmation.textContent = "Verified solution; cube reaches solved state";
    elements.referenceConfirmation.classList.add("reference-confirmed");
    elements.cubePuzzleId.textContent = result.puzzle_id;
    elements.cubeScramble.textContent = result.scramble_moves.length ? result.scramble_moves.join(" ") : "Already solved";

    renderCube(steps[0].facelets, { label: `Reference solver start state for ${result.puzzle_id}` });
    elements.cubeStateTag.textContent = "Reference start";
    for (const step of steps.slice(1)) {
      if (!window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
        await new Promise((resolve) => window.setTimeout(resolve, 480));
      }
      renderCube(step.facelets, { label: `Reference solver state after ${step.move}` });
      elements.cubeStateTag.textContent = `After ${step.move}`;
      elements.referenceProgress.textContent = `Applying verified move ${step.move} through the cube-engine replay…`;
    }
    renderCube(result.final_state, { label: `Solved cube verified by the backend cube engine for ${result.puzzle_id}` });
    elements.cubeStateTag.textContent = "Solved ✓";
    elements.referenceStatus.textContent = "Solved ✓";
    elements.referenceProgress.textContent = `Verified solved state after ${result.move_count} reference move${result.move_count === 1 ? "" : "s"}. Playback timing is visual only and is not model latency.`;
  } catch (error) {
    elements.referenceStatus.textContent = "Unavailable";
    elements.referenceConfirmation.textContent = "No verified solve displayed";
    elements.referenceConfirmation.classList.add("reference-failed");
    elements.referenceProgress.textContent = error.message;
    setError(error.message);
  } finally {
    elements.referenceButton.disabled = false;
  }
}

async function initialize() {
  elements.form.addEventListener("submit", submitBenchmark);
  elements.executionMode.addEventListener("change", () => {
    updateModeDisplay(elements.executionMode.value);
    try { updateSetupSummary(); } catch { /* Invalid settings are rejected on submit. */ }
    void handlePreview();
  });
  elements.actionPreviews.addEventListener("change", () => {
    try { updateSetupSummary(); } catch { /* Invalid settings are rejected on submit. */ }
  });
  elements.liveUsageConfirmed.addEventListener("change", () => {
    if (elements.executionMode.value === "state_aware_experiment") clearError();
  });
  elements.previewButton.addEventListener("click", handlePreview);
  elements.referenceButton.addEventListener("click", playReferenceSolution);
  elements.retryStatusButton.addEventListener("click", retryStatusCheck);
  elements.length.addEventListener("input", () => {
    elements.lengthValue.textContent = `${elements.length.value} HTM`;
    try { updateSetupSummary(); } catch { /* The form's error message is shown on submit. */ }
  });
  elements.count.addEventListener("change", () => {
    try { updateSetupSummary(); } catch { /* Invalid values are rejected on submit. */ }
  });
  elements.puzzlePicker.addEventListener("change", () => renderSelectedPuzzle(elements.puzzlePicker.value));

  try {
    updateSetupSummary();
    updateModeDisplay(elements.executionMode.value);
  } catch (error) {
    setError(error.message);
  }

  try {
    const config = await requestJson("/api/ui-config");
    setIdentity(config);
  } catch {
    setIdentity({});
  }
  try {
    primaryDataset = await requestJson("/api/benchmark/primary-dataset");
  } catch {
    primaryDataset = null;
  }
  await handlePreview();
}

initialize();
