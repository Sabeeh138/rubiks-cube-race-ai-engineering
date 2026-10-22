import assert from "node:assert/strict";
import test from "node:test";

import { faceletsToFaces } from "../../app/static/cube.mjs";
import { trialPlaybackSteps } from "../../app/static/playback.mjs";

const start = "UUUUUUFFFUBBRRRRRRRRRFFDFFDDDBDDBDDBFFDLLLLLLLLLUBBUBB";
const afterUPrime = "UUFUUFUUFRRRRRRRRRFFDFFDFFDDDBDDBDDBLLLLLLLLLUBBUBBUBB";

test("playback uses persisted engine states for valid moves and omits terminal failures", () => {
  const trial = {
    initial_state: start,
    final_state: afterUPrime,
    attempts: [
      { outcome: "valid", move: "U'", cube_state_before: start },
      { outcome: "timeout", move: null, cube_state_before: afterUPrime },
    ],
  };

  const steps = trialPlaybackSteps(trial);
  assert.deepEqual(steps, [
    { label: "Start", move: null, facelets: start },
    { label: "After U'", move: "U'", facelets: afterUPrime },
  ]);
  for (const step of steps) faceletsToFaces(step.facelets);
});

test("playback renders only the initial state when no valid move was applied", () => {
  const trial = {
    initial_state: start,
    final_state: start,
    attempts: [{ outcome: "invalid_move", move: null, cube_state_before: start }],
  };

  assert.deepEqual(trialPlaybackSteps(trial), [{ label: "Start", move: null, facelets: start }]);
});

test("playback rejects missing state data instead of inventing a cube state", () => {
  assert.throws(() => trialPlaybackSteps({ attempts: [] }), /initial state and attempt list/);
  assert.throws(() => trialPlaybackSteps({ initial_state: start, attempts: [{ outcome: "valid", move: "R" }] }), /state is missing/);
});
