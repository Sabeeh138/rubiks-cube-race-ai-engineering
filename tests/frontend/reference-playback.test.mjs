import assert from "node:assert/strict";
import test from "node:test";

import { isSolvedFacelets } from "../../app/static/cube.mjs";
import { referencePlaybackSteps } from "../../app/static/reference-playback.mjs";

const solved = "U".repeat(9) + "R".repeat(9) + "F".repeat(9) + "D".repeat(9) + "L".repeat(9) + "B".repeat(9);
const scrambled = "UUUUUUFFFUBBRRRRRRRRRFFDFFDDDBDDBDDBFFDLLLLLLLLLUBBUBB";

test("reference playback uses the returned move sequence and engine states", () => {
  const result = {
    solved: true,
    initial_state: scrambled,
    solution_moves: ["R'"],
    move_count: 1,
    steps: [{ move: "R'", state_after: solved }],
    final_state: solved,
  };
  assert.equal(isSolvedFacelets(result.final_state), true);
  assert.deepEqual(referencePlaybackSteps(result), [
    { label: "Start", move: null, facelets: scrambled },
    { label: "After R'", move: "R'", facelets: solved },
  ]);
});

test("reference playback rejects unverified, incomplete, or inconsistent results", () => {
  const valid = {
    solved: true,
    initial_state: solved,
    solution_moves: [],
    move_count: 0,
    steps: [],
    final_state: solved,
  };
  assert.deepEqual(referencePlaybackSteps(valid), [{ label: "Start", move: null, facelets: solved }]);
  assert.throws(() => referencePlaybackSteps({ ...valid, solved: false }), /marked solved/);
  assert.throws(() => referencePlaybackSteps({ ...valid, final_state: scrambled }), /not solved/);
  assert.throws(() => referencePlaybackSteps({ ...valid, move_count: 1 }), /move count/);
  assert.throws(() => referencePlaybackSteps({ ...valid, solution_moves: ["R"], move_count: 1, steps: [{ move: "U", state_after: solved }] }), /does not match/);
});
