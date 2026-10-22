import { faceletsToFaces, isSolvedFacelets } from "./cube.mjs";

/** Validate and shape a verified backend reference solution for display. */
export function referencePlaybackSteps(result) {
  if (!result || result.solved !== true || !Array.isArray(result.solution_moves) || !Array.isArray(result.steps)) {
    throw new TypeError("reference solution must be marked solved and contain moves and replay steps");
  }
  if (result.move_count !== result.solution_moves.length || result.steps.length !== result.solution_moves.length) {
    throw new TypeError("reference solution move count does not match its replay");
  }
  faceletsToFaces(result.initial_state);
  faceletsToFaces(result.final_state);
  if (!isSolvedFacelets(result.final_state)) {
    throw new TypeError("reference solution final cube state is not solved");
  }

  const playback = [{ label: "Start", move: null, facelets: result.initial_state }];
  result.steps.forEach((step, index) => {
    if (!step || step.move !== result.solution_moves[index]) {
      throw new TypeError("reference solution replay move does not match its sequence");
    }
    faceletsToFaces(step.state_after);
    playback.push({ label: `After ${step.move}`, move: step.move, facelets: step.state_after });
  });
  const lastState = playback.at(-1).facelets;
  if (lastState !== result.final_state) {
    throw new TypeError("reference solution replay does not end at the verified final state");
  }
  return playback;
}
