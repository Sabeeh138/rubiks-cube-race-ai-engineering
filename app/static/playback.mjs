/** Build replay steps from the backend's persisted pre-turn and final states. */
export function trialPlaybackSteps(trial) {
  if (!trial || typeof trial.initial_state !== "string" || !Array.isArray(trial.attempts)) {
    throw new TypeError("trial must contain an initial state and attempt list");
  }
  const steps = [{ label: "Start", move: null, facelets: trial.initial_state }];
  trial.attempts.forEach((attempt, index) => {
    if (attempt.outcome !== "valid") return;
    const following = trial.attempts[index + 1];
    const facelets = following ? following.cube_state_before : trial.final_state;
    if (typeof facelets !== "string") throw new TypeError("recorded move state is missing");
    steps.push({ label: `After ${attempt.move}`, move: attempt.move, facelets });
  });
  return steps;
}
