"""Versioned move-selection instructions for controlled cube experiments."""

RUBIKS_MOVE_OBJECTIVE_V1 = "rubiks-move-objective-v1"
RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE = "rubiks-move-objective-v2-state-aware"

PROMPT_V1_INSTRUCTIONS = (
    "Given the current cube state and recent moves, choose exactly one move "
    "from the listed legal moves that best advances the cube toward the fully "
    "solved state. Return only that move in the required format."
)

PROMPT_V2_INSTRUCTIONS = (
    "Given the current Rubik's Cube state, original scramble, full applied move "
    "history, and (when provided) the resulting state of each listed legal move, "
    "choose exactly one listed legal move that best advances the entire cube toward the fully "
    "solved target. Consider all six faces and the whole cube, not one face in "
    "isolation. Avoid pointless repetition or reversal of recent moves, while "
    "allowing any legal move that genuinely helps the current state. Return only "
    "one legal move in the API's required choice or JSON format."
)

SUPPORTED_PROMPT_VERSIONS = frozenset(
    {RUBIKS_MOVE_OBJECTIVE_V1, RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE}
)


def instructions_for_prompt(prompt_version: str) -> str:
    """Return the frozen instruction for a known experimental condition."""

    if prompt_version == RUBIKS_MOVE_OBJECTIVE_V1:
        return PROMPT_V1_INSTRUCTIONS
    if prompt_version == RUBIKS_MOVE_OBJECTIVE_V2_STATE_AWARE:
        return PROMPT_V2_INSTRUCTIONS
    raise ValueError(f"unsupported prompt version {prompt_version!r}")
