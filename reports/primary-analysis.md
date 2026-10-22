# Rubik’s Cube Race: Primary Experiment Analysis

- Dataset: `rubiks-cube-race-primary-v1`
- Analysis status: **PENDING**
- Selected run: `none`
- Persisted result files scanned: 7

Assignment analysis is pending. Only a completed live run matching all five pinned puzzles, all four exact model IDs, a 30-valid-move cap, and a 30-second turn timeout is eligible.
Reason: `no_eligible_live_primary_run_is_persisted`.
Mock demonstrations, legacy runs without a saved execution mode, and one-request diagnostics are not treated as model benchmark measurements.

## Usage and cost audit

Per-attempt request counts and failure outcomes come from persisted trial records. Provider-reported token counters are retained only when the response supplies safe non-negative integer fields; absent usage is missing data, not zero. Different counter names are kept separate and are not inferred or combined.

Actual spend is not calculated from token counts or catalog prices. Verify it in OpenRouter account usage/activity for the experiment time window. The $1 API key limit is a safeguard, not an estimate that the full experiment will cost $1 or less.

The report excludes mocks and unknown-mode records. A legacy result cannot be classified as live from its puzzle name or model IDs alone.

## Result files considered

| File | Classification | Reason |
| --- | --- | --- |
| `10bbac46-4650-4741-aae2-a679a25bf685.json` | excluded_unknown_mode | execution_mode_was_not_persisted |
| `22cf3859-d140-400f-8b72-2b8e45c24353.json` | excluded_unknown_mode | execution_mode_was_not_persisted |
| `5fab96aa-0724-47c2-966c-6a37df30b89f.json` | excluded_unknown_mode | execution_mode_was_not_persisted |
| `63acc67c-a4c3-4582-988b-0f3d5fce6b15.json` | excluded_unknown_mode | execution_mode_was_not_persisted |
| `8a619d23-2f33-4072-ada6-0ce3ce4f00fc.json` | excluded_unknown_mode | execution_mode_was_not_persisted |
| `badb4c34-fbd8-4c18-976a-fb9421f24485.json` | excluded_unknown_mode | execution_mode_was_not_persisted |
| `e25c8426-cbde-4588-a641-f119643bd82a.json` | excluded_unknown_mode | execution_mode_was_not_persisted |
