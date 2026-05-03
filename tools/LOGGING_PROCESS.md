# Logging Process For Local Generated Layout Tests

This process exists because visual observation and saved logs must agree before we use a run to debug the algorithm.

## Ground Rules

- Do not change the navigation algorithm while validating logs.
- Do not edit `controllers/main/main.py`, the Webots world, or the gate proto for logging changes.
- Run one visible Webots test at a time when visual feedback is needed.
- Treat a run as final only if the archived simulator log contains the final Webots gate progress.
- If a visible run is left open, the archive is only a snapshot, not a final result.

## Files Produced Per Run

The runner creates a unique folder:

`tools/generated_layout_logs/<run_id>/`

Inside it:

- `run_manifest.txt`: run settings, selected layouts, timestamps.
- `generated_layout_results.csv`: one row per generated layout.
- `layout_XXXX/layout_manifest.txt`: per-layout metadata.
- `layout_XXXX/sim_progress.log`: Webots gate truth and gate pass progress.
- `layout_XXXX/assignment_debug.log`: controller-side decisions, detections, estimates, and commands.
- `layout_XXXX/stdout.log` and `layout_XXXX/stderr.log`: process output streams.

The latest run results are also copied to:

`tools/generated_layout_results_latest.csv`

## Source Of Truth

For pass/fail, use `sim_progress.log`, not the controller's own opinion.

A local generated layout passes only if Webots records all 15 gate checks:

- 5 gates on lap 0,
- 5 gates on lap 1,
- 5 gates on lap 2,
- final `finished` or `debug_finished` event.

The assignment debug log is for explaining why the controller behaved that way. It is not the scoring truth.

## How To Run One Visible Layout

Use this only after explicit approval:

```powershell
powershell -ExecutionPolicy Bypass -File tools\run_generated_layout_tests.ps1 -StartLayout 0 -Count 1 -Mode realtime -Visible -DebugController -TimeoutSeconds 260
```

For visual feedback, the expected workflow is:

1. Run exactly one visible layout.
2. The observer says what happened visually.
3. Compare that visual report with `layout_XXXX/sim_progress.log`.
4. If they disagree, fix the logging process before touching navigation code.
5. If they agree and the run failed, classify the failure using `COURSE_ALIGNMENT_REPORT.md`.

## What The Assignment Debug Log Explains

`assignment_debug.log` should answer:

- which run/layout produced the log,
- what gate the controller believed it was working on,
- what phase/stage it was in,
- what the camera detected,
- why a detection was rejected,
- how each gate estimate was accepted or rejected,
- which direction the controller chose to fly through a gate,
- when the controller thought it completed a gate or lap,
- what command it sent to the PID controller.

## Important Failure Mode Avoided

The old runner could copy `sim_progress.log` while visible Webots was still running. That made the archive look like a failed run even though Webots continued and later passed more gates.

The new rule is:

- final run: Webots is finished or stopped by the runner, then logs are archived,
- left-open run: archived logs are marked as a non-final snapshot.
