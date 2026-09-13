# Archive

Nothing here is part of the live bench. It is kept because deleting the record of how a
result was reached is worse than carrying a folder, and because git history is easier to
search when the file still has a name.

Two groups.

## Superseded, and unsafe to run against the current bench

| File | Superseded by | Why it must not be run |
|---|---|---|
| `score_trials.py` | `score_rq1.py` | Scores with a threshold in **frames**, the defect v1.3 exists to fix, and writes `<prefix>scores.csv` - the same filename the live scorer writes, with an incompatible schema. Running it silently replaces a real result with a differently-defined one. |
| `run_trials.ps1` | `run_interleaved.ps1` | Writes **no manifest row**, so its trials are invisible to every analysis tool and get scored at an assumed episode length. Its default task text is the tube transfer that was never recorded. |
| `make_review_sheet.py` | `make_filmstrip.py` | A single final frame cannot show whether a grasp held. Worse, a failed ffmpeg call left the **previous run's thumbnail** in place and the sheet presented it as current - a silent error in an artifact used for human judgement. |
| `ping_motors.py` | `health_motors.py` | A strict subset: `health_motors.py` pings and also reports voltage, temperature and error flags. |

## Finished work, kept as evidence

| File | What it established |
|---|---|
| `window_selection_check.py` | The measurement behind scorer v1.4. Compared three ways of choosing the grasp window on the demonstrations and showed the "longest closure" rule was picking up carried-over gripper state. Cited by name in `score_rq1.py`. |
| `calibrate_from_demos.py` | Checked the position thresholds (closure 40, lift 10, descend 15, travel 25) against the demonstrations' own distributions. Its windows were chosen by the pre-v1.4 rule, so its numbers should not be re-quoted without re-running it. |
| `plot_traces.py` | Plotted gripper and lift traces when the summary statistics disagreed with expectation. It highlights the longest closure, which is no longer the window the scorer uses. |
| `baseline_waypoints.py` | The protocol's scripted non-AI baseline: record waypoints by hand, replay them N times. Never executed - no waypoint file was ever saved. |
| `check_scene.py` | Red-cap detection and zone occupancy for the closed-loop condition C3, against the tube-transfer task that was never recorded. The current task's object is a blue carton, so the colour test does not transfer. |

To bring one back, move it out and give it the same treatment the live scripts got: paths
and scoring constants from `labbench.py`, and a test in `test_harness.py`.
