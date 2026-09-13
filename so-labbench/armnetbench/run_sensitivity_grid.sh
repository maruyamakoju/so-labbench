#!/bin/sh
# The whole ArmnetBench single-arm grid, one cell at a time, resumable.
#
#   sh run_sensitivity_grid.sh act          # ~18 min per task, 8 tasks
#   sh run_sensitivity_grid.sh smolvla      # ~70 min per task
#
# A cell that already has a CSV is skipped, so an interrupted run picks up where it stopped.
HERE="$(dirname "$0")"
# These analyses do not touch a robot; what they need is a LeRobot new enough to load the
# published checkpoints. Naming the interpreter through the active config once sent this
# whole loop at the OLD environment, where it died on an import eight times in a row and
# left eight empty cells - so the interpreter is checked before the first task, not after.
PY="${LABBENCH_PYTHON:-$(python "$HERE/../harness/labbench.py" --get paths.env_python 2>/dev/null)}"
if [ ! -x "$PY" ]; then
  echo "No interpreter. Set LABBENCH_PYTHON to a python with lerobot >= 0.5, or point"
  echo "LABBENCH_CONFIG at a config whose paths.env_python is one."
  exit 1
fi
if ! "$PY" -c "from lerobot.policies.factory import get_policy_class, make_pre_post_processors" 2>/dev/null; then
  echo "$PY cannot load the published checkpoints:"
  "$PY" -c "from lerobot.policies.factory import make_pre_post_processors" 2>&1 | tail -2
  echo
  echo "These analyses need lerobot >= 0.5. Set LABBENCH_PYTHON to that environment's python."
  exit 1
fi
echo "interpreter: $PY  ($("$PY" -c "import lerobot; print('lerobot', lerobot.__version__)"))"
POLICY="${1:-act}"
STRIDE="${2:-25}"
mkdir -p "$HERE/grid"
for TASK in eye_drops_to_basket cable_clip block_stack ring_insert tool_removal tool_insert cable_unclip eye_drops_to_shelf; do
  OUT="$HERE/grid/${TASK}__${POLICY}.csv"
  if [ -s "$OUT" ]; then echo "skip $TASK/$POLICY (already have $(basename "$OUT"))"; continue; fi
  echo "=== $TASK / $POLICY  $(date +%H:%M)"
  "$PY" "$HERE/rig_sensitivity.py" --task "$TASK" --policies "$POLICY" --stride "$STRIDE" --out "$OUT" \
    > "$HERE/grid/${TASK}__${POLICY}.out" 2>&1
  if [ -s "$OUT" ]; then echo "    done $(date +%H:%M)"; else echo "    FAILED - see grid/${TASK}__${POLICY}.out"; fi
done
echo "grid pass for $POLICY complete $(date +%H:%M)"
