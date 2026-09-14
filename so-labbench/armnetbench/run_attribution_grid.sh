#!/bin/sh
# The preregistered input-attribution run: eight tasks, both policies, stride 50, resumable.
#
#   sh run_attribution_grid.sh
#
# Preregistered in so-labbench/prereg_input_attribution.md. Verdicts come only from
# judge_input_attribution.py, after all eight tasks are written.
HERE="$(dirname "$0")"
PY="${LABBENCH_PYTHON:-$(python "$HERE/../harness/labbench.py" --get paths.env_python 2>/dev/null)}"
if [ ! -x "$PY" ]; then
  echo "No interpreter. Set LABBENCH_PYTHON to a python with lerobot >= 0.5."
  exit 1
fi
if ! "$PY" -c "from lerobot.policies.factory import get_policy_class, make_pre_post_processors" 2>/dev/null; then
  echo "$PY cannot load the published checkpoints. Set LABBENCH_PYTHON."
  exit 1
fi
echo "interpreter: $PY  ($("$PY" -c "import lerobot; print('lerobot', lerobot.__version__)"))"
mkdir -p "$HERE/attribution"
EXPECTED=18      # 9 perturbations x 2 policies; a file with fewer is a cell that died mid-run
for TASK in eye_drops_to_basket cable_clip block_stack ring_insert tool_removal tool_insert cable_unclip eye_drops_to_shelf; do
  OUT="$HERE/attribution/${TASK}.csv"
  # Input_attribution writes after each policy, so a crash during the second leaves a file with
  # only the first. Existence is not completion; the row count is.
  if [ -s "$OUT" ] && [ "$(($(wc -l < "$OUT") - 1))" -ge "$EXPECTED" ]; then
    echo "skip $TASK (complete)"; continue
  fi
  echo "=== $TASK  $(date +%H:%M)"
  "$PY" "$HERE/input_attribution.py" --task "$TASK" --policies act smolvla --stride 50 --out "$OUT" \
    > "$HERE/attribution/${TASK}.out" 2>&1
  ROWS=$(( $(wc -l < "$OUT" 2>/dev/null || echo 1) - 1 ))
  if [ "$ROWS" -ge "$EXPECTED" ]; then echo "    done $(date +%H:%M)"; else echo "    FAILED ($ROWS rows) - see attribution/${TASK}.out"; fi
done
echo "attribution grid complete $(date +%H:%M)"
