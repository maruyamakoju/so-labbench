# PR body draft

**Title:** Fix infinite reset loop in `record_loop` when no teleoperator is provided (root cause of #2597)

## Context

This was reported in #2597 (the reporter correctly suggested updating the timestamp). #3042 rate-limited the warning message, but the underlying non-terminating loop is still present on `main`: with a policy and no teleoperator (the workflow of `examples/evaluate.py` / headless real-robot evaluation), the reset phase never ends. This PR fixes the root cause.

## What this does

`record_loop()` updates its loop-exit variable `timestamp` only at the bottom of the `while` body. The no-teleop branch ends with `continue`, which skips both the `precise_sleep()` call and the `timestamp` update, so when `lerobot-record` runs with a policy and no teleoperator (real-robot policy evaluation), the reset phase:

- never terminates (`timestamp` stays 0 forever instead of respecting `reset_time_s`), and
- busy-spins at 100% CPU while emitting the "No teleoperator provided" warning every 10 iterations.

This PR updates the timestamp and keeps loop pacing before the `continue`.

## How it was tested

- Reproduced on a real SO-101 follower (Windows 11, ACT & SmolVLA policies, `reset_time_s=10`): before the fix the process stayed inside the first reset phase for >50 minutes until killed; after applying the equivalent fix, the same command records all episodes and exits normally.
- Interactive keyboard use (`exit_early` via right arrow) still works: the branch behavior is unchanged apart from time bookkeeping.

(If maintainers prefer, `busy_wait`/`precise_sleep` could be dropped from the branch to keep the reset phase CPU-light in a different way — happy to adjust.)
