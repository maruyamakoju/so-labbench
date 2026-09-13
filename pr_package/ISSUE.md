# Issue draft — 更新: 新規Issueは不要
# 既存Issue #2597 が同一バグ（closed、修正はログ抑制のみ=PR #3042）。
# 新規Issueを立てず、PR本文で #2597 を参照する方針に変更（PR_BODY.md参照）。
# 以下は当初ドラフト（PR本文の素材として保持）。

# Issue draft (post to https://github.com/huggingface/lerobot/issues)

**Title:** `record_loop` never exits the reset phase when no teleoperator is provided (infinite busy-spin loop)

## Describe the bug

In `src/lerobot/scripts/lerobot_record.py`, `record_loop()` runs `while timestamp < control_time_s:` and updates `timestamp` only at the **bottom** of the loop body. The no-teleop branch:

```python
else:
    no_action_count += 1
    if no_action_count == 1 or no_action_count % 10 == 0:
        logging.warning("No teleoperator provided, skipping action generation. ...")
    continue
```

uses `continue`, which skips both `precise_sleep(...)` **and** the `timestamp = time.perf_counter() - start_episode_t` update. `timestamp` therefore stays `0` forever and the loop:

1. never terminates (the reset phase runs indefinitely instead of `reset_time_s`), and
2. busy-spins at 100% CPU on one core (no sleep), emitting the warning every 10 iterations.

This is hit whenever `lerobot-record` is used **with a policy and no teleoperator** (i.e. real-robot policy evaluation runs) and `reset_time_s > 0`: after episode 0 the reset phase never ends. Interactive users can mask the bug by pressing the right-arrow key (`exit_early`), but in headless/automated runs the process hangs forever.

## Reproduction

Windows 11 / lerobot with a SO-101 follower + one OpenCV camera, but the bug is platform-independent and visible from the code path alone:

```
lerobot-record \
  --robot.type=so101_follower --robot.port=<PORT> --robot.id=<ID> \
  --robot.cameras="{fixed: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=<user>/eval_test --dataset.single_task="..." \
  --policy.path=<trained_policy> \
  --dataset.num_episodes=3 --dataset.episode_time_s=60 --dataset.reset_time_s=10 \
  --dataset.push_to_hub=false
```

Observed (real log from our run, lerobot 0.3.4-era code with the identical control flow):

```
INFO 2026-08-12 09:58:55 Recording episode 0
INFO 2026-08-12 09:59:55 Reset the environment
INFO 2026-08-12 10:06:53 ... "No policy or teleoperator provided, skipping action generation." (still looping)
INFO 2026-08-12 10:49:xx ... (process killed manually after >50 min inside a 10 s reset phase)
```

After applying the fix below, the same command completes all 3 episodes and exits normally (verified on the real robot; episodes at 10:10:18 / 10:12:13 / 10:14:06, "Stop recording" at 10:15:51).

## Fix

Update the timestamp (and keep loop pacing) before `continue`:

```python
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(...)
            precise_sleep(control_interval)
            timestamp = time.perf_counter() - start_episode_t
            continue
```

Patch attached / PR opened.

## Environment
- lerobot: reproduced on 0.3.4-derived code; the identical control flow exists on current `main` (`src/lerobot/scripts/lerobot_record.py`, `record_loop`)
- OS: Windows 11 (bug is OS-independent)
- Hardware: SO-101 follower (Feetech STS3215), 1x OpenCV camera, policy = ACT / SmolVLA
