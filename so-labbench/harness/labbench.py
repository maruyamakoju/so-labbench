# The one place this bench states a fact about itself.
#
# Two kinds of fact live here and they are kept apart on purpose:
#
#   * Facts about THIS MACHINE - dataset root, python, serial ports, camera indices. These
#     come from labbench_config.json, so someone reproducing the work edits one file rather
#     than hunting absolute paths through twenty. LABBENCH_CONFIG selects which config, so a
#     second study is a second config file rather than a fork of the scripts.
#
#   * DECISIONS that must not drift - what counts as a closed gripper, how long a grasp has
#     to hold, how a run id is spelled, where an episode's file is. These were duplicated
#     across a dozen scripts, and duplication is how a benchmark ends up measuring its
#     independent variable with one threshold and its dependent variable with another.
#     (It nearly happened: the demo quality audit called a closure stable at 12 frames while
#     the scorer called it stable at 0.5 s, which is 15 frames at the rate those demos
#     declare. No episode's verdict actually differed - checked, 0 of 60 - but nothing in
#     the code prevented it. Now one definition exists and it is stated in seconds.)
#
# What deliberately does NOT live here: anything measured from the machine at run time. The
# control rate is not a constant of this rig, so there is no CONTROL_HZ; there is
# true_hz(frames, seconds) and every caller measures it per trial.
import csv
import json
import math
import os
import re
from pathlib import Path

HERE = Path(__file__).parent
CONFIG_PATH = Path(os.environ.get("LABBENCH_CONFIG", HERE / "labbench_config.json"))


class ConfigError(RuntimeError):
    """Raised with the key that is missing, rather than handing back None for something
    downstream to trip over inside an unrelated script."""


try:
    CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
except FileNotFoundError as exc:
    raise ConfigError(f"no config at {CONFIG_PATH}. Set LABBENCH_CONFIG or create it.") from exc
except json.JSONDecodeError as exc:
    raise ConfigError(f"{CONFIG_PATH} is not valid JSON: {exc}") from exc


def _lookup(*keys, default=None, required=False):
    node = CFG
    for i, key in enumerate(keys):
        if not isinstance(node, dict) or key not in node:
            if required:
                raise ConfigError(f"{CONFIG_PATH.name} is missing {'.'.join(keys[:i + 1])}")
            return default
        node = node[key]
    return node


def _path(env_var, *keys, default=None, required=False):
    if os.environ.get(env_var):
        return Path(os.environ[env_var])
    value = _lookup(*keys, default=default, required=required)
    return Path(value) if isinstance(value, str) else None


def _value(env_var, *keys, default=None, required=False):
    if os.environ.get(env_var):
        return os.environ[env_var]
    return _lookup(*keys, default=default, required=required)


# ---------------------------------------------------------------------------- this machine

DATASETS = _path("LABBENCH_DATASETS", "paths", "hf_cache", required=True)
ENV_PYTHON = _path("LABBENCH_PYTHON", "paths", "env_python")
WORK_DIR = _path("LABBENCH_WORK", "paths", "work_dir")
FFMPEG = _path("LABBENCH_FFMPEG", "paths", "ffmpeg")

ROBOT_PORT = _value("LABBENCH_ROBOT_PORT", "robot", "port", required=True)
ROBOT_ID = _value("LABBENCH_ROBOT_ID", "robot", "id", default="my_follower")
LEADER_PORT = _value("LABBENCH_LEADER_PORT", "robot", "leader_port", default="")
LEADER_ID = _value("LABBENCH_LEADER_ID", "robot", "leader_id", default="my_leader")

# The camera the single-camera tools work on, by NAME in the active config: "fixed" for the
# RQ0/RQ1 rig, "front" / "top" / "wrist" for the ArmnetBench reproduction.
CAMERA_NAME = os.environ.get("LABBENCH_CAMERA", "fixed")
CAMERA_INDEX = int(_value("LABBENCH_CAMERA_INDEX", "cameras", CAMERA_NAME, "index_or_path", default=0))
FRAME_W = int(_value("LABBENCH_FRAME_W", "cameras", CAMERA_NAME, "width", default=640))
FRAME_H = int(_value("LABBENCH_FRAME_H", "cameras", CAMERA_NAME, "height", default=480))
CAMERA_NAMES = [name for name in _lookup("cameras", default={})
                if not name.startswith("_")
                and _lookup("cameras", name, "enabled", default=True)]

# Serial protocol for the Feetech bus. Register numbers, not policy: the same on every SO-101.
MOTOR_IDS = range(1, 7)
BAUD = 1_000_000
CENTER_TICK = 2047


class Reg:
    TORQUE_ENABLE = 40
    ACC = 41
    GOAL_POSITION = 42
    GOAL_SPEED = 46
    PRESENT_POSITION = 56
    PRESENT_VOLTAGE = 62
    PRESENT_TEMPERATURE = 63


TEMP_PAUSE_C = float(_value("LABBENCH_TEMP_PAUSE", "safety", "temp_pause_threshold_c", default=62))
TEMP_RESUME_C = float(_value("LABBENCH_TEMP_RESUME", "safety", "temp_resume_threshold_c", default=57))
MAX_CONNECT_RETRIES = int(_value("LABBENCH_MAX_RETRIES", "safety", "max_connect_retries", default=3))

TASK_NAME = _value("LABBENCH_TASK_NAME", "task", "name", default="pick_remote")
TASK_INSTRUCTION = _value("LABBENCH_TASK", "task", "instruction", default="Pick up the remote control")
EPISODE_SEC = float(_value("LABBENCH_EPISODE_SEC", "task", "episode_time_s", default=45))
RESET_SEC = float(_value("LABBENCH_RESET_SEC", "task", "reset_time_s", default=20))
DATASET_FPS = float(_value("LABBENCH_FPS", "dataset", "fps", default=30))
DATASET_NAMESPACE = _value("LABBENCH_NAMESPACE", "dataset", "namespace", default="maruo")


def policy_path(name: str) -> str:
    """A local checkpoint directory or a Hugging Face id, whichever the config names."""
    value = _lookup("policies", name)
    if value is None:
        known = [k for k in _lookup("policies", default={}) if not k.startswith("_")]
        raise ConfigError(f"no policy '{name}' in {CONFIG_PATH.name}; known: {', '.join(known)}")
    return value


def is_hub_id(path: str) -> bool:
    """"user/model" is a hub id; anything with a drive letter or a separator is on disk."""
    return "/" in path and "\\" not in path and ":" not in path


# ------------------------------------------------------------------ decisions that must not drift
#
# Stated in SECONDS, never in frames. The recording loop runs for a wall-clock duration and
# appends one frame per iteration, so a heavier policy produces fewer frames for the same
# episode; a threshold written as a frame count silently asks a slow policy to hold longer
# than a fast one. Use stable_frames()/hold_frames() with the trial's own measured rate.

CLOSE_T = float(_value("LABBENCH_CLOSE_T", "scoring", "grip_close_threshold", default=40.0))
STABLE_S = float(_value("LABBENCH_STABLE_S", "scoring", "stable_closure_s", default=0.5))
# HOLD_S is the length of THE GRASP, not the total time the gripper spent closed anywhere in
# the episode. A policy that closes for 0.6 s, drops the object, and closes again on nothing
# has not held it for 1.2 s. The window select_grasp_window() picks is what gets measured.
HOLD_S = float(_value("LABBENCH_HOLD_S", "scoring", "hold_closure_s", default=1.0))
LIFT_GAIN = float(_value("LABBENCH_LIFT_GAIN", "scoring", "lift_gain", default=10.0))
DESCEND_D = float(_value("LABBENCH_DESCEND_D", "scoring", "descend_drop", default=15.0))
DESCEND_FLOOR = float(_value("LABBENCH_DESCEND_FLOOR", "scoring", "descend_lift_threshold", default=-30.0))
MOVE_D = float(_value("LABBENCH_MOVE_D", "scoring", "move_total", default=25.0))
# Uncalibrated, and reached only by a trial that moved a lot without ever descending: the
# taxonomy resolves by stage first, so a policy that descends AND thrashes is a grasp
# failure, which says more. Measured reversal rates on real trials are 7-12/s, well above
# this, so anything that reaches the branch lands in it. Deliberately not tuned by guesswork
# - it is meant to be calibrated against human labels, and the pilot that would have done
# that never produced a trial in this branch.
INSTABILITY_REV_PER_S = float(_value("LABBENCH_INSTABILITY", "scoring", "instability_rev_per_s", default=2.7))

# The demonstrations carry no wall-clock duration, so any frame-based reading of them rests
# on an assumption. Stating it once, here, is the difference between an assumption and a
# scattering of magic numbers. 722 frames at the declared 30 fps is 24.1 s.
DEMO_ASSUMED_HZ = float(_value("LABBENCH_DEMO_HZ", "scoring", "demo_assumed_hz", default=30.0))


def true_hz(n_frames: int, episode_sec: float) -> float:
    """The rate this trial actually ran at. Not a constant of the rig: it fell from 23.8 to
    14.9 Hz inside one session when the room got dark and the camera stretched its exposure."""
    if episode_sec is None or episode_sec <= 0:
        raise ValueError(f"episode_sec must be positive, got {episode_sec!r}")
    return n_frames / episode_sec


def _frames_for(seconds: float, hz: float) -> int:
    # ceil, not round: a threshold of "at least 0.5 s" must not be met by 0.49 s because the
    # frame count happened to round down, and round() in Python is banker's rounding, which
    # makes the boundary behave differently at 23 Hz than at 31 Hz.
    return max(1, math.ceil(seconds * hz - 1e-9))


def stable_frames(hz: float) -> int:
    return _frames_for(STABLE_S, hz)


def hold_frames(hz: float) -> int:
    return _frames_for(HOLD_S, hz)


def closure_runs(grip, close_t: float = None):
    """Every stretch of closed gripper as (onset, length). One implementation, because there
    were four and one of them silently disagreed with the others."""
    close_t = CLOSE_T if close_t is None else close_t
    runs, start = [], None
    for i, value in enumerate(grip):
        closed = value < close_t
        if closed and start is None:
            start = i
        elif not closed and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(grip) - start))
    return runs


def lift_gain(lift, onset: int, length: int) -> float:
    """How far the arm rose during a closure, measured from its height at the closure's onset."""
    if length <= 0:
        return 0.0
    segment = lift[onset:onset + length]
    return float(max(segment) - lift[onset])


def select_grasp_window(runs, lift, min_frames: int, exclude_frame_zero: bool = False):
    """Which closure to call the grasp: of those long enough to be stable, the one that
    lifted the most. Returns (onset, length) or None.

    Not the longest one. 27 of the 60 demonstrations begin with the gripper still closed
    from the previous recording, and that idle stretch is usually the longest, so the older
    rule measured the lift over a window that was never a grasp. Scoring the demonstrations
    with this rule agrees with the independent quality audit on all 60.

    exclude_frame_zero drops a closure that was already in progress when recording started.
    Off by default: the largest-lift rule already ignores an idle carry-over, and turning it
    on would also discard a genuine grasp that happened to survive from the previous episode.
    """
    candidates = [(o, n) for o, n in runs if n >= min_frames and not (exclude_frame_zero and o == 0)]
    if not candidates:
        return None
    return max(candidates, key=lambda r: lift_gain(lift, *r))


def wilson_ci(k: int, n: int, z: float = 1.96):
    """Binomial interval that stays inside [0, 1] at the extremes, which is where these
    success rates live."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


# ---------------------------------------------------------------------------- run ids

RUN_ID_RE = re.compile(r"^eval_(?P<study>[^_]+)_(?P<condition>[^_]+)_(?P<model>.+)_(?P<index>\d+)$")


def run_id(study: str, condition: str, model: str, index: int) -> str:
    return f"eval_{study}_{condition}_{model}_{index}"


def run_prefix(study: str, condition: str, model: str = "") -> str:
    return f"eval_{study}_{condition}_" + (f"{model}_" if model else "")


def parse_run_id(value: str):
    """-> dict(study, condition, model, index), or None if it is not a run id. The model may
    contain underscores ("23complete"), the index never does, so the index anchors the parse."""
    m = RUN_ID_RE.match(value)
    if not m:
        return None
    return {"study": m["study"], "condition": m["condition"], "model": m["model"], "index": int(m["index"])}


def repo_id(run: str) -> str:
    return f"{DATASET_NAMESPACE}/{run}"


# ---------------------------------------------------------------------------- dataset layout
#
# The RQ0/RQ1 recordings are format v2.1 (one file per episode) and lerobot >= 0.4 writes
# v3.0 (several episodes per file, videos under a different nesting). Every script that
# built these paths itself was v2.1-only, and would have reported every reproduction trial
# as "missing" rather than failing.


def dataset_dir(run: str) -> Path:
    return DATASETS / run


def dataset_exists(run: str) -> bool:
    return (DATASETS / run / "meta" / "info.json").exists()


def dataset_info(run: str) -> dict:
    info = DATASETS / run / "meta" / "info.json"
    if not info.exists():
        raise FileNotFoundError(f"no dataset '{run}' under {DATASETS}")
    return json.loads(info.read_text(encoding="utf-8"))


def dataset_version(run: str) -> str:
    """"v2.1" or "v3.0". A dataset that is not there has no version, and saying so beats
    guessing v2.1 and handing back a path that will never exist."""
    return dataset_info(run).get("codebase_version", "v2.1")


def episode_parquet(run: str, episode: int = 0) -> Path:
    # v3.0 packs several episodes per file; an evaluation recording holds exactly one, so its
    # single data file is that episode. Multi-episode v3 datasets need LeRobotDataset.
    if dataset_version(run).startswith("v3"):
        return DATASETS / run / "data" / "chunk-000" / f"file-{episode:03d}.parquet"
    return DATASETS / run / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"


def episode_video(run: str, episode: int = 0, camera: str = None) -> Path:
    camera = CAMERA_NAME if camera is None else camera
    if dataset_version(run).startswith("v3"):
        return DATASETS / run / "videos" / f"observation.images.{camera}" / "chunk-000" / f"file-{episode:03d}.mp4"
    return DATASETS / run / "videos" / "chunk-000" / f"observation.images.{camera}" / f"episode_{episode:06d}.mp4"


def episode_indices(dataset: str):
    """Every episode index in a multi-episode dataset, in order."""
    if dataset_version(dataset).startswith("v3"):
        return list(range(dataset_info(dataset).get("total_episodes", 0)))
    files = sorted((DATASETS / dataset / "data" / "chunk-000").glob("episode_*.parquet"))
    return [int(f.stem.split("_")[-1]) for f in files]


def iter_episode_parquets(dataset: str):
    """(episode, path) for each episode of a recorded dataset. v2.1 only: a v3.0 dataset
    packs many episodes into one file and must be read with LeRobotDataset."""
    if dataset_version(dataset).startswith("v3"):
        raise NotImplementedError(
            f"{dataset} is format v3.0, where one parquet holds several episodes. "
            "Read it with lerobot's LeRobotDataset rather than per-episode files.")
    return [(ep, episode_parquet(dataset, ep)) for ep in episode_indices(dataset)]


# ---------------------------------------------------------------------------- the trial manifest

MANIFEST_PATH = HERE / "rq1_manifest.csv"
MANIFEST_FIELDS = ["run_id", "timestamp", "model", "condition", "start_position",
                   "repo_id", "episode_sec", "video_path"]


def read_manifest(study: str = None, condition: str = None, model: str = None):
    """Trial rows, newest write winning for a repeated run_id.

    The filters matter more than they look: the manifest holds every trial ever recorded, so
    a tool called with no arguments processes RQ0 and the reproduction together and writes a
    single output file over both. Post-processing for one condition should pass that
    condition.
    """
    if not MANIFEST_PATH.exists():
        return []
    with MANIFEST_PATH.open(encoding="ascii") as f:
        rows = list(csv.DictReader(f))
    unique = {}
    for row in rows:
        parsed = parse_run_id(row["run_id"]) or {}
        row = dict(row, **{f"parsed_{k}": v for k, v in parsed.items()})
        if study and parsed.get("study") != study:
            continue
        if condition and row.get("condition") != condition and parsed.get("condition") != condition:
            continue
        if model and row.get("model") != model:
            continue
        unique[row["run_id"]] = row
    # Recording order, not sorted: the manifest is append-only, so its order is the order the
    # trials actually ran in, and that is information - a session drifts, and a table sorted
    # by name hides the drift. A retried trial keeps its later row and its original place.
    return list(unique.values())


def episode_seconds():
    """run_id -> the wall-clock episode length it was recorded with."""
    return {r["run_id"]: float(r["episode_sec"]) for r in read_manifest() if r.get("episode_sec")}


def append_manifest_row(**row):
    exists = MANIFEST_PATH.exists()
    with MANIFEST_PATH.open("a", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in MANIFEST_FIELDS})


def study_output(name: str, study: str = None, condition: str = None) -> Path:
    """An output file named for what produced it. Alignment residuals and commanded-closure
    readings used to land on one filename regardless of study, so finishing the reproduction
    overwrote the RQ0 record with different trials."""
    parts = [p for p in (study, condition) if p]
    stem = "_".join(parts + [name]) if parts else name
    return HERE / stem


# ---------------------------------------------------------------------------- cameras


def reference_frame(camera: str = None) -> Path:
    """The frame a camera is aimed at. For RQ0/RQ1 it is built from the training data; for
    the reproduction it is a frame of the benchmark's own demonstrations."""
    camera = CAMERA_NAME if camera is None else camera
    configured = _lookup("alignment", camera)
    if configured:
        base = _lookup("alignment", "reference_dir", default=".")
        return (HERE / base / configured).resolve()
    return HERE / "_frames" / "camera_reference.png"


def open_camera(index: int = None, width: int = None, height: int = None, fps: float = None):
    """One way to open a camera, so exposure and format behave the same in every tool.
    DirectShow specifically: the Media Foundation backend reports different exposure units
    on this machine, and a tool that opened the camera differently measured it differently."""
    import cv2
    index = CAMERA_INDEX if index is None else index
    capture = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W if width is None else width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H if height is None else height)
    capture.set(cv2.CAP_PROP_FPS, DATASET_FPS if fps is None else fps)
    return capture


def require(module: str, why: str):
    """Import a module, or say which interpreter has it.

    The scripts here need one of two environments, and a plain `python` on Windows resolves
    to the Store build with nothing installed. A bare ModuleNotFoundError sends the reader
    looking for a missing package when the package is installed and the interpreter is wrong.
    """
    import importlib
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        import sys
        raise SystemExit("\n".join([
            f"'{module}' is not available to this interpreter, which is",
            f"    {sys.executable}",
            why,
            f"The environment for the active config ({CONFIG_PATH.name}) is:",
            f"    {ENV_PYTHON}",
            f'Run it as:  & "{ENV_PYTHON}" <script>.py',
        ])) from exc


def describe():
    return "\n".join([
        f"config      {CONFIG_PATH}",
        f"datasets    {DATASETS}",
        f"env python  {ENV_PYTHON}",
        f"work dir    {WORK_DIR}",
        f"ffmpeg      {FFMPEG}",
        f"robot       {ROBOT_ID} on {ROBOT_PORT}   leader {LEADER_ID} on {LEADER_PORT}",
        f"cameras     {', '.join(CAMERA_NAMES) or 'none'}   working: {CAMERA_NAME} "
        f"(index {CAMERA_INDEX}, {FRAME_W}x{FRAME_H})",
        f"task        {TASK_NAME}: {TASK_INSTRUCTION!r} at {DATASET_FPS:g} fps, "
        f"{EPISODE_SEC:g}s episode, {RESET_SEC:g}s reset",
        f"scoring     closed < {CLOSE_T:g}, stable {STABLE_S:g}s, hold {HOLD_S:g}s, "
        f"lift {LIFT_GAIN:g}, descend {DESCEND_D:g}",
        f"safety      pause at {TEMP_PAUSE_C:g}C, resume at {TEMP_RESUME_C:g}C, "
        f"{MAX_CONNECT_RETRIES} connect retries",
    ])


def _cli():
    """A little of this module is reachable from PowerShell, so the runners do not have to
    reimplement the dataset layout or the config lookup in a second language."""
    import sys
    args = sys.argv[1:]
    if not args:
        print(describe())
        missing = [n for n, p in [("datasets", DATASETS), ("env python", ENV_PYTHON),
                                  ("work dir", WORK_DIR), ("ffmpeg", FFMPEG)]
                   if p is None or not p.exists()]
        print("\nmissing: " + (", ".join(missing) if missing else "nothing"))
        return 0
    command, rest = args[0], args[1:]
    if command == "--get":
        value = _lookup(*rest[0].split("."))
        if value is None:
            raise SystemExit(f"no such key: {rest[0]}")
        print(value)
    elif command == "--episode-video":
        camera = rest[1] if len(rest) > 1 else None
        print(episode_video(rest[0], camera=camera))
    elif command == "--episode-parquet":
        print(episode_parquet(rest[0]))
    elif command == "--cameras":
        print(" ".join(CAMERA_NAMES))
    else:
        raise SystemExit("usage: labbench.py [--get <dotted.key> | --episode-video <run> [camera] "
                         "| --episode-parquet <run> | --cameras]")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
