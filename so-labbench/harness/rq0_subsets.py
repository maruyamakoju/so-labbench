# RQ0: which episodes go into each training arm.
#   python rq0_subsets.py            # writes rq0_subsets.json and prints the composition
#
# The arms are derived, not typed: "23complete" is every episode the quality audit
# (dataset_quality_audit.py -> so101_pick_remote_quality.csv) marked complete_pickup,
# and "23random" is a seeded draw of the same size from all 60. The random arm is the
# size control - without it a better 23complete model could mean "fewer episodes of a
# narrower distribution" rather than "no incomplete demonstrations".
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).parent
QUALITY = HERE / "so101_pick_remote_quality.csv"
OUT = HERE / "rq0_subsets.json"
SEED = 1000   # same seed the frozen checkpoints were trained with


def main():
    with QUALITY.open(encoding="ascii") as f:
        rows = list(csv.DictReader(f))
    all_eps = [int(r["ep"]) for r in rows]
    complete = sorted(int(r["ep"]) for r in rows if r["verdict"] == "complete_pickup")
    n = len(complete)
    rng = random.Random(SEED)
    rand = sorted(rng.sample(all_eps, n))
    verdict = {int(r["ep"]): r["verdict"] for r in rows}

    subsets = {
        "source": QUALITY.name,
        "seed": SEED,
        "arms": {
            "60all": {"episodes": all_eps, "note": "frozen act_aug_v2_60ep / smolvla_v2_60ep; no retraining"},
            f"{n}complete": {"episodes": complete, "note": "verdict == complete_pickup"},
            f"{n}random": {"episodes": rand, "note": f"random.Random({SEED}).sample(all, {n}); size control"},
        },
    }
    for name, arm in subsets["arms"].items():
        comp = {}
        for e in arm["episodes"]:
            comp[verdict[e]] = comp.get(verdict[e], 0) + 1
        arm["composition"] = dict(sorted(comp.items()))
        print(f"{name:12} n={len(arm['episodes']):2}  {arm['composition']}")
        print(f"{'':12} {arm['episodes']}")
    OUT.write_text(json.dumps(subsets, indent=2), encoding="ascii")
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
