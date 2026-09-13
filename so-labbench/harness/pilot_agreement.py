# The pilot's actual deliverable: does the scorer label trials the way a human does?
#   python pilot_agreement.py [pilot_human_labels.md]
# Reads the operator's visual calls from the markdown table, pairs them with the
# scorer's taxonomy for the same run, and reports agreement. The gate is 10/10.
#
# Success rates are not computed here on purpose - the pilot measures the instrument,
# not the policy, and the preregistration forbids reading performance off it.
import sys
import re
import pandas as pd
from pathlib import Path

HERE = Path(__file__).parent
DEFAULT_MD = HERE / "pilot_human_labels.md"

# scorer taxonomy -> the seven words the operator writes in the sheet
AUTO_TO_HUMAN = {
    "success": "success",
    "approach_failure": "approach",
    "grasp_failure": "grasp",
    "lift_failure": "lift",
    "hold_failure": "hold",
    "policy_instability": "instability",
    "invalid_trial": "invalid",
}
VALID = set(AUTO_TO_HUMAN.values())


def read_human(md: Path):
    rows = []
    for line in md.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("| PILOT-"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue
        trial_id, pair, pos, model, label = cells[0], cells[1], cells[2], cells[3], cells[4]
        rows.append(dict(trial_id=trial_id, pair=int(pair), position=pos,
                         model=model.lower(), human=label.lower()))
    return rows


def read_auto():
    auto = {}
    for model in ("act", "smolvla"):
        f = HERE / f"eval_rq1_pilot_{model}_scores.csv"
        if not f.exists():
            continue
        for _, r in pd.read_csv(f).iterrows():
            status = r.get("status")
            tax = "invalid_trial" if status == "invalid_trial" else (
                r.get("taxonomy") if status == "ok" else None)
            auto[(model, int(r["trial"]))] = tax
    return auto


def main():
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MD
    human = read_human(md)
    auto = read_auto()
    if not human:
        raise SystemExit(f"no PILOT rows found in {md}")

    unfilled = [h["trial_id"] for h in human if not h["human"]]
    bad = [(h["trial_id"], h["human"]) for h in human if h["human"] and h["human"] not in VALID]
    if bad:
        print("labels outside the seven allowed words:")
        for t, v in bad:
            print(f"  {t}: '{v}'")
    if unfilled:
        print(f"not yet labelled: {', '.join(unfilled)}")
    if unfilled or bad:
        print("\nFill those in before reading agreement.")
        return 2

    print(f"{'trial':10} {'model':9} {'pos':4} {'human':12} {'auto':12} match")
    agree = 0
    missing = 0
    for h in human:
        a_tax = auto.get((h["model"], h["pair"]))
        if a_tax is None:
            a_word, mark = "MISSING", "-"
            missing += 1
        else:
            a_word = AUTO_TO_HUMAN.get(a_tax, a_tax)
            ok = (a_word == h["human"])
            agree += ok
            mark = "ok" if ok else "MISMATCH"
        print(f"{h['trial_id']:10} {h['model']:9} {h['position']:4} {h['human']:12} {a_word:12} {mark}")

    n = len(human) - missing
    print(f"\nagreement: {agree}/{n}" + (f"  ({missing} trials missing scorer output)" if missing else ""))
    if missing:
        print("Score the runs first: python score_rq1.py eval_rq1_pilot_act_ 5")
        return 2
    if agree == n and n == 10:
        print("\n10/10. The scorer and the operator label these trials the same way.")
        print("Per the preregistration this version is the final freeze - do not change")
        print("the success definition, taxonomy or scorer until C0 is complete.")
        return 0
    print("\nNot 10/10. Each mismatch is a finding about the instrument, not a bad trial.")
    print("Decide per row whether the scorer or the label was wrong, fix the scorer,")
    print("bump the version, and rerun all ten. Do not mix old and new pilots.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
