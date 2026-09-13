# RQ-P1 primary outcome: human-judged success per policy on this rig, against ArmnetBench.
#   python rqp1_results.py [condition]            # default: eyedrops
#
# Reads the manifest rows of study rqp1, joins the human labels from rqp1_human_labels.md
# (successful / suboptimal / failure / invalid, one row per trial), and prints per-policy
# success (strict: only "successful" counts, as in the benchmark), Wilson 95% CI, and the
# difference to the published rate in ../armnetbench/reference_results.csv. Secondary
# columns come from the scorer CSV when present. Writes rqp1_results.csv.
import csv
import sys
from pathlib import Path

from labbench import read_manifest, study_output, wilson_ci

HERE = Path(__file__).parent
MANIFEST = HERE / "rq1_manifest.csv"
LABELS = HERE / "rqp1_human_labels.md"
REFERENCE = HERE.parent / "armnetbench" / "reference_results.csv"
TASK_TEXT = {"eyedrops": "Put the eye drops into the basket", "ring": "Insert the colourful ring into the central wooden peg"}
LABEL_SET = {"successful", "suboptimal", "failure", "invalid"}


def read_labels(condition: str):
    """run_id -> label, preferring the blind review set.

    Blind (blind_review.py prepare): the sheet names shuffled review ids and nothing else,
    and a separate key maps them back to runs. Reading it here is the only place the two
    are joined, which keeps the judgement itself free of any knowledge of the policy.
    The older per-run sheet is still accepted so that anything labelled before the blind
    flow existed can be scored, but it is reported as what it is.
    """
    key = HERE / f"rqp1_{condition}_review_key.csv"
    sheet = HERE / f"rqp1_{condition}_labels.md"
    if key.exists() and sheet.exists():
        import blind_review
        mapping = {}
        with key.open(encoding="ascii") as f:
            for r in csv.DictReader(f):
                mapping[r["review_id"]] = r["run_id"]
        labels = {mapping[rid]: lab for rid, lab in blind_review.read_labels(sheet).items()
                  if rid in mapping and lab in LABEL_SET}
        return labels, "blind"
    out = {}
    if LABELS.exists():
        for line in LABELS.read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 5 or not cells[0].startswith("eval_"):
                continue
            lab = cells[4].lower()
            if lab in LABEL_SET:
                out[cells[0]] = lab
    return out, "unblinded (the sheet names the policy on every row)"


def main():
    condition = sys.argv[1] if len(sys.argv) > 1 else "eyedrops"
    labels, judging = read_labels(condition)
    trials = read_manifest(study="rqp1", condition=condition)
    if not trials:
        raise SystemExit(f"no rqp1 trials for condition '{condition}' in the manifest")
    ref = {}
    if REFERENCE.exists():
        with REFERENCE.open() as f:
            for r in csv.DictReader(f):
                if r["task"] == TASK_TEXT.get(condition):
                    ref[r["policy_type"]] = (int(r["successful"]), int(r["rollouts"]))
    rows = []
    print(f"judging: {judging}\n")
    print(f"{'policy':9} {'n':>3} {'success':>8} {'rate':>6} {'95% CI':>12} {'ArmnetBench':>12} {'delta':>7}  unlabelled/invalid")
    for policy in sorted({r["model"] for r in trials}):
        ids = [r["run_id"] for r in trials if r["model"] == policy]
        labs = [labels.get(i) for i in ids]
        valid = [l for l in labs if l in ("successful", "suboptimal", "failure")]
        k = sum(l == "successful" for l in valid)
        n = len(valid)
        lo, hi = wilson_ci(k, n)
        rk, rn = ref.get(policy, (None, None))
        rrate = rk / rn if rn else float("nan")
        delta = (k / n - rrate) if n and rn else float("nan")
        unl = sum(l is None for l in labs)
        inv = sum(l == "invalid" for l in labs)
        print(f"{policy:9} {n:3d} {k:8d} {k/n if n else float('nan'):6.2f} {100*lo:5.0f}-{100*hi:<5.0f} "
              f"{(f'{rk}/{rn}={rrate:.2f}' if rn else 'n/a'):>12} {delta:+7.2f}  {unl}/{inv}")
        rows.append(dict(condition=condition, policy=policy, n=n, successful=k, suboptimal=sum(l == "suboptimal" for l in valid),
                         failure=sum(l == "failure" for l in valid), rate=round(k / n, 3) if n else "", ci_lo=round(lo, 3), ci_hi=round(hi, 3),
                         ref_successful=rk, ref_rollouts=rn, ref_rate=round(rrate, 3) if rn else "", delta=round(delta, 3) if n and rn else "",
                         unlabelled=unl, invalid=inv, judging=judging))
    out = study_output("results.csv", "rqp1", condition)
    with out.open("w", newline="", encoding="ascii") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {out.name}   (n=30 resolves differences of about 25 points; smaller ones are 'no difference')")


if __name__ == "__main__":
    main()
