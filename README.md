# SO-LabBench

Careful measurement of low-cost imitation-learning policies, on one SO-ARM101 and one GPU.

Public results in this field are almost all self-reported, and almost none are re-run by
anyone else. This repository is an attempt at the other half: pre-registered measurements,
thresholds that are stated before the data, negative results published at the same weight as
positive ones, and code someone else can run.

Everything here is reproducible from public assets — the ArmnetBench benchmark, its
checkpoints, and the LeRobot source. Every finding below was obtained without touching a robot.

## Findings

### Published SO-101 action values are not comparable across arms
[`so-labbench/finding_calibration_portability.md`](so-labbench/finding_calibration_portability.md)

A LeRobot joint value is a percentage of the calibrated range of the arm it was recorded on,
and `drive_mode` can invert that mapping. Across ten public SO-101 datasets the maximum
commanded gripper value runs from 28.4 to 100; one is on a radian scale, so the units are not
shared either. **None of the datasets or checkpoints examined ships the calibration file**
that would make its action values recoverable — including a 633-file benchmark. The file is
under a kilobyte.

Concretely: a policy trained on one of those datasets asks our gripper for a third to a half
of the opening our own teleoperation used.

### These policies barely respond to camera aiming, and lean on different cameras
[`so-labbench/finding_rig_sensitivity.md`](so-labbench/finding_rig_sensitivity.md)

387 frames of the benchmark's own demonstrations, 34 perturbations, two policies, each
measured against two floors: the policy's own output noise on a repeated frame, and what its
output does between two consecutive frames anyway.

- An 80 px camera shift — larger than the benchmark's own camera-shift condition intends —
  moves neither policy's output as much as one frame of ordinary motion. Nor do 5° rotations,
  10% zooms, or a 30% lighting change.
- Blur does, by more than any aiming error. Optics matter more than position.
- On this task, blanking the wrist moves ACT most and blanking the front moves SmolVLA most.
  Across all eight tasks ACT's dominant view changes with the task: wrist five times, front
  twice, top once. SmolVLA's is the front camera on all eight. So which view a policy leans on
  is a property of the task **for ACT** and a stable preference **for SmolVLA** - see the
  correction note under the next finding, because this sentence has now been wrong twice.
- Blanking all three moves ACT 20.6x its natural step — 1.9x the sum of the single-camera
  effects. **ACT** uses vision heavily and redundantly: losing one view is survivable because
  the others carry it. SmolVLA does not share this - its combined effect is at or below the
  sum of the single views on seven of eight tasks.

### SmolVLA responds to its cameras five to eleven times less than ACT, on every task
[`so-labbench/finding_vision_reliance_act_vs_smolvla.md`](so-labbench/finding_vision_reliance_act_vs_smolvla.md)

The eight-task grid, for both policies this machine can run. Blanking all three cameras moves
ACT's commanded action by about one standard deviation of the human demonstrations on that
task (82-132%). It moves SmolVLA's by an eighth of one (9-17%). ACT is higher on **8 of 8**
tasks, by 5.4x to 10.7x, and the closest pair is still 3.8x apart.

The grid normally reports effects as multiples of the policy's own one-frame step, which is
the wrong scale for comparing two policies: the denominator belongs to the thing being
measured. So the result is stated against three denominators that fail differently - raw
action units, the policy's own step, and the demonstrations' spread, which belongs to neither
policy. All three agree, and the two policies' natural steps are close enough (0.76-1.19
against 0.64-1.04) to rule the artefact out directly.

- **SmolVLA leans on the front camera on all eight tasks**; ACT's preference changes with the
  task on the same rig, which rules out the camera placement as the cause. One view winning
  all eight by chance is 1 in 2,187. The lean is small, though: on five tasks the front
  camera's lead is smaller than SmolVLA's own per-frame sampling noise.
- **ACT's views back each other up; SmolVLA's do not.** Blanking all three divided by the sum
  of blanking each: ACT 1.65-2.94, above 1 on 8 of 8. SmolVLA 0.73-1.04, above 1 on 1 of 8.
- **Neither policy's vision reliance predicts its success rate** (ACT r = -0.22, SmolVLA
  r = -0.28, both intervals spanning zero).

**A claim that dissolved on the way.** SmolVLA's reliance looked task-invariant - 4.3 to 5.3
times its natural step across eight tasks, against ACT's 17.5 to 33.2. Checked across the
three denominators, the invariance exists in that one only: SmolVLA's coefficient of
variation across tasks is 6% there, but 20% raw and 18% task-relative, matching ACT's 21% and
17%. It was a property of the denominator. The tool was built to catch exactly that, and
what it caught was its author.

**This corrected two earlier sentences in this README** - "which view a policy leans on is a
property of the task, not the architecture" and "these policies use vision redundantly" - both
generalised from ACT alone. The first was itself already a correction of a one-task
generalisation, fixed with a one-policy generalisation.

What this cannot say: moving less is not using vision less well, and one VLA against one
non-VLA is n=1 per class, so nothing here is about VLAs in general. Each checkpoint is a
per-task fine-tune, so the architecture cannot be separated from the recipe.

### On the task nobody solves, all seven policies get there and come back
[`so-labbench/finding_failure_anatomy.md`](so-labbench/finding_failure_anatomy.md)

`cable_clip` is scored 0.000 by every one of the seven published policies. ArmnetBench
records each rollout's action trace, not only its video, so all seven can be analysed here -
including the four this machine cannot run.

Measured against the 50 human demonstrations, none of them leaves the demonstrated set of arm
configurations. A held-out demonstration sits 0.29 from the others; the policies sit 0.28 to
0.49, and pi0.5 is closer to the demonstrations than a demonstration is. Every one reaches
58-91% of the way through the demonstrated sequence, then spends its final quarter back at
20-46% of it, taking up to twice the human's duration to do so.

They are not failing to reach the decisive moment. They reach it, cannot complete it, retreat
and retry. Across 14 policy-task cells, where a rollout ends correlates with the published
success rate at r = +0.90, while how far it ever got does not (r = +0.07) - and the note is
explicit about which half of that is circular and which is not.

The gripper trace sharpens it. A human finishes this task with a median of **3** gripper
closures; the policies use **5 to 17.5**, over twice the duration. They grasp, approach, fail
to seat the connector, release, retreat and grasp again. On a task most of them solve, the
same count is 3 to 6.5 against the human's 3.

So the retrying itself works. The trajectory is indistinguishable from a human's, the
perception is live, the grasping happens. What is missing is whatever makes one attempt
succeed - millimetres, contact, force - which is exactly what inserting a DisplayPort
connector is made of, and exactly what joint angles do not record.

A note on method: the wrist-camera frames suggested the opposite conclusion, that the
policies were never holding the cable. The action trace refuted it. Had the order been
reversed, a plausible mistake would have been published.

**Corrected by the next finding:** the grasp-count observation above is real for
`cable_clip` and does not generalise. Tested across all 56 cells it does not predict success,
so grasp count is not usable as a health indicator for a policy or a task.

### Grasping more often than the human does not predict failure (negative)
[`so-labbench/finding_retry_does_not_predict.md`](so-labbench/finding_retry_does_not_predict.md)

The obvious next question after `cable_clip`: do policies that retry more than the human fail
more? Asked of all 2,499 rollouts, eight tasks by seven policies, the answer is no, and the
way it is no is worth more than the answer.

Inside a policy-task cell, where task difficulty and policy identity cancel, failed rollouts
grasp more often than successful ones in 24 of 43 cells. A coin flip (sign test p = 0.271),
median difference 0.007 closures per second.

Between tasks the correlation depends entirely on a choice nobody would question. Raw
closures per second correlates with success at **r = +0.34** [+0.09, +0.56]. The same
quantity expressed as a multiple of the human's rate on the same task correlates at
**r = -0.38** [-0.58, -0.12]. Same data, same 56 cells, opposite signs, both excluding zero.
The reason is measurable: the human's own grasp rate correlates with the task's mean success
rate at r = +0.89, so it is a proxy for task difficulty, and dividing by it flips the sign.

Vary the two remaining free choices - where the open/closed threshold sits, and how long a
closure must last to count - and the correlation moves between +0.46 and -0.40 while the
within-cell test wanders from p = 0.033 to p = 0.937. Drop the one task the hypothesis came
from and the headline correlation falls from -0.38 to -0.22, crossing zero.

The threshold is not assumed anywhere: it is measured per task from that task's own
demonstrations, because an action value is a percentage of one arm's calibrated range and a
constant imported from elsewhere would mean nothing.

A negative result is what a broken instrument produces most easily, so the instrument is
checked rather than trusted. `mutation_check.py` breaks the analysis 19 ways and every one
must be caught. The first version of the tests caught 8 of 19: they covered three leaf
functions and none of the pipeline, and the mutation that removes the division by duration -
which would destroy this finding's central claim - survived. The suite is now 44 checks and
catches all 19. Re-running on the real data after that work left every reported number
unchanged.

**For anyone analysing ArmnetBench traces:** publish both the threshold and the normalisation,
and check whether your quantity correlates with task difficulty before reporting a
cross-task correlation. Three defensible choices here span a full sign change, and most of
that range "excludes zero".

### Policies that never succeed still respond strongly to what they see
[`so-labbench/finding_sensitivity_vs_success.md`](so-labbench/finding_sensitivity_vs_success.md)

ArmnetBench publishes a success rate for every policy on every task, so the same sensitivity
measurement across all eight tasks asks whether what a policy attends to predicts whether it
succeeds. For ACT it does not: vision dependence correlates with the published rate at
r = -0.22, an interval spanning zero.

The three tasks where ACT scores exactly 0.000 have vision dependence of 19.3, 25.2 and 33.7
times their natural step - mid-range to highest. `cable_clip`, which **all seven published
policies fail completely**, has the highest of all. A policy whose output moves 34x when its
cameras are blanked is not failing because it cannot see the task; whatever defeats it is
downstream of perception. That rules out a whole class of explanation, and "low-cost arms
fail because they need better cameras" is not supported here.

### The published successes are not explained by where the object started (negative)
[`so-labbench/finding_success_vs_position.md`](so-labbench/finding_success_vs_position.md)

A hypothesis, proposed and refuted in one night. Detecting the object in the first frame of
51 labelled rollouts, ACT's successes sit 13.6 px closer to the demonstrations' centre than
its failures (interval spans zero) and SmolVLA's sit 14.8 px further. Signs disagree.

The more useful half is the test's own weakness: the rollouts span 227 px horizontally where
the demonstrations span 421. **The published rates were measured over a narrower band of
positions than the policies were trained on** — worth knowing when quoting them.

### A flow-matching policy's output moves when nothing else does
SmolVLA's action moves 0.48 units when the same frame is fed twice, against a 1.04-unit step
between consecutive frames. ACT moves 0.000000. Averaged over fifty episodes the noise
cancels, so an offline fit statistic is unaffected; in a closed loop it accumulates instead.

## What the harness is careful about

- **Thresholds are in seconds, never frames**, converted with each trial's own measured rate.
  LeRobot's record loop runs on wall clock and appends one frame per iteration, so a heavier
  policy produces fewer frames: a frame-count threshold silently asks a slow policy to hold
  longer than a fast one. Measured back to back on one rig: ACT 23.6 Hz, SmolVLA 18.4 Hz, so
  a "0.5 s" threshold written as 15 frames meant 0.63 s and 0.81 s.
- **A nuisance variable is recorded, not gated on.** Camera alignment never stops a trial;
  every trial carries its residual, so a surprising result can be checked against drift after
  the fact rather than argued about.
- **Machine facts are measured.** There is no `CONTROL_HZ`; there is `true_hz(frames, seconds)`.
- **Decisions live in one file.** `harness/labbench.py` holds every threshold, dataset path
  and run-id spelling. They were once in up to eight places, and one pair disagreed — the
  demonstration audit called a closure stable at 12 frames while the scorer called it stable
  at 0.5 s, which is 15. No verdict actually differed, but nothing prevented it.
- **Judging is blind.** Trials are shuffled and renamed before a human labels them; the key
  that maps them back is written once and not opened until the labels are in.
- **The decision table is written before the data**, including the branch where the result
  means the experiment did not work.

```
python so-labbench/harness/test_harness.py      # 157 checks, no pytest, no install
python so-labbench/harness/mutation_check.py    # breaks the code 19 ways; each must be caught
```

The second matters as much as the first. A suite that passes proves nothing on its own —
tests written after the code tend to assert what it does rather than what it should. Each
mutation is a defect that was real here or would be easy to reintroduce, and every one must
fail at least one check. Two survived the first run, and both were genuine gaps.

## Running it

Two environments, because one study needs a newer LeRobot than another and the older one is
frozen so its results stay reproducible.

```bash
cp so-labbench/harness/labbench_config.example.json so-labbench/harness/labbench_config.json
# edit the paths, serial port and camera indices for your machine
python so-labbench/harness/labbench.py           # prints what it resolved to
```

`LABBENCH_CONFIG` selects a config, which is how a second study becomes a second file rather
than a fork. The GPU analyses under `so-labbench/armnetbench/` need no robot and no config
beyond a LeRobot install:

```bash
python so-labbench/armnetbench/rig_sensitivity.py --task eye_drops_to_basket
python so-labbench/armnetbench/summarize_sensitivity.py
```

## What is in here

```
so-labbench/
  finding_*.md              the results above, each with what it does and does not show
  research_roadmap.md       what is being measured and what follows what
  rqp1_*.md                 a pre-registered independent reproduction of ArmnetBench,
                            with everything known to be wrong with it written before running
  rq0_*.md, rq1_*.md        closed and frozen experiments, kept as the record
  harness/                  the measurement code (README inside); archive/ says what is retired
  armnetbench/              analyses of the public benchmark, and its published results
pr_package/                 two LeRobot bugs found here, with patches and reproductions
```

The frozen experiments are kept deliberately. RQ1 measures degradation from a baseline whose
policy succeeded zero times in ten, so it is frozen rather than deleted; the reasoning is
more useful than a tidy repository.

## Licence

Apache 2.0. The ArmnetBench data and checkpoints analysed here are the work of Armnet and its
contributors, also Apache 2.0; the analyses are ours and any error in them is ours.

Corrections are welcome, especially to the calibration note — the check behind it is a few
lines against `meta/stats.json` and anyone can run it.
