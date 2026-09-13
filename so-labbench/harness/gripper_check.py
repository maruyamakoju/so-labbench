# Does this arm open far enough for the policy we are about to run?
#
#   python gripper_check.py                 # step through the values the policy will command
#   python gripper_check.py --hold 34.4     # park at one value so it can be measured
#   python gripper_check.py --list          # print the tick mapping without moving anything
#
# Why this check exists. A policy's action is a percentage of the arm it was trained on,
# and lerobot-calibrate defines that percentage from however far the operator moved each
# joint. Two SO-101s are only interchangeable if they were calibrated over the same travel.
#
# For the ArmnetBench eye-drops policies the numbers are not reassuring: their demonstrations
# command the gripper open to a mean of 34.4 and never past 48.4, while our own
# demonstrations used 85.7 to 100 of our range. If our calibration spans the full travel and
# theirs did not, the policy will ask our gripper for a third to a half of the opening our
# own teleoperation used - and if that is narrower than the object, nothing the policy does
# can succeed, for a reason that has nothing to do with generalization.
#
# So: move the gripper to each value the policy will actually command, and measure the jaw
# opening with a ruler. The object has to fit, with margin, at the policy's OPEN value.
#
# Nothing else moves. The arm should already be in its home pose.
import argparse
import sys

from labbench import Reg
from motors import calibration, normalized_to_tick, open_bus

GRIPPER = "gripper"
GRIPPER_ID = 6

# The values that matter for this reproduction, from the benchmark's own demonstrations.
POLICY_VALUES = [
    (0.0, "fully closed on their scale"),
    (1.6, "their mean while grasping"),
    (15.0, "the closed/open threshold we score with"),
    (34.4, "their MEAN open command - the object must fit here"),
    (48.4, "their MAXIMUM open command - the widest the policy ever asks for"),
    (85.7, "our own demonstrations' mean open, for comparison"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hold", type=float, help="park the gripper at this normalized value and exit")
    ap.add_argument("--list", action="store_true", help="print the mapping without moving the arm")
    ap.add_argument("--settle", type=float, default=2.5, help="seconds to wait at each value")
    args = ap.parse_args()

    cal = calibration()
    spec = cal[GRIPPER]
    span = spec["range_max"] - spec["range_min"]
    print(f"gripper calibrated range: {spec['range_min']}..{spec['range_max']} ({span} ticks)\n")

    if args.list:
        for value, why in POLICY_VALUES:
            print(f"  {value:5.1f} -> tick {normalized_to_tick(GRIPPER, value, cal):5d} "
                  f"({value / 100 * span:5.0f} ticks from closed)   {why}")
        return 0

    values = [(args.hold, "requested")] if args.hold is not None else POLICY_VALUES
    with open_bus() as bus:
        held = bus.hold_where_it_is(GRIPPER_ID)
        print(f"gripper is at tick {held}; torque on, moving slowly.\n")
        bus.settle(0.5)
        for value, why in values:
            tick = normalized_to_tick(GRIPPER, value, cal)
            bus.write_position(GRIPPER_ID, tick)
            bus.settle(args.settle)
            reached = bus.read_position(GRIPPER_ID)
            print(f"  normalized {value:5.1f} -> tick {tick:5d}, reached {reached}   {why}")
            if args.hold is None:
                input("     measure the jaw opening in mm, then press Enter for the next value...")

    print("\nSpeed caps restored. The reading that decides the experiment is the one at 34.4:")
    print("the object must fit through that opening with a few millimetres to spare, because")
    print("that is the average width the policy actually asks for. If it does not, choose a")
    print("thinner object and record both widths - the deviation is part of the result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
