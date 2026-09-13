# Put the arm in its centre pose before a trial.
#
#   python home_pose.py
#
# Every evaluation starts from here. Without it the policy inherits whatever pose the
# previous trial ended in, and a policy that starts from a collapsed arm behaves differently
# from one that starts upright - which turns the starting pose into an uncontrolled variable
# that changes across a session as failures accumulate.
#
# The joints move in order from the end of the arm inwards, so the arm does not swing its
# own mass through the workspace on the way to the middle.
import sys
import time

from labbench import CENTER_TICK, MOTOR_IDS
from motors import open_bus

SETTLE_WRIST = 1.2      # seconds per wrist/elbow joint
SETTLE_SHOULDER = 2.5   # the shoulder carries the arm's weight and needs longer
SETTLE_BASE = 2.0
ON_TARGET_TICKS = 150   # how close to the centre counts as arrived


def main():
    # The bus context manager restores the speed caps on every exit path, including an
    # exception. It matters here: this script deliberately throttles the servos, those
    # registers live inside the servo and outlast the process, and an arm left throttled
    # runs every later trial slowly with nothing in the record saying so.
    with open_bus() as bus:
        for motor in MOTOR_IDS:
            bus.hold_where_it_is(motor)
        time.sleep(0.5)

        for motor in [6, 5, 4, 3]:
            bus.write_position(motor, CENTER_TICK)
            time.sleep(SETTLE_WRIST)
        time.sleep(1.0)
        bus.write_position(2, CENTER_TICK)
        time.sleep(SETTLE_SHOULDER)
        bus.write_position(1, CENTER_TICK)
        time.sleep(SETTLE_BASE)

        off = []
        for motor in MOTOR_IDS:
            position = bus.read_position(motor)
            if position is None:
                print(f"motor {motor}: no response")
                off.append(motor)
                continue
            drift = position - CENTER_TICK
            status = "OK" if abs(drift) < ON_TARGET_TICKS else f"OFF by {drift}"
            if abs(drift) >= ON_TARGET_TICKS:
                off.append(motor)
            print(f"motor {motor}: pos={position} {status}")

    print("home pose done (torque ON to hold, speed caps restored to full)")
    if off:
        # Silence here would let a trial start from a pose nobody chose.
        print(f"WARNING: motor(s) {off} did not reach the centre. Check for an obstruction or "
              f"a stalled servo before recording; a 12 V power cycle clears a hung one.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
