# Talking to the follower's Feetech bus, with the cleanup that must not be skipped.
#
# Every tool here throttles the servos before it moves them (gentle acceleration, low speed)
# and must put the caps back afterwards. Those registers live INSIDE the servo and survive
# the process, so a script that exits through an exception leaves the arm permanently slow -
# and the next evaluation runs at a speed nothing in its record mentions. That is not
# hypothetical: a diagnostic script did exactly this in August and the teleoperation lag it
# caused was first blamed on the hardware.
#
# So: one context manager, one place where the caps come back, and it comes back on the way
# out of an exception too.
import time
from contextlib import contextmanager

from labbench import BAUD, MOTOR_IDS, ROBOT_PORT, Reg, require

scs = require("scservo_sdk", "It is the Feetech servo SDK, and it talks to the arm's motor bus.")

SLOW_SPEED = 350        # ticks/s while a script is positioning the arm
SLOW_ACC = 15
UNCAPPED_SPEED = 0      # 0 means "no limit" to these servos
FULL_ACC = 254


class Bus:
    def __init__(self, port, packet):
        self.port, self.packet = port, packet

    def read_position(self, motor):
        value, result, _ = self.packet.read2ByteTxRx(self.port, motor, Reg.PRESENT_POSITION)
        return value if result == scs.COMM_SUCCESS else None

    def write_position(self, motor, tick):
        self.packet.write2ByteTxRx(self.port, motor, Reg.GOAL_POSITION, int(tick))

    def hold_where_it_is(self, motor):
        """Enable torque without a jump: aim at wherever the joint already is."""
        self.packet.write1ByteTxRx(self.port, motor, Reg.ACC, SLOW_ACC)
        self.packet.write2ByteTxRx(self.port, motor, Reg.GOAL_SPEED, SLOW_SPEED)
        self.packet.write1ByteTxRx(self.port, motor, Reg.TORQUE_ENABLE, 1)
        position = self.read_position(motor)
        if position is not None:
            self.write_position(motor, position)
        return position

    def settle(self, seconds):
        time.sleep(seconds)


@contextmanager
def open_bus(port_name: str = None):
    """Open the follower bus, and restore the speed caps on every exit path."""
    port_name = ROBOT_PORT if port_name is None else port_name
    port = scs.PortHandler(port_name)
    packet = scs.PacketHandler(0)
    if not port.openPort():
        raise SystemExit(f"cannot open {port_name}. Is the follower powered and plugged in?")
    if not port.setBaudRate(BAUD):
        port.closePort()
        raise SystemExit(f"cannot set {BAUD} baud on {port_name}")
    bus = Bus(port, packet)
    try:
        yield bus
    finally:
        for motor in MOTOR_IDS:
            packet.write2ByteTxRx(port, motor, Reg.GOAL_SPEED, UNCAPPED_SPEED)
            packet.write1ByteTxRx(port, motor, Reg.ACC, FULL_ACC)
        port.closePort()


def calibration(robot_id: str = None):
    """The follower's calibrated tick range per joint, which is what turns a normalized
    action into a physical position. lerobot moved this directory between versions, so both
    spellings are tried."""
    import json
    from pathlib import Path

    from labbench import ROBOT_ID
    robot_id = ROBOT_ID if robot_id is None else robot_id
    root = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots"
    for folder in ("so_follower", "so101_follower"):
        path = root / folder / f"{robot_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise SystemExit(f"no calibration for '{robot_id}' under {root}")


def normalized_to_tick(joint: str, value: float, cal: dict = None) -> int:
    """A policy's action is a percentage of THIS arm's calibrated range, not an angle. Two
    SO-101s calibrated over different travel will put the same number in different places -
    which is why a policy trained on one arm cannot be assumed to mean the same thing on
    another. See rqp1_design_review.md, A1."""
    cal = calibration() if cal is None else cal
    spec = cal[joint]
    span = spec["range_max"] - spec["range_min"]
    return int(round(spec["range_min"] + value / 100.0 * span))
