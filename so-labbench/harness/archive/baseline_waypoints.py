# SO-LabBench scripted baseline (no AI): record waypoints by hand, then replay them.
#
#   record mode:  python baseline_waypoints.py record waypoints.json
#       -> torque OFF; physically pose the arm, press Enter to save each waypoint
#          (includes gripper), type 'q' + Enter to finish.
#   replay mode:  python baseline_waypoints.py replay waypoints.json [n_repeats]
#       -> moves through the saved waypoints slowly, N times, reporting per-run timing.
#
# This is the deterministic non-AI baseline required by the protocol: for a FIXED
# object position, compare its success rate and cycle time against learned policies.
import sys
import json
import time
from pathlib import Path
import scservo_sdk as scs

TORQUE, ACC, GOAL, GOAL_SPEED, PRESENT = 40, 41, 42, 46, 56
from labbench import ROBOT_PORT as PORT

def open_port():
    port = scs.PortHandler(PORT)
    packet = scs.PacketHandler(0)
    assert port.openPort() and port.setBaudRate(1000000), f"cannot open {PORT}"
    return port, packet

def record(path: Path):
    port, packet = open_port()
    for mid in range(1, 7):
        packet.write1ByteTxRx(port, mid, TORQUE, 0)   # free-move
    print("Torque OFF. Pose the arm by hand. Enter=save waypoint, q+Enter=finish.")
    wps = []
    while True:
        s = input(f"[{len(wps)} saved] > ").strip().lower()
        if s == "q":
            break
        pose = []
        for mid in range(1, 7):
            p, res, _ = packet.read2ByteTxRx(port, mid, PRESENT)
            pose.append(p if res == scs.COMM_SUCCESS else None)
        if None in pose:
            print("  read error, try again")
            continue
        wps.append(pose)
        print(f"  saved: {pose}")
    path.write_text(json.dumps(wps))
    port.closePort()
    print(f"{len(wps)} waypoints -> {path}")

def replay(path: Path, repeats: int):
    wps = json.loads(path.read_text())
    port, packet = open_port()
    for mid in range(1, 7):
        packet.write1ByteTxRx(port, mid, ACC, 20)
        packet.write2ByteTxRx(port, mid, GOAL_SPEED, 500)
        packet.write1ByteTxRx(port, mid, TORQUE, 1)
    for rep in range(1, repeats + 1):
        t0 = time.perf_counter()
        for wp in wps:
            for mid, pos in zip(range(1, 7), wp):
                packet.write2ByteTxRx(port, mid, GOAL, pos)
            time.sleep(1.6)
        print(f"run {rep}/{repeats}: {time.perf_counter() - t0:.1f}s")
        input("reset the scene, Enter for next run...") if rep < repeats else None
    # restore full speed
    for mid in range(1, 7):
        packet.write2ByteTxRx(port, mid, GOAL_SPEED, 0)
        packet.write1ByteTxRx(port, mid, ACC, 254)
    port.closePort()

if __name__ == "__main__":
    mode, path = sys.argv[1], Path(sys.argv[2])
    if mode == "record":
        record(path)
    else:
        replay(path, int(sys.argv[3]) if len(sys.argv) > 3 else 1)
