from labbench import require

scs = require("scservo_sdk", "It is the Feetech servo SDK, and it talks to the arm's motor bus.")

PRESENT_VOLTAGE = 62
PRESENT_TEMPERATURE = 63

from labbench import ROBOT_PORT
port = scs.PortHandler(ROBOT_PORT)
packet = scs.PacketHandler(0)
assert port.openPort(), f"cannot open {ROBOT_PORT}"
assert port.setBaudRate(1000000), "cannot set baud"
for i in range(1, 7):
    model, res, err = packet.ping(port, i)
    if res != scs.COMM_SUCCESS:
        print(f"motor {i}: NO RESPONSE")
        continue
    v, res_v, _ = packet.read1ByteTxRx(port, i, PRESENT_VOLTAGE)
    t, res_t, _ = packet.read1ByteTxRx(port, i, PRESENT_TEMPERATURE)
    volt = f"{v / 10.0:.1f}" if res_v == scs.COMM_SUCCESS else "?"
    if res_t == scs.COMM_SUCCESS:
        print(f"motor {i}: OK  voltage={volt}V  temp={t}C  err_status={err}")
    else:
        # The runners gate on temp=<n>C and take the maximum over the motors they can read.
        # Printing "temp=NoneC" matched nothing, so a failed sensor quietly dropped that
        # motor from the maximum - and a bus where every sensor failed reported -1, which is
        # below the pause threshold. A thermal gate that opens when the thermometer breaks
        # is worse than no gate, so say so in a way the caller cannot mistake for cool.
        print(f"motor {i}: OK  voltage={volt}V  temp=UNREADABLE  err_status={err}")
port.closePort()
