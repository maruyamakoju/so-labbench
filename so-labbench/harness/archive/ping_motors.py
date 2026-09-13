import scservo_sdk as scs

from labbench import ROBOT_PORT
port = scs.PortHandler(ROBOT_PORT)
packet = scs.PacketHandler(0)
assert port.openPort(), f"cannot open {ROBOT_PORT}"
assert port.setBaudRate(1000000), "cannot set baud"
for i in range(1, 7):
    model, res, err = packet.ping(port, i)
    status = "OK" if res == scs.COMM_SUCCESS else "NO RESPONSE"
    print(f"motor {i}: {status}" + (f" (model {model})" if res == scs.COMM_SUCCESS else ""))
port.closePort()
