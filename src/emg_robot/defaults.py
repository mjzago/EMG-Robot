A0 = 0x48
A1 = 0x49
A2 = 0x4A
A3 = 0x4B
A4 = 0x4C
A5 = 0x4D
A6 = 0x4E
A7 = 0x4F


# Addresses of the EMG sensors to read on every cycle. Order matters! 
I2C_ADDRESSES = [A4, A5, A2, A6, A0]
EMG_CHANNEL_NAMES = ["biceps", "triceps", "pronator teres", "brachioradialis", "supinator"]
ROBOT_IP = "192.168.2.12"
