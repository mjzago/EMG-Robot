#!/usr/bin/env python3
import smbus
import time
import sys

A0 = 0x48
A1 = 0x49
A2 = 0x4A
A3 = 0x4B
A4 = 0x4C
A5 = 0x4D
A6 = 0x4E
A7 = 0x4F

bus = smbus.SMBus(1)
address = locals()[sys.argv[1]]

while True:
    data = bus.read_i2c_block_data(address, 0, 2)
    value = (data[0] << 8) | data[1]
    print(value)
    time.sleep(1)
