#!/usr/bin/env python3
import smbus
import signal 
import math
import time 
import sys 
import csv

A0 = 0x48
A1 = 0x49
A2 = 0x4A 
A3 = 0x4B
A4 = 0x4C 
A5 = 0x4D 
A6 = 0x4E
A7 = 0x4F 

IMU = 0x68

bus = smbus.SMBus(1)
# get the address from program arguments 
addresses = []
for a in sys.argv[1:-1]:
    addresses.append(locals()[a])

print(addresses)

# Last argument is used as the name for the file
FILENAME = f"./data-{sys.argv[-1]}.csv"
print(FILENAME)
file = open(FILENAME, "w")
writer = csv.writer(file)

def get_value(data, idx, scaling_f):
    v = (data[idx] << 8) | data[idx+1]
    if v >= 0x8000:
        v = -(65535 - v) + 1
    return v / scaling_f

def get_imu_acc_raw(address):
    try:
        if address == A1 or address == A3 or address == A7:
            data = bus.read_i2c_block_data(address, 0x00, 1)  # Modified to read 1 byte instead of 6 bytes
            return [data[0]]  # Returns only the first byte as value
        else:
            data = bus.read_i2c_block_data(address, 0x3b, 6)  
            return [get_value(data, i*2, 16384.0) for i in range(3)]
    except IOError as e:
        print(f"I/O error while trying to read data from device at address {hex(address)}:", e)
        return [None, None, None]

def get_imu_gyro_raw(address):
    try:
        data = bus.read_i2c_block_data(address, 0x43, 6)
        return [get_value(data, i*2, 131.0) for i in range(3)]
    except IOError as e:
        print(f"I/O error while trying to read data from device at address {hex(address)}:", e)
        return [None, None, None]

def signal_handler(sig, frame):
    print('Average sampling rate: ', (distances / i))
    sys.exit(0)

# Handle kill signals
for kill_signal in [signal.SIGINT, signal.SIGTERM]:
    signal.signal(kill_signal, signal_handler)

# Sampling rate in kiloHertz
ABTAST = 1.5
DISTANCE = 1 / (ABTAST * 1000)

last_sensor_read = int(time.time())
distances = 0
i = 0

bus.write_byte_data(IMU, 0x6b, 0)

# CSV header
header = ["emg_0", "emg_1", "emg_2", "emg_3", "emg_4", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "dt"]
writer.writerow(header)

while True:
    delta = time.time() - last_sensor_read
    if delta < DISTANCE:
        continue

    last_sensor_read = time.time()
    values = []

    for a in addresses:
        try:
            data = bus.read_i2c_block_data(a, 0x00, 2)
            value = (data[0] << 8) | data[1]
            values.append(value)
        except IOError as e:
            print(f"I/O error while trying to read data from device at address {hex(a)}:", e)
            values.append(None)

    values += get_imu_acc_raw(IMU)
    values += get_imu_gyro_raw(IMU)

    values.append(delta)
    writer.writerow(values)
  
    distances += delta
    i += 1

    # 1-second delay
    time.sleep(1)
