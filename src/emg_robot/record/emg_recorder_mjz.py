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

# Function to read a single byte from a device
def read_byte(address, register):
    try:
        return bus.read_byte_data(address, register)
    except IOError as e:
        print(f"I/O error while trying to read from device at address {hex(address)}:", e)
        return None

# Function to read accelerometer data from IMU
def read_accelerometer_raw():
    try:
        data = bus.read_i2c_block_data(IMU, 0x3B, 6)
        acc_x = (data[0] << 8 | data[1]) / 16384.0
        acc_y = (data[2] << 8 | data[3]) / 16384.0
        acc_z = (data[4] << 8 | data[5]) / 16384.0
        return acc_x, acc_y, acc_z
    except IOError as e:
        print(f"I/O error while trying to read accelerometer data:", e)
        return None, None, None

# Function to read gyroscope data from IMU
def read_gyroscope_raw():
    try:
        data = bus.read_i2c_block_data(IMU, 0x43, 6)
        gyro_x = (data[0] << 8 | data[1]) / 131.0
        gyro_y = (data[2] << 8 | data[3]) / 131.0
        gyro_z = (data[4] << 8 | data[5]) / 131.0
        return gyro_x, gyro_y, gyro_z
    except IOError as e:
        print(f"I/O error while trying to read gyroscope data:", e)
        return None, None, None

# Function to handle Ctrl+C signal
def signal_handler(sig, frame):
    print('Exiting...')
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Open CSV file for writing
filename = f"data-{sys.argv[-1]}.csv"
with open(filename, "w", newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "Acc_X", "Acc_Y", "Acc_Z", "Gyro_X", "Gyro_Y", "Gyro_Z", "Timestamp"])

    # Main loop
    while True:
        timestamp = time.time()
        values = []

        for address in [A0, A1, A2, A3, A4, A5, A6, A7]:
            value = read_byte(address, 0x00)
            values.append(value)

        acc_x, acc_y, acc_z = read_accelerometer_raw()
        gyro_x, gyro_y, gyro_z = read_gyroscope_raw()

        row = values + [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, timestamp]
        writer.writerow(row)

        time.sleep(1)  # Delay for 1 second
