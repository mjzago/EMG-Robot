#!/usr/bin/env python3
# Beispielhafte Benutzung:
# python main.py A0 A1 A2 A3 A4 Bewegung_1
# python main.py A2 A4 A0 A3 A1 Bewegung_1_1
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
# get the address from programm arguments 
addresses = []
for a in sys.argv[1:-1]:
  addresses.append(locals()[a])

print(addresses)

# ################################################
# # letztes argument wird als name für die datei benutzt
# filename = f"./data-{sys.argv[-1]}.csv"
# print(filename)
# file = open(filename, "w")
# writer = csv.writer(file)

# def get_value(data, idx, scaling_f):
  # v = (data[idx] << 8) | data[idx+1]
  # if v >= 0x8000:
    # v = -(65535 - v) + 1
  # return v / scaling_f

# def get_imu_acc_raw(address):
  # # acceleration: 0x3b .. 0x40, 16384 lsb/g
  # data = bus.read_i2c_block_data(address, 0x3b, 6)  
  # return [get_value(data, i*2, 16384.0) for i in range(3)]

# def get_imu_acc(address):
  # # acceleration: 0x3b .. 0x40, 16384 lsb/g
  # data = bus.read_i2c_block_data(address, 0x3b, 6)
  # ax, ay, az = [get_value(data, i*2, 16384.0) for i in range(3)]
  # rx = math.atan2(ay, math.sqrt(ax**2 + az**2))
  # ry = math.atan2(ax, math.sqrt(ay**2 + az**2))
  # return (rx, ry)

# def get_imu_gyro_raw(address):
  # data = bus.read_i2c_block_data(address, 0x43, 6)
  # return [get_value(data, i*2, 131.0) for i in range(3)]

# def get_imu_gyro(address):
  # # gyroscope: 0x43 .. 0x48, 131 lsb/°/s
  # data = bus.read_i2c_block_data(address, 0x43, 4)  # 4 -> only x and y
  # gx, gy = [get_value(data, i*2, 131.0) for i in range(2)]
  # return (gx, gy)

# def signal_handler(sig, frame):
    # print('mittlere smampling rate: ', (distances / i))
    # sys.exit(0)

# # handle kill signals
# for kill_signal in [signal.sigint, signal.sigterm]:
  # signal.signal(kill_signal, signal_handler)


# # in kilohertz
# abtast = 1.5
# distance = 1/(abtast*1000)

# last_sensor_read = int(time.time())
# distances = 0
# i = 0

# bus.write_byte_data(imu, 0x6b, 0)

# # csv header
# header = ["emg_0", "emg_1", "emg_2", "emg_3", "emg_4", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "dt"]
# writer.writerow(header)

# while true:
  # delta = time.time() - last_sensor_read
  # if delta < distance:
    # continue

  # last_sensor_read = time.time()
  # values = []

  # for a in addresses:
# #    print(a)
    # data = bus.read_i2c_block_data(a, 0, 2)
    # values.append((data[0] << 8) | data[1])

# #  values += get_imu_acc(imu)
# #  values += get_imu_gyro(imu)
  # values += get_imu_acc_raw(imu)
  # values += get_imu_gyro_raw(imu)

  # # zeit für eine iteration
  
  # values.append(delta)
  # writer.writerow(values)
  
  # distances += delta
  # i += 1
