#!/usr/bin/env python3
# Beispielhafte Benutzung:
# python main.py A0 A1 A2 A3 A4 Bewegung_1
# python main.py A2 A4 A0 A3 A1 Bewegung_1_1
import smbus 
import time 
import sys 
import csv

# wir müssen darauf achten dass insgesamt 5 adressen angegeben wurden
if len(sys.argv) < 7:
  print("Bitte alle 5 Sensor Adressen angeben und die Bewgung die ausgeführt wird")
  exit(1)

A0 = 0x48
A1 = 0x49
A2 = 0x4A 
A3 = 0x4B
A4 = 0x4C 
A5 = 0x4D 
A6 = 0x4E
A7 = 0x4F 

bus = smbus.SMBus(1)
# get the address from programm arguments 
address1 = locals()[sys.argv[1]]
address2 = locals()[sys.argv[2]]
address3 = locals()[sys.argv[3]]
address4 = locals()[sys.argv[4]]
address5 = locals()[sys.argv[5]]

################################################
# letztes Argument wird als name für die datei benutzt
FILENAME = f"./data-{sys.argv[6]}.csv"
print(FILENAME)
file = open(FILENAME, "w")
writer = csv.writer(file)

# in kiloHertz
ABTAST = 1.5
INTERVAL = 1/(ABTAST*1000)

countdown_timer = 0
last_time = int(time.time())
while True:
  if int(time.time()) - last_time >= 1 and countdown_timer < 13 :
    countdown_timer += 1
    print(countdown_timer)
    last_time = int(time.time())
  if countdown_timer == 13:
    print("Bewegung jetzt starten")
    countdown_timer = 14
  data1 = bus.read_i2c_block_data(address1, 0, 2)
  data2 = bus.read_i2c_block_data(address2, 0, 2)
  data3 = bus.read_i2c_block_data(address3, 0, 2)
  data4 = bus.read_i2c_block_data(address4, 0, 2)
  data5 = bus.read_i2c_block_data(address5, 0, 2)
  value1 = (data1[0] << 8) | data1[1]
  value2 = (data2[0] << 8) | data2[1]
  value3 = (data3[0] << 8) | data3[1]
  value4 = (data4[0] << 8) | data4[1]
  value5 = (data5[0] << 8) | data5[1]
  # print dauert zu lange und verlangsamt den loop zu sehr
  # print(value1, value2, value3, value4, value5)
  writer.writerow([ value1, value2, value3, value4, value5 ])
  time.sleep(INTERVAL) 

