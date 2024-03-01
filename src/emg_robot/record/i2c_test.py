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

# Inicialização do objeto bus
bus = smbus.SMBus(1)

addresses = []
for a in sys.argv[1:-1]:
    addresses.append(locals()[a])
print(addresses)

while True:
    for a in addresses:
        try:
            # Leia os dados do bloco I2C
            data = bus.read_i2c_block_data(a, 0, 2)
            value = (data[0] << 8) | data[1]
            print(value)
        except IOError as e:
            print("Erro de I/O ao tentar ler dados do dispositivo:", e)
        time.sleep(1)
