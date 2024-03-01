import smbus

# Create an instance of the SMBus
bus = smbus.SMBus(1)

for device in range(128):
    try:
        bus.read_byte(device)
        print(f"Device found at address 0x{device:02x}")
    except Exception as e:
        pass
