from pylorawan.lorawan import LoRaWAN
from LoRaRF import SX126x
import time

# Initialize SX1262 radio
lora = SX126x()
lora.begin()

# Configure radio parameters
lora.setFrequency(915000000)  # Set for your region
lora.setTxPower(14)
lora.setSpreadingFactor(7)
lora.setBandwidth(125000)
lora.setCodingRate(5)
lora.setHeaderMode(lora.HEADER_EXPLICIT)
lora.setCrc(True)

# LoRaWAN parameters (from your AWS IoT Core settings)
dev_addr = bytes.fromhex("01020304")
nwk_s_key = bytes.fromhex("0102030405060708090A0B0C0D0E0F10")
app_s_key = bytes.fromhex("0102030405060708090A0B0C0D0E0F10")
frame_counter = 0

# Create LoRaWAN instance
lorawan = LoRaWAN(nwk_s_key, app_s_key)

while True:
    # Create your payload
    sensor_data = bytes([0x01, 0x02, 0x03, 0x04])  # Example payload
    
    # Create LoRaWAN packet
    lorawan_packet = lorawan.create_data_message(
        dev_addr=dev_addr,
        payload=sensor_data,
        fport=1,
        fcnt=frame_counter
    )
    
    # Send packet using SX1262
    lora.beginPacket()
    lora.write(lorawan_packet)
    lora.endPacket()
    
    # Increment frame counter
    frame_counter += 1
    
    # Wait before next transmission
    time.sleep(60)
