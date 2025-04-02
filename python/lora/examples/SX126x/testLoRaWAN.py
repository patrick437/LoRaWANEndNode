import os
import sys
import base64
import json
import struct
import time
from datetime import datetime

# Append path to import LoRaRF
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))

from LoRaRF import SX126x
import time
from pylorawan.encryption import aes128_encrypt, generate_mic
from pylorawan.common import encrypt_frm_payload, generate_mic_mac_payload
from pylorawan.message import MType, MHDR, FCtrlUplink, FHDRUplink, MACPayloadUplink, PHYPayload

# AWS IoT Core LoRaWAN Configuration
DEV_EUI = "2ccf67fefec3ab32"  # Replace with your DevEUI
WIRELESS_DEVICE_ID = "d0989744-3e86-4feb-a47e-e231ec08584b"  # Replace with your Wireless Device ID

# LoRaWAN parameters
dev_addr = 0x01020304  # Your device address
app_s_key = (0x01020304050607080910111213141516).to_bytes(16, "big")  # Your AppSKey
nwk_s_key = (0x01020304050607080910111213141516).to_bytes(16, "big")  # Your NwkSKey
frame_counter = 0

def create_sensor_payload(temperature=None, humidity=None, battery=None):
    """
    Create a structured sensor payload
    
    Args:
        temperature (float, optional): Temperature in °C
        humidity (float, optional): Humidity percentage
        battery (float, optional): Battery voltage
    
    Returns:
        bytes: Structured sensor payload
    """
    sensor_data = {
        "timestamp": int(time.time()),
        "temperature": temperature if temperature is not None else 0.0,
        "humidity": humidity if humidity is not None else 0.0,
        "battery": battery if battery is not None else 0.0,
        "device_eui": DEV_EUI
    }
    
    # Convert to JSON and then to bytes
    return json.dumps(sensor_data).encode('utf-8')

def prepare_lorawan_packet(payload):
    """
    Prepare LoRaWAN packet with encryption
    
    Args:
        payload (bytes): Sensor data to transmit
    
    Returns:
        bytes: Encrypted LoRaWAN packet
    """
    global frame_counter
    
    # Convert to bytes if needed
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    
    # Create LoRaWAN packet
    dev_addr_bytes = dev_addr.to_bytes(4, "little")
    mhdr = MHDR(mtype=MType.UnconfirmedDataUp, major=0)
    encrypted_payload = encrypt_frm_payload(payload, app_s_key, dev_addr, frame_counter, 0)
    f_ctrl = FCtrlUplink(adr=False, adr_ack_req=False, ack=False, class_b=False, f_opts_len=0)
    fhdr = FHDRUplink(dev_addr=dev_addr, f_ctrl=f_ctrl, f_cnt=frame_counter, f_opts=b"")
    mac_payload = MACPayloadUplink(fhdr=fhdr, f_port=1, frm_payload=encrypted_payload)
    mic = generate_mic_mac_payload(mhdr, mac_payload, nwk_s_key)
    phy_payload = PHYPayload(mhdr=mhdr, payload=mac_payload, mic=mic)
    lorawan_packet = phy_payload.generate()
    
    # Increment counter for next transmission
    frame_counter += 1
    
    return lorawan_packet

def prepare_aws_payload(lorawan_packet):
    """
    Prepare payload in AWS IoT Core LoRaWAN format
    
    Args:
        lorawan_packet (bytes): Encrypted LoRaWAN packet
    
    Returns:
        dict: AWS IoT Core compatible payload
    """
    return {
        "WirelessDeviceId": WIRELESS_DEVICE_ID,
        "PayloadData": base64.b64encode(lorawan_packet).decode('utf-8'),
        "WirelessMetadata": {
            "LoRaWAN": {
                "DevEui": DEV_EUI,
                "DevAddr": f"{dev_addr:08x}",
                "Timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    }

def send_data(data_bytes):
    """
    Transmit data via LoRa radio
    
    Args:
        data_bytes (bytes): Packet to transmit
    """
    # LoRa radio configuration
    busId = 0; csId = 0 
    resetPin = 18; busyPin = 20; irqPin = 16; txenPin = 6; rxenPin = -1 
    LoRa = SX126x()
    
    try:
        # Initialize LoRa radio
        print("Begin LoRa radio")
        if not LoRa.begin(busId, csId, resetPin, busyPin, irqPin, txenPin, rxenPin):
            raise Exception("Something wrong, can't begin LoRa radio")

        LoRa.setDio2RfSwitch()
        
        # Set frequency to 868 Mhz
        print("Set frequency to 868 Mhz")
        LoRa.setFrequency(868000000)

        # Set TX power
        print("Set TX power to +22 dBm")
        LoRa.setTxPower(22, LoRa.TX_POWER_SX1262)

        # Configure modulation parameters
        print("Set modulation parameters:\n\tSpreading factor = 7\n\tBandwidth = 125 kHz\n\tCoding rate = 4/5")
        sf = 7                  # LoRa spreading factor
        bw = 125000             # Bandwidth
        cr = 5                  # Coding rate
        LoRa.setLoRaModulation(sf, bw, cr)

        # Configure packet parameters
        print("Set packet parameters:\n\tExplicit header type\n\tPreamble length = 12\n\tPayload Length = 15\n\tCRC on")
        headerType = LoRa.HEADER_EXPLICIT
        preambleLength = 12
        payloadLength = 15
        crcType = True
        LoRa.setLoRaPacket(headerType, preambleLength, payloadLength, crcType)

        # Set synchronize word for public network
        print("Set synchronize word to 0x3444")
        LoRa.setSyncWord(0x3444)

        # Transmit packet
        print("\n-- LoRa Transmitter --\n")
        LoRa.beginPacket()
        data_list = list(data_bytes)
        LoRa.write(data_list, len(data_list))
        LoRa.endPacket()

        # Wait for transmission to complete
        LoRa.wait()

        # Print transmission details
        print("Transmit time: {0:0.2f} ms | Data rate: {1:0.2f} byte/s".format(
            LoRa.transmitTime(), LoRa.dataRate()))

    except Exception as e:
        print(f"Transmission error: {e}")
    finally:
        # Ensure radio is properly closed
        LoRa.end()

def main():
    """
    Main transmission loop
    """
    while True:
        try:
            # Simulate sensor data collection
            # Replace with actual sensor reading
            temperature = 23.5  # Example temperature
            humidity = 45.2     # Example humidity
            battery = 3.7       # Example battery voltage
            
            # Create sensor payload
            sensor_payload = create_sensor_payload(
                temperature, humidity, battery
            )
            
            # Prepare LoRaWAN packet
            lorawan_packet = prepare_lorawan_packet(sensor_payload)
            
            # Prepare AWS IoT Core payload
            aws_payload = prepare_aws_payload(lorawan_packet)
            
            # Print payload for debugging
            print("AWS Payload:")
            print(json.dumps(aws_payload, indent=2))
            
            # Decode and print the original payload for verification
            print("\nDecoded Payload:")
            decoded_payload = json.loads(base64.b64decode(aws_payload['PayloadData']).decode('utf-8'))
            print(json.dumps(decoded_payload, indent=2))
            
            # Send data via LoRa
            send_data(lorawan_packet)
            
            # Wait between transmissions
            time.sleep(300)  # 5-minute interval
        
        except Exception as e:
            print(f"Error in main loop: {e}")
            time.sleep(60)  # Wait before retrying

if __name__ == "__main__":
    main()
