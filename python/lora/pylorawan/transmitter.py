import sys
sys.path.append('/home/patrick/pylorawan')  # Add the parent directory

from pylorawan.encryption import aes128_encrypt, generate_mic
from pylorawan.common import encrypt_frm_payload, generate_mic_mac_payload
from pylorawan.message import MType, MHDR, FCtrlUplink, FHDRUplink, MACPayloadUplink, PHYPayload

# Import your existing LoRaRF modules
from pylorawan.LoRaRF import SX126x  # Or whatever specific module you're using

# Initialize your LoRa module
lora = SX126x()
lora.begin()

# Configure radio parameters
lora.setFrequency(868000000)  # Set for your region
lora.setTxPower(14)
lora.setSpreadingFactor(7)
lora.setBandwidth(125000)
lora.setCodeRate(5)
lora.setHeaderMode(lora.HEADER_EXPLICIT)
lora.setCrc(True)

# LoRaWAN parameters (from AWS IoT Core)
dev_addr = 0x01020304  # Replace with your actual device address
app_s_key = (0x01020304050607080910111213141516).to_bytes(16, "big")  # Replace with your AppSKey
nwk_s_key = (0x01020304050607080910111213141516).to_bytes(16, "big")  # Replace with your NwkSKey
frame_counter = 0  # Initialize frame counter

def send_lorawan_packet(payload):
    global frame_counter
    
    # Convert your payload to bytes if it isn't already
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    
    # Convert device address to bytes
    dev_addr_bytes = dev_addr.to_bytes(4, "little")
    
    # Create LoRaWAN packet structure
    mtype = MType.UnconfirmedDataUp
    mhdr = MHDR(mtype=mtype, major=0)
    
    # Encrypt payload
    encrypted_payload = encrypt_frm_payload(
        payload, app_s_key, dev_addr_bytes, frame_counter, 0  # 0 for uplink
    )
    
    # Create frame header
    f_ctrl = FCtrlUplink(
        adr=False, adr_ack_req=False, ack=False, class_b=False, f_opts_len=0
    )
    fhdr = FHDRUplink(
        dev_addr=dev_addr_bytes, f_ctrl=f_ctrl, f_cnt=frame_counter, f_opts=b""
    )
    
    # Create MAC payload
    mac_payload = MACPayloadUplink(
        fhdr=fhdr, f_port=1, frm_payload=encrypted_payload
    )
    
    # Generate MIC
    mic = generate_mic_mac_payload(mhdr, mac_payload, nwk_s_key)
    
    # Create complete physical payload
    phy_payload = PHYPayload(mhdr=mhdr, payload=mac_payload, mic=mic)
    lorawan_packet = phy_payload.generate()
    
    # Send the packet using your LoRaRF
    lora.beginPacket()
    lora.write(lorawan_packet)
    lora.endPacket()
    
    # Increment frame counter for next transmission
    frame_counter += 1
    
    return lorawan_packet  # Return for debugging if needed

# Main program
if __name__ == "__main__":
    import time
    
    try:
        while True:
            # Example: Read sensor data or create your payload
            # Replace this with your actual payload creation code
            sensor_data = b"\x01\x02\x03\x04"  # Example data bytes
            
            # Send as LoRaWAN packet
            packet = send_lorawan_packet(sensor_data)
            print(f"Sent LoRaWAN packet: {packet.hex()}")
            
            # Wait before next transmission (respect duty cycle if in EU region)
            time.sleep(60)  # Wait 60 seconds between transmissions
    
    except KeyboardInterrupt:
        print("Program terminated by user")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        # Clean shutdown if needed
        print("Shutting down radio...")
        # Add any cleanup code for your radio if needed
