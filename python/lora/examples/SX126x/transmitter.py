import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from LoRaRF import SX126x
import time
from pylorawan.encryption import aes128_encrypt, generate_mic
from pylorawan.common import encrypt_frm_payload, generate_mic_mac_payload
from pylorawan.message import MType, MHDR, FCtrlUplink, FHDRUplink, MACPayloadUplink, PHYPayload
# LoRaWAN parameters
dev_addr = 0x01020304 # Your device address
app_s_key = (0x01020304050607080910111213141516).to_bytes(16, "big")  # Your AppSKey
nwk_s_key = (0x01020304050607080910111213141516).to_bytes(16, "big")  # Your NwkSKey
frame_counter = 0

# New function that prepares LoRaWAN packet
def prepare_lorawan_packet(payload):
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

def send_data(data_bytes):
    # Begin LoRa radio and set NSS, reset, busy, IRQ, txen, and rxen pin with connected Raspberry Pi gpio pins
    # IRQ pin not used in this example (set to -1). Set txen and rxen pin to -1 if RF module doesn't have one
    busId = 0; csId = 0 
    resetPin = 18; busyPin = 20; irqPin = 16; txenPin = 6; rxenPin = -1 
    LoRa = SX126x()
    print("Begin LoRa radio")
    if not LoRa.begin(busId, csId, resetPin, busyPin, irqPin, txenPin, rxenPin) :
        raise Exception("Something wrong, can't begin LoRa radio")

    LoRa.setDio2RfSwitch()
    # Set frequency to 868 Mhz
    print("Set frequency to 868 Mhz")
    LoRa.setFrequency(868000000)

    # Set TX power, default power for SX1262 and SX1268 are +22 dBm and for SX1261 is +14 dBm
    # This function will set PA config with optimal setting for requested TX power
    print("Set TX power to +22 dBm")
    LoRa.setTxPower(22, LoRa.TX_POWER_SX1262)                       # TX power +17 dBm using PA boost pin

    # Configure modulation parameter including spreading factor (SF), bandwidth (BW), and coding rate (CR)
    # Receiver must have same SF and BW setting with transmitter to be able to receive LoRa packet
    print("Set modulation parameters:\n\tSpreading factor = 7\n\tBandwidth = 125 kHz\n\tCoding rate = 4/5")
    sf = 7                                                          # LoRa spreading factor: 7
    bw = 125000                                                     # Bandwidth: 125 kHz
    cr = 5                                                          # Coding rate: 4/5
    LoRa.setLoRaModulation(sf, bw, cr)

    # Configure packet parameter including header type, preamble length, payload length, and CRC type
    # The explicit packet includes header contain CR, number of byte, and CRC type
    # Receiver can receive packet with different CR and packet parameters in explicit header mode
    print("Set packet parameters:\n\tExplicit header type\n\tPreamble length = 12\n\tPayload Length = 15\n\tCRC on")
    headerType = LoRa.HEADER_EXPLICIT                               # Explicit header mode
    preambleLength = 12                                             # Set preamble length to 12
    payloadLength = 15                                              # Initialize payloadLength to 15
    crcType = True                                                  # Set CRC enable
    LoRa.setLoRaPacket(headerType, preambleLength, payloadLength, crcType)

    # Set syncronize word for public network (0x3444)
    print("Set syncronize word to 0x3444")
    LoRa.setSyncWord(0x3444)

    print("\n-- LoRa Transmitter --\n")

    # Message to transmit
    message = data_bytes
    #messageList = list(message)
    #for i in range(len(messageList)) : messageList[i] = ord(messageList[i])
    

    # Transmit message continuously


    # Transmit message and counter
    # write() method must be placed between beginPacket() and endPacket()
    LoRa.beginPacket()
    data_list = list(data_bytes)
    LoRa.write(data_list, len(data_list))
    LoRa.endPacket()

    # Wait until modulation process for transmitting packet finish
    LoRa.wait()

    # Print transmit time and data rate
    print("Transmit time: {0:0.2f} ms | Data rate: {1:0.2f} byte/s".format(LoRa.transmitTime(), LoRa.dataRate()))

    # Don't load RF module with continous transmit
    time.sleep(5)

    try :
        pass
    except :
        LoRa.end()
        
def main():
    while True:
        # Your sensor data collection code
        sensor_data = b"\x01\x02\x03\x04"  # Replace with actual sensor data
        
        # Prepare LoRaWAN packet
        lorawan_packet = prepare_lorawan_packet(sensor_data)
        
        # Send using your working function
        send_data(lorawan_packet)
        
        # Wait as needed
        import time
        time.sleep(60)

if __name__ == "__main__":
    main()
