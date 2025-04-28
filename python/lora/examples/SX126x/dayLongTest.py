import sys
import os
import time
import atexit
import argparse
import cv2
import numpy as np
from functools import lru_cache
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
# Import IMX500 Camera modules
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# Import LoRaWAN modules
from LoRaRF import SX126x
from pylorawan.encryption import aes128_encrypt, generate_mic
from pylorawan.common import encrypt_frm_payload, generate_mic_mac_payload
from pylorawan.message import MType, MHDR, FCtrlUplink, FHDRUplink, MACPayloadUplink, PHYPayload

# Custom utilities
from itkacher.date_utils import DateUtils
from itkacher.file_utils import FileUtils
from itkacher.video_recorder import VideoRecorder

# LoRaWAN parameters
dev_addr = 0x01020304  # Your device address
app_s_key = (0x01020304050607080910111213141516).to_bytes(16, "big")  # Your AppSKey
nwk_s_key = (0x01020304050607080910111213141516).to_bytes(16, "big")  # Your NwkSKey
frame_counter = 0  # In-memory counter, no file I/O

# Car counting parameters
car_count = 0
last_transmit_time = time.time()
TRANSMIT_INTERVAL = 60  # Changed to 60 seconds

# Detection parameters
last_detections = []
threshold = 0.55
iou = 0.65
max_detections = 10
car_class_id = None  # Will be determined based on the model

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

def prepare_lorawan_packet(payload):
    """Prepare LoRaWAN packet with encryption and framing"""
    global frame_counter
    
    # Convert to bytes if needed
    if not isinstance(payload, bytes):
        payload = bytes(payload)
    
    # Create LoRaWAN packet
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
    
    # Reset frame counter if it gets too large (prevent overflow in 24+ hour runs)
    if frame_counter > 0xFFFF:  # Use 16-bit limit (65535)
        frame_counter = 0
        
    return lorawan_packet

def send_data(data_bytes):
    """Initialize LoRa radio and transmit data"""
    # Begin LoRa radio and set pins
    busId = 0; csId = 0 
    resetPin = 18; busyPin = 20; irqPin = 16; txenPin = 6; rxenPin = -1 
    LoRa = SX126x()
    
    print("Begin LoRa radio")
    if not LoRa.begin(busId, csId, resetPin, busyPin, irqPin, txenPin, rxenPin):
        raise Exception("Something wrong, can't begin LoRa radio")

    LoRa.setDio2RfSwitch()
    # Set frequency to 868 Mhz
    print("Set frequency to 868 Mhz")
    LoRa.setFrequency(868100000)

    # Set TX power
    print("Set TX power to +22 dBm")
    LoRa.setTxPower(22, LoRa.TX_POWER_SX1262)

    # Configure modulation parameters
    print("Set modulation parameters:\n\tSpreading factor = 7\n\tBandwidth = 125 kHz\n\tCoding rate = 4/5")
    sf = 7
    bw = 125000
    cr = 5
    LoRa.setLoRaModulation(sf, bw, cr)

    # Configure packet parameters
    print("Set packet parameters")
    headerType = LoRa.HEADER_EXPLICIT
    preambleLength = 12
    payloadLength = 32
    crcType = True
    LoRa.setLoRaPacket(headerType, preambleLength, payloadLength, crcType)

    # Set syncronize word for public network
    print("Set syncronize word to 0x3444")
    LoRa.setSyncWord(0x3444)

    print("\n-- Transmitting Detection Data --\n")

    # Transmit message
    LoRa.beginPacket()
    data_list = list(data_bytes)
    LoRa.write(data_list, len(data_list))
    LoRa.endPacket()

    # Wait until modulation process for transmitting packet finish
    LoRa.wait()

    # Print transmit time and data rate
    print("Transmit time: {0:0.2f} ms | Data rate: {1:0.2f} byte/s".format(
        LoRa.transmitTime(), LoRa.dataRate()))

# Frame counter is now maintained in memory only
# This avoids file descriptor issues during long runs
frame_counter = 0  # Global counter initialized at 0

def warm_up_camera():
    """Warm up the camera and model by capturing a few frames"""
    print("Warming up camera and model...")
    for i in range(3):  # Capture 3 warm-up frames
        frame = picam2.capture_array()
        metadata = picam2.capture_metadata()
        # Try to get outputs but don't process them yet
        outputs = imx500.get_outputs(metadata, add_batch=True)
        print(f"Warm-up frame {i+1}: {'outputs received' if outputs is not None else 'no outputs'}")
        time.sleep(0.5)
    print("Warm-up complete")
        
def exit_handler():
    """Handle clean exit"""
    print("Exiting program")

def parse_detections(metadata: dict):
    """Parse the output tensor into detected objects, with better error handling."""
    global last_detections
    global intrinsics
    
    # Handle None intrinsics
    if intrinsics is None:
        print("WARNING: intrinsics is None, cannot parse detections")
        return last_detections
    
    # Safely get attributes with default values
    bbox_normalization = getattr(intrinsics, 'bbox_normalization', False)
    bbox_order = getattr(intrinsics, 'bbox_order', 'yx')
    
    # Get model outputs
    np_outputs = imx500.get_outputs(metadata, add_batch=True)
    input_w, input_h = imx500.get_input_size()
    if np_outputs is None:
        print("No outputs received from model")
        return last_detections
    
    # Handle different postprocessing methods
    try:
        if hasattr(intrinsics, 'postprocess') and intrinsics.postprocess == "nanodet":
            boxes, scores, classes = \
                postprocess_nanodet_detection(outputs=np_outputs[0], conf=threshold, iou_thres=iou,
                                            max_out_dets=max_detections)[0]
            from picamera2.devices.imx500.postprocess import scale_boxes
            boxes = scale_boxes(boxes, 1, 1, input_h, input_w, False, False)
        else:
            # Standard processing
            boxes, scores, classes = np_outputs[0][0], np_outputs[1][0], np_outputs[2][0]
            if bbox_normalization:
                boxes = boxes / input_h

            # Add the bbox_order handling
            if bbox_order == "xy":
                boxes = boxes[:, [1, 0, 3, 2]]

            boxes = np.array_split(boxes, 4, axis=1)
            boxes = zip(*boxes)

        # Create detection objects
        last_detections = [
            Detection(box, category, score, metadata)
            for box, score, category in zip(boxes, scores, classes)
            if score > threshold
        ]
        
        print(f"Detected {len(last_detections)} objects")
        return last_detections
    
    except Exception as e:
        print(f"Error in parse_detections: {e}")
        import traceback
        traceback.print_exc()
        return last_detections

@lru_cache
def get_labels():
    """Get model labels with better error handling"""
    global intrinsics
    
    if intrinsics is None:
        print("WARNING: intrinsics is None, returning default labels")
        return ["car"]  # Default for car-only model
        
    if not hasattr(intrinsics, 'labels'):
        print("WARNING: intrinsics has no 'labels' attribute, returning default labels")
        return ["car"]  # Default for car-only model
    
    labels = intrinsics.labels
    if hasattr(intrinsics, 'ignore_dash_labels') and intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    
    return labels

def find_car_class_id():
    """Find the class ID corresponding to 'car' in the model labels"""
    labels = get_labels()
    
    # For a single-class 'car' model, the class ID is typically 0
    if len(labels) == 1 and labels[0].lower() == 'car':
        return 0
        
    # Otherwise, look for car-related classes
    for i, label in enumerate(labels):
        if label.lower() in ['car', 'automobile', 'vehicle', 'truck', 'bus']:
            return i
            
    # Default to 0 if no car class found
    print("No specific car class found, using class 0")
    return 0

def draw_detections(request, stream="main"):
    """Draw the detections on the display"""
    detections = last_results
    if detections is None:
        return
    labels = get_labels()
    with MappedArray(request, stream) as m:
        for detection in detections:
            x, y, w, h = detection.box
            label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

            # Calculate text size and position
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = x + 5
            text_y = y + 15

            # Create a copy of the array to draw the background with opacity
            overlay = m.array.copy()

            # Draw the background rectangle on the overlay
            cv2.rectangle(overlay,
                          (text_x, text_y - text_height),
                          (text_x + text_width, text_y + baseline),
                          (255, 255, 255),  # Background color (white)
                          cv2.FILLED)

            alpha = 0.30
            cv2.addWeighted(overlay, alpha, m.array, 1 - alpha, 0, m.array)

            # Draw text on top of the background
            cv2.putText(m.array, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Draw detection box
            cv2.rectangle(m.array, (x, y), (x + w, y + h), (0, 255, 0, 0), thickness=2)

def draw_detections_on_frame(frame, detections):
    """Draw the detections on the frame and return the modified frame"""
    if detections is None:
        return frame
    
    labels = get_labels()
    frame_copy = frame.copy()
    
    for detection in detections:
        x, y, w, h = detection.box
        label = f"{labels[int(detection.category)]} ({detection.conf:.2f})"

        # Calculate text size and position
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        text_x = x + 5
        text_y = y + 15

        # Draw the background rectangle
        cv2.rectangle(frame_copy,
                     (text_x, text_y - text_height),
                     (text_x + text_width, text_y + baseline),
                     (255, 255, 255),  # Background color (white)
                     cv2.FILLED)

        # Draw text on top of the background
        cv2.putText(frame_copy, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw detection box
        cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    
    return frame_copy

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_tensors', action='store_true', help='Save tensor data')
    parser.add_argument('--record_video', action='store_true', help='Record video from images')
    parser.add_argument('--model', type=str, default="traffic.rpk", 
                      help='Path to the detection model')
    parser.add_argument('--display', action='store_true', help='Display video with detections using cv2.imshow')
    parser.add_argument('--display_every', type=int, default=10, 
                      help='Display every N frames to reduce processing load (default: 10)')
    parser.add_argument('--save_detections', action='store_true', help='Save images with detection boxes')
    args = parser.parse_args()

    # Register exit handler to save frame counter
    atexit.register(exit_handler)
    
    # Set the model path - you can override with --model argument
    model = args.model if args.model else "traffic.rpk"
    
    print(f"Using model: {model}")
    print(f"Model exists: {os.path.exists(model)}")
    
    # Initialize IMX500 with the model
    imx500 = IMX500(model)
    intrinsics = imx500.network_intrinsics
    
    # Create fallback intrinsics if needed
    if intrinsics is None:
        from picamera2.devices.imx500 import NetworkIntrinsics
        intrinsics = NetworkIntrinsics()
        intrinsics.task = "object detection"
        intrinsics.bbox_normalization = False
        intrinsics.labels = ["car"]  # Default to car detection
        intrinsics.ignore_dash_labels = False
        intrinsics.bbox_order = "yx"
        print("Created fallback intrinsics")
    
    # Check for labels.txt in the model directory
    try:
        model_dir = os.path.dirname(model) if os.path.dirname(model) else "."
        labels_path = os.path.join(model_dir, "labels.txt")
        
        if os.path.exists(labels_path):
            print(f"Found labels file: {labels_path}")
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
            
            # Set labels in intrinsics
            intrinsics.labels = labels
            print(f"Loaded labels: {labels}")
    except Exception as e:
        print(f"Error loading labels file: {e}")
        # Default to car if there's any issue
        intrinsics.labels = ["car"]
    
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls = {},
        buffer_count=12
    )

    imx500.show_network_fw_progress_bar()
    
    # Start the camera
    picam2.start(config, show_preview=False)
    
    # Warm up the camera and model
    warm_up_camera()
    
    # Get car class ID
    car_class_id = find_car_class_id()
    print(f"Car class ID: {car_class_id}")
    
    # Initialize video recorder if needed
    video_recorder = VideoRecorder() if args.record_video else None
    
    # Initialize frame counter for display
    frame_count = 0
    
    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            frame_count += 1
            
            # Capture and parse detections
            metadata = picam2.capture_metadata()
            last_results = parse_detections(metadata)
            
            # Display frame if needed
            if args.display and frame_count % args.display_every == 0:
                display_frame = draw_detections_on_frame(frame, last_results)
                cv2.imshow("Car Detection", display_frame)
                
                # Break loop if 'q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    print("Display closed by user")
            
            # Filter detections for cars and increment counter
            car_detections = [
                detection for detection in last_results
                if int(detection.category) == car_class_id
            ]
            
            # If cars are detected, increment the count
            if car_detections:
                car_count += len(car_detections)
                print(f"Car count: {car_count}")
                
            # Transmit car count if it's time (every 60 seconds)
            current_time = time.time()
            if current_time - last_transmit_time >= TRANSMIT_INTERVAL:
                print(f"Transmitting car count: {car_count}")
                
                # Create payload with just the car count (as 2 bytes to handle larger numbers)
                payload = car_count.to_bytes(2, byteorder='big')
                
                # Prepare and send LoRaWAN packet
                lorawan_packet = prepare_lorawan_packet(payload)
                send_data(lorawan_packet)
                
                # Reset car count and update last transmit time
                car_count = 0
                last_transmit_time = current_time
            
            # Record image to SD card if needed
            if args.record_video or args.save_tensors:
                data_folder = f"./data/images/{DateUtils.get_date()}/"
                try:
                    # Save image
                    current_time = DateUtils.get_time()
                    image_path = f"{data_folder}/{current_time}.jpg"
                    FileUtils.create_folders(data_folder)
                    picam2.capture_file(image_path)
                    
                    # Process video if needed
                    if args.record_video and video_recorder:
                        video_recorder.process_image(image_path)
                except Exception as e:
                    print(f"Error saving image: {e}")

            # Save detections if enabled
            if args.save_detections and (frame_count % args.display_every == 0 or car_detections):
                # Create folder if it doesn't exist
                detections_folder = f"./data/detections/{DateUtils.get_date()}/"
                FileUtils.create_folders(detections_folder)
                
                # Draw detections on the frame
                annotated_frame = draw_detections_on_frame(frame, last_results)
                
                # Save the annotated frame
                detection_path = f"{detections_folder}/{DateUtils.get_time()}_annotated.jpg"
                cv2.imwrite(detection_path, annotated_frame)
                print(f"Saved annotated frame to {detection_path}")
            
            # Wait before next capture - shorter wait to ensure we don't miss cars
            # but not too short to avoid overwhelming the system
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("Program terminated by user")
        cv2.destroyAllWindows()  # Close any open windows
        
        # Send final car count if there are any
        if car_count > 0:
            print(f"Transmitting final car count: {car_count}")
            payload = car_count.to_bytes(2, byteorder='big')
            try:
                lorawan_packet = prepare_lorawan_packet(payload)
                send_data(lorawan_packet)
            except Exception as e:
                print(f"Error sending final packet: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()  # Close any open windows
        
        # Try to recover from errors by restarting the main loop
        print("Attempting to recover and continue...")
        try:
            if 'picam2' in locals() and picam2 is not None:
                # Try to restart the camera
                picam2.stop()
                time.sleep(2)
                picam2.start(config, show_preview=False)
                time.sleep(1)
                continue  # Return to the start of the loop
        except:
            print("Could not recover. Exiting program.")
