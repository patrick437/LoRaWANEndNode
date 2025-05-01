import sys
import os
import time
import argparse
import cv2
import numpy as np
from functools import lru_cache

# Import IMX500 Camera modules
from picamera2 import MappedArray, Picamera2
from picamera2.devices import IMX500
from picamera2.devices.imx500 import (NetworkIntrinsics, postprocess_nanodet_detection)

# Custom utilities
from itkacher.date_utils import DateUtils
from itkacher.file_utils import FileUtils
from itkacher.video_recorder import VideoRecorder

# Detection parameters
last_detections = []
threshold = 0.55
iou = 0.65
max_detections = 10

class Detection:
    def __init__(self, coords, category, conf, metadata):
        """Create a Detection object, recording the bounding box, category and confidence."""
        self.category = category
        self.conf = conf
        self.box = imx500.convert_inference_coords(coords, metadata, picam2)

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
        
        # Save tensor data if enabled
        if args.save_tensors and len(last_detections) > 0:
            try:
                timestamp = DateUtils.get_time()
                tensor_folder = f"./data/tensors/{DateUtils.get_date()}/"
                FileUtils.create_folders(tensor_folder)
                tensor_outputs = [boxes, scores, classes]
                
                if video_recorder:
                    video_recorder.save_tensor_data(tensor_outputs, timestamp, tensor_folder)
            except Exception as e:
                print(f"Error saving tensor data: {e}")
        
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
        return ["object"]  # Default label
        
    if not hasattr(intrinsics, 'labels'):
        print("WARNING: intrinsics has no 'labels' attribute, returning default labels")
        return ["object"]  # Default label
    
    labels = intrinsics.labels
    if hasattr(intrinsics, 'ignore_dash_labels') and intrinsics.ignore_dash_labels:
        labels = [label for label in labels if label and label != "-"]
    
    return labels

def draw_detections_on_frame(frame, detections):
    """Draw the detections on the frame and return the modified frame"""
    if detections is None:
        return frame
    
    labels = get_labels()
    frame_copy = frame.copy()
    
    for detection in detections:
        x, y, w, h = detection.box
        
        # Ensure label index is valid
        label_idx = int(detection.category)
        if 0 <= label_idx < len(labels):
            label_text = labels[label_idx]
        else:
            label_text = "Unknown"
            
        label = f"{label_text} ({detection.conf:.2f})"

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

def create_video_from_frames(image_folder, output_path, framerate=30):
    """Create a video from a folder of images"""
    try:
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        if not images:
            print(f"No images found in {image_folder}")
            return False
            
        # Sort images by timestamp in filename
        images.sort()
        
        # Get the dimensions of the first image
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
        video = cv2.VideoWriter(output_path, fourcc, framerate, (width, height))
        
        # Add each image to the video
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
            
        # Release the video writer
        video.release()
        print(f"Video created: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

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
    parser.add_argument('--video_duration', type=int, default=10, 
                      help='Duration of video in seconds (default: 10)')
    args = parser.parse_args()
    
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
        intrinsics.labels = ["object"]  # Default label
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
        # Default to generic label if there's any issue
        intrinsics.labels = ["object"]
    
    # Initialize the camera
    picam2 = Picamera2(imx500.camera_num)
    config = picam2.create_preview_configuration(
        controls = {},
        buffer_count=12
    )

    if hasattr(imx500, 'show_network_fw_progress_bar'):
        imx500.show_network_fw_progress_bar()
    
    # Start the camera
    picam2.start(config, show_preview=False)
    
    # Warm up the camera and model
    warm_up_camera()
    
    # Initialize video recorder if needed
    video_recorder = VideoRecorder() if args.record_video else None
    
    # Calculate frames per video based on duration
    FRAMES_PER_VIDEO = 30 * args.video_duration  # 30fps × duration
    
    # Initialize frame counter and image counter
    frame_count = 0
    image_count = 0
    
    # Create temporary folder for video frames
    video_frames_folder = f"./data/video_frames/{DateUtils.get_date()}/"
    FileUtils.create_folders(video_frames_folder)
    
    try:
        print("Starting object detection...")
        while True:
            # Capture frame
            frame = picam2.capture_array()
            frame_count += 1
            
            # Capture and parse detections
            metadata = picam2.capture_metadata()
            last_results = parse_detections(metadata)
            
            # Draw detections on frame
            annotated_frame = draw_detections_on_frame(frame, last_results)
            
            # Display frame if needed
            if args.display and frame_count % args.display_every == 0:
                cv2.imshow("Object Detection", annotated_frame)
                
                # Break loop if 'q' is pressed
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    print("Display closed by user")
                    break
            
            # Print detected objects
            if last_results:
                labels = get_labels()
                for result in last_results:
                    label_idx = int(result.category)
                    if 0 <= label_idx < len(labels):
                        label_text = labels[label_idx]
                    else:
                        label_text = "Unknown"
                    
                    confidence = result.conf
                    x, y, w, h = result.box
                    print(f"Detected {label_text} with confidence {confidence:.2f} at position ({x}, {y}, {w}, {h})")
            
            # Save annotated frame if video recording is enabled
            if args.record_video:
                current_time = DateUtils.get_time()
                frame_path = f"{video_frames_folder}/{current_time}.jpg"
                cv2.imwrite(frame_path, annotated_frame)
                image_count += 1
                
                # Create video if enough frames collected
                if image_count >= FRAMES_PER_VIDEO:
                    video_folder = f"./data/videos/{DateUtils.get_date()}/"
                    FileUtils.create_folders(video_folder)
                    output_video = f"{video_folder}/video_{current_time}.mp4"
                    
                    if create_video_from_frames(video_frames_folder, output_video):
                        # Clear the frames folder after creating video
                        for file in os.listdir(video_frames_folder):
                            file_path = os.path.join(video_frames_folder, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.unlink(file_path)
                            except Exception as e:
                                print(f"Error deleting {file_path}: {e}")
                        
                        # Reset counter
                        image_count = 0
            
            # Save detections if enabled
            if args.save_detections and last_results:
                # Create folder if it doesn't exist
                detections_folder = f"./data/detections/{DateUtils.get_date()}/"
                FileUtils.create_folders(detections_folder)
                
                # Save the annotated frame
                detection_path = f"{detections_folder}/{DateUtils.get_time()}_annotated.jpg"
                cv2.imwrite(detection_path, annotated_frame)
                
            # Short delay
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("Program terminated by user")
        cv2.destroyAllWindows()  # Close any open windows
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()  # Close any open windows
    finally:
        # Clean up
        try:
            picam2.stop()
            print("Camera stopped")
        except:
            pass
