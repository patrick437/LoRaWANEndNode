from picamera2 import Picamera2
from picamera2.devices import IMX500
import time
import cv2

# Simple test script for IMX500 with RPK model
def test_imx500_model(model_path):
    print(f"Testing model: {model_path}")
    
    try:
        # Initialize IMX500 with the model
        imx500 = IMX500(model_path)
        print("IMX500 initialized successfully")
        
        # Get network intrinsics and print them
        intrinsics = imx500.network_intrinsics
        print(f"Network intrinsics: {intrinsics}")
        
        # Initialize camera
        picam2 = Picamera2(imx500.camera_num)
        config = picam2.create_preview_configuration(
            controls={},
            buffer_count=4
        )
        
        # Start camera
        print("Starting camera...")
        picam2.start(config, show_preview=False)
        print("Camera started")
        
        # Capture a few frames and process them
        for i in range(3):
            print(f"Capturing frame {i+1}")
            # Capture frame
            frame = picam2.capture_array()
            
            # Capture metadata with model inference
            metadata = picam2.capture_metadata()
            
            # Try to get model outputs
            outputs = imx500.get_outputs(metadata, add_batch=True)
            print(f"Model outputs: {outputs}")
            
            # Short pause between captures
            time.sleep(1)
        
        # Clean up
        picam2.stop()
        print("Test completed successfully")
        return True
        
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test the model
    model_path = "traffic.rpk"  # Replacth your model path
    success = test_imx500_model(model_path)
    print(f"Model test {'succeeded' if success else 'failed'}")
