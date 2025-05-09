import json
import base64
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
    try:
        # Extract payload data - handles both direct invocation and IoT Core formats
        base64_payload = None
        if 'PayloadData' in event:
            # Standard AWS IoT Core LoRaWAN format
            base64_payload = event['PayloadData']
        elif 'payload' in event and 'PayloadData' in event['payload']:
            # Nested payload format
            base64_payload = event['payload']['PayloadData']
        else:
            # For testing, allow direct base64 string
            base64_payload = event
            
        # Convert base64 to bytes
        if isinstance(base64_payload, str):
            bytes_data = base64.b64decode(base64_payload)
        else:
            bytes_data = base64.b64decode(json.dumps(base64_payload))
        
        logger.info(f"Decoded bytes: {list(bytes_data)}")
        
        if len(bytes_data) < 2:
            raise ValueError("Payload too short - expected at least 2 bytes for car count")
        
        # Extract the car count from the first two bytes (big-endian)
        car_count = int.from_bytes(bytes_data[0:2], byteorder='big')
        logger.info(f"Number of cars detected in the last 60 seconds: {car_count}")
        
        # Create final response with metadata
        wireless_metadata = event.get('WirelessMetadata', {}).get('LoRaWAN', {})
        
        # Get current timestamp
        timestamp = datetime.now().isoformat()
        
        result = {
            "deviceId": event.get('WirelessDeviceId', 'unknown'),
            "timestamp": timestamp,
            "carCount": car_count,
            "interval": "60 seconds",
            "frameInfo": {
                "rssi": wireless_metadata.get('Rssi'),
                "snr": wireless_metadata.get('Snr'),
                "freq": wireless_metadata.get('Frequency')
            },
            "rawPayloadHex": bytes_data.hex()
        }
        
        logger.info(f"Decoded result: {json.dumps(result)}")
        return result
        
    except Exception as e:
        logger.error(f"Error decoding payload: {str(e)}")
        import traceback
        return {
            "error": True,
            "message": str(e),
            "stack": traceback.format_exc(),
            "input": event
        }
