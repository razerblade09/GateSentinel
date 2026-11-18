import cv2
from ultralytics import YOLO
import pytesseract
import numpy as np
import re
import requests
import time  # Added for cooldown tracking

class CameraFeed:
    def __init__(self):
        # Load YOLO model (use relative path from this file's location)
        _model_path = os.path.join(os.path.dirname(__file__), '..', 'runs', 'detect', 'train16', 'weights', 'best.pt')
        self.model = YOLO(_model_path)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)  # Use 0 for default, 1 for external camera
        self.license_plate_text = ""
        self.captured_plate_image = None
        self.last_log_times = {}  # Track last log time per plate
        if not self.cap.isOpened():
            raise Exception("Camera not accessible")

    def is_valid_ph_plate(self, text):
        """Check if the license plate is a valid Philippine plate format."""
        four_wheel_pattern = r'^[A-Z]{3}-?\d{4}$'
        motorcycle_pattern = r'(^\d{3}[A-Z]{3}$)|(^\d{4}[A-Z]{2}$)|(^[A-Z]{2}\d{5}$)'
        return (
            bool(re.match(four_wheel_pattern, text)) or
            bool(re.match(motorcycle_pattern, text))
        )

    def process_license_plate_text(self, text):
        """Clean and validate license plate text."""
        # Remove all non-alphanumeric/dash characters and convert to uppercase
        cleaned = re.sub(r'[^A-Z0-9]', '', text.upper().strip())
        if self.is_valid_ph_plate(cleaned):
            return cleaned
        return None

    def log_plate_scan(self, plate_number, status):
        """Send plate scan log to the server with cooldown check."""
        try:
            current_time = time.time()
            last_time = self.last_log_times.get(plate_number, 0)
            
            # Client-side cooldown check
            if current_time - last_time < 300:
                print(f"Plate {plate_number} on cooldown. Skipping log.")
                return

            # Send log data to the server
            response = requests.post(
                'http://localhost:5001/api/logs',
                json={'plate_number': plate_number, 'status': status}
            )
            if response.status_code == 200:
                print(f"Logged plate: {plate_number} - Status: {status}")
                self.last_log_times[plate_number] = current_time  # Update on success
            elif response.status_code == 400 and response.json().get('error') == 'Cooldown active':
                print(f"Plate {plate_number} on cooldown (server-side).")
            else:
                print(f"Failed to log plate: {response.json().get('error')}")
        except requests.exceptions.RequestException as e:
            print(f"Error logging plate: {e}")

    def update_frame(self):
        """Capture and process a frame from the camera feed."""
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image from webcam.")
                return None

            # Perform YOLO detection
            results = self.model.predict(frame, conf=0.3)

            for result in results:
                image = result.plot()  # Draw detections on the frame
                for box in result.boxes:
                    conf = box.conf[0].item()  # Get confidence score
                    if conf < 0.75:
                        continue  # Skip low-confidence detections

                    # Extract license plate region
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    license_plate_region = frame[y1:y2, x1:x2]

                    # Preprocess for OCR
                    license_plate_gray = cv2.cvtColor(license_plate_region, cv2.COLOR_BGR2GRAY)
                    license_plate_blur = cv2.GaussianBlur(license_plate_gray, (5, 5), 0)
                    license_plate_thresh = cv2.adaptiveThreshold(
                        license_plate_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                        cv2.THRESH_BINARY, 11, 2
                    )

                    # Sharpening filter
                    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                    license_plate_sharp = cv2.filter2D(license_plate_thresh, -1, kernel)

                    # OCR
                    self.captured_plate_image = license_plate_sharp
                    raw_text = pytesseract.image_to_string(
                        self.captured_plate_image, 
                        config='--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    )
                    
                    # Validate plate format
                    valid_plate = self.process_license_plate_text(raw_text)
                    if valid_plate:
                        current_time = time.time()
                        last_time = self.last_log_times.get(valid_plate, 0)
                        if current_time - last_time >= 300:
                            self.license_plate_text = valid_plate
                            print(f"Valid Plate: {valid_plate} (Confidence: {conf:.2f})")
                        else:
                            print(f"Plate {valid_plate} on cooldown. Skipping.")
                            self.license_plate_text = ""
                    else:
                        self.license_plate_text = ""
                        print(f"Rejected invalid format: {raw_text.strip()}")

            return image
        except Exception as e:
            print(f"Error in update_frame: {e}")
            return None

    def get_license_plate_text(self):
        """Return the detected license plate text."""
        return self.license_plate_text

    def get_captured_plate_image(self):
        """Return the captured license plate image."""
        return self.captured_plate_image

    def release(self):
        """Release the camera feed."""
        self.cap.release()