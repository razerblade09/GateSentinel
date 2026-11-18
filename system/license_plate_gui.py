import sys
import cv2
import sqlite3
import requests
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                            QGridLayout, QLineEdit, QMessageBox, QGroupBox,
                            QVBoxLayout, QHBoxLayout)
from PyQt5.QtGui import QPixmap, QFont, QImage, QPalette, QColor
from PyQt5.QtCore import Qt, QTimer
from camera_feed import CameraFeed
import os

# Define the absolute path to the database
DB_PATH = os.path.join(os.path.dirname(__file__), 'admin', 'license_plate_system.db')

class AutoCloseMessageBox(QMessageBox):
    def __init__(self, *args, timeout=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.timeout = timeout
        self.setModal(False)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.close)
        self.timer.start(timeout * 1000)

def fetch_car_details(plate_number):
    """Fetch car details from the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM cars WHERE plate_number = ?", (plate_number,))
            car = c.fetchone()
            if car:
                return dict(zip([column[0] for column in c.description], car))
    except sqlite3.Error as e:
        print(f"Error fetching car details: {e}")
    return None

def fetch_guest_details(plate_number):
    """Fetch guest details from the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM guests WHERE plate_number = ?", (plate_number,))
            guest = c.fetchone()
            if guest:
                return dict(zip([column[0] for column in c.description], guest))
    except sqlite3.Error as e:
        print(f"Error fetching guest details: {e}")
    return None

def add_guest(plate_number):
    """Add a new guest to the database."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            c = conn.cursor()
            c.execute("INSERT INTO guests (plate_number, access_status) VALUES (?, ?)", (plate_number, 'Active'))
            conn.commit()
    except sqlite3.Error as e:
        print(f"Error adding guest: {e}")

def log_plate_scan(plate_number, status):
    """Improved logging with cooldown handling"""
    try:
        response = requests.post(
            'http://localhost:5001/api/logs',
            json={'plate_number': plate_number, 'status': status}
        )
        if response.status_code == 200:
            print(f"Logged: {plate_number} - {status}")
            return True
        elif response.status_code == 400 and "Cooldown" in response.text:
            print(f"Cooldown active for {plate_number}")
            return False
        else:
            print(f"Logging failed: {response.text}")
            return False
    except Exception as e:
        print(f"Logging error: {e}")
        return False
    
class LicensePlateGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initCamera()
        self.guest_counter = 0  
        self.last_detected_plate = None
        self.last_detection_time = 0
        self.cooldown_duration = 5  # 5-second cooldown
        self.active_message_box = None

    def initUI(self):
        """Initialize the GUI with improved contrast"""
        self.setWindowTitle("License Plate Recognition System")
        self.setGeometry(100, 100, 1280, 720)
        
        # Enhanced color palette
        self.dark_palette = QPalette()
        self.dark_palette.setColor(QPalette.Window, QColor(18, 18, 18))
        self.dark_palette.setColor(QPalette.WindowText, QColor(240, 240, 240))
        self.setPalette(self.dark_palette)

        main_layout = QHBoxLayout()
        left_panel = QVBoxLayout()
        right_panel = QVBoxLayout()

        # Camera Feed Section
        camera_group = QGroupBox("Live Camera Feed")
        camera_group.setStyleSheet("""
            QGroupBox { 
                color: #42a5f5;
                font: bold 14px;
                border: 1px solid #404040;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 360)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; border-radius: 4px;")
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(self.camera_label)
        camera_group.setLayout(camera_layout)
        left_panel.addWidget(camera_group)

        # Captured Plate Section
        capture_group = QGroupBox("Captured Plate")
        capture_group.setStyleSheet("""
            QGroupBox { 
                color: #42a5f5;
                font: bold 14px;
                border: 1px solid #404040;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        self.captured_label = QLabel()
        self.captured_label.setMinimumSize(320, 180)
        self.captured_label.setAlignment(Qt.AlignCenter)
        self.captured_label.setStyleSheet("background-color: black; border-radius: 4px;")
        capture_layout = QVBoxLayout()
        capture_layout.addWidget(self.captured_label)
        capture_group.setLayout(capture_layout)
        left_panel.addWidget(capture_group)

        # Control Panel
        control_group = QGroupBox("System Controls")
        control_group.setStyleSheet("""
            QGroupBox { 
                color: #42a5f5;
                font: bold 14px;
                border: 1px solid #404040;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        control_layout = QVBoxLayout()

        # Plate Input
        plate_layout = QHBoxLayout()
        self.plate_label = QLabel("Detected Plate:")
        self.plate_label.setFont(QFont("Segoe UI", 11))
        self.plate_label.setStyleSheet("color: #e0e0e0;")
        self.plate_input = QLineEdit()
        self.plate_input.setFont(QFont("Segoe UI", 11))
        self.plate_input.setStyleSheet("""
            QLineEdit {
                background: #2d2d2d;
                color: #ffffff;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        plate_layout.addWidget(self.plate_label)
        plate_layout.addWidget(self.plate_input)
        control_layout.addLayout(plate_layout)

        # Search Button
        self.search_btn = QPushButton("Search Database")
        self.search_btn.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.search_btn.setStyleSheet("""
            QPushButton {
                background-color: #42a5f5;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 12px;
                margin-top: 10px;
            }
            QPushButton:hover {
                background-color: #1e88e5;
            }
        """)
        self.search_btn.clicked.connect(self.search_plate)
        control_layout.addWidget(self.search_btn)

        # Details Panel
        details_group = QGroupBox("Vehicle Details")
        details_group.setStyleSheet("""
            QGroupBox { 
                color: #42a5f5;
                font: bold 14px;
                border: 1px solid #404040;
                margin-top: 10px;
                padding-top: 15px;
            }
        """)
        self.details_label = QLabel("No vehicle information available")
        self.details_label.setFont(QFont("Segoe UI", 11))
        self.details_label.setStyleSheet("""
            QLabel {
                background: #2d2d2d;
                color: #e0e0e0;
                border-radius: 4px;
                padding: 15px;
            }
        """)
        self.details_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        details_layout = QVBoxLayout()
        details_layout.addWidget(self.details_label)
        details_group.setLayout(details_layout)
        control_layout.addWidget(details_group)

        control_group.setLayout(control_layout)
        right_panel.addWidget(control_group)

        main_layout.addLayout(left_panel, 60)
        main_layout.addLayout(right_panel, 40)
        self.setLayout(main_layout)

    def initCamera(self):
        """Initialize camera components"""
        self.camera_feed = CameraFeed()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def show_auto_close_message(self, title, message, icon=QMessageBox.Information, timeout=3):
        """Show a message box that automatically closes after timeout seconds"""
        if self.active_message_box:
            self.active_message_box.close()
        
        msg = AutoCloseMessageBox(icon, title, message, timeout=timeout)
        msg.show()
        self.active_message_box = msg

    def update_frame(self):
        """Update the camera feed and process detected plates."""
        frame = self.camera_feed.update_frame()
        if frame is None:
            return  

        # Display the processed frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.camera_label.setPixmap(QPixmap.fromImage(q_img))

        # Process detected plate with cooldown check
        detected_plate = self.camera_feed.get_license_plate_text()
        current_time = time.time()

        if detected_plate and detected_plate != self.last_detected_plate:
            if current_time - self.last_detection_time >= self.cooldown_duration:
                self.last_detection_time = current_time
                self.last_detected_plate = detected_plate
                self.plate_input.setText(detected_plate)
                
                car = fetch_car_details(detected_plate)
                if car:
                    status = car['status']
                    self.details_label.setText(f"Car Details:\nName: {car['owner_name']}\nModel: {car['car_model']}\nColor: {car['car_color']}\nStatus: {car['status']}")
                    if status == 'Expired':
                        self.show_auto_close_message("Registration Expired", f"Warning: {car['owner_name']}'s registration has expired!\nPlease renew.", QMessageBox.Warning)
                    else:
                        self.show_auto_close_message("Thank You", f"Thank you, {car['owner_name']}!")
                else:
                    guest = fetch_guest_details(detected_plate)
                    if guest:
                        if guest['access_status'] == 'Active':
                            self.details_label.setText("Car Details:\nGuest (Access Active)")
                            self.show_auto_close_message("Thank You", f"Thank you, Guest!\nPlease register within 24 hours.")
                        else:
                            self.details_label.setText("Car Details:\nGuest (Access Expired)")
                            self.show_auto_close_message("Access Denied", "Guest access has expired. Please register.", QMessageBox.Warning)
                    else:
                        self.details_label.setText("Car Details:\nNot Found")
                        self.show_auto_close_message("Thank You", f"Thank you, Guest!\nPlease register within 24 hours.")
                        add_guest(detected_plate)

                # Log the detected plate
                log_plate_scan(detected_plate, status if car else "Guest")

        # Display captured plate image
        captured_img = self.camera_feed.get_captured_plate_image()
        if captured_img is not None:
            captured_img_rgb = cv2.cvtColor(captured_img, cv2.COLOR_BGR2RGB)
            h, w, ch = captured_img_rgb.shape
            bytes_per_line = ch * w
            q_img = QImage(captured_img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.captured_label.setPixmap(QPixmap.fromImage(q_img))

    def search_plate(self):
        """Search and display car details."""
        plate_number = self.plate_input.text().strip()
        car = fetch_car_details(plate_number)
        if car:
            self.details_label.setText(
                f"Car Details:\nName: {car['owner_name']}\nModel: {car['car_model']}\nColor: {car['car_color']}\nStatus: {car['status']}"
            )
        else:
            self.details_label.setText("Car not found.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = LicensePlateGUI()
    window.show()
    sys.exit(app.exec_())