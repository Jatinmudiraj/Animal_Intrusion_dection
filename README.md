# 🛡️ Animal Intrusion Detection System
This is a Flask web application for detecting animal intrusion across user-defined boundary lines or areas using YOLOv8 object detection and DeepSORT tracking.
It supports:

Live webcam feed 📷

Uploaded video files 🎥

Alert sounds 🔔

SMS notifications 📱 (via Twilio)

# 📋 Features
Object Detection: Detects animals based on the COCO dataset.

Object Tracking: Uses DeepSORT for assigning consistent IDs to moving animals.

Line Crossing Detection: Detects when an animal crosses a virtual boundary.

Area Intrusion Detection: Detects when an animal enters a defined area.

Sound Alerts: Plays a sound (one.mp3) on intrusion.

SMS Alerts: Sends a Twilio SMS when an intrusion is detected.

Click-Based ROI Selection: Define lines and areas interactively through the web app.

Live Stream: Stream live webcam or uploaded video with detection overlays.

# 🛠 Installation


pip install -r requirements.txt

## Main libraries used:

### Flask

### opencv-python

### ultralytics

### deep_sort_realtime

### twilio

### pygame

### numpy

### scipy

### munkres

Download YOLOv8 model: The code uses yolov8n.pt (Nano version). It's automatically handled by the ultralytics package, but make sure you have access.

Configure Twilio: Set your Twilio credentials in the script:


account_sid = 'YOUR_TWILIO_ACCOUNT_SID'
auth_token = 'YOUR_TWILIO_AUTH_TOKEN'
twilio_number = 'YOUR_TWILIO_PHONE_NUMBER'
recipient_number = 'RECIPIENT_PHONE_NUMBER'
Ensure one.mp3 (alert sound) exists in your project folder.

# 🚀 Running the Application
Start the server:


python app.py
Visit your browser:


http://127.0.0.1:5000/

# 🧩 Usage Flow
## 📍 1. Define Boundaries
Click points on the live feed to define:

Two boundary lines (each needs 2 clicks = 4 total).

Area polygon (multiple clicks).

## 📍 2. Submit Coordinates
Enter additional parameters (_N, _C, _H, _W dimensions if needed).

Submit via form.

## 📍 3. Live Detection
Live webcam (/video_feed) or uploaded video (/stream) starts.

Animals crossing the line or entering the area will:

Trigger a sound alert.

Send an SMS (rate limited to once every 10 minutes).

# 📁 Project Structure

animal-intrusion-detection/
│
├── static/uploads/         # Uploaded videos are saved here
├── templates/
│   ├── index.html           # Home and click input page
│   ├── video.html           # Video stream page
├── one.mp3                  # Sound file played on intrusion
├── app.py                   # Main application script
├── requirements.txt         # List of Python dependencies
└── README.md                 # This file


# 🔔 Important Notes
SMS sending is rate-limited: an SMS will not be sent more than once every 10 minutes per event type.

Make sure your Twilio number is verified and SMS sending is allowed to the recipient number.

pygame is used for sound playback — ensure audio output is available.

The application supports live streaming from a webcam or uploaded video (MP4 format recommended).

