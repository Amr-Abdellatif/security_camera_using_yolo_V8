# security_camera_using_yolo_V8

## Video Explaining the whole project in arabic

https://www.youtube.com/watch?v=61GI5U-yX2c&t=1s

![cv](https://github.com/Amr-Abdellatif/security_camera_using_yolo_V8/assets/92921252/63867958-c0a7-4f2a-a129-db6ddbe65a38)

# Steps

 YOLOv8 Object Detection with Email Alerts

This project uses the YOLOv8 model for real-time object detection with an optional email alert feature to notify you of detected objects. The code in this repository captures video from a camera feed (webcam) and detects objects, specifically focusing on detecting persons.

## Prerequisites

- Python 3.7 or higher
- PyTorch
- OpenCV (cv2)
- [Ultralytics YOLO V8]
- `supervision` library (ensure you have it installed)

## Getting Started

1. Clone this repository to your local machine:

```bash
git clone (https://github.com/Amr-Abdellatif/security_camera_using_yolo_V8.git)
```

2. Install the required dependencies using pip install -r requirments.txt

3. Edit the email_settings.py file with your email configuration.

## Usage

Run the object detection script: ## security_cam.py

1.The script will capture video from the specified camera feed (change capture_index if necessary) and perform real-time object detection.

2.Detected objects, specifically persons, are highlighted in the video feed.

3.The FPS (frames per second) is displayed in the top-left corner of the video feed.

4.Press Esc to exit the script

## Email Alerts

You can configure the script to send email alerts when persons are detected in the video feed. Make sure to set up your email settings in the email_settings.py file.
