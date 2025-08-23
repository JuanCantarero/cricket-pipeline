from ultralytics import YOLO
from picamera2 import Picamera2, Preview
import cv2
import RPi.GPIO as GPIO
from time import sleep
import numpy as np
from collections import deque

# -------------------- Parameters --------------------
CONF_THRESHOLD = 0.5
ACTIVATION_Y_THRESHOLD = 0
CENTER_ANGLE = 70
STEP = 45
MIN_ANGLE = 20
MAX_ANGLE = 135
DETECTION_WINDOW_SIZE = 10

# -------------------- Servo Setup --------------------
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(16, GPIO.OUT)
pwm = GPIO.PWM(16, 50)
pwm.start(0)

def angle_to_duty_cycle(angle):
    return 2.5 + (angle / 180.0) * 10

def move_to_angle(angle):
    angle = max(MIN_ANGLE, min(MAX_ANGLE, angle))
    duty = angle_to_duty_cycle(angle)
    pwm.ChangeDutyCycle(duty)
    sleep(0.5)
    pwm.ChangeDutyCycle(0)

current_angle = CENTER_ANGLE
move_to_angle(current_angle)

# -------------------- YOLO Model and Camera --------------------
model = YOLO("/home/pi/Desktop/crickets/train5/weights/best.pt")

picam2 = Picamera2()
sensor_width, sensor_height = picam2.sensor_resolution
zoom_factor = 1
zoom_width = int(sensor_width / zoom_factor)
zoom_height = int(sensor_height / zoom_factor)
x = (sensor_width - zoom_width) // 2
y = (sensor_height - zoom_height) // 2

config = picam2.create_preview_configuration(
    main={"size": (640, 480)},
    controls={"ScalerCrop": (x, y, zoom_width, zoom_height)}
)

picam2.configure(config)
picam2.start_preview(Preview.NULL)
picam2.start()

detection_window = deque(maxlen=DETECTION_WINDOW_SIZE)

# -------------------- Main Loop --------------------
try:
    while True:
        # Capture and preprocess frame
        frame = picam2.capture_array()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        h, w, _ = frame_rgb.shape
        margin = int(0.15 * w)
        cropped_frame = frame_rgb[:, margin:w - margin]

        gray = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)
        input_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        results = model(input_frame)
        boxes = results[0].boxes
        detected_label = None

        if boxes is not None:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                if conf < CONF_THRESHOLD:
                    continue

                label = model.names[cls_id].lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                y_center = (y1 + y2) // 2

                if y_center > ACTIVATION_Y_THRESHOLD and label in ['male', 'female']:
                    detected_label = label
                    print(f"Detected: {detected_label}")
                    break

        detection_window.append(detected_label)
        males = detection_window.count('male')
        females = detection_window.count('female')

        # Decision logic
        if males > females and males > 0:
            desired_angle = CENTER_ANGLE - STEP
            if current_angle != desired_angle:
                current_angle = desired_angle
                move_to_angle(current_angle)
        elif females > males and females > 0:
            desired_angle = CENTER_ANGLE + STEP
            if current_angle != desired_angle:
                current_angle = desired_angle
                move_to_angle(current_angle)

except KeyboardInterrupt:
    pass

finally:
    picam2.close()
    pwm.stop()
    GPIO.cleanup()







