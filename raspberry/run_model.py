from ultralytics import YOLO
from picamera2 import Picamera2, Preview
import cv2

# Load the YOLOv8 model trained on grayscale images
model = YOLO("/home/pi/Desktop/crickets/train5/weights/best.pt")

# Initialize the camera with zoom
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

print("Cricket detector running. Press 'q' to exit.")

while True:
    frame = picam2.capture_array()
    
    # Convert from BGRA to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    # Crop 15% from the left and right sides
    h, w, _ = frame_rgb.shape
    margin = int(0.15 * w)
    cropped_frame = frame_rgb[:, margin:w - margin]

    # Convert to grayscale
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)

    # YOLO expects an RGB image, so convert the grayscale image
    # back to 3-channel format (necessary if your model was trained this way)
    gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detection with the YOLO model
    results = model(gray_frame_3ch)

    # Visualization (draw results on the cropped original image)
    annotated_frame = results[0].plot()

    cv2.imshow("Cricket Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()






