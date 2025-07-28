from ultralytics import YOLO
from picamera2 import Picamera2, Preview
import cv2

# Cargar el modelo YOLOv8 entrenado en imágenes en escala de grises
model = YOLO("/home/pi/Desktop/crickets/train5/weights/best.pt")

# Iniciar la cámara con zoom
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

print("Detector de grillos corriendo. Presiona 'q' para salir.")

while True:
    frame = picam2.capture_array()
    
    # Convertir de BGRA a RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

    # Recortar el 15% de los lados izquierdo y derecho
    h, w, _ = frame_rgb.shape
    margin = int(0.15 * w)
    cropped_frame = frame_rgb[:, margin:w - margin]

    # Convertir a escala de grises
    gray_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_RGB2GRAY)

    # YOLO espera una imagen RGB, así que convertimos la imagen en escala de grises
    # de nuevo a formato de 3 canales (necesario si tu modelo fue entrenado así)
    gray_frame_3ch = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detección con el modelo YOLO
    results = model(gray_frame_3ch)

    # Visualización (marcamos sobre la imagen original recortada)
    annotated_frame = results[0].plot()

    cv2.imshow("Cricket Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.close()






