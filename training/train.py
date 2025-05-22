
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

# Try the model
# Install the ultralytics package
from ultralytics import YOLO
model = YOLO("yolov8n.pt") #YOLOv8n is a small model, ideal for a Raspberry Pi

results = model(r"C:\Users\Juan Cantarero\Desktop\foto_golf.jpg")  # sample image
results[0].show()  # shows image with detections


# Test the model on CPU and GPU

import time
# Path to your image (you can use any image)
image_path = "https://ultralytics.com/images/bus.jpg"

# Test on CPU
model.to("cpu")
start_cpu = time.time()
results_cpu = model(image_path)
end_cpu = time.time()
print(f"🧠 CPU inference time: {end_cpu - start_cpu:.4f} seconds")

# Test on GPU (if available)
if torch.cuda.is_available():
    model.to("cuda")
    start_gpu = time.time()
    results_gpu = model(image_path)
    end_gpu = time.time()
    print(f"⚡ GPU inference time: {end_gpu - start_gpu:.4f} seconds")
else:
    print("🚫 CUDA not available — skipping GPU test")

# Show the result
results_gpu[0].show() if torch.cuda.is_available() else results_cpu[0].show()

# Load base model (YOLOv8 nano version)
model = YOLO("yolov8n.pt")

# Train on your custom dataset
model.train(
    data=r"C:\Users\Juan Cantarero\Desktop\CHICAGO\IIT\COURSES\TFM\Crickets\datasets\My First Project.v1i.yolov8\data.yaml",
    epochs=50,
    imgsz=512,       # or even 416 to save memory
    batch=4,         # safe value for 2GB VRAM
    device=0         # in this case, I use my MX150 GPU
)

# Save the model in PyTorch format
model.save(r"C:\Users\Juan Cantarero\Desktop\CHICAGO\IIT\COURSES\TFM\Code\modelo_entrenado.pt")

