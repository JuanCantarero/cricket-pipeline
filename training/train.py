
import torch
from ultralytics import YOLO

def train_custom_model():
    model = YOLO("yolov8n.pt")  # Carga el modelo base YOLOv8 nano

    model.train(
        data=r"C:\Users\Juan Cantarero\Desktop\CHICAGO\IIT\COURSES\TFM\Crickets\datasets\auto_crickets.v3i.yolov8\data.yaml",
        epochs=50,
        imgsz=512,    # Reduce tamaño para ahorrar VRAM
        batch=4,      # Batch pequeño para GPU con poca memoria
        device=0,     # GPU ID 0 (tu MX150)
        workers=0,
        cache=False,  # Desactiva el cacheo para ahorrar memoria  
    )

    # Guarda el modelo entrenado
    model.save(r"C:\Users\Juan Cantarero\Desktop\CHICAGO\IIT\COURSES\TFM\Code\modelo_entrenado.pt")

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()  # Necesario en Windows
    train_custom_model()

