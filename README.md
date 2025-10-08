# 🦗 Real-Time Cricket Detection on Raspberry Pi

This project uses a Raspberry Pi equipped with the official Pi Camera and a custom-trained YOLOv8 nano model to detect crickets in real time. It leverages picamera2, OpenCV, and the ultralytics library to capture video frames, run inference, and display bounding boxes over detected insects.
After detection, a mechanical flipper arm—similar to those found in pinball machines—is triggered to sort the crickets into two separate boxes based on their predicted sex (male or female).

---

## 🚀 Features

- 🧠 Real-time object detection using YOLOv8 nano 
- 🎥 Seamless integration with the official Raspberry Pi Camera using `picamera2`  
- 💡 Minimal dependencies — optimized for Raspberry Pi OS (Bookworm or later)  
- 🐍 Clean, modular Python code, easy to modify or extend  

---

## 🖥️ Requirements & Installation

On the Raspberry Pi:  
Install system dependencies, create a virtual environment, and install required Python packages:

```bash
sudo apt update
sudo apt install -y python3-picamera2

python3 -m venv --system-site-packages ~/yoloenv
source ~/yoloenv/bin/activate
pip install ultralytics opencv-python
```

---

## 🧠 Model

The model used in this project was trained with Ultralytics YOLOv8 nano on a custom dataset of crickets.  
You can explore and download the dataset from Roboflow:

🔗 **[Cricket Detection Dataset on Roboflow](https://universe.roboflow.com/cricket-sex-classification/auto_crickets-cf8wj)**

If you'd like to use your own data, check out the [Ultralytics training guide](https://docs.ultralytics.com).

Example model path:

```
/home/pi/Desktop/crickets/train5/weights/best.pt
```

---

## ▶️ Run the Detector

From your project folder:

```bash
source ~/yoloenv/bin/activate
python run_model.py
```

Exit detection by pressing **`q`**.

---

## 📂 Project Structure

```
cricket-pipeline/
├── model/
│   └── results/
│       └── ...
│   └── weights/
│       └── best.pt
│       └── last.pt
├── training/
│   └── train.py
├── raspberry/
│   └── real_time_inference.py
├── docs/
│   └── cricket_demo.gif # Optional demo animation
└── README.md
```

---

## 🛠️ Troubleshooting

- ❌ **No frame captured** → Ensure the camera is connected and enabled via `raspi-config`  
- ❌ **Qt plugin error** → Use `Preview.NULL` instead of `Preview.QTGL`  
- ❌ **“Expected 3 channels, got 4”** → Add `frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)` before inference  

---

## 👤 Author

This project was developed by **Juan Cantarero**  
**Supervisor:** Dr. Matthew Smith  
Illinois Institute of Technology – Master's in Artificial Intelligence                      
Universidad Politecnica de Madrid – Master's in Industrial Engineering

---

## 📜 License

This project is publicly available under the [Creative Commons Attribution 4.0 International (CC BY 4.0) License](https://creativecommons.org/licenses/by/4.0/).

You are free to **use, share, and adapt** the material for any purpose, including commercial use, as long as you provide **appropriate credit**.

