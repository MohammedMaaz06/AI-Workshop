import os
from roboflow import Roboflow
from ultralytics import YOLO

# Load API key securely from environment variables
API_KEY = os.getenv("ROBOFLOW_API_KEY")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace("yolov5-recognition").project("hand-detection-asf0x")
version = project.version(5)
dataset = version.download("yolov8")

model = YOLO("yolov8n.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=80,
    imgsz=640,
    batch=-1,
)
