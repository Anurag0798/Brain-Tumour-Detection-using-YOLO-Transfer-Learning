from  ultralytics import YOLO 
from PIL import Image

model = YOLO('runs/detect/train/weights/best.pt')
model(r'D:\yolocustome\brain-tumor\train\images\61.jpg', conf=.007, save=True, show=True)