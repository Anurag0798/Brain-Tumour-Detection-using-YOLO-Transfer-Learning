# Brain Tumour Detection using YOLO & Transfer Learning

## Overview

This project provides an end-to-end solution for automated brain tumour detection in MRI images, leveraging the YOLO (You Only Look Once) object detection model with transfer learning. By fine-tuning a pre-trained YOLO model on a labelled MRI dataset, the system is capable of localizing and classifying brain tumours with high accuracy and real-time performance.

## Features

- **Transfer Learning:** Quickly adapts YOLOv8 to the task of brain tumour detection on MRI scans.
- **Object Detection and Localization:** Draws bounding boxes around tumours in MRI images and provides confidence scores.
- **Custom Dataset Support:** Train and evaluate on your own labelled medical datasets (YOLO format).
- **Training & Inference Scripts:** Easy-to-use scripts for transfer learning, model export, and running predictions in batch or individually.
- **API Deployment:** RESTful API with FastAPI for easy integration with web or clinical tools.
- **Evaluation & Visualization:** Generates annotated result images and returns detection outputs in JSON.

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Anurag0798/Brain-Tumour-Detection-using-YOLO-Transfer-Learning.git

cd Brain-Tumour-Detection-using-YOLO-Transfer-Learning
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv .venv

# On Linux/Mac
source .venv/bin/activate

# On Windows
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
Key packages include: `numpy`, `matplotlib`, `pandas`, `ultralytics`, `torch`, `opencv-python`, `pillow`, `fastapi`, `uvicorn`, `python-multipart`.

### 4. Prepare dataset
- Place your MRI images and labels in the `dataset/images/train/` and `dataset/images/val/` directories as per YOLOv8 format.
- Label classes as `tumour` and `no tumour` (`data.yaml` defines these).
- Edit `data.yaml` if you need to change paths or class names.

### 5. Train the YOLOv8 model
```bash
python train.py
```
- This runs YOLOv8 training on the configured dataset for your defined number of epochs (default 50) and image size (default 640).
- After training, best weights are saved in `weights/best.pt` and exported as ONNX.

### 6. Run Inference API (FastAPI)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
- POST an MRI image to `/predict` endpoint for real-time tumour detection.

#### API Example (using api_testing.py)
```bash
python api_testing.py
```
- This will POST a test image and print JSON results: bounding boxes, scores, and classes.

## File & Script Descriptions

- **train.py**  
  Trains YOLOv8 with your dataset and exports the model to ONNX format.

- **test.py**  
  Tests the model prediction on a provided random test image by exporting the tested image along with the bounding boxes as predicted tumour or not (including the count of both tumours and no tumours, if any).

- **app/yolov8_handler.py**  
  Loads the trained model, processes the incoming image, and runs YOLOv8 detection. Returns results in a ready-to-serve format.

- **app/main.py**  
  FastAPI server that loads the trained model and exposes the `/predict` endpoint for image inference.

- **api_testing.py**  
  Testing utility to POST an image to the API and print results.

## Inference API Example

- **Endpoint:** `POST /predict`
- **Request:** multipart/form-data with image file
- **Response:** JSON with detection results:
```json
{
  "boxes": [[x1, y1, x2, y2], ...],
  "scores": [0.95, ...],
  "classes": [1, ...]  // 0 = no tumour, 1 = tumour
}
```

## Deployment (Docker/Cloud)

A `render.yaml` is provided for deploying on platforms like Render.com (see file for build and start commands).

## Tips & Customization

- Change epochs, batch size, and YOLO model architecture in `train.py`.
- For custom classes, update `data.yaml` and re-label your data.
- The exported model (`best.pt`) can also be loaded directly with Ultralytics YOLO tools, or used for conversion/optimization.

## License

This project is licensed under the MIT License.  
See [LICENSE](LICENSE) for details.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- Medical images from publicly available datasets and open-access sources.

## Contributing

Pull requests are welcome!  
For feature suggestions or bug reports, please open an issue.   
If you find this project useful, consider starring the repository and citing if used in your research or products.