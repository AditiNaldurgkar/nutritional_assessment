# Removed import of yolov5 modules to avoid circular import error
# Instead, import DetectMultiBackend and other yolov5 utilities directly from yolov5 package if installed

try:
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.general import non_max_suppression, scale_coords
    from yolov5.utils.torch_utils import select_device
except ImportError:
    DetectMultiBackend = None
    non_max_suppression = None
    scale_coords = None
    select_device = None

import torch
import cv2

from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image as keras_image
import numpy as np

def process_image(image_path):
    if DetectMultiBackend is None:
        raise ImportError("YOLOv5 package not found. Please install or clone yolov5 repository properly.")
    device = select_device('')
    model = DetectMultiBackend('yolov5s.pt', device=device)  # or yolov5n.pt etc.
    model.eval()

    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))  # resize as needed
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    pred = model(img_tensor, augment=False)
    pred = non_max_suppression(pred)[0]

    results = []
    if pred is not None:
        for *xyxy, conf, cls in pred:
            results.append({
                "bbox": [float(x) for x in xyxy],
                "confidence": float(conf),
                "class": int(cls)
            })
    return results

from keras.applications.vgg16 import decode_predictions

vgg_model = VGG16(weights='imagenet', include_top=True)

def predict_with_vgg16(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = vgg_model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]  # Get top 3 predictions

    # Format decoded predictions as list of dicts
    results = []
    for _, label, prob in decoded_preds:
        results.append({'label': label, 'probability': float(prob)})

    return results
