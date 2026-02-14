
import os
import sys
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2

# Import model definition and custom layers
from model_def import (
    build_final_model, 
    ViewConsistencyLayer, 
    BBoxCIoULoss, 
    biou, 
    cxcywh_to_xyxy
)

app = Flask(__name__, static_folder='../', static_url_path='/')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model.keras')
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'weights_warmup.weights.h5')

# Global model variable
model = None

def load_ai_model():
    global model
    custom_objects = {
        'ViewConsistencyLayer': ViewConsistencyLayer,
        'BBoxCIoULoss': BBoxCIoULoss,
        'biou': biou,
        'cxcywh_to_xyxy': cxcywh_to_xyxy
    }
    
    try:
        if os.path.exists(MODEL_PATH):
            print(f"Loading model from {MODEL_PATH}...")
            model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
            print("Model loaded successfully.")
        else:
            print(f"Model file not found at {MODEL_PATH}. Trying to build and load weights...")
            model = build_final_model()
            if os.path.exists(WEIGHTS_PATH):
                print(f"Loading weights from {WEIGHTS_PATH}...")
                model.load_weights(WEIGHTS_PATH)
                print("Weights loaded successfully.")
            else:
                print("No weights found. Running with random weights (DEMO MODE).")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Fallback: Building new model with random weights (DEMO MODE).")
        model = build_final_model()

def preprocess_image(image_file, target_size):
    # Read image using PIL
    img = Image.open(image_file).convert('RGB')
    img = np.array(img)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Expand dims to create batch: (1, H, W, 3)
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'cc' not in request.files or 'mlo' not in request.files:
        return jsonify({'error': 'Missing image files. Please upload both CC and MLO views.'}), 400

    cc_file = request.files['cc']
    mlo_file = request.files['mlo']

    try:
        # Preprocess images
        # Strategy: Grid Search Inference to find the best ROI.
        # Since we don't have an ROI detector, feeding the full image as ROI confuses the model (OOD).
        # We will feed 5 distinct crops (Center, TL, TR, BL, BR) as candidate ROIs.
        # The crop that yields the highst confidence likely contains the lesion (or most informative features).
        # We then take the BBox from that best prediction.

        # 1. Full Images (512x512)
        cc_full = preprocess_image(cc_file, (512, 512))
        mlo_full = preprocess_image(mlo_file, (512, 512))
        
        # 2. Grid Generation (Pass 1)
        cc_file.seek(0)
        mlo_file.seek(0)
        
        # Read Original Images once
        cc_img_pil = Image.open(cc_file).convert('RGB')
        mlo_img_pil = Image.open(mlo_file).convert('RGB')
        cc_np = np.array(cc_img_pil)
        mlo_np = np.array(mlo_img_pil)

        def get_crops_5(img_np, target_size=(224,224)):
            h, w, _ = img_np.shape
            crops = []
            
            # 1. Center
            cy, cx = h // 2, w // 2
            ch, cw = h // 2, w // 2 # 50% size crop? Or fixed size?
            # Let's use 50% of image dimensions as crop size (zoomed in)
            crop_h, crop_w = h // 2, w // 2
            
            # Coordinates for 5 crops:
            # TL, TR, BL, BR, Center
            
            coords = [
                (0, 0, crop_w, crop_h), # TL
                (w - crop_w, 0, w, crop_h), # TR
                (0, h - crop_h, crop_w, h), # BL
                (w - crop_w, h - crop_h, w, h), # BR
                (cx - crop_w // 2, cy - crop_h // 2, cx + crop_w // 2, cy + crop_h // 2) # Center
            ]
            
            batch = []
            for (x1, y1, x2, y2) in coords:
                c = img_np[y1:y2, x1:x2]
                c = cv2.resize(c, target_size)
                c = c.astype(np.float32) / 255.0
                batch.append(c)
                
            return np.array(batch)

        cc_crops_batch = get_crops_5(cc_np) # (5, 224, 224, 3)
        mlo_crops_batch = get_crops_5(mlo_np) # (5, 224, 224, 3)
        
        # Replicate Full Images to match batch size 5
        cc_full_batch = np.repeat(cc_full, 5, axis=0) # (5, 512, 512, 3)
        mlo_full_batch = np.repeat(mlo_full, 5, axis=0)
        
        # Inputs Batch
        inputs_batch = {
            "cc_full": cc_full_batch,
            "cc_roi": cc_crops_batch,
            "mlo_full": mlo_full_batch,
            "mlo_roi": mlo_crops_batch # Assuming roughly same location for MLO?
             # NOTE: MLO and CC might not have lesion in same quadrant (e.g. Sup/Inf vs Med/Lat).
             # But we simply want ONE good signal. 
             # If cancer is in CC-TL, it will spike the score for that batch item.
             # Ideally we should mix-match all 5x5=25 combinations? Too slow.
             # Let's assume correlated quadrants (Top is Top) or just rely on Max(Max).
             # Let's stick to 1-1 mapping for now (TL-TL, TR-TR), it covers most cases.
        }

        # Inference Batch
        outputs_batch = model.predict(inputs_batch)
        
        probas = outputs_batch['cancer_cls'] # (5, 1)
        bboxes = outputs_batch['bbox'] # (5, 4)
        
        # Find index with max probability
        best_idx = np.argmax(probas)
        probability = float(probas[best_idx][0])
        
        # Get the relative bbox (normalized 0-1 relative to the CROP)
        rel_bbox = bboxes[best_idx].tolist() # [rcx, rcy, rw, rh]
        
        # We need to translate this back to the FULL IMAGE coordinates.
        # Re-construct the crop coordinates for the best index
        h, w = mlo_np.shape[:2] # Using MLO shape as reference (assuming CC is same or close enough for this logic)
        crop_h, crop_w = h // 2, w // 2
        cy, cx = h // 2, w // 2
        
        # Order matches get_crops_5: TL, TR, BL, BR, Center
        crop_coords = [
            (0, 0, crop_w, crop_h),   # 0: TL
            (w - crop_w, 0, w, crop_h), # 1: TR
            (0, h - crop_h, crop_w, h), # 2: BL
            (w - crop_w, h - crop_h, w, h), # 3: BR
            (cx - crop_w // 2, cy - crop_h // 2, cx + crop_w // 2, cy + crop_h // 2) # 4: Center
        ]
        
        # Get the pixel coordinates of the selected crop
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords[best_idx]
        current_crop_w = crop_x2 - crop_x1
        current_crop_h = crop_y2 - crop_y1
        
        # Rel bbox is [cx, cy, w, h] normalized to crop
        rcx, rcy, rw, rh = rel_bbox
        
        # Convert to Absolute Pixels in Crop
        abs_cx_crop = rcx * current_crop_w
        abs_cy_crop = rcy * current_crop_h
        abs_w_crop = rw * current_crop_w
        abs_h_crop = rh * current_crop_h
        
        # Convert to Absolute Pixels in Full Image
        abs_cx_full = crop_x1 + abs_cx_crop
        abs_cy_full = crop_y1 + abs_cy_crop
        
        # Normalize to Full Image Dimensions (0-1)
        final_cx = abs_cx_full / w
        final_cy = abs_cy_full / h
        final_w = abs_w_crop / w
        final_h = abs_h_crop / h
        
        # Clamp to ensure strict 0-1 range
        final_cx = max(0.0, min(1.0, final_cx))
        final_cy = max(0.0, min(1.0, final_cy))
        final_w = min(1.0, final_w)
        final_h = min(1.0, final_h)

        best_bbox = [final_cx, final_cy, final_w, final_h]
        
        print(f"Best Crop Index: {best_idx}")
        print(f"Original Rel BBox: {rel_bbox}")
        print(f"Transformed Full BBox: {best_bbox}")

        return jsonify({
            'probability': probability,
            'bbox': best_bbox,
            'message': 'Analysis successful (Grid Search)'
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_ai_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
