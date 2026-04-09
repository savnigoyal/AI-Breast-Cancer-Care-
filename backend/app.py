import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2

app = Flask(__name__, static_folder="../", static_url_path="/")
CORS(app)


@app.route("/")
def index():
    return app.send_static_file("index.html")


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, "mammo_classifier.keras")
LEGACY_MODEL_PATH = os.path.join(ROOT_DIR, "best_model.keras")
IMG_SIZE = (224, 224)

# Global model variable
model = None


# 5 fixed regions (same order expected by frontend)
REGION_LABELS = ["Top-Left", "Top-Right", "Bottom-Left", "Bottom-Right", "Center"]

TABULAR_FEATURE_KEYS = [
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

SKLEARN_FEATURE_ORDER = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]

TO_SKLEARN_NAME = {
    "radius_mean": "mean radius",
    "texture_mean": "mean texture",
    "perimeter_mean": "mean perimeter",
    "area_mean": "mean area",
    "smoothness_mean": "mean smoothness",
    "compactness_mean": "mean compactness",
    "concavity_mean": "mean concavity",
    "concave points_mean": "mean concave points",
    "symmetry_mean": "mean symmetry",
    "fractal_dimension_mean": "mean fractal dimension",
    "radius_se": "radius error",
    "texture_se": "texture error",
    "perimeter_se": "perimeter error",
    "area_se": "area error",
    "smoothness_se": "smoothness error",
    "compactness_se": "compactness error",
    "concavity_se": "concavity error",
    "concave points_se": "concave points error",
    "symmetry_se": "symmetry error",
    "fractal_dimension_se": "fractal dimension error",
    "radius_worst": "worst radius",
    "texture_worst": "worst texture",
    "perimeter_worst": "worst perimeter",
    "area_worst": "worst area",
    "smoothness_worst": "worst smoothness",
    "compactness_worst": "worst compactness",
    "concavity_worst": "worst concavity",
    "concave points_worst": "worst concave points",
    "symmetry_worst": "worst symmetry",
    "fractal_dimension_worst": "worst fractal dimension",
}

FROM_SKLEARN_NAME = {value: key for key, value in TO_SKLEARN_NAME.items()}

BASELINE_FEATURES = {
    "radius_mean": 14.13,
    "texture_mean": 19.29,
    "perimeter_mean": 91.97,
    "area_mean": 654.89,
    "smoothness_mean": 0.096,
    "compactness_mean": 0.104,
    "concavity_mean": 0.089,
    "concave points_mean": 0.049,
    "symmetry_mean": 0.181,
    "fractal_dimension_mean": 0.063,
    "radius_se": 0.405,
    "texture_se": 1.216,
    "perimeter_se": 2.866,
    "area_se": 40.34,
    "smoothness_se": 0.007,
    "compactness_se": 0.025,
    "concavity_se": 0.032,
    "concave points_se": 0.012,
    "symmetry_se": 0.021,
    "fractal_dimension_se": 0.004,
    "radius_worst": 16.27,
    "texture_worst": 25.68,
    "perimeter_worst": 107.26,
    "area_worst": 880.58,
    "smoothness_worst": 0.132,
    "compactness_worst": 0.254,
    "concavity_worst": 0.272,
    "concave points_worst": 0.114,
    "symmetry_worst": 0.290,
    "fractal_dimension_worst": 0.084,
}

FEATURE_WEIGHTS = {
    "radius_mean": 1.20,
    "texture_mean": 0.90,
    "perimeter_mean": 1.20,
    "area_mean": 1.10,
    "smoothness_mean": 0.70,
    "compactness_mean": 1.00,
    "concavity_mean": 1.10,
    "concave points_mean": 1.20,
    "symmetry_mean": 0.70,
    "fractal_dimension_mean": 0.60,
    "radius_se": 0.80,
    "texture_se": 0.70,
    "perimeter_se": 0.85,
    "area_se": 0.85,
    "smoothness_se": 0.60,
    "compactness_se": 0.80,
    "concavity_se": 0.85,
    "concave points_se": 0.90,
    "symmetry_se": 0.60,
    "fractal_dimension_se": 0.55,
    "radius_worst": 1.30,
    "texture_worst": 1.00,
    "perimeter_worst": 1.30,
    "area_worst": 1.20,
    "smoothness_worst": 0.90,
    "compactness_worst": 1.00,
    "concavity_worst": 1.15,
    "concave points_worst": 1.25,
    "symmetry_worst": 0.80,
    "fractal_dimension_worst": 0.70,
}


def clamp01(value):
    return float(max(0.0, min(1.0, value)))


def safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(np.std(values))


def compute_boxcount_fractal(binary_img):
    binary = (binary_img > 0).astype(np.uint8)
    h, w = binary.shape
    sizes = [2, 4, 8, 16]
    counts = []

    for size in sizes:
        if size > min(h, w):
            continue
        cnt = 0
        for y in range(0, h, size):
            for x in range(0, w, size):
                block = binary[y : y + size, x : x + size]
                if block.size and np.any(block):
                    cnt += 1
        if cnt > 0:
            counts.append((size, cnt))

    if len(counts) < 2:
        return 1.0

    log_sizes = np.log([1.0 / c[0] for c in counts])
    log_counts = np.log([c[1] for c in counts])
    slope = np.polyfit(log_sizes, log_counts, 1)[0]
    return float(max(0.0, slope))


def region_stats_from_gray(gray_region):
    if gray_region.size == 0:
        return {
            "radius": 0.0,
            "texture": 0.0,
            "perimeter": 0.0,
            "area": 0.0,
            "smoothness": 0.0,
            "compactness": 0.0,
            "concavity": 0.0,
            "concave_points": 0.0,
            "symmetry": 0.0,
            "fractal_dimension": 0.0,
        }

    blur = cv2.GaussianBlur(gray_region, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    h, w = gray_region.shape
    area_pixels = float(np.count_nonzero(mask))
    total_pixels = float(h * w)

    if area_pixels < 0.05 * total_pixels:
        mask = np.ones_like(mask, dtype=np.uint8) * 255
        area_pixels = total_pixels

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea) if contours else None

    radius = np.sqrt(area_pixels / np.pi)
    texture = float(np.std(gray_region))
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0
    area = area_pixels

    lap = cv2.Laplacian(blur, cv2.CV_32F)
    grad_var = float(np.var(lap))
    smoothness = 1.0 / (1.0 + grad_var)

    compactness = 0.0
    if area_pixels > 0:
        compactness = (perimeter * perimeter) / (4.0 * np.pi * area_pixels + 1e-7)

    concavity = 0.0
    concave_points = 0.0
    if contour is not None and len(contour) >= 5:
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is not None and len(hull) >= 3:
            defects = cv2.convexityDefects(contour, hull)
            if defects is not None and len(defects) > 0:
                depths = defects[:, 0, 3] / 256.0
                concavity = float(np.mean(depths))
                concave_points = float(np.sum(depths > 2.0))

    left = mask[:, : w // 2]
    right = cv2.flip(mask[:, w - w // 2 :], 1)
    min_w = min(left.shape[1], right.shape[1])
    if min_w > 0:
        left = left[:, :min_w] > 0
        right = right[:, :min_w] > 0
        symmetry = float(np.mean(left == right))
    else:
        symmetry = 0.0

    fractal_dimension = compute_boxcount_fractal(mask)

    return {
        "radius": float(radius),
        "texture": float(texture),
        "perimeter": float(perimeter),
        "area": float(area),
        "smoothness": float(smoothness),
        "compactness": float(compactness),
        "concavity": float(concavity),
        "concave_points": float(concave_points),
        "symmetry": float(symmetry),
        "fractal_dimension": float(fractal_dimension),
    }


def feature_vector_from_uploaded_views(cc_np, mlo_np):
    samples = []
    for img in [cc_np, mlo_np]:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        samples.append(region_stats_from_gray(gray))

        h, w = gray.shape
        coords = get_region_coords(h, w)
        for label in REGION_LABELS:
            x1, y1, x2, y2 = coords[label]
            region = gray[y1:y2, x1:x2]
            samples.append(region_stats_from_gray(region))

    feature_values = {}
    base_keys = [
        "radius",
        "texture",
        "perimeter",
        "area",
        "smoothness",
        "compactness",
        "concavity",
        "concave_points",
        "symmetry",
        "fractal_dimension",
    ]

    for key in base_keys:
        vals = [item[key] for item in samples]
        feature_values[f"{key}_mean"] = float(np.mean(vals))
        feature_values[f"{key}_se"] = safe_std(vals)
        feature_values[f"{key}_worst"] = float(np.max(vals))

    # Match expected column naming exactly: "concave points" with space.
    feature_values["concave points_mean"] = feature_values.pop("concave_points_mean")
    feature_values["concave points_se"] = feature_values.pop("concave_points_se")
    feature_values["concave points_worst"] = feature_values.pop("concave_points_worst")

    return feature_values


def shap_style_summary(feature_values):
    contributions = []
    total_score = 0.0

    for key in TABULAR_FEATURE_KEYS:
        value = float(feature_values.get(key, 0.0))
        baseline = float(BASELINE_FEATURES.get(key, 1.0))
        weight = float(FEATURE_WEIGHTS.get(key, 1.0))

        denom = max(abs(baseline) * 0.5, 1e-6)
        normalized_delta = (value - baseline) / denom
        contrib = normalized_delta * weight
        total_score += contrib

        contributions.append(
            {
                "feature": key,
                "value": value,
                "contribution": float(contrib),
                "direction": "increase_risk" if contrib >= 0 else "decrease_risk",
            }
        )

    # Map accumulated contribution into [0, 1] with a smooth sigmoid.
    shap_probability = clamp01(1.0 / (1.0 + np.exp(-(total_score / 8.0))))

    summary_rows = []
    summary_rows.extend(contributions)

    summary_rows = sorted(summary_rows, key=lambda item: abs(item["contribution"]), reverse=True)
    top_summary = summary_rows[:10]

    return {
        "probability": shap_probability,
        "top_features": top_summary,
    }


def preprocess_np_image(image_np, target_size=IMG_SIZE):
    resized = cv2.resize(image_np, target_size)
    resized = resized.astype(np.float32) / 255.0
    return np.expand_dims(resized, axis=0)


def build_legacy_multiview_input(image_np):
    full_512 = preprocess_np_image(image_np, (512, 512))
    roi_224 = preprocess_np_image(image_np, (224, 224))

    return {
        "cc_full": full_512,
        "cc_roi": roi_224,
        "mlo_full": full_512,
        "mlo_roi": roi_224,
    }


def extract_probability(model_output):
    # Support dict/list/tensor outputs to keep compatibility with different model exports.
    if isinstance(model_output, dict):
        if "cancer_cls" in model_output:
            arr = model_output["cancer_cls"]
        else:
            arr = list(model_output.values())[0]
    else:
        arr = model_output

    arr = np.asarray(arr).reshape(-1)
    if arr.size == 0:
        return 0.0

    return clamp01(float(arr[0]))


def predict_probability_from_np(image_np):
    return predict_probabilities_from_list([image_np])[0]


def predict_probabilities_from_list(images_np):
    if not images_np:
        return []

    # Legacy model expects 4 named tensors; rebuilt models can use single-image input.
    if isinstance(model.inputs, list) and len(model.inputs) == 4:
        full_batch = np.concatenate(
            [preprocess_np_image(img, (512, 512)) for img in images_np], axis=0
        )
        roi_batch = np.concatenate(
            [preprocess_np_image(img, (224, 224)) for img in images_np], axis=0
        )

        model_input = {
            "cc_full": full_batch,
            "cc_roi": roi_batch,
            "mlo_full": full_batch,
            "mlo_roi": roi_batch,
        }
        pred = model.predict(model_input, verbose=0)
    else:
        batch = np.concatenate([preprocess_np_image(img) for img in images_np], axis=0)
        pred = model.predict(batch, verbose=0)

    if isinstance(pred, dict):
        arr = pred.get("cancer_cls", list(pred.values())[0])
    else:
        arr = pred

    arr = np.asarray(arr).reshape(-1)
    return [clamp01(float(v)) for v in arr]


def get_region_coords(h, w):
    crop_h, crop_w = h // 2, w // 2
    cx, cy = w // 2, h // 2

    return {
        "Top-Left": (0, 0, crop_w, crop_h),
        "Top-Right": (w - crop_w, 0, w, crop_h),
        "Bottom-Left": (0, h - crop_h, crop_w, h),
        "Bottom-Right": (w - crop_w, h - crop_h, w, h),
        "Center": (
            cx - crop_w // 2,
            cy - crop_h // 2,
            cx + crop_w // 2,
            cy + crop_h // 2,
        ),
    }


def region_to_bbox(coords, full_w, full_h):
    x1, y1, x2, y2 = coords
    cx = ((x1 + x2) / 2.0) / full_w
    cy = ((y1 + y2) / 2.0) / full_h
    bw = (x2 - x1) / full_w
    bh = (y2 - y1) / full_h
    return [clamp01(cx), clamp01(cy), clamp01(bw), clamp01(bh)]


def evaluate_view(image_np):
    h, w = image_np.shape[:2]
    regions = get_region_coords(h, w)

    full_prob = predict_probability_from_np(image_np)

    crop_images = []
    crop_meta = []
    for label in REGION_LABELS:
        x1, y1, x2, y2 = regions[label]
        crop_np = image_np[y1:y2, x1:x2]
        crop_images.append(crop_np)
        crop_meta.append((label, crop_np.size > 0))

    batch_probs = predict_probabilities_from_list(
        [img if img.size > 0 else np.zeros((16, 16, 3), dtype=np.uint8) for img in crop_images]
    )

    crop_scores = []
    for idx, (label, is_valid) in enumerate(crop_meta):
        crop_prob = batch_probs[idx] if is_valid else 0.0

        crop_scores.append({"crop": label, "probability": clamp01(crop_prob)})

    best_crop = max(crop_scores, key=lambda item: item["probability"])
    sorted_scores = sorted(crop_scores, key=lambda item: item["probability"], reverse=True)
    second_best = sorted_scores[1] if len(sorted_scores) > 1 else None

    best_coords = regions[best_crop["crop"]]
    best_bbox = region_to_bbox(best_coords, w, h)

    return {
        "full_probability": clamp01(full_prob),
        "crop_scores": crop_scores,
        "best_crop": best_crop,
        "second_best": second_best,
        "bbox": best_bbox,
    }


def load_ai_model():
    global model

    selected_path = MODEL_PATH if os.path.exists(MODEL_PATH) else LEGACY_MODEL_PATH

    if not os.path.exists(selected_path):
        raise FileNotFoundError(
            "No model file found. Expected mammo_classifier.keras (preferred) or best_model.keras."
        )

    print(f"Loading model from {selected_path}...")

    if selected_path == LEGACY_MODEL_PATH:
        from model_def import ViewConsistencyLayer, BBoxCIoULoss, biou, cxcywh_to_xyxy

        custom_objects = {
            "ViewConsistencyLayer": ViewConsistencyLayer,
            "BBoxCIoULoss": BBoxCIoULoss,
            "biou": biou,
            "cxcywh_to_xyxy": cxcywh_to_xyxy,
        }
        model = tf.keras.models.load_model(
            selected_path,
            custom_objects=custom_objects,
            compile=False,
        )
    else:
        model = tf.keras.models.load_model(selected_path, compile=False)

    print("Model loaded successfully.")
    print("Feature-based SHAP-style explainer initialized.")


def risk_text(probability):
    if probability < 0.35:
        return "Low Risk"
    if probability < 0.65:
        return "Moderate Risk (Consultation Recommended)"
    return "High Risk (Immediate Attention Required)"


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "cc" not in request.files or "mlo" not in request.files:
        return jsonify({"error": "Missing image files. Please upload both CC and MLO views."}), 400

    cc_file = request.files["cc"]
    mlo_file = request.files["mlo"]

    try:
        cc_np = np.array(Image.open(cc_file).convert("RGB"))
        mlo_np = np.array(Image.open(mlo_file).convert("RGB"))

        cc_result = evaluate_view(cc_np)
        mlo_result = evaluate_view(mlo_np)
        feature_values = feature_vector_from_uploaded_views(cc_np, mlo_np)
        shap_summary = shap_style_summary(feature_values)

        # Choose the view with stronger suspicious signal (higher best-region score).
        if cc_result["best_crop"]["probability"] >= mlo_result["best_crop"]["probability"]:
            selected_view = "CC"
            selected_result = cc_result
            other_result = mlo_result
        else:
            selected_view = "MLO"
            selected_result = mlo_result
            other_result = cc_result

        probability = clamp01(selected_result["best_crop"]["probability"])
        selected_crop_label = selected_result["best_crop"]["crop"]
        crop_scores = selected_result["crop_scores"]
        best_bbox = selected_result["bbox"]

        second_best = selected_result["second_best"]
        gap_text = ""
        if second_best is not None:
            gap = (selected_result["best_crop"]["probability"] - second_best["probability"]) * 100.0
            gap_text = (
                f" In the same view, the next strongest area was {second_best['crop']} "
                f"at {second_best['probability'] * 100:.2f}% (gap: {gap:.2f}%)."
            )

        analysis_basis = (
            f"The AI checked both CC and MLO images. It split the {selected_view} view into 5 areas "
            f"and selected {selected_crop_label} because it had the highest cancer score "
            f"({probability * 100:.2f}%).{gap_text}"
        )

        evidence = [
            f"Selected view: {selected_view}",
            f"Best area: {selected_crop_label}",
            f"Selected area score: {probability * 100:.2f}%",
            f"CC full-image score: {cc_result['full_probability'] * 100:.2f}%",
            f"MLO full-image score: {mlo_result['full_probability'] * 100:.2f}%",
        ]

        if shap_summary is not None:
            evidence.append(f"SHAP-style tabular score: {shap_summary['probability'] * 100:.2f}%")

        prediction_label = "Cancer Signal Detected" if probability >= 0.50 else "No Strong Cancer Signal"

        response = {
            "probability": probability,
            "prediction": prediction_label,
            "risk_level": risk_text(probability),
            "bbox": best_bbox,
            "selected_view": selected_view,
            "selected_crop_label": selected_crop_label,
            "selected_crop_index": REGION_LABELS.index(selected_crop_label),
            "crop_scores": crop_scores,
            "analysis_basis": analysis_basis,
            "evidence": evidence,
            "feature_values": {
                key: float(feature_values.get(key, 0.0))
                for key in TABULAR_FEATURE_KEYS
            },
            "shap_summary": shap_summary,
            "view_scores": {
                "cc_full": cc_result["full_probability"],
                "mlo_full": mlo_result["full_probability"],
                "cc_best_region": cc_result["best_crop"]["probability"],
                "mlo_best_region": mlo_result["best_crop"]["probability"],
            },
            "message": "Analysis successful",
        }

        return jsonify(response)

    except Exception as exc:
        print(f"Prediction error: {exc}")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    load_ai_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
