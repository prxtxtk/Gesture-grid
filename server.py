# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import cv2
import numpy as np
import joblib
import mediapipe as mp

# --- CONFIG ---
MODEL_FILE = "asl_model_improved_continued.joblib"  # place your model here
CONF_THRESH = 0.0  # frontend may check confidence too

app = Flask(__name__)
CORS(app)

# --- load model ---
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}. Put it next to server.py or change MODEL_FILE")

model = joblib.load(MODEL_FILE)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ---- feature helpers (same as your cam.py) ----
def unit_vector(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def angle_between(v1, v2):
    v1u = unit_vector(v1); v2u = unit_vector(v2)
    cos = np.clip(np.dot(v1u, v2u), -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def compute_features_from_landmarks(pts):
    wrist = pts[0]
    pairwise = pts[:, :2]
    dists = np.linalg.norm(pairwise[:, None, :] - pairwise[None, :, :], axis=2)
    hand_span = max(np.max(dists), 1e-6)
    rel = (pts - wrist) / hand_span
    feats = []
    feats.extend(rel.flatten().tolist())
    fingertip_idx = [4, 8, 12, 16, 20]
    for i in fingertip_idx:
        feats.append(np.linalg.norm((pts[i] - wrist)[:2]) / hand_span)
    for i in range(len(fingertip_idx)):
        for j in range(i + 1, len(fingertip_idx)):
            a = fingertip_idx[i]; b = fingertip_idx[j]
            feats.append(np.linalg.norm((pts[a] - pts[b])[:2]) / hand_span)
    finger_triplets = [(2,3,4),(5,6,8),(9,10,12),(13,14,16),(17,18,20)]
    for a,b,c in finger_triplets:
        v1 = pts[a] - pts[b]; v2 = pts[c] - pts[b]
        feats.append(angle_between(v1[:3], v2[:3]) / 180.0)
    for i in fingertip_idx:
        v = pts[i][:3] - wrist[:3]
        v = v / (np.linalg.norm(v) + 1e-9)
        feats.extend(v.tolist())
    return np.array(feats, dtype=float)

def extract_feats_from_image_bgr(bgr):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = hands.process(img_rgb)
        if not res.multi_hand_landmarks:
            raise ValueError("no_hand")
        lm = res.multi_hand_landmarks[0].landmark
        pts = np.array([[p.x, p.y, p.z] for p in lm], dtype=float)
        return compute_features_from_landmarks(pts)

def get_model_classes(m):
    if hasattr(m, "classes_"):
        return m.classes_
    try:
        if hasattr(m, "named_steps") and 'randomforestclassifier' in m.named_steps:
            est = m.named_steps['randomforestclassifier']
            if hasattr(est, "classes_"):
                return est.classes_
    except Exception:
        pass
    return None

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "error": "no image"}), 400

    b64 = data["image"]
    if "," in b64:
        b64 = b64.split(",")[1]
    try:
        img_bytes = base64.b64decode(b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("bad_image")
    except Exception as e:
        return jsonify({"success": False, "error": "invalid_image", "detail": str(e)}), 400

    try:
        feats = extract_feats_from_image_bgr(bgr)
    except ValueError as e:
        # IMPORTANT: return success=True so frontend counts this as an attempt.
        # Still include "error" so UI can show "no hand detected".
        return jsonify({
            "success": True,
            "error": "no_hand_detected",
            "pred": "",
            "confidence": None,
            "top": []
        }), 200
    except Exception as e:
        return jsonify({"success": False, "error": "feat_error", "detail": str(e)}), 500

    classes = get_model_classes(model)
    try:
        probs = model.predict_proba([feats])[0]
        if classes is not None:
            idx = int(np.argmax(probs))
            pred = str(classes[idx])
            conf = float(probs[idx])
            top_k = 3
            top_idx = np.argsort(probs)[-top_k:][::-1]
            top_labels = [[str(classes[i]), float(probs[i])] for i in top_idx]
        else:
            idx = int(np.argmax(probs))
            pred = str(idx)
            conf = float(probs[idx])
            top_k = 3
            top_idx = np.argsort(probs)[-top_k:][::-1]
            top_labels = [[str(i), float(probs[i])] for i in top_idx]
    except Exception:
        try:
            pred = model.predict([feats])[0]
            conf = None
            top_labels = [[str(pred), None]]
        except Exception as e:
            return jsonify({"success": False, "error": "model_error", "detail": str(e)}), 500

    return jsonify({"success": True, "pred": pred, "confidence": conf, "top": top_labels}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
