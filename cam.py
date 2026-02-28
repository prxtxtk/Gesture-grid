# cam.py
"""
Webcam demo for continued ASL model `asl_model_improved_continued.joblib`.
Uses the same feature engineering as train_improved.py / continue_training.py.
Press 'q' to quit.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # reduce TF/MediaPipe info logs

import cv2
import joblib
import numpy as np
import mediapipe as mp

# --- CONFIG ---
MODEL_FILE = "asl_model_improved_continued.joblib"  # <- updated model filename
CONF_THRESH = 0.0  # set to e.g. 0.6 to only show higher-confidence preds

# --- load model ---
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}\n"
        f"Make sure the file is in the same folder as cam.py or change MODEL_FILE."
    )
model = joblib.load(MODEL_FILE)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# --- feature helpers (must match training) ---
def unit_vector(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def angle_between(v1, v2):
    v1u = unit_vector(v1); v2u = unit_vector(v2)
    cos = np.clip(np.dot(v1u, v2u), -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def compute_features_from_landmarks(pts):
    # pts: numpy array (21,3)
    wrist = pts[0]
    pairwise = pts[:, :2]
    dists = np.linalg.norm(pairwise[:, None, :] - pairwise[None, :, :], axis=2)
    hand_span = max(np.max(dists), 1e-6)
    rel = (pts - wrist) / hand_span
    feats = []
    feats.extend(rel.flatten().tolist())
    fingertip_idx = [4, 8, 12, 16, 20]
    # wrist->fingertip distances
    for i in fingertip_idx:
        feats.append(np.linalg.norm((pts[i] - wrist)[:2]) / hand_span)
    # pairwise fingertip distances
    for i in range(len(fingertip_idx)):
        for j in range(i + 1, len(fingertip_idx)):
            a = fingertip_idx[i]; b = fingertip_idx[j]
            feats.append(np.linalg.norm((pts[a] - pts[b])[:2]) / hand_span)
    # finger angles
    finger_triplets = [(2,3,4),(5,6,8),(9,10,12),(13,14,16),(17,18,20)]
    for a,b,c in finger_triplets:
        v1 = pts[a] - pts[b]; v2 = pts[c] - pts[b]
        feats.append(angle_between(v1[:3], v2[:3]) / 180.0)
    # wrist->fingertip direction vectors
    for i in fingertip_idx:
        v = pts[i][:3] - wrist[:3]
        v = v / (np.linalg.norm(v) + 1e-9)
        feats.extend(v.tolist())
    return np.array(feats, dtype=float)

def extract_feats_from_mediapipe_landmarks(landmarks):
    pts = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=float)
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

# --- camera loop ---
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    classes = get_model_classes(model)

    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)  # mirror
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(img_rgb)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0]
                feats = extract_feats_from_mediapipe_landmarks(lm.landmark)

                pred = None
                conf = None
                top_labels = []

                # try predict_proba first
                try:
                    probs = model.predict_proba([feats])[0]
                    if classes is None:
                        classes = get_model_classes(model)
                    if classes is not None:
                        idx = int(np.argmax(probs))
                        pred = classes[idx]
                        conf = float(probs[idx])
                        top_k = 3
                        top_idx = np.argsort(probs)[-top_k:][::-1]
                        top_labels = [(str(classes[i]), float(probs[i])) for i in top_idx]
                    else:
                        idx = int(np.argmax(probs))
                        pred = str(idx)
                        conf = float(probs[idx])
                        top_k = 3
                        top_idx = np.argsort(probs)[-top_k:][::-1]
                        top_labels = [(str(i), float(probs[i])) for i in top_idx]
                except Exception:
                    try:
                        pred = model.predict([feats])[0]
                    except Exception:
                        pred = None

                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                if pred is not None and (conf is None or conf >= CONF_THRESH):
                    label_text = f"{pred}" + (f" ({conf:.2f})" if conf is not None else "")
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
                    x0, y0 = 10, 40
                    cv2.rectangle(frame, (x0-6, y0-th-6), (x0+tw+6, y0+6), (0,0,0), -1)
                    cv2.putText(frame, label_text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,0,255), 2)

                    # top-3 below
                    y_start = y0 + 30
                    for i, (lab, p) in enumerate(top_labels):
                        line = f"{i+1}. {lab} {p:.2f}"
                        cv2.putText(frame, line, (10, y_start + i*28), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200,200,200), 2)

            cv2.imshow("ASL Live (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
