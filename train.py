# train.py
# Heavy, proper ASL training with checkpointing:
# - builds expanded dataset (augmentation)
# - optional hyperparameter search (RandomizedSearchCV)
# - trains final large RandomForest
# - SAVES the final trained model IMMEDIATELY after .fit()
# Usage:
#   python train.py --force-extract --augment --tune
#   (omit flags to use defaults; flags printed at start)

from pathlib import Path
import argparse, time, math, random, sys
import numpy as np, pandas as pd, cv2, joblib
import mediapipe as mp
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report

# ---------- CONFIG: heavy settings (match your long run) ----------
DATA_DIR = "asl"
FEATURES_CSV = "landmarks_dataset_expanded.csv"   # expanded CSV created from images + augmentations
MODEL_FILE = "asl_model.joblib"                   # final saved model (overwritten when done)
BASELINE_MODEL = "asl_model_baseline.joblib"      # checkpoint baseline model (optional)
RANDOM_STATE = 42

# augmentation (heavy => multiplies dataset)
AUGMENT_PER_IMAGE = 4       # keep at 4 to match your previous heavy run
AUG_PROB = 0.9
IMG_ROTATE = 8
IMG_SHIFT = 0.04
BRIGHTNESS_RANGE = (0.85, 1.15)

# RandomForest settings
RF_N_ESTIMATORS_FINAL = 800    # large forest; increases runtime but improves stability
RF_N_ESTIMATORS_BASELINE = 300

# hyperparameter search (heavy)
RANDOMIZED_ITERS = 20
CV_FOLDS = 3
# ----------------------------------------------------------------

mp_hands = mp.solutions.hands

parser = argparse.ArgumentParser()
parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
parser.add_argument("--no-tune", action="store_true", help="Disable hyperparameter tuning")
parser.add_argument("--force-extract", action="store_true", help="Force re-extract features CSV")
parser.add_argument("--no-baseline", action="store_true", help="Do not save baseline")
args = parser.parse_args()

DO_AUG = not args.no_augment
DO_TUNE = not args.no_tune
FORCE_EXTRACT = args.force_extract
SAVE_BASELINE = not args.no_baseline

def now(): return time.strftime("%Y-%m-%d %H:%M:%S")

# ---------- feature helpers ----------
def normalize_landmarks(pts):
    wrist = pts[0].copy()
    rel = pts - wrist
    dists = np.linalg.norm(rel[:, :2], axis=1)
    scale = dists.max() if dists.max() > 1e-9 else 1.0
    rel[:, :2] /= scale
    rel[:, 2] /= scale
    return rel.flatten()

def angle_between(a,b,c):
    ba = a - b; bc = c - b
    nba = ba / (np.linalg.norm(ba) + 1e-9)
    nbc = bc / (np.linalg.norm(bc) + 1e-9)
    cosang = np.clip(np.dot(nba, nbc), -1.0, 1.0)
    return math.acos(cosang)

def compute_angles(pts):
    triplets = [(1,2,3),(2,3,4),(5,6,7),(6,7,8),(9,10,11),(10,11,12),(13,14,15),(14,15,16),(17,18,19)]
    return np.array([angle_between(pts[a,:2], pts[b,:2], pts[c,:2]) for a,b,c in triplets])

def feature_vector_from_landmarks(pts21x3):
    coords = normalize_landmarks(pts21x3)    # 63
    angles = compute_angles(pts21x3)         # 9
    tips = [4,8,12,16,20]
    pairwise = [np.linalg.norm(pts21x3[t,:2] - pts21x3[0,:2]) for t in tips]  # 5
    return np.concatenate([coords, angles, np.array(pairwise)])  # => 77 dims

# ---------- augmentation helpers ----------
def rrotate(img, angle_range=IMG_ROTATE):
    h,w = img.shape[:2]
    ang = random.uniform(-angle_range, angle_range)
    M = cv2.getRotationMatrix2D((w/2,h/2), ang, 1.0)
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def rtranslate(img, max_shift=IMG_SHIFT):
    h,w = img.shape[:2]
    tx = random.uniform(-max_shift, max_shift) * w
    ty = random.uniform(-max_shift, max_shift) * h
    M = np.float32([[1,0,tx],[0,1,ty]])
    return cv2.warpAffine(img, M, (w,h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def rbrightness(img, rng=BRIGHTNESS_RANGE):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(float)
    f = random.uniform(*rng)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * f, 0, 255)
    return cv2.cvtColor(hsv.astype('uint8'), cv2.COLOR_HSV2BGR)

# ---------- extract single image landmarks ----------
def extract_landmarks_from_image(img_bgr, hands_proc):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    res = hands_proc.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0]
    pts = np.array([[p.x, p.y, p.z] for p in lm.landmark], dtype=float)
    return feature_vector_from_landmarks(pts)

# ---------- build expanded CSV (heavy) ----------
def build_features_csv(data_dir=DATA_DIR, out_csv=FEATURES_CSV, augment=DO_AUG, aug_per=AUGMENT_PER_IMAGE, aug_prob=AUG_PROB):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} not found")
    labels = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    proc = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    rows = []
    total = sum(len(list((data_dir / L).glob('*'))) for L in labels)
    print(f"{now()} Building expanded CSV from {total} images (augment={augment}, aug_per={aug_per})")
    for L in labels:
        img_paths = [p for p in (data_dir / L).glob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png')]
        print(f"Label {L}: {len(img_paths)} images")
        for p in img_paths:
            img = cv2.imread(str(p))
            if img is None: 
                continue
            fv = extract_landmarks_from_image(img, proc)
            if fv is not None:
                rows.append(list(fv) + [L] + [str(p)])
            if augment:
                for k in range(aug_per):
                    if random.random() > aug_prob: 
                        continue
                    aug = rbrightness(rtranslate(rrotate(img)))
                    fv2 = extract_landmarks_from_image(aug, proc)
                    if fv2 is not None:
                        rows.append(list(fv2) + [L] + [str(p) + f"_aug{k}"])
    proc.close()
    if len(rows) == 0:
        raise RuntimeError("No landmarks extracted.")
    nfeat = len(rows[0]) - 2
    cols = [f"f{i}" for i in range(nfeat)] + ["label", "file"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False)
    print(f"{now()} Saved expanded CSV -> {out_csv}  rows={len(df)}")
    return out_csv

# ---------- load numeric features from CSV ----------
def load_features_csv(csv_path):
    df = pd.read_csv(csv_path)
    if 'label' not in df.columns:
        raise ValueError("CSV must contain 'label' column")
    # keep only numeric feature columns (drop 'file' and any other non-numeric)
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] == 0:
        candidates = [c for c in df.columns if c != 'label']
        num_df = df[candidates].apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    X = num_df.values
    y = df['label'].values
    print(f"{now()} Loaded CSV: X.shape={X.shape}, labels={len(y)}")
    return X, y

# ---------- baseline train & checkpoint ----------
def train_baseline_save(X_train, y_train, X_val, y_val):
    print(f"{now()} Training baseline RF (n_estimators={RF_N_ESTIMATORS_BASELINE})")
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=RF_N_ESTIMATORS_BASELINE, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    t0 = time.time()
    clf.fit(X_train, y_train)
    t = time.time() - t0
    val_acc = accuracy_score(y_val, clf.predict(X_val))
    print(f"{now()} Baseline trained in {t:.1f}s, val_acc={val_acc:.4f}")
    if SAVE_BASELINE:
        try:
            joblib.dump(clf, BASELINE_MODEL)
            print(f"{now()} Baseline saved -> {BASELINE_MODEL}")
        except Exception as e:
            print(f"{now()} WARNING: could not save baseline model: {e}")
    return clf

# ---------- hyperparameter search ----------
def randomized_search_rf(X, y, n_iter=RANDOMIZED_ITERS):
    print(f"{now()} Running RandomizedSearchCV: {n_iter} iters × {CV_FOLDS}-fold CV")
    pipe = make_pipeline(StandardScaler(), RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1))
    param_dist = {
        "randomforestclassifier__n_estimators": [200,300,400,600,800],
        "randomforestclassifier__max_depth": [None, 12, 18, 24],
        "randomforestclassifier__min_samples_leaf": [1,2,3,4],
        "randomforestclassifier__max_features": ["sqrt","log2", None]
    }
    rs = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=n_iter, cv=CV_FOLDS, scoring="accuracy", n_jobs=-1, random_state=RANDOM_STATE, verbose=2)
    t0 = time.time()
    rs.fit(X, y)
    print(f"{now()} RandomizedSearchCV done in {time.time()-t0:.1f}s; best_score={rs.best_score_:.4f}")
    print(f"{now()} Best params: {rs.best_params_}")
    return rs.best_estimator_

# ---------- final fit + immediate save ----------
def final_fit_and_save(estimator, X_train_full, y_train_full, out_path=MODEL_FILE, n_estimators_override=None):
    # If estimator is a pipeline with RF inside, optionally set estimator params
    if n_estimators_override is not None:
        # try to set param if estimator has named step
        try:
            # works for pipeline with 'randomforestclassifier' step
            est = estimator
            # if pipeline, set underlying classifier params via set_params
            if hasattr(estimator, "set_params"):
                estimator.set_params(randomforestclassifier__n_estimators=n_estimators_override)
        except Exception:
            pass
    print(f"{now()} Final fit: n_samples={X_train_full.shape[0]} ... (this may take long)")
    t0 = time.time()
    estimator.fit(X_train_full, y_train_full)
    t_fit = time.time() - t0
    print(f"{now()} Final fit done in {t_fit:.1f}s")
    # SAVE IMMEDIATELY after fit
    try:
        joblib.dump(estimator, out_path)
        print(f"{now()} ✅ Final model saved immediately -> {out_path}")
    except Exception as e:
        print(f"{now()} ERROR saving final model: {e}")
        raise
    return estimator

# ---------- main ----------
def main():
    print(f"{now()} START: augment={DO_AUG}, tune={DO_TUNE}, force_extract={FORCE_EXTRACT}")
    # 1) build or reuse expanded CSV
    if not Path(FEATURES_CSV).exists() or FORCE_EXTRACT:
        csv_path = build_features_csv(DATA_DIR, FEATURES_CSV, augment=DO_AUG, aug_per=AUGMENT_PER_IMAGE, aug_prob=AUG_PROB)
    else:
        csv_path = FEATURES_CSV
        print(f"{now()} Using existing features CSV -> {csv_path}")

    # 2) load numeric features & labels
    X, y = load_features_csv(csv_path)

    # 3) train/test split (holdout)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    print(f"{now()} Train/test split: train={X_train.shape[0]} test={X_test.shape[0]}")

    # 4) baseline fit & save checkpoint
    baseline_clf = train_baseline_save(X_train, y_train, X_test, y_test)

    # 5) optionally do hyperparameter tuning on X_train
    best_estimator = baseline_clf
    if DO_TUNE:
        try:
            best = randomized_search_rf(X_train, y_train, n_iter=RANDOMIZED_ITERS)
            best_estimator = best
        except Exception as e:
            print(f"{now()} WARNING: tuning failed: {e}")
            print(f"{now()} Proceeding with baseline estimator.")

    # 6) final fit on training set with final large forest (ensure n_estimators large)
    # override n_estimators to final value (if pipeline supports set_params)
    try:
        best_estimator = final_fit_and_save(best_estimator, X_train, y_train, out_path=MODEL_FILE, n_estimators_override=RF_N_ESTIMATORS_FINAL)
    except Exception as e:
        print(f"{now()} ERROR during final fit/save: {e}")
        print(f"{now()} Attempting to save current estimator as fallback.")
        try:
            joblib.dump(best_estimator, MODEL_FILE)
            print(f"{now()} Fallback saved -> {MODEL_FILE}")
        except Exception as ee:
            print(f"{now()} Fallback save also failed: {ee}")
            raise

    # 7) evaluate on holdout
    preds = best_estimator.predict(X_test)
    print(f"{now()} Final test accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds, zero_division=0))
    print(f"{now()} END")

if __name__ == "__main__":
    main()
