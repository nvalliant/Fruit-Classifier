import os
import base64
import traceback
import numpy as np
import cv2
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ─── Konfigurasi Path Model (Gunakan Absolute Path agar aman) ─────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "fruit_svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "fruit_scaler.pkl")

model  = None
scaler = None
classes = None

def load_model():
    global model, scaler, classes
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model tidak ditemukan di {MODEL_PATH}")
        return False
    try:
        model   = joblib.load(MODEL_PATH)
        scaler  = joblib.load(SCALER_PATH)
        # Ambil list kelas langsung dari atribut bawaan model Scikit-Learn
        classes = model.classes_.tolist() 
        print(f"[OK] Model loaded: {MODEL_PATH}")
        print(f"[OK] Classes found: {classes}")
        return True
    except Exception as e:
        print(f"[ERROR] Gagal load model: {e}")
        return False

model_loaded = load_model()


# ─── Helper: Mapping Label Kaggle ke Format Frontend ─────────────────────────
def map_kaggle_label(raw_label):
    mapping = {
        'RipeBanana':      ('banana',     'ripe'),
        'RottenBanana':    ('banana',     'rotten'),
        'UnripeBanana':    ('banana',     'unripe'),
        'RipeStrawberry':  ('strawberry', 'ripe'),
        'RottenStrawberry':('strawberry', 'rotten'),
        'UnripeStrawberry':('strawberry', 'unripe'),
        'RipeOrange':      ('orange',     'ripe'),
        'RottenOrange':    ('orange',     'rotten'),
        'UnripeOrange':    ('orange',     'unripe'),
    }
    ft, rs = mapping.get(raw_label, (raw_label, 'unknown'))
    return f"{ft}_{rs}", ft, rs


# ─── Feature extraction — exact copy dari Cell 2 notebook ────────────────────
def extract_features_from_array(img_array, size=(128, 128)):
    img_resized = cv2.resize(img_array, size)
    gray    = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aspect_ratio, extent = 0, 0
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h if h != 0 else 0
        area = cv2.contourArea(c)
        rect_area = w * h
        extent = float(area) / rect_area if rect_area != 0 else 0

    fp = mask > 0

    # HSV (6)
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    hsv_feats = [
        np.mean(h_ch[fp]) if fp.any() else 0, np.mean(s_ch[fp]) if fp.any() else 0,
        np.mean(v_ch[fp]) if fp.any() else 0, np.std(h_ch[fp])  if fp.any() else 0,
        np.std(s_ch[fp])  if fp.any() else 0, np.std(v_ch[fp])  if fp.any() else 0,
    ]

    # LAB (5) ← NEW
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    lab_feats = [
        np.mean(l_ch[fp]) if fp.any() else 0, np.mean(a_ch[fp]) if fp.any() else 0,
        np.mean(b_ch[fp]) if fp.any() else 0, np.std(a_ch[fp])  if fp.any() else 0,
        np.std(b_ch[fp])  if fp.any() else 0,
    ]

    # Hue histogram 18 bins (18) ← NEW
    h_hist = cv2.calcHist([h_ch], [0], mask, [18], [0, 180])
    h_hist = cv2.normalize(h_hist, h_hist).flatten().tolist()

    # GLCM with distances=[1,3,5], 4 angles, 6 props (6) ← EXPANDED
    # Quantise to 32 levels: reduces sparsity and speeds up computation.
    # ⚠️  Must match the notebook — retrain if you change this value.
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask).astype(np.uint8)
    masked_gray = (masked_gray // 8).astype(np.uint8)   # 256 → 32 levels
    glcm = graycomatrix(masked_gray, distances=[1, 3, 5],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)
    glcm_feats = [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'correlation').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'dissimilarity').mean(),  # NEW
        graycoprops(glcm, 'ASM').mean(),             # NEW
    ]

    # LBP histogram 10 bins (10) ← NEW
    lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
    lbp_pixels = lbp[fp] if fp.any() else lbp.ravel()
    lbp_hist, _ = np.histogram(lbp_pixels, bins=10, range=(0, 10), density=True)

    features = hsv_feats + lab_feats + h_hist + glcm_feats + lbp_hist.tolist() + [aspect_ratio, extent]
    
    # raw dict for frontend display (keep the same keys you already use)
    raw = {
        'h_mean': hsv_feats[0], 's_mean': hsv_feats[1], 'v_mean': hsv_feats[2],
        'h_std':  hsv_feats[3], 's_std':  hsv_feats[4], 'v_std':  hsv_feats[5],
        'contrast': glcm_feats[0], 'correlation': glcm_feats[1],
        'energy': glcm_feats[2], 'homogeneity': glcm_feats[3],
        'aspect_ratio': aspect_ratio, 'extent': extent,
    }
    return features, raw


# ─── CORS headers helper ──────────────────────────────────────────────────────
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/')
def index():
    return render_template('index.html')

@app.after_request
def after_request(response):
    return add_cors(response)

@app.route('/predict', methods=['OPTIONS'])
@app.route('/health',  methods=['OPTIONS'])
@app.route('/classes', methods=['OPTIONS'])
def options():
    resp = jsonify({})
    return add_cors(resp)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    # Frontend sekarang akan menerima mapped classes
    mapped_classes = [map_kaggle_label(c)[0] for c in classes] if classes else []
    return jsonify({
        'status':       'ok',
        'model_loaded': model_loaded,
        'classes':      mapped_classes, 
        'model_path':   MODEL_PATH,
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    mapped_classes = [map_kaggle_label(c)[0] for c in classes] if classes else []
    return jsonify({'classes': mapped_classes})


@app.route('/predict', methods=['POST'])
def predict():
    if not model_loaded or model is None:
        return jsonify({'error': 'Model belum diload.'}), 503

    try:
        if request.is_json:
            data = request.get_json()
            image_b64 = data.get('image', '')
            if ',' in image_b64:
                image_b64 = image_b64.split(',', 1)[1]
            img_bytes = base64.b64decode(image_b64)
            nparr     = np.frombuffer(img_bytes, np.uint8)
            img_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'Input tidak valid.'}), 400

        if img_array is None:
            return jsonify({'error': 'Gagal decode gambar.'}), 400

        # Ekstraksi & Scaling
        features, raw = extract_features_from_array(img_array)
        features_scaled = scaler.transform([features])

        # Prediksi dari model
        prediction_raw = model.predict(features_scaled)[0]
        
        # Format ke frontend
        frontend_prediction, fruit_type, ripeness_stage = map_kaggle_label(prediction_raw)

        probs = {}
        confidence = None
        
        if hasattr(model, 'predict_proba'):
            # Jika training dengan probability=True
            prob_array = model.predict_proba(features_scaled)[0]
            for cls_raw, p in zip(classes, prob_array):
                front_cls, _, _ = map_kaggle_label(cls_raw)
                probs[front_cls] = float(p)
            confidence = float(max(prob_array))
        else:
            # Fallback jika model SVC tidak dilatih dengan probability=True
            dec = model.decision_function(features_scaled)[0]
            dec_shifted = dec - dec.min()
            total = dec_shifted.sum()
            for cls_raw, sc in zip(classes, dec_shifted):
                front_cls, _, _ = map_kaggle_label(cls_raw)
                probs[front_cls] = float(sc / total) if total > 0 else 0.0
            confidence = float(probs.get(frontend_prediction, 0.0))

        return jsonify({
            'prediction':     frontend_prediction,
            'fruit_type':     fruit_type,
            'ripeness_stage': ripeness_stage,
            'confidence':     confidence,
            'probabilities':  probs,
            'features':       raw,
            'source':         'python_svm_model',
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ─── Run ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("RIPE.AI — Flask API Server")
    print("=" * 60)
    print(f"Model loaded: {model_loaded}")
    app.run(host='0.0.0.0', port=5000, debug=False)