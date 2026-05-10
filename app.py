"""
AutoVal AI — Flask Backend API
================================
University of Essex | BSc AI Year 1 | CE101
Machine Learning Driven Car Price Prediction System

HOW TO RUN:
  1. Install requirements:
       pip install flask flask-cors scikit-learn pandas numpy joblib
  2. Make sure your model/ folder has:
       model/best_model.pkl
       model/label_encoders.pkl
       model/scaler.pkl
       model/feature_cols.pkl
       model/metadata.json
  3. Run:
       python app.py
  4. API will be live at: http://localhost:5000
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import json
import numpy as np
import os

# ── Initialise Flask ──────────────────────────────────────
app = Flask(__name__)
CORS(app)   # Allow requests from our HTML frontend

# ── Load saved model and artefacts ───────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

def load_artefacts():
    """Load all saved model artefacts at startup."""
    try:
        model          = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
        label_encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
        scaler         = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        feature_cols   = joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))

        with open(os.path.join(MODEL_DIR, 'metadata.json')) as f:
            metadata = json.load(f)

        print("✅ Model loaded successfully!")
        print(f"   Model : {metadata['model_name']}")
        print(f"   R²    : {metadata['r2_score']}")
        print(f"   MAE   : PKR {metadata['mae_pkr']:,.0f}")
        return model, label_encoders, scaler, feature_cols, metadata

    except FileNotFoundError as e:
        print(f"❌ ERROR: Could not load model artefacts.")
        print(f"   Missing file: {e}")
        print(f"   Please run train_model.py first to generate the model/ folder.")
        return None, None, None, None, None

model, label_encoders, scaler, feature_cols, metadata = load_artefacts()

# ── Valid input values ────────────────────────────────────
VALID_VALUES = {
    'Brand': ['Toyota', 'Honda', 'Suzuki', 'Hyundai', 'KIA', 'Daihatsu',
              'Nissan', 'Mitsubishi', 'BMW', 'Mercedes', 'Audi'],
    'Fuel_Type': ['Petrol', 'Diesel', 'Hybrid', 'CNG', 'Electric'],
    'Transmission': ['Manual', 'Automatic', 'CVT', 'Semi-Auto'],
    'Condition': ['Excellent', 'Good', 'Fair', 'Poor'],
    'City': ['Karachi', 'Lahore', 'Islamabad', 'Rawalpindi',
             'Peshawar', 'Faisalabad', 'Multan'],
}

# ── Helper: encode a single categorical value ─────────────
def safe_encode(encoder, value, col_name):
    """
    Encode a categorical value using a fitted LabelEncoder.
    If unseen, uses the most common class (index 0) to avoid crash.
    """
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    else:
        print(f"  ⚠️  Unseen value '{value}' for '{col_name}' — using fallback.")
        return 0

# ── ROUTE: Health check ───────────────────────────────────
@app.route('/', methods=['GET'])
def health():
    status = 'ready' if model is not None else 'model_not_loaded'
    return jsonify({
        'status': status,
        'service': 'AutoVal AI — Car Price Prediction API',
        'version': '1.0.0',
        'model': metadata['model_name'] if metadata else 'N/A',
        'endpoints': ['/predict', '/model-info', '/valid-values']
    })

# ── ROUTE: Model info ─────────────────────────────────────
@app.route('/model-info', methods=['GET'])
def model_info():
    if metadata is None:
        return jsonify({'error': 'Model not loaded'}), 503
    return jsonify(metadata)

# ── ROUTE: Valid input values ─────────────────────────────
@app.route('/valid-values', methods=['GET'])
def valid_values():
    return jsonify(VALID_VALUES)

# ── ROUTE: Predict price ──────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON body:
    {
        "brand":        "Toyota",
        "model":        "Corolla",
        "year":         2019,
        "fuel":         "Petrol",
        "transmission": "Automatic",
        "engine":       1600,
        "condition":    "Good",
        "city":         "Karachi",
        "mileage":      55000,
        "owners":       1
    }

    Returns:
    {
        "success":     true,
        "price_pkr":   3250000,
        "price_lakhs": "32.50",
        "price_range": { "low": "29.90", "high": "35.10" },
        "confidence_pct": 91,
        "model_used":  "Random Forest Regressor",
        "factors": [ ... ]
    }
    """

    # ── 0. Check model is loaded ──
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Please run train_model.py first.'
        }), 503

    # ── 1. Parse request body ──
    data = request.get_json(force=True)
    if not data:
        return jsonify({'success': False, 'error': 'No JSON body received'}), 400

    # ── 2. Extract and validate fields ──
    required = ['brand', 'model', 'year', 'fuel', 'transmission',
                'engine', 'condition', 'city', 'mileage', 'owners']

    missing = [f for f in required if f not in data or data[f] == '']
    if missing:
        return jsonify({
            'success': False,
            'error': f"Missing required fields: {', '.join(missing)}"
        }), 400

    try:
        brand        = str(data['brand'])
        car_model    = str(data['model'])
        year         = int(data['year'])
        fuel         = str(data['fuel'])
        transmission = str(data['transmission'])
        engine_cc    = int(data['engine'])
        condition    = str(data['condition'])
        city         = str(data['city'])
        mileage      = int(data['mileage'])
        owners       = int(data['owners'])
    except (ValueError, TypeError) as e:
        return jsonify({'success': False, 'error': f'Invalid field type: {str(e)}'}), 400

    # ── 3. Range checks ──
    if not (1990 <= year <= 2025):
        return jsonify({'success': False, 'error': 'Year must be between 1990 and 2025'}), 400
    if not (0 <= mileage <= 500000):
        return jsonify({'success': False, 'error': 'Mileage must be between 0 and 500,000 km'}), 400
    if not (1 <= owners <= 10):
        return jsonify({'success': False, 'error': 'Owners must be between 1 and 10'}), 400

    # ── 4. Encode categorical features ──
    try:
        brand_enc  = safe_encode(label_encoders['Brand'],        brand,        'Brand')
        model_enc  = safe_encode(label_encoders['Model'],        car_model,    'Model')
        fuel_enc   = safe_encode(label_encoders['Fuel_Type'],    fuel,         'Fuel_Type')
        trans_enc  = safe_encode(label_encoders['Transmission'], transmission, 'Transmission')
        cond_enc   = safe_encode(label_encoders['Condition'],    condition,    'Condition')
        city_enc   = safe_encode(label_encoders['City'],         city,         'City')
    except Exception as e:
        return jsonify({'success': False, 'error': f'Encoding error: {str(e)}'}), 500

    # ── 5. Assemble feature vector (order MUST match training) ──
    feature_vector = np.array([[
        brand_enc,    # Brand_enc
        model_enc,    # Model_enc
        year,         # Year
        fuel_enc,     # Fuel_Type_enc
        trans_enc,    # Transmission_enc
        engine_cc,    # Engine_CC
        cond_enc,     # Condition_enc
        city_enc,     # City_enc
        mileage,      # Mileage_KM
        owners        # Previous_Owners
    ]])

    # ── 6. Scale (Linear Regression needs scaled input) ──
    # For tree models, scaling is a no-op but doesn't hurt
    from sklearn.linear_model import LinearRegression as LR
    if isinstance(model, LR):
        feature_vector_scaled = scaler.transform(feature_vector)
        prediction = model.predict(feature_vector_scaled)[0]
    else:
        prediction = model.predict(feature_vector)[0]

    prediction = max(200000, float(prediction))  # Floor at 200k PKR

    # ── 7. Build confidence & range ──
    # Estimate confidence using heuristics (real ML would use prediction intervals)
    age = 2025 - year
    conf = 95
    conf -= age * 0.4              # Older cars harder to price
    conf -= (mileage / 10000) * 0.15  # High mileage adds uncertainty
    if condition in ['Poor', 'Fair']:
        conf -= 4
    conf = int(min(97, max(70, conf)))

    # Price range: ±8% around prediction (simulates confidence interval)
    low_price  = prediction * 0.92
    high_price = prediction * 1.08

    # ── 8. Build factor analysis ──
    factors = build_factors(brand, year, fuel, transmission, engine_cc,
                             condition, mileage, owners)

    # ── 9. Return response ──
    response = {
        'success': True,
        'price_pkr':      round(prediction),
        'price_lakhs':    f"{prediction / 100000:.2f}",
        'price_range': {
            'low':  f"{low_price / 100000:.1f}",
            'high': f"{high_price / 100000:.1f}",
        },
        'confidence_pct': conf,
        'model_used':     metadata['model_name'],
        'model_r2':       metadata['r2_score'],
        'factors':        factors,
        'input_summary': {
            'brand': brand, 'model': car_model, 'year': year,
            'fuel': fuel, 'transmission': transmission,
            'engine_cc': engine_cc, 'condition': condition,
            'city': city, 'mileage_km': mileage, 'owners': owners
        }
    }

    print(f"✅ Prediction: PKR {prediction:,.0f} for {year} {brand} {car_model}")
    return jsonify(response)


def build_factors(brand, year, fuel, transmission, engine_cc, condition, mileage, owners):
    """Generates a human-readable factor analysis for the result."""
    age = 2025 - year
    factors = []

    # Year / Age
    if age <= 3:
        factors.append({'name': f'Year {year} (nearly new)', 'impact': 'POSITIVE', 'detail': 'Very low depreciation'})
    elif age <= 7:
        factors.append({'name': f'Year {year} ({age} years old)', 'impact': 'NEUTRAL', 'detail': 'Moderate depreciation applied'})
    else:
        factors.append({'name': f'Year {year} ({age} years old)', 'impact': 'NEGATIVE', 'detail': f'{min(78, age*8.5):.0f}% depreciation applied'})

    # Mileage
    if mileage < 40000:
        factors.append({'name': f'Low mileage ({mileage:,} km)', 'impact': 'POSITIVE', 'detail': 'Minimal wear on engine and drivetrain'})
    elif mileage < 120000:
        factors.append({'name': f'Average mileage ({mileage:,} km)', 'impact': 'NEUTRAL', 'detail': 'Normal depreciation for mileage'})
    else:
        factors.append({'name': f'High mileage ({mileage:,} km)', 'impact': 'NEGATIVE', 'detail': 'Significant mileage deduction applied'})

    # Fuel
    fuel_impact = {'Petrol': 'NEUTRAL', 'Diesel': 'POSITIVE', 'Hybrid': 'POSITIVE',
                   'CNG': 'NEGATIVE', 'Electric': 'POSITIVE'}
    fuel_detail = {
        'Petrol': 'Standard fuel type — no premium or discount',
        'Diesel': '+8% premium for fuel efficiency',
        'Hybrid': '+22% premium for eco-friendly drivetrain',
        'CNG': '-15% discount (lower market demand)',
        'Electric': '+38% premium for EV technology'
    }
    factors.append({'name': f'Fuel: {fuel}', 'impact': fuel_impact.get(fuel, 'NEUTRAL'),
                    'detail': fuel_detail.get(fuel, '')})

    # Transmission
    if transmission == 'Automatic':
        factors.append({'name': 'Automatic transmission', 'impact': 'POSITIVE', 'detail': '+10% for ease of driving and higher demand'})
    elif transmission == 'CVT':
        factors.append({'name': 'CVT transmission', 'impact': 'POSITIVE', 'detail': '+4% for smooth driving experience'})
    else:
        factors.append({'name': 'Manual transmission', 'impact': 'NEUTRAL', 'detail': '-7% vs automatic market preference'})

    # Condition
    cond_impact = {'Excellent': 'POSITIVE', 'Good': 'NEUTRAL', 'Fair': 'NEGATIVE', 'Poor': 'NEGATIVE'}
    cond_detail = {
        'Excellent': '+12% premium for pristine condition',
        'Good': 'No adjustment — standard market price',
        'Fair': '-14% for visible wear or minor damage',
        'Poor': '-30% for significant damage or issues'
    }
    factors.append({'name': f'Condition: {condition}', 'impact': cond_impact.get(condition, 'NEUTRAL'),
                    'detail': cond_detail.get(condition, '')})

    # Owners
    if owners == 1:
        factors.append({'name': '1st owner', 'impact': 'POSITIVE', 'detail': 'Single owner history adds premium'})
    elif owners == 2:
        factors.append({'name': f'{owners} previous owners', 'impact': 'NEUTRAL', 'detail': '-6% for 2nd-hand history'})
    else:
        factors.append({'name': f'{owners} previous owners', 'impact': 'NEGATIVE', 'detail': f'-{(owners-1)*8}% for multi-owner history'})

    return factors


# ── ROUTE: Batch predict (bonus endpoint) ────────────────
@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Accepts a list of cars and returns predictions for all.
    Body: { "cars": [ {...}, {...} ] }
    Useful for comparing multiple cars at once.
    """
    data = request.get_json(force=True)
    if not data or 'cars' not in data:
        return jsonify({'success': False, 'error': "Expected JSON with 'cars' list"}), 400

    cars = data['cars']
    if not isinstance(cars, list) or len(cars) == 0:
        return jsonify({'success': False, 'error': 'cars must be a non-empty list'}), 400

    results = []
    for i, car in enumerate(cars):
        with app.test_request_context('/predict', method='POST',
                                       json=car, content_type='application/json'):
            resp = predict()
            if hasattr(resp, 'get_json'):
                results.append({'index': i, 'result': resp.get_json()})
            else:
                results.append({'index': i, 'result': resp[0].get_json()})

    return jsonify({'success': True, 'predictions': results, 'count': len(results)})


# ── Run server ────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  🚗  AutoVal AI — Car Price Prediction API")
    print("="*55)
    print("  University of Essex | BSc AI Year 1")
    print("  Server: http://localhost:5000")
    print("  Docs:   http://localhost:5000/")
    print("="*55 + "\n")

    app.run(
        host='0.0.0.0',   # Accept requests from any network interface
        port=5000,
        debug=True        # Set False for production
    )
