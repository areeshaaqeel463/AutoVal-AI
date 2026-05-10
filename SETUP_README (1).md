# AutoVal AI — Backend Setup Guide
# University of Essex | BSc AI Year 1 | Team Project Challenge

## ── requirements.txt ────────────────────────────────────
# Save these as requirements.txt and run:
#   pip install -r requirements.txt

flask==3.0.0
flask-cors==4.0.0
scikit-learn==1.4.0
pandas==2.1.4
numpy==1.26.3
joblib==1.3.2
matplotlib==3.8.2
seaborn==2.0.0

## ── Project Folder Structure ────────────────────────────

AutoVal-AI/
│
├── data/
│   └── used_cars_pakistan.csv       ← Dataset (download from Kaggle)
│
├── model/                           ← Created automatically by train_model.py
│   ├── best_model.pkl
│   ├── label_encoders.pkl
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   └── metadata.json
│
├── train_model.py                   ← Step 1: Run this first
├── app.py                           ← Step 2: Run this as the server
├── car-price-predictor-connected.html ← Step 3: Open this in browser
└── requirements.txt

## ── Step-by-Step Setup ───────────────────────────────────

STEP 1 — Install Python
  Download from: https://python.org/downloads
  Version: Python 3.10 or 3.11 recommended

STEP 2 — Install dependencies
  Open terminal/command prompt in the project folder:
  pip install flask flask-cors scikit-learn pandas numpy joblib matplotlib seaborn

STEP 3 — Get the Dataset (choose one option)

  OPTION A (Recommended — Pakistan specific):
    URL: https://www.kaggle.com/datasets/imadkhattak/pakwheel-cars-data
    Name: Pakistan Used Car Market Dataset (2024)
    Source: PakWheels.com listings
    Records: ~5,000+ car listings
    Columns: brand, model, year, mileage, fuel, price (PKR), city, transmission
    Download: pakwheels_cars.csv → rename to used_cars_pakistan.csv

  OPTION B (Pakistan — OLX data):
    URL: https://www.kaggle.com/datasets/muhammadawaistayyab/used-car-price-prediction-pakistan
    Records: ~3,000 listings from OLX Pakistan

  OPTION C (No download — synthetic dataset):
    Just run train_model.py as-is — it auto-generates 2,000 realistic records
    Good for testing and development!

STEP 4 — Train the model
  python train_model.py
  
  This will:
  ✓ Load / generate the dataset
  ✓ Run EDA and save 4 plots (PNG files)
  ✓ Train Linear Regression, Random Forest, Gradient Boosting
  ✓ Print performance metrics for each
  ✓ Save best model + encoders to model/ folder
  ✓ Print final summary

  Expected output example:
  ┌─ Random Forest Regressor ─────────────────────────────
  │  MAE  : PKR      280,000
  │  RMSE : PKR      420,000
  │  R²   : 0.8740  (87.4% variance explained)
  └─────────────────────────────────────────────────────

STEP 5 — Start the Flask API server
  python app.py

  You should see:
  =====================================================
    🚗  AutoVal AI — Car Price Prediction API
  =====================================================
    Server: http://localhost:5000
  =====================================================

  Test the API is working:
  Open browser → http://localhost:5000
  You should see a JSON response like:
  {
    "status": "ready",
    "service": "AutoVal AI — Car Price Prediction API",
    "model": "Random Forest Regressor"
  }

STEP 6 — Open the frontend
  Open car-price-predictor-connected.html in your browser.
  The API Status indicator in the top-right should show:
    ● API Online · Random Forest Regressor

  Select your car details and click "Predict Price Now"
  The real ML model will respond with a PKR price!

## ── API Endpoints Reference ─────────────────────────────

GET  /                 → Health check, model info
GET  /model-info       → Full model metadata + scores
GET  /valid-values     → Valid dropdown options
POST /predict          → Single car prediction

POST /predict — Request body:
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

POST /predict — Response:
{
  "success":        true,
  "price_pkr":      3250000,
  "price_lakhs":    "32.50",
  "price_range":    { "low": "29.90", "high": "35.10" },
  "confidence_pct": 91,
  "model_used":     "Random Forest Regressor",
  "model_r2":       0.874,
  "factors":        [ ... ],
  "input_summary":  { ... }
}

## ── Troubleshooting ──────────────────────────────────────

PROBLEM: "ModuleNotFoundError: No module named 'flask'"
SOLUTION: pip install flask flask-cors

PROBLEM: "FileNotFoundError: model/best_model.pkl"
SOLUTION: Run train_model.py first before app.py

PROBLEM: Frontend shows "API Offline"
SOLUTION: Make sure app.py is running in a terminal. 
          The terminal must stay open while you use the site.

PROBLEM: CORS error in browser console
SOLUTION: flask-cors is installed and enabled. 
          If still an issue: pip install flask-cors --upgrade

PROBLEM: "Port 5000 is already in use"
SOLUTION: Change port in app.py: app.run(port=5001)
          Then update API_BASE in the HTML to: http://localhost:5001

## ── Kaggle Dataset Columns to Expect ───────────────────

After downloading the PakWheels dataset, check column names.
Common column mappings (edit train_model.py if columns differ):

Kaggle column name   →  Variable name in our code
─────────────────────────────────────────────────
make / brand         →  Brand
model                →  Model
year / Year          →  Year
mileage / km_driven  →  Mileage_KM
fuel_type / fuel     →  Fuel_Type
transmission         →  Transmission
engine_cc / cc       →  Engine_CC
condition            →  Condition
city / location      →  City
price / Price_PKR    →  Price_PKR

##  Project Management
- **Jira Board:** https://bic-team-t6b2f6in.atlassian.net/jira/software/projects/AVA/boards
- **Total Tasks:** 54 tasks completed
- **Sprints:** 8 sprints
- **Story Points:** 173 pts delivered

## ── Team Notes (for JIRA / Report) ────────────────────

Hasan Ali Khan (ML Engineer):
  - Run and document train_model.py results
  - Screenshot the comparison table of all 3 models
  - Paste metrics (MAE, RMSE, R²) into the JIRA ticket AVA-012
  - Save the 4 EDA plots for the team report

Arham Salman (Data Engineer):
  - Download dataset from Kaggle (Option A or B above)
  - Update train_model.py to load real CSV instead of synthetic
  - Map column names using the table above
  - Document dataset source, size, and columns in JIRA AVA-004

Jabber Arif Khan (Frontend Developer):
  - Test car-price-predictor-connected.html with the live API
  - Verify all 11 car brands and their models work
  - Test on mobile device for responsiveness
  - Screenshot the working prediction for JIRA AVA-019

Areesha Aqeel (Project Lead):
  - Coordinate dataset download and verify it works
  - Update all JIRA tickets with progress
  - Write the integration section of Assignment 2
  - team report
  - github
  - cordinate with all members

---
AutoVal AI | University of Essex | BSc Artificial Intelligence | Year 1
Team Project Challenge | Instructor: Sir Sajid Ali | 2025
