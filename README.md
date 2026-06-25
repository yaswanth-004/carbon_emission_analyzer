Carbon Emission Analyzer

A full-stack Machine Learning web application that predicts CO₂ emissions for Household lifestyle habits and Industrial activity using a Decision Tree algorithm — served through a Flask web interface.

What It Does

Takes household lifestyle inputs (diet, transport, energy use, etc.) or industrial source data (coal, oil, gas CO₂ etc.)
Predicts the exact CO₂ value in kg/year using a Decision Tree Regressor
Classifies the result into LOW / MEDIUM / HIGH emission band using a Decision Tree Classifier
Displays results in a modern dark-themed web UI with animated output, gauge bar, and personalised reduction tips

Data Pipeline
Two real-world datasets were cleaned and merged into one unified frame of 15,563 rows and 29 features.
Household (Carbon Emission.csv)

Standardised column names (lowercase + underscores)
Removed duplicate rows
Filled missing vehicle_type → none (users without a personal vehicle)
Parsed list-formatted columns (recycling, cooking_with) using ast.literal_eval
Removed outliers using 1st–99th percentile IQR clipping
Label-encoded all categorical columns and saved maps to label_maps.pkl

Industrial (owid-co2-data.csv)

Selected 8 relevant columns from 70+ raw columns
Filtered to years ≥ 1990 and dropped all-zero rows
Converted CO₂ from million tonnes → kg (× 10⁹)
Padded missing household columns with 0 for a unified schema


🤖 ML Models
Both models trained with:
max_depth=8  |  min_samples_split=10  |  min_samples_leaf=5  |  random_state=42
Training split: 80% train / 20% test with 5-fold cross-validation.
Emission Band Thresholds
Type🟢 LOW🟡 MEDIUM🔴 HIGHHousehold (kg/year)< 1,5001,500 – 3,500> 3,500Industrial (kg/year)< 1×10¹¹1×10¹¹ – 1×10¹²> 1×10¹²
Top Features by Importance Score
oil_co2 (~0.73) → country (~0.15) → flaring_co2 (~0.07) → lifestyle features

🚀 Getting Started
1. Clone the repository
bashgit clone https://github.com/yaswanth-004/carbon_emission_analyzer.git
cd carbon_emission_analyzer
2. Install dependencies
bashpip install -r requirements.txt
3. Run the Flask app
bashpython app.py
Open your browser at http://localhost:5000

🖥️ Web Interface
The app has two prediction modes accessible via tabs:
🏠 Household Mode — 19 input fields covering:

Body type, sex, diet
Transport type, vehicle type, monthly distance
Air travel frequency, heating energy source
Grocery bill, waste bags, screen time
Recycling materials, cooking appliances
Energy efficiency practices

🏭 Industrial Mode — 8 industrial fields:

Country code, year
Coal, oil, gas, cement, flaring, other industry CO₂ (million tonnes)

Result panel shows:

Exact CO₂ prediction in kg/year with animated counter
Emission band badge (🟢 LOW / 🟡 MEDIUM / 🔴 HIGH)
Visual gauge bar
Personalised reduction tips per band


📦 Requirements
flask
joblib
pandas
numpy
scikit-learn
matplotlib
seaborn
bashpip install -r requirements.txt

🔧 Using the Models Directly
pythonimport joblib
import pandas as pd

# Load saved models
reg_model     = joblib.load('model/model_regression.pkl')
cls_model     = joblib.load('model/model_classification.pkl')
feature_names = joblib.load('model/feature_names.pkl')
label_maps    = joblib.load('model/label_maps.pkl')

# Prepare your input (must match feature_names order, all numeric)
sample = pd.DataFrame([your_encoded_dict])[feature_names]

# Predict
co2_kg = reg_model.predict(sample)[0]       # e.g. 2238.5 kg/year
band   = cls_model.predict(sample)[0]       # 0 = LOW, 1 = MEDIUM, 2 = HIGH

📊 Dataset Sources
DatasetSourceCarbon Emission (Household)Kaggle — Carbon Footprint DatasetCO₂ by Country (Industrial)Our World in Data

📄 License
This project is open source under the MIT License — free to use for educational and research purposes.
