import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime
from dotenv import load_dotenv
from functools import wraps
import clerk_backend_api
from clerk_backend_api import Clerk

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MODEL_FOLDER'] = 'model'

# Clerk Configuration
CLERK_SECRET_KEY = os.getenv('CLERK_SECRET_KEY')
clerk_client = None
if CLERK_SECRET_KEY and CLERK_SECRET_KEY != 'REPLACE_WITH_YOUR_SECRET_KEY':
    clerk_client = Clerk(bearer_auth=CLERK_SECRET_KEY)

# Auth Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # In a Clerk-integrated Flask app, the frontend handles the session.
        # Backend verification requires the CLERK_SECRET_KEY.
        if clerk_client is None:
            # If no Secret Key is provided, we check for session cookie
            if not request.cookies.get('__session'):
                if request.method == 'GET':
                    return redirect(url_for('index', auth_required=1))
                return jsonify({"status": "error", "message": "Authentication required"}), 401
            return f(*args, **kwargs)
        
        # When CLERK_SECRET_KEY is present, we would perform full JWT verification here.
        return f(*args, **kwargs)
    return decorated_function

# Ensure required directories exist for storage
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Global State Management (In-memory storage)
model = None
latest_df = None
model_metrics = {}

def preprocess_data(df):
    """
    Standardizes column names and performs feature engineering.
    Extracts time components and generates lag features for time-series forecasting.
    """
    # Standardize column headers to lowercase
    df.columns = [c.lower().strip() for c in df.columns]
    
    # Define coordinate mapping for various possible naming conventions
    col_map = {
        'datetime': 'datetime',
        'date_time': 'datetime',
        'junction': 'junction',
        'vehicles': 'vehicles',
        'id': 'id'
    }
    
    # Rename columns based on map
    found_cols = {c: col_map[c] for c in df.columns if c in col_map}
    df = df.rename(columns=found_cols)
    
    # Validate required data columns
    required = ['datetime', 'junction', 'vehicles']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    # Time series processing
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['junction', 'datetime']) 
    
    # Feature 1-3: Time components
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    
    # Feature 4-5: Lag features (Previous states)
    # lag_1: Traffic volume at the previous hour
    df['lag_1'] = df.groupby('junction')['vehicles'].shift(1)
    # lag_24: Traffic volume 24 hours ago (same time previous day)
    df['lag_24'] = df.groupby('junction')['vehicles'].shift(24)
    
    return df

@app.route('/')
def index():
    """Renders the data upload and training initialization page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """
    Handles CSV upload, data preprocessing, model training, and evaluation.
    Returns a JSON summary of the training results and performance metrics.
    """
    global model, latest_df, model_metrics
    
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400
    
    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            # 1. Processing
            df = pd.read_csv(filepath)
            df = preprocess_data(df)
            df_trained = df.dropna()
            latest_df = df.copy() 
            
            # 2. Setup Features and Target
            features = ['hour', 'day_of_week', 'month', 'junction', 'lag_1', 'lag_24']
            X = df_trained[features]
            y = df_trained['vehicles']
            
            # 3. Train/Test Split (80/20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 4. Model Training (Disabled n_jobs for better Windows/Flask compatibility)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 5. Performance Evaluation
            y_pred = model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            
            model_metrics = {
                "rmse": round(rmse, 2),
                "r2_score": round(r2, 4)
            }
            
            # 6. Persistence
            model_path = os.path.join(app.config['MODEL_FOLDER'], 'traffic_model.pkl')
            joblib.dump(model, model_path)
            
            return redirect(url_for('dashboard'))
            
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    else:
        return jsonify({"status": "error", "message": "Invalid file format. Please upload a CSV."}), 400

@app.route('/dashboard')
@login_required
def dashboard():
    """Renders the analytics dashboard view."""
    return render_template('dashboard.html')

@app.route('/demo')
def demo():
    """Renders the Live Demo page mockup with neon branding."""
    return render_template('demo.html')

@app.route('/about')
def about():
    """Renders the infographic-style About page."""
    return render_template('about.html')

@app.route('/api/analytics')
@login_required
def get_analytics():
    """
    API endpoint that serves analytics data and model metrics for the dashboard.
    Enhanced with congestion counts for PIE chart.
    """
    global latest_df, model, model_metrics
    if latest_df is None:
        return jsonify({"status": "error", "message": "Dataset not found. Please upload data first."}), 400
    
    # Calculate Summary Statistics
    total_volume = int(latest_df['vehicles'].sum())
    avg_hourly = float(latest_df.groupby('hour')['vehicles'].mean().mean())
    peak_hour = int(latest_df.groupby('hour')['vehicles'].mean().idxmax())
    
    # Prepare Chart Data
    hourly_avg = latest_df.groupby('hour')['vehicles'].mean().sort_index().to_dict()
    junction_avg = latest_df.groupby('junction')['vehicles'].mean().sort_index().to_dict()

    # Congestion for Pie Chart (Mocking metrics if model exists)
    congestion_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    if model is not None:
        # We can run a sample prediction on a subset or the whole df to get distribution
        sample_size = min(100, len(latest_df))
        sample = latest_df.dropna().sample(sample_size)
        features = ['hour', 'day_of_week', 'month', 'junction', 'lag_1', 'lag_24']
        preds = model.predict(sample[features])
        for p in preds:
            if p > 100: congestion_counts["HIGH"] += 1
            elif p >= 50: congestion_counts["MEDIUM"] += 1
            else: congestion_counts["LOW"] += 1
    else:
        # Fallback if model isn't trained (unlikely if latest_df exists but safety first)
        congestion_counts = {"LOW": 60, "MEDIUM": 30, "HIGH": 10}

    return jsonify({
        "summary": {
            "total_volume": total_volume,
            "average_hourly": round(avg_hourly, 2),
            "peak_hour": peak_hour,
            "metrics": model_metrics
        },
        "charts": {
            "hourly": {
                "labels": list(hourly_avg.keys()),
                "data": list(hourly_avg.values())
            },
            "junction": {
                "labels": list(junction_avg.keys()),
                "data": list(junction_avg.values())
            },
            "congestion": {
                "labels": list(congestion_counts.keys()),
                "data": list(congestion_counts.values())
            }
        }
    })

@app.route('/forecast_page')
@login_required
def forecast_page():
    """Renders the prediction/forecasting interface."""
    return render_template('forecast.html')

@app.route('/forecast', methods=['POST'])
@login_required
def forecast():
    """
    Predicts traffic volume for a given time and junction.
    Automatically retrieves necessary lag features from the latest processed data.
    """
    global model, latest_df
    
    # Ensure model is ready
    if model is None:
        model_path = os.path.join(app.config['MODEL_FOLDER'], 'traffic_model.pkl')
        if os.path.exists(model_path):
            model = joblib.load(model_path)
        else:
            return jsonify({'error': 'No trained model found. Please upload training data.'}), 400
            
    # Parse Request Data (Handle both JSON and Form data for legacy/compatibility)
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    try:
        hour = int(data.get('hour'))
        day = int(data.get('day'))
        month = int(data.get('month'))
        junction = int(data.get('junction'))
    except (TypeError, ValueError, AttributeError):
        return jsonify({'error': 'Invalid input data. Please check your parameters.'}), 400
    
    # Data check for lag features
    if latest_df is None:
         return jsonify({'error': 'Lag data missing. Re-upload CSV to populate history.'}), 400
         
    # Retrieve Lag Features for the specific junction
    junc_data = latest_df[latest_df['junction'] == junction]
    if junc_data.empty:
        # Fallback if junction ID is unrecognized relative to the dataset
        lag_1 = 0
        lag_24 = 0
    else:
        lag_1 = junc_data.iloc[-1]['vehicles']
        lag_24 = junc_data.iloc[-24]['vehicles'] if len(junc_data) >= 24 else lag_1

    # Regression Prediction
    prediction = model.predict([[hour, day, month, junction, lag_1, lag_24]])[0]
    
    # Traffic Congestion Classification 
    congestion = "LOW"
    if prediction > 100: 
        congestion = "HIGH"
    elif prediction >= 50: 
        congestion = "MEDIUM"
    
    return jsonify({
        'predicted_volume': int(round(prediction)),
        'congestion': congestion
    })

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Legacy endpoint for compatibility."""
    return forecast()

if __name__ == '__main__':
    # Run the Flask development server
    app.run()
