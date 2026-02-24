# Smart Traffic Analytics & Forecasting System

An AI-powered web platform for analyzing historical traffic data and predicting future vehicle volumes at various junctions. Built with Flask, Scikit-Learn (Random Forest), and Chart.js.

## 🚀 Features

- **Automated ML Training**: Upload any traffic dataset (CSV) to automatically train a `RandomForestRegressor`.
- **Intelligent Forecasting**: Predict vehicle volume based on Hour, Day, Month, and Junction.
- **Congestion Analysis**: Real-time classification of traffic flow (LOW, MEDIUM, HIGH) with visual indicators.
- **Analytics Dashboard**: Interactive visualizations of hourly trends and junction distributions using Chart.js.
- **Performance Metrics**: Real-time evaluation of model accuracy using RMSE and R² Score.
- **Premium UI**: Modern dark-themed, glassy interface with smooth transitions and responsive design.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Pandas, Scikit-Learn
- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript, Chart.js
- **Model**: Random Forest Regressor with Lag Features

## 📋 Required CSV Structure

The system expects a CSV file with the following columns (case-insensitive):
- `DateTime`: The timestamp of the reading.
- `Junction`: Integer ID of the traffic junction.
- `Vehicles`: Number of vehicles recorded.

## 🏁 Setup Instructions

1. **Clone the project** and navigate to the directory:
   ```bash
   cd smart_traffic_web
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app.py
   ```

4. **Access the application**:
   Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## 📊 Testing

- A `sample_traffic.csv` is provided in the root directory for immediate testing.
- Upload the file on the home page to train the model and unlock the dashboard and forecasting features.

---
*Developed for Smart Cities Solutions*
