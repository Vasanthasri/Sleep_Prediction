from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model
with open('sleep_health_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load label encoders
with open('label_encoders.pkl', 'rb') as le_file:
    label_encoders = pickle.load(le_file)

# Feature names used during model training (excluding 'Quality of Sleep')
model_features = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 
    'Physical Activity Level', 'Stress Level', 'BMI Category', 
    'Blood Pressure Systolic', 'Blood Pressure Diastolic', 'Heart Rate', 
    'Daily Steps', 'Sleep Disorder'
]

@app.route('/')
def home():
    return render_template('index.html')

def predict():
    try:
        # Extract form data
        data = request.form
        sleep_duration = float(data.get('Sleep Duration', 0))
        physical_activity_level = data.get('Physical Activity Level')
        stress_level = int(data.get('Stress Level', 0))
        bmi_category = data.get('BMI Category')
        blood_pressure_systolic = int(data.get('Blood Pressure Systolic', 0))
        blood_pressure_diastolic = int(data.get('Blood Pressure Diastolic', 0))
        heart_rate = int(data.get('Heart Rate', 0))
        daily_steps = int(data.get('Daily Steps', 0))
        sleep_disorder = data.get('Sleep Disorder')
        
        # Initialize prediction
        prediction = "Average"
        
        # Check sleep duration
        if sleep_duration < 4:
            prediction = "Very Poor"
        elif sleep_duration < 6:
            prediction = "Poor"
        elif sleep_duration <= 7:
            prediction = "Average"
        else:
            prediction = "Good"
        
        # Check blood pressure
        if blood_pressure_systolic < 90 or blood_pressure_diastolic < 60:
            prediction = "Poor"
        elif blood_pressure_systolic >= 120 or blood_pressure_diastolic >= 80:
            prediction = "Poor"

        # Check heart rate
        if heart_rate < 60 or heart_rate > 100:
            prediction = "Poor"
        
        # Check daily steps
        if daily_steps < 5000:
            prediction = "Poor"

        # Check stress level
        if stress_level > 5:
            prediction = "Poor"

        # Check sleep disorder
        if sleep_disorder in ["Insomnia", "Sleep Apnea"]:
            prediction = "Poor"

        return jsonify(prediction=prediction)

    except Exception as e:
        return jsonify(error=f"An error occurred: {str(e)}"), 400

def preprocess_input(df):
    # Replace empty strings with NaN
    df.replace('', pd.NA, inplace=True)

    # Handle Blood Pressure
    if 'Blood Pressure' in df.columns:
        df[['Blood Pressure Systolic', 'Blood Pressure Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
        df['Blood Pressure Systolic'] = pd.to_numeric(df['Blood Pressure Systolic'], errors='coerce').fillna(0)
        df['Blood Pressure Diastolic'] = pd.to_numeric(df['Blood Pressure Diastolic'], errors='coerce').fillna(0)
        df.drop(columns=['Blood Pressure'], inplace=True)
    else:
        df['Blood Pressure Systolic'] = 0
        df['Blood Pressure Diastolic'] = 0

    # Convert categorical features using LabelEncoders
    for col in ['Gender', 'Occupation', 'BMI Category', 'Physical Activity Level', 'Sleep Disorder']:
        if col in df.columns:
            le = label_encoders.get(col, None)
            if le is not None:
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else 'Unknown')
                df[col] = le.transform(df[col])
            else:
                df[col] = 0  # Default value for unknown encoder
        else:
            df[col] = 0

    # Convert numerical columns to float if necessary
    if 'Stress Level' in df.columns:
        df['Stress Level'] = pd.to_numeric(df['Stress Level'], errors='coerce').fillna(0)
    else:
        df['Stress Level'] = 0

    # Ensure all expected columns are present
    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    return df

def validate_input(data):
    required_fields = model_features
    for field in required_fields:
        if field not in data:
            return False, f"Missing required field: {field}"
    return True, None

if __name__ == '__main__':
    app.run(debug=True)
