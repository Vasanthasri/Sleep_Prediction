import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv(r'D:\Sleep\Sleep_health_and_lifestyle_dataset.csv')

# Print the column names to check available columns
print("Columns in the dataset:", df.columns)

# Handle categorical features using LabelEncoder
label_encoders = {}
categorical_columns = ['Gender', 'Occupation', 'Physical Activity Level', 'BMI Category', 'Sleep Disorder']

for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = df[col].astype(str)  # Convert to string if necessary
        df[col] = df[col].fillna('Unknown')  # Handle missing values
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Convert 'Blood Pressure' from '130/85' to two separate columns
if 'Blood Pressure' in df.columns:
    df[['Blood Pressure Systolic', 'Blood Pressure Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
    df[['Blood Pressure Systolic', 'Blood Pressure Diastolic']] = df[['Blood Pressure Systolic', 'Blood Pressure Diastolic']].astype(float)

    # Drop the original 'Blood Pressure' column
    df = df.drop('Blood Pressure', axis=1)

# Features (input columns) - Exclude 'Person ID' and 'Quality of Sleep'
model_features = [
    'Gender', 'Age', 'Occupation', 'Sleep Duration', 
    'Physical Activity Level', 'Stress Level', 'BMI Category', 
    'Blood Pressure Systolic', 'Blood Pressure Diastolic', 'Heart Rate', 
    'Daily Steps', 'Sleep Disorder'
]

# Check if all features exist in the DataFrame
missing_features = [feature for feature in model_features if feature not in df.columns]
if missing_features:
    raise ValueError(f"Missing features in the dataset: {missing_features}")

X = df[model_features]
y = df['Quality of Sleep']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a machine learning model (RandomForestClassifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model (check the accuracy)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the trained model and label encoders to files using pickle
with open('sleep_health_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
with open('label_encoders.pkl', 'wb') as le_file:
    pickle.dump(label_encoders, le_file)

print("Model and label encoders saved successfully!")

# Test the saved model by loading it back and predicting on a sample
with open('sleep_health_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('label_encoders.pkl', 'rb') as le_file:
    loaded_encoders = pickle.load(le_file)

# Use a sample data point from the test set to make a prediction
sample_data = X_test.iloc[0].values.reshape(1, -1)  # Get the first sample from X_test
prediction = loaded_model.predict(sample_data)

print(f"Predicted Quality of Sleep for the sample data: {prediction[0]}")
