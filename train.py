import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

print("Starting to build the Customer Churn Prediction Models...")

# 1. Load Data
print("Loading data...")
df = pd.read_csv('data/Telco-Customer-Churn.csv')

# 2. Data Cleaning
print("Cleaning data...")
# Convert TotalCharges to numeric, coerce errors to NaN
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
# Drop rows with NaN in TotalCharges (only ~11 rows)
df.dropna(subset=['TotalCharges'], inplace=True)

# Drop CustomerID as it has no predictive power
df.drop('customerID', axis=1, inplace=True)

# Separate Target mapping (Churn: Yes/No to 1/0)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
X = df.drop('Churn', axis=1)
y = df['Churn']

# 3. Feature Engineering Setup
print("Setting up Feature Engineering pipelines...")
# Define numerical and categorical columns
numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                        'PhoneService', 'MultipleLines', 'InternetService', 
                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'Contract', 'PaperlessBilling', 'PaymentMethod']

# Create Preprocessor using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
    ])

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. Build Pipeline with SMOTE and Classifier
print("\nTraining models and evaluating...")
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

best_model = None
best_acc = 0
best_model_name = ""

for name, model in models.items():
    # We use ImbPipeline to integrate SMOTE into the training process securely
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', model)
    ])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f" -> {name} Accuracy: {acc:.4f}")
    if acc > best_acc:
        best_acc = acc
        best_model = pipeline
        best_model_name = name

print(f"\n✅ Best Model Selected: {best_model_name} with Accuracy Score: {best_acc:.4f}")

# Save the best model and pipeline to disk
print("\nSaving the model to model.pkl...")
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("🎯 Model saved successfully! You can now run the Streamlit app.")
