import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd

df = pd.read_excel(r'C:\Users\naray\OneDrive\Pictures\Desktop\01-College_Stuff\Loan_Approval\Personal Bank Loan Classification.xlsx', sheet_name='Data')
df.head(5)
print(df) 

# Display the cleaned dataframe
print(df.head())
print("Updated Column Names:", df.columns.tolist())

# Drop unnecessary columns
df = df.drop(columns=["ID", "ZIP Code"])  # Remove non-relevant features

# Define features and target
X = df.drop(columns=["Personal Loan"])  # Features
y = df["Personal Loan"]  # Target variable

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model and scaler
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully!")
