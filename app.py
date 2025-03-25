from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form inputs
        inputs = [float(request.form.get(field)) for field in [
            "age", "experience", "income", "family", "ccavg", "education",
            "mortgage", "securities_account", "cd_account", "online", "credit_card"
        ]]

        # Preprocess input
        inputs = np.array(inputs).reshape(1, -1)
        inputs = scaler.transform(inputs)

        # Predict
        probability = model.predict_proba(inputs)[0][1]  # Loan approval probability
        prediction = model.predict(inputs)[0]  # 1 = Approved, 0 = Not Approved

        # Risk level logic
        if probability > 0.7:
            risk_level = "Low Risk"
        elif probability > 0.4:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        return render_template("result.html", prediction=prediction, probability=f"{probability:.2%}", risk_level=risk_level)

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
