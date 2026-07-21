from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ---------------------------------------------------
# Load Model and Preprocessor
# ---------------------------------------------------

MODEL_PATH = os.path.join("final_model", "model.pkl")
PREPROCESSOR_PATH = os.path.join("final_model", "preprocessor.pkl")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Numerical columns
NUMERICAL_COLUMNS = [
    "months_as_customer",
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "capital_gains",
    "capital_loss",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "auto_year",
    "policy_bind_year",
    "policy_bind_month",
    "policy_bind_day",
    "incident_month",
    "incident_day"
]

# ---------------------------------------------------
# Home Page
# ---------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = {}

        for key, value in request.form.items():

            if key in NUMERICAL_COLUMNS:

                if value.strip() == "":
                    data[key] = 0

                elif "." in value:
                    data[key] = float(value)

                else:
                    data[key] = int(value)

            else:
                data[key] = value

        input_df = pd.DataFrame([data])

        transformed = preprocessor.transform(input_df)

        prediction = model.predict(transformed)[0]

        prediction_probability = model.predict_proba(transformed)[0]

        if prediction == 1:
            result = "🚨 Fraud Claim Detected"
            confidence = prediction_probability[1]
        else:
            result = "✅ Genuine Claim"
            confidence = prediction_probability[0]

        return render_template(
            "index.html",
            prediction_text=result,
            confidence=f"{confidence*100:.2f}%"
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text="Prediction Failed",
            confidence=str(e)
        )


# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)