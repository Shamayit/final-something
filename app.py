from flask import Flask, render_template, request
import numpy as np
import pickle

# -----------------------------
# Load Model + Scaler
# -----------------------------
model = pickle.load(open("heard_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

app = Flask(__name__)

# -----------------------------
# Lifestyle Recommendation
# -----------------------------
def life_style(pred):
    if pred == 1:
        return {
            "status": "Heart Disease Detected",
            "morning": [
                "Drink warm water with lemon.",
                "10 minutes slow walking.",
                "5 minutes deep breathing.",
                "Light breakfast: oats, fruits, nuts."
            ],
            "diet": [
                "Low salt food.",
                "Avoid red meat and fried items.",
                "Eat green leafy vegetables daily.",
                "Include almonds, walnuts, oats."
            ],
            "exercise": [
                "Walk 30 minutes twice a day.",
                "Avoid heavy weight lifting.",
                "Light yoga & stretching for 15 mins."
            ],
            "sleep": [
                "Sleep 7–8 hours.",
                "Avoid screens 1 hour before bedtime.",
                "Maintain consistent sleep schedule."
            ]
        }
    else:
        return {
            "status": "No Heart Disease",
            "morning": [
                "Drink 1 glass warm water.",
                "20 minutes normal walk or jogging.",
                "Healthy breakfast with protein."
            ],
            "diet": [
                "Balanced diet.",
                "Limit sugar and oily foods.",
                "Eat fruits daily."
            ],
            "exercise": [
                "Jogging 20–30 mins.",
                "Yoga or stretching.",
                "Weekend cycling/swimming."
            ],
            "sleep": [
                "Sleep 7 hours.",
                "Avoid heavy food at night.",
                "Limit caffeine after 7 PM."
            ]
        }

# -----------------------------
# Home Route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form.get("age")),
            int(request.form.get("sex")),
            int(request.form.get("cp")),
            float(request.form.get("trestbps")),
            float(request.form.get("chol")),
            int(request.form.get("fbs")),
            int(request.form.get("restecg")),
            float(request.form.get("thalach")),
            int(request.form.get("exang")),
            float(request.form.get("oldpeak")),
            int(request.form.get("slope")),
            int(request.form.get("ca")),
            int(request.form.get("thal"))
        ]
    except:
        return "Error: Missing or invalid input values"

    # Convert to array
    arr = np.array([features])

    # Scale
    arr_scaled = scaler.transform(arr)

    # Predict probability
    prob = model.predict(arr_scaled)[0][0]
    pred = 1 if prob > 0.5 else 0

    # Lifestyle plan
    plan = life_style(pred)

    # Send to result page
    return render_template(
        "result.html",
        prediction=pred,
        prob=prob,
        plan=plan
    )

# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
