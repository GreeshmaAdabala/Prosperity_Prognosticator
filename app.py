from flask import Flask, render_template, request
import joblib

# Create Flask App
app = Flask(__name__)

# Load Saved ML Model
model = joblib.load("random_forest_model.pkl")


# ---------------- HOME PAGE ----------------
@app.route("/")
def home():
    return render_template("home.html")


# ---------------- PREDICTION FORM PAGE ----------------
@app.route("/predict_form")
def predict_form():
    return render_template("index.html")


# ---------------- PREDICTION LOGIC ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        inputs = [
            float(request.form['state']),
            float(request.form['funding']),
            float(request.form['rounds']),
            float(request.form['vc']),
            float(request.form['angel']),
            float(request.form['roundA']),
            float(request.form['roundB']),
            float(request.form['roundC']),
            float(request.form['roundD']),
            float(request.form['milestones']),
            float(request.form['relationships'])
        ]

        # Make prediction
        prediction = model.predict([inputs])[0]

        # Convert output to readable result
        if prediction == 1:
            result = "Startup Likely Acquired 🚀"
        else:
            result = "Startup Likely Closed ❌"

        return render_template("result.html", result=result)

    except Exception as e:
        return f"Error occurred: {e}"


# ---------------- RUN SERVER ----------------
if __name__ == "__main__":
    app.run(debug=True)