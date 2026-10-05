"""
Module 3 - Basic web application (Flask)  (CivicFix AI, Review 2)

Run:   python app.py      then open http://127.0.0.1:5000
Routes:  GET /          -> complaint form
         POST /predict  -> JSON {"complaint": "..."}  returns the predicted category
"""
import joblib
from flask import Flask, jsonify, render_template, request
from predict import predict

app = Flask(__name__)
model = joblib.load("model.joblib")          # run train_category.py first


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.get_json(silent=True) or {}
    text = str(data.get("complaint", "")).strip()
    if len(text) < 3:
        return jsonify({"error": "Please enter a complaint (at least 3 characters)."}), 400
    return jsonify(predict(text[:600], model))


if __name__ == "__main__":
    app.run(debug=True)
