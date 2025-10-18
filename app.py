from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("models/saved_rf.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]  # input as list
    prediction = model.predict(np.array(data).reshape(1, -1))
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
