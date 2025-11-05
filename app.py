# ==========================================
# APLIKASI FLASK - PREDIKSI DIABETES
# Menggunakan model Random Forest
# ==========================================
from flask import Flask, render_template, request
import numpy as np
import joblib

# Inisialisasi Flask
app = Flask(__name__)

# Load model yang sudah dilatih
model = joblib.load("model/rf_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil input dari form HTML
        features = [float(x) for x in request.form.values()]
        data = np.array([features])
        
        # Prediksi dengan model
        prediction = model.predict(data)[0]
        
        # Tentukan hasil
        output = "Positif Diabetes" if prediction == 1 else "Negatif Diabetes"
        return render_template("result.html", result=output)
    
    except Exception as e:
        return render_template("result.html", result=f"Terjadi error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
