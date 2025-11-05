# ==========================================
# RANDOM FOREST CLASSIFICATION - DIABETES
# Versi Stabil untuk VS Code & Jupyter
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-GUI agar aman di Linux/VS Code
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import joblib

# Buat folder output jika belum ada
os.makedirs("output", exist_ok=True)
os.makedirs("model", exist_ok=True)

print("=== Memuat Dataset ===")
df = pd.read_csv("diabetes.csv")
print(df.head())

# ==========================================
# 1️⃣ Data Preprocessing
# ==========================================
cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# ==========================================
# 2️⃣ Split Data
# ==========================================
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 3️⃣ Training Random Forest
# ==========================================
print("\n=== Melatih Model Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ==========================================
# 4️⃣ Evaluasi Model
# ==========================================
y_pred = rf.predict(X_test)

print("\n=== Hasil Training & Testing ===")
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)
print(f"Akurasi Training : {train_acc:.2f}")
print(f"Akurasi Testing  : {test_acc:.2f}")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("output/confusion_matrix.png")
print("✅ Confusion Matrix disimpan di: output/confusion_matrix.png")

# ==========================================
# 5️⃣ ROC Curve dan AUC
# ==========================================
y_prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}", color='blue')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("output/roc_curve.png")

print(f"✅ ROC Curve disimpan di: output/roc_curve.png")
print(f"✅ Nilai ROC AUC Score: {auc:.3f}")

# ==========================================
# 6️⃣ Simpan Model
# ==========================================
joblib.dump(rf, "model/rf_model.pkl")
print("✅ Model tersimpan di: model/rf_model.pkl")

print("\n=== SELESAI ===")
print("Semua hasil evaluasi dan grafik disimpan di folder 'output/'")
