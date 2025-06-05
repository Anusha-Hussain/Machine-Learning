# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# 2. Load Dataset
df = pd.read_csv(r'C:\Users\Hp\Downloads\heart.csv')  # Use raw string for Windows path
print(df.head())

# 3. Explore Dataset
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 4. Visualize Correlation
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# 5. Prepare Data
X = df.drop('target', axis=1)   # Features
y = df['target']                # Target

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 9. Evaluate Model
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 10. Make a Prediction on New Data
# Example new patient data: [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
new_patient = np.array([[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]])
new_patient_scaled = scaler.transform(new_patient)
prediction = model.predict(new_patient_scaled)

if prediction[0] == 1:
    print("\n🔴 Likely to have heart disease.")
else:
    print("\n🟢 Unlikely to have heart disease.")

# 11. Save Model and Scaler
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n✅ Model and Scaler saved successfully!")
