import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
df = pd.read_csv("data/onlinefraud.csv")

print("📌 Columns in dataset:", df.columns.tolist())

# Select only required columns
df = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'isFraud']]

# Encode categorical column
df['type'] = df['type'].astype('category').cat.codes

# Split into features and target
X = df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig']]
y = df['isFraud']

# Fix class imbalance
sm = SMOTE()
X, y = sm.fit_resample(X, y)

# Scale values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\n📍 Model Results:\n", classification_report(y_test, y_pred))

# Save trained model + scaler
joblib.dump(model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("\n🎉 Training Complete! Model saved successfully in /models/")
