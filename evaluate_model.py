import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

print("1. Loading the dataset...")
df = pd.read_excel("Land prices of Colombo district-Sri Lanka.xlsx", sheet_name="Sheet1")
df = df.sort_values('Posted_Date_new').reset_index(drop=True)

# IQR Outlier Removal
Q1 = df['Price per Perch'].quantile(0.25)
Q3 = df['Price per Perch'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[(df['Price per Perch'] >= Q1 - 1.5 * IQR) & (df['Price per Perch'] <= Q3 + 1.5 * IQR)].copy()

# Feature preparation
drop_columns = ['Address_ID', 'Posted_Date_new', 'Mentioned Price(Rs)', 'Price_Scale', 'Price per Perch']
X = df_clean.drop(columns=drop_columns)
y = df_clean['Price per Perch']

cat_features = ['Address', 'Land_type']
num_features = [col for col in X.columns if col not in cat_features]

n = len(df_clean)
train_idx = int(n * 0.70)
val_idx = int(n * 0.85)

X_train_raw, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
X_test_raw, y_test = X.iloc[val_idx:], y.iloc[val_idx:]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ]
)

X_train = preprocessor.fit_transform(X_train_raw)
X_test = preprocessor.transform(X_test_raw)

# Model Loading
print("2. The trained model is being loaded...")
model = XGBRegressor()
model.load_model("colombo_land_valuation_xgb.json")

# Predict on Testing Set 
y_pred = model.predict(X_test)

# Metrics checking
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\n" + "="*40)
print("     MODEL PERFORMANCE & ACCURACY REPORT     ")
print("="*40)
print(f"🔹 R2 Score (Determination): {r2:.4f} ")
print(f"🔹 RMSE (Root Mean Squared Error): LKR {rmse:,.2f}")
print(f"🔹 MAE (Mean Absolute Error): LKR {mae:,.2f}")
print("="*40)

# 5-Fold Cross Validation Testing
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=kf, scoring='r2')
print(f"🔹 5-Fold Cross-Validation Mean R2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print("="*40)