import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

engine = create_engine("sqlite:///cosmetic.db")
df = pd.read_sql("cosmetics", engine)

# Map usage frequency
usage_map = {
    "Daily": 3,
    "Weekly": 2,
    "Monthly": 1.5,
    "Occasional": 1
}
df["Usage_Score"] = df["Usage_Frequency"].map(usage_map)

# Drop rows with NaN if any (safety measure)
df = df.dropna(subset=["Usage_Score", "Rating", "Number_of_Reviews"])

# Encode categorical columns
categorical_cols = [
    "Brand", "Category", "Skin_Type",
    "Gender_Target", "Packaging_Type",
    "Country_of_Origin"
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Features & target
X = df[
    categorical_cols +
    ["Price_USD", "Rating", "Number_of_Reviews"]
]

y = (
    df["Usage_Score"] * 0.5 +
    df["Rating"] * 0.3 +
    df["Number_of_Reviews"] * 0.2
)

# Train model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X, y)

joblib.dump((model, encoders), "models/popularity_model.pkl")

print("✅ Popularity prediction model trained")
