import pandas as pd
import joblib
import os
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# ---------- CONFIGURATION ----------
DB_PATH = "sqlite:///cosmetic.db"
CSV_PATH = "most_used_beauty_cosmetics_products_extended.csv"
MODEL_PATH = "models/popularity_model.pkl"
OUTPUT_PATH = "output/top_predicted_products.csv"

# Ensure directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("output", exist_ok=True)

def load_data():
    """Load CSV and store in SQLite database."""
    print("📂 Loading data...")
    if not os.path.exists(CSV_PATH):
        print(f"❌ Error: {CSV_PATH} not found!")
        return False
    
    df = pd.read_csv(CSV_PATH)
    engine = create_engine(DB_PATH)
    df.to_sql("cosmetics", engine, if_exists="replace", index=False)
    print("✅ CSV data loaded into database.")
    return True

def train_popularity_model():
    """Train the RandomForest model for product popularity."""
    print("🤖 Training model...")
    engine = create_engine(DB_PATH)
    df = pd.read_sql("cosmetics", engine)

    # Map usage frequency
    usage_map = {
        "Daily": 3,
        "Weekly": 2,
        "Monthly": 1.5,
        "Occasional": 1
    }
    df["Usage_Score"] = df["Usage_Frequency"].map(usage_map)
    
    # Clean data
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
    X = df[categorical_cols + ["Price_USD", "Rating", "Number_of_Reviews"]]
    y = (df["Usage_Score"] * 0.5 + df["Rating"] * 0.3 + df["Number_of_Reviews"] * 0.2)

    # Train model
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)

    # Save model and encoders
    joblib.dump((model, encoders), MODEL_PATH)
    print(f"✅ Model saved to {MODEL_PATH}")

def predict_and_save():
    """Predict popularity for all products and save top 10 to a file."""
    print("📈 Generating predictions...")
    engine = create_engine(DB_PATH)
    df = pd.read_sql("cosmetics", engine)

    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found. Train the model first.")
        return

    model, encoders = joblib.load(MODEL_PATH)
    categorical_cols = list(encoders.keys())

    # Encode data
    df_encoded = df.copy()
    for col in categorical_cols:
        # Handle unseen labels by mapping them to the most frequent or a default if necessary
        # But here we are using the same data for simplicity
        df_encoded[col] = encoders[col].transform(df[col])

    X = df_encoded[categorical_cols + ["Price_USD", "Rating", "Number_of_Reviews"]]
    df["Predicted_Popularity"] = model.predict(X)

    # Get Top 10
    top_products = df.sort_values(by="Predicted_Popularity", ascending=False).head(10)

    # Save to file
    top_products.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Top 10 products saved to {OUTPUT_PATH}")

    # Print results summary
    print("\n🔥 TOP 10 PREDICTED PRODUCTS 🔥")
    print(top_products[["Product_Name", "Brand", "Category", "Predicted_Popularity"]])

if __name__ == "__main__":
    if load_data():
        train_popularity_model()
        predict_and_save()
        print("\n🚀 Pipeline finished successfully!")
