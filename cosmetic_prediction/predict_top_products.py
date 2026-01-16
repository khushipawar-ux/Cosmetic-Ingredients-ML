import pandas as pd
import joblib
from sqlalchemy import create_engine

engine = create_engine("sqlite:///cosmetic.db")
df = pd.read_sql("cosmetics", engine)

model, encoders = joblib.load("models/popularity_model.pkl")

categorical_cols = [
    "Brand", "Category", "Skin_Type",
    "Gender_Target", "Packaging_Type",
    "Country_of_Origin"
]

# Encode data
for col in categorical_cols:
    df[col] = encoders[col].transform(df[col])

X = df[
    categorical_cols +
    ["Price_USD", "Rating", "Number_of_Reviews"]
]

df["Predicted_Popularity"] = model.predict(X)

# Top used products
top_products = df.sort_values(
    by="Predicted_Popularity",
    ascending=False
).head(10)

print("\n🔥 TOP 10 MOST USED BEAUTY PRODUCTS 🔥\n")
print(top_products[[
    "Product_Name", "Brand", "Category",
    "Rating", "Number_of_Reviews",
    "Predicted_Popularity"
]])
