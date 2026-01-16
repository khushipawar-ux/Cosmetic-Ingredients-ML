import pandas as pd
from sqlalchemy import create_engine

# Load CSV
df = pd.read_csv("most_used_beauty_cosmetics_products_extended.csv")

# Create SQLite DB
engine = create_engine("sqlite:///cosmetic.db")

# Store data
df.to_sql("cosmetics", engine, if_exists="replace", index=False)

print("✅ CSV data loaded into database")
