"""Upload existing CSV data to Supabase PostgreSQL."""
import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import sys

# ---------- Connection Settings ----------
DB_HOST = "db.jofomorypkzedpzcemmr.supabase.co"
DB_PORT = "5432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASS = "Password12345678910"   # alphanumeric only, no special chars

DATABASE_URL = f"postgresql://{DB_USER}:{quote_plus(DB_PASS)}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print(f"Connecting to: {DB_HOST}:{DB_PORT}/{DB_NAME}")
print(f"User: {DB_USER}")
print(f"Password length: {len(DB_PASS)} chars")

try:
    engine = create_engine(DATABASE_URL, connect_args={"connect_timeout": 15})
    with engine.connect() as conn:
        result = conn.execute(text("SELECT version()"))
        version = result.fetchone()[0]
        print(f"\n✅ Connected successfully")
        print(f"   PostgreSQL version: {version[:60]}...")
except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    print("\nTroubleshooting:")
    print("1. Verify password is exactly: Password12345678910")
    print("2. Verify Supabase project is not paused (check dashboard)")
    print("3. Try the Connection Pooler endpoint if IPv6 is blocked")
    sys.exit(1)

# ---------- Clear existing data ----------
print("\nClearing existing data from 'predictions' table...")
try:
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE predictions RESTART IDENTITY"))
        conn.commit()
    print("   Cleared.")
except Exception as e:
    print(f"   ⚠️  Could not truncate: {e}")
    print("   (Table might not exist yet — create it in Supabase SQL Editor first)")

# ---------- Upload ----------
for name in ['NIFTY50', 'SP500']:
    print(f"\nUploading {name}...")
    base = pd.read_csv(f'reports/baseline_{name}.csv')
    cust = pd.read_csv(f'reports/custom_{name}.csv')

    merged = base.merge(cust, on='Date', suffixes=('_base', '_cust'))
    merged['ticker'] = name
    merged['strategy_return'] = (merged['Pred_cust'] > 0).astype(int) * merged['Actual_cust']

    final_df = pd.DataFrame({
        'date': pd.to_datetime(merged['Date']).dt.date,
        'ticker': name,
        'actual_return': merged['Actual_cust'].astype(float),
        'predicted_return': merged['Pred_cust'].astype(float),
        'strategy_return': merged['strategy_return'].astype(float),
    })

    try:
        final_df.to_sql('predictions', engine, if_exists='append', index=False)
        print(f"   ✅ Uploaded {len(final_df)} rows")
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")
        print(f"   Make sure the 'predictions' table exists in Supabase.")
        sys.exit(1)

# ---------- Verify ----------
print("\nVerifying...")
with engine.connect() as conn:
    result = conn.execute(text(
        "SELECT ticker, COUNT(*) FROM predictions GROUP BY ticker ORDER BY ticker"
    ))
    for row in result:
        print(f"   {row[0]}: {row[1]} rows")

print("\n✅ Upload complete.")