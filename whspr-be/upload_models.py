"""
Upload trained .pkl model files to Supabase Storage bucket 'models'.
Run once: python upload_models.py
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from supabase import create_client

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

if not url or not key:
    raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env first.")

client = create_client(url, key)

MODELS_DIR = Path(__file__).parent / "models"
FILES = ["svm_emotion_model.pkl", "knn_emotion_model.pkl", "rf_emotion_model.pkl"]

for name in FILES:
    path = MODELS_DIR / name
    if not path.exists():
        print(f"⚠️  {name} not found locally, skipping.")
        continue
    data = path.read_bytes()
    res = client.storage.from_("models").upload(
        name, data, {"content-type": "application/octet-stream", "upsert": "true"}
    )
    print(f"✅  Uploaded {name}")

print("Done.")
