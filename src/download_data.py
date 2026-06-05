import os
import kaggle
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────
RAW = Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Drug Review Sentiment — Data Download")
print("=" * 60)

# ── Kaggle Dataset ─────────────────────────────────────────
print("\n[1/1] Downloading UCI Drug Review Dataset...")
print("      Dataset: jessicali9530/kuc-hackathon-winter-2018")

try:
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "jessicali9530/kuc-hackathon-winter-2018",
        path=str(RAW),
        unzip=True,
        quiet=False,
    )
    print("\n✓ Download complete!")

except Exception as e:
    print(f"\n✗ Kaggle error: {e}")
    print("\nManual download:")
    print("1. https://www.kaggle.com/datasets/jessicali9530/kuc-hackathon-winter-2018")
    print("2. Download karke data/raw/ mein rakh do")

# ── Verify ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Files in data/raw/:")
for f in sorted(RAW.glob("*")):
    size = f.stat().st_size / (1024*1024)
    print(f"  {f.name:<45} {size:.2f} MB")
print("=" * 60)
print("\nNext → notebooks/01_eda.ipynb")