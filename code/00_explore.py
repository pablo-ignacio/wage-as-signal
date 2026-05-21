"""
Run from project root: python code/00_explore.py
"""

import sys
import io
import pandas as pd
from pathlib import Path

# Force UTF-8 output so non-ASCII characters in the data don't crash the print
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RAW = Path("data/raw")

FILES = {
    "individual/positions":  RAW / "revelio_academic_individual" / "academic_individual_position_academic.csv",
    "individual/users":      RAW / "revelio_academic_individual" / "academic_individual_user_academic.csv",
    "individual/education":  RAW / "revelio_academic_individual" / "academic_individual_user_education_academic.csv",
    "individual/skills":     RAW / "revelio_academic_individual" / "academic_individual_user_skill_academic.csv",
    "postings/indeed":       RAW / "revelio_academic_postings"   / "academic_postings_indeed_individual_academic.csv",
    "postings/linkedin":     RAW / "revelio_academic_postings"   / "academic_postings_linkedin_individual_academic.csv",
    "postings/unified":      RAW / "revelio_academic_postings"   / "academic_postings_unified_individual_academic.csv",
}

SEP = "=" * 70


def explore(name: str, path: Path) -> None:
    print(f"\n{SEP}")
    print(f"  {name}")
    print(f"  {path.name}")
    print(SEP)

    df = pd.read_csv(path, low_memory=False)

    print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n--- Columns & dtypes ---")
    for col, dtype in df.dtypes.items():
        n_missing = df[col].isna().sum()
        pct = n_missing / len(df) * 100
        missing_str = f"  ({pct:.1f}% missing)" if n_missing > 0 else ""
        print(f"  {col:<45} {str(dtype):<10}{missing_str}")

    # Date range for columns that look like dates
    date_cols = [c for c in df.columns if any(k in c.lower() for k in ("date", "start", "end", "year"))]
    if date_cols:
        print("\n--- Date / time ranges ---")
        for col in date_cols:
            try:
                parsed = pd.to_datetime(df[col], errors="coerce")
                print(f"  {col}: {parsed.min()} → {parsed.max()}")
            except Exception:
                pass

    print("\n--- Sample rows (5) ---")
    print(df.head(5).to_string(max_colwidth=40))


for name, path in FILES.items():
    explore(name, path)

print(f"\n{SEP}")
print("Done.")
