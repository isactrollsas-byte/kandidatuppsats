"""
Slår ihop alla datakällor till ett master-dataset för PCA-analys
=================================================================
Indata:
  - macro_data_sweden.csv   (fetch_macro_data.py)
  - sector_proxy_data.csv   (fetch_sector_data.py)
  - sales_data.csv          (Nordea-data, läggs till manuellt)

Utdata:
  - master_dataset.csv      (komplett, rådata)
  - master_clean.csv        (interpolerat, redo för PCA)
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUTPUT_RAW   = "data/master_dataset.csv"
OUTPUT_CLEAN = "data/master_clean.csv"

# ══════════════════════════════════════════════════════════════════════════════
# Ladda filer
# ══════════════════════════════════════════════════════════════════════════════

def load_csv(path: str) -> pd.DataFrame | None:
    if not Path(path).exists():
        print(f"  ! {path} saknas – hoppas över")
        return None
    df = pd.read_csv(path, sep=";", decimal=",")
    print(f"  ✓ {path}: {df.shape[0]} rader, kolumner: {df.columns.tolist()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Rensa och förbered
# ══════════════════════════════════════════════════════════════════════════════

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolerar saknade värden och loggar täckning per kolumn.
    """
    numeric_cols = [c for c in df.columns if c != "period"]

    print("\nTäckning per variabel (% icke-saknade):")
    for col in numeric_cols:
        pct = df[col].notna().mean() * 100
        print(f"  {col:<30} {pct:.0f}%")

    # Linjär interpolation (fungerar bra för tidsserie)
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")

    # Om fortfarande saknas (t.ex. kolumn helt tom) → median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    remaining = df[numeric_cols].isnull().sum().sum()
    if remaining > 0:
        print(f"\n  ! {remaining} saknade värden kvar efter interpolation")
    else:
        print("\n  ✓ Inga saknade värden efter interpolation")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Huvudfunktion
# ══════════════════════════════════════════════════════════════════════════════

def merge_all() -> pd.DataFrame:
    print("="*60)
    print("  SLÅR IHOP DATAKÄLLOR")
    print("="*60 + "\n")

    # Ladda
    print("Laddar filer:")
    macro  = load_csv("data/macro_data_sweden.csv")
    sector = load_csv("data/sector_proxy_data.csv")
    sales  = load_csv("data/sales_data.csv")

    # Bygg period-index 2010Q1–2025Q4
    all_quarters = pd.period_range(start="2010Q1", end="2025Q4", freq="Q").astype(str)
    master = pd.DataFrame({"period": all_quarters})

    # Slå ihop
    print("\nSlår ihop:")
    for df, label in [(macro, "Makrodata"), (sector, "Sektordata"), (sales, "Försäljningsdata")]:
        if df is not None and "period" in df.columns:
            merge_cols = df.columns.tolist()
            master = master.merge(df[merge_cols].drop_duplicates("period"), on="period", how="left")
            added = [c for c in merge_cols if c != "period"]
            print(f"  ✓ {label}: {added}")

    # Spara rådata
    master.to_csv(OUTPUT_RAW, index=False, sep=";", decimal=",")
    print(f"\nRådata sparad: {OUTPUT_RAW} ({master.shape[0]}×{master.shape[1]})")

    # Rensa
    print("\nInterpolerar saknade värden:")
    clean = clean_dataset(master.copy())
    clean.to_csv(OUTPUT_CLEAN, index=False, sep=";", decimal=",")
    print(f"\nRent dataset sparat: {OUTPUT_CLEAN}")

    # Sammanfattning
    print("\n" + "─"*60)
    print("VARIABLER I MASTER-DATASET:")
    print("─"*60)
    groups = {
        "Makroekonomiska": ["bnp_tillvaxt_pct", "inflation_pct", "arbetsloshet_pct",
                            "investeringar_mnkr", "reporate_pct", "stibor3m_pct"],
        "Sektorproxies":   ["provision_netto_mnkr", "foretag_inlaning_mnkr", "ecb_betalningar"],
        "Beroende variabel": ["sales"],
    }
    for group, cols in groups.items():
        present = [c for c in cols if c in master.columns]
        missing = [c for c in cols if c not in master.columns]
        if present:
            print(f"\n  {group}:")
            for c in present:
                pct = master[c].notna().mean() * 100
                print(f"    {'✓' if pct > 50 else '!'} {c} ({pct:.0f}% täckning)")
        if missing:
            print(f"    Saknas: {missing}")

    return master


if __name__ == "__main__":
    df = merge_all()
    print(f"\nFörhandsgranskning:")
    print(df.head(8).to_string())
