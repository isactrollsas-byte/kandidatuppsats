"""
Hämtar makroekonomiska data för Sverige (2010-2025)
Datakällor: SCB och Riksbanken SWEA API

Variabler:
  - BNP-tillväxt, volymförändring mot föregående år (%) — SCB, kvartal
  - KPI, förändring mot 12 mån tidigare (%)             — SCB, månad → kvartal
  - Arbetslöshet 16-64 år (%)                           — SCB, kvartal
  - Fasta bruttoinvesteringar, volymförändring (%)      — SCB, kvartal
  - Reporänta (%)                                       — Riksbanken, dag → kvartal
"""

import requests
import pandas as pd
import time

START_YEAR = 2010
END_YEAR   = 2025
OUTPUT_CSV = "data/macro_data_sweden.csv"
SCB_BASE   = "https://api.scb.se/OV0104/v1/doris/sv/ssd"
RIKS_BASE  = "https://api.riksbank.se/swea/v1"


def scb_post(table_path: str, query: dict) -> pd.DataFrame:
    url = f"{SCB_BASE}/{table_path}"
    r = requests.post(url, json=query, timeout=30)
    if r.status_code != 200:
        print(f"    HTTP {r.status_code}: {r.text[:200]}")
        r.raise_for_status()
    data = r.json()
    cols = [c["text"] for c in data["columns"]]
    rows = [row["key"] + row["values"] for row in data["data"]]
    return pd.DataFrame(rows, columns=cols)


# ══════════════════════════════════════════════════════════════════════════════
# BNP-tillväxt (kvartal, volymförändring mot samma kvartal föregående år)
# Tabell: NR/NR0103/NR0103S/NR0103ENS10SnabbStat
# EkoIndikator BNP10, ContentsCode NR0103A!
# ══════════════════════════════════════════════════════════════════════════════

def fetch_gdp() -> pd.DataFrame:
    print("Hämtar BNP-tillväxt (SCB)...")
    table   = "NR/NR0103/NR0103S/NR0103ENS10SnabbStat"
    periods = [f"{y}K{q}" for y in range(START_YEAR, END_YEAR + 1) for q in range(1, 5)]

    query = {
        "query": [
            {"code": "EkoIndikator", "selection": {"filter": "item", "values": ["BNP10"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["NR0103A!"]}},
            {"code": "Tid",          "selection": {"filter": "item", "values": periods}},
        ],
        "response": {"format": "json"}
    }
    try:
        df = scb_post(table, query)
        # Kolumner: EkoIndikator, tid, värde
        val_col = df.columns[-1]
        tid_col = df.columns[-2]
        df["bnp_tillvaxt_pct"] = pd.to_numeric(df[val_col], errors="coerce")
        df["period"] = df[tid_col].str.replace("K", "Q")
        df = df[["period", "bnp_tillvaxt_pct"]].dropna()
        print(f"  ✓ BNP: {len(df)} kvartal")
        return df
    except Exception as e:
        print(f"  ✗ BNP misslyckades: {e}")
        return pd.DataFrame(columns=["period", "bnp_tillvaxt_pct"])


# ══════════════════════════════════════════════════════════════════════════════
# KPI / Inflation (månad, förändring mot 12 mån tidigare → aggregeras kvartal)
# Tabell: PR/PR0101/PR0101A/KPI2020M
# ContentsCode 00000804 = årsförändring (%)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_cpi() -> pd.DataFrame:
    print("Hämtar KPI/Inflation (SCB)...")
    table  = "PR/PR0101/PR0101A/KPI2020M"
    months = [f"{y}M{str(m).zfill(2)}" for y in range(START_YEAR, END_YEAR + 1) for m in range(1, 13)]

    query = {
        "query": [
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["00000804"]}},
            {"code": "Tid",          "selection": {"filter": "item", "values": months}},
        ],
        "response": {"format": "json"}
    }
    try:
        df = scb_post(table, query)
        val_col = df.columns[-1]
        tid_col = df.columns[-2]
        df["inflation_pct"] = pd.to_numeric(df[val_col], errors="coerce")
        df["datum"]  = pd.to_datetime(df[tid_col], format="%YM%m", errors="coerce")
        df["period"] = df["datum"].dt.to_period("Q").astype(str)
        df = df.groupby("period")["inflation_pct"].mean().reset_index()
        print(f"  ✓ KPI: {len(df)} kvartal")
        return df
    except Exception as e:
        print(f"  ✗ KPI misslyckades: {e}")
        return pd.DataFrame(columns=["period", "inflation_pct"])


# ══════════════════════════════════════════════════════════════════════════════
# Arbetslöshet (kvartal, %)
# Tabell: AM/AM0401/AM0401A/AKURLBefK
# Arbetskraftstillh ALÖSP, TypData O_DATA, Kon 1+2, Alder tot16-64
# ══════════════════════════════════════════════════════════════════════════════

def fetch_unemployment() -> pd.DataFrame:
    print("Hämtar Arbetslöshet (SCB)...")
    table   = "AM/AM0401/AM0401A/AKURLBefK"
    periods = [f"{y}K{q}" for y in range(START_YEAR, END_YEAR + 1) for q in range(1, 5)]

    query = {
        "query": [
            {"code": "Arbetskraftstillh", "selection": {"filter": "item", "values": ["ALÖSP"]}},
            {"code": "TypData",           "selection": {"filter": "item", "values": ["O_DATA"]}},
            {"code": "Kon",               "selection": {"filter": "item", "values": ["1+2"]}},
            {"code": "Alder",             "selection": {"filter": "item", "values": ["tot16-64"]}},
            {"code": "Tid",               "selection": {"filter": "item", "values": periods}},
        ],
        "response": {"format": "json"}
    }
    try:
        df = scb_post(table, query)
        val_col = df.columns[-1]
        tid_col = df.columns[-2]
        df["arbetsloshet_pct"] = pd.to_numeric(df[val_col], errors="coerce")
        df["period"] = df[tid_col].str.replace("K", "Q")
        df = df[["period", "arbetsloshet_pct"]].dropna()
        print(f"  ✓ Arbetslöshet: {len(df)} kvartal")
        return df
    except Exception as e:
        print(f"  ✗ Arbetslöshet misslyckades: {e}")
        return pd.DataFrame(columns=["period", "arbetsloshet_pct"])


# ══════════════════════════════════════════════════════════════════════════════
# Fasta bruttoinvesteringar (kvartal, volymförändring %)
# Tabell: NR/NR0103/NR0103A/NR0103ENS2010T04Kv
# Typ 1 = totalt, ContentsCode NR0103U7 = volymförändring mot föregående år
# ══════════════════════════════════════════════════════════════════════════════

def fetch_investments() -> pd.DataFrame:
    print("Hämtar Bruttoinvesteringar (SCB)...")
    table   = "NR/NR0103/NR0103A/NR0103ENS2010T04Kv"
    periods = [f"{y}K{q}" for y in range(START_YEAR, END_YEAR + 1) for q in range(1, 5)]

    query = {
        "query": [
            {"code": "Typ",          "selection": {"filter": "item", "values": ["1"]}},
            {"code": "ContentsCode", "selection": {"filter": "item", "values": ["NR0103U7"]}},
            {"code": "Tid",          "selection": {"filter": "item", "values": periods}},
        ],
        "response": {"format": "json"}
    }
    try:
        df = scb_post(table, query)
        val_col = df.columns[-1]
        tid_col = df.columns[-2]
        df["investeringar_pct"] = pd.to_numeric(df[val_col], errors="coerce")
        df["period"] = df[tid_col].str.replace("K", "Q")
        df = df[["period", "investeringar_pct"]].dropna()
        print(f"  ✓ Investeringar: {len(df)} kvartal")
        return df
    except Exception as e:
        print(f"  ✗ Investeringar misslyckades: {e}")
        return pd.DataFrame(columns=["period", "investeringar_pct"])


# ══════════════════════════════════════════════════════════════════════════════
# Reporänta (dag → kvartalsmedeltal)
# Riksbanken SWEA API, serie SECBREPOEFF
# ══════════════════════════════════════════════════════════════════════════════

def fetch_repo_rate() -> pd.DataFrame:
    print("Hämtar Reporänta (Riksbanken)...")
    url = f"{RIKS_BASE}/Observations/SECBREPOEFF/{START_YEAR}-01-01/{END_YEAR}-12-31"
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
        records = [{"datum": obs["date"], "reporate_pct": float(obs["value"])}
                   for obs in data if obs.get("value") not in (None, "")]
        df = pd.DataFrame(records)
        df["datum"]  = pd.to_datetime(df["datum"])
        df["period"] = df["datum"].dt.to_period("Q").astype(str)
        df = df.groupby("period")["reporate_pct"].mean().reset_index()
        print(f"  ✓ Reporänta: {len(df)} kvartal")
        return df
    except Exception as e:
        print(f"  ✗ Reporänta misslyckades: {e}")
        return pd.DataFrame(columns=["period", "reporate_pct"])


# ══════════════════════════════════════════════════════════════════════════════
# Bygg master-dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_master_dataset() -> pd.DataFrame:
    print("\n" + "="*60)
    print("  HÄMTAR MAKROEKONOMISK DATA – SVERIGE 2010–2025")
    print("="*60 + "\n")

    df_gdp   = fetch_gdp();        time.sleep(0.3)
    df_cpi   = fetch_cpi();        time.sleep(0.3)
    df_unemp = fetch_unemployment(); time.sleep(0.3)
    df_inv   = fetch_investments(); time.sleep(0.3)
    df_repo  = fetch_repo_rate()

    all_quarters = pd.period_range(start=f"{START_YEAR}Q1", end=f"{END_YEAR}Q4", freq="Q").astype(str)
    master = pd.DataFrame({"period": all_quarters})

    for df, name in [
        (df_gdp,   "BNP-tillväxt"),
        (df_cpi,   "KPI"),
        (df_unemp, "Arbetslöshet"),
        (df_inv,   "Investeringar"),
        (df_repo,  "Reporänta"),
    ]:
        if not df.empty:
            master = master.merge(df.drop_duplicates("period"), on="period", how="left")
            print(f"  ✓ {name} inlagd")

    print(f"\nMaster dataset: {master.shape[0]} rader × {master.shape[1]} kolumner")
    print(f"Täckning:\n{master.notna().mean().mul(100).round(0).astype(int).to_string()}")
    return master


if __name__ == "__main__":
    df = build_master_dataset()
    df.to_csv(OUTPUT_CSV, index=False, sep=";", decimal=",")
    print(f"\nData sparad till: {OUTPUT_CSV}")
    print("\nFörhandsgranskning:")
    print(df.head(8).to_string())
