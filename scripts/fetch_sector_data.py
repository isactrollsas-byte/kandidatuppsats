"""
Hämtar sektordata som PROXY för cash management-efterfrågan i Sverige
======================================================================
Exakt "cash management"-statistik rapporteras inte separat av myndigheter.
Dessa tre proxies triangulerar efterfrågan på tjänsterna:

  PROXY 1 – SCB: Nettoprovisioner från banker (avgiftsintäkter minus avgiftskostnader)
            Tabell: FM/FM0402/FinResAr
            Kontopost: E00133 (B5 Provisionsintäkter) - E00134 (B6 Provisionskostnader)
            Frekvens: År (1996–2024) → linjärt interpolerat till kvartal
            Källa: SCB

  PROXY 2 – ECB SDMX: Betalningsvolymer Sverige (kredittransferingar)
            Dataset: PSS (Payment and Securities Settlement Statistics)
            Frekvens: År | Källa: Europeiska centralbanken

  PROXY 3 – SCB: MFI-inlåning från icke-finansiella företag (S11)
            Tabell: FM/FM0401/FM0401X/MFIM1
            Kontopost: 201014 (In-/upplåning, Sv, Icke-finansiella företag)
            Frekvens: Månad → aggregeras till kvartal
            Källa: SCB

Alla proxies sparas i: sector_proxy_data.csv
"""

import requests
import pandas as pd
import numpy as np
import time
from io import StringIO

START_YEAR = 2010
END_YEAR   = 2025
OUTPUT_CSV = "sector_proxy_data.csv"
SCB_BASE   = "https://api.scb.se/OV0104/v1/doris/sv/ssd"
ECB_BASE   = "https://data-api.ecb.europa.eu/service/data"


# ══════════════════════════════════════════════════════════════════════════════
# PROXY 1 – SCB: Bankernas provisionsintäkter netto (B5 minus B6)
# Tabell: FM/FM0402/FinResAr  |  Årsdata → interpoleras till kvartal
# ══════════════════════════════════════════════════════════════════════════════

def fetch_scb_bank_commissions() -> pd.DataFrame:
    """
    Hämtar provisionsintäkter (B5) och provisionskostnader (B6) för banker,
    beräknar netto, och interpolerar årsdata linjärt till kvartalsfrekvens.

    Tabell  : FM/FM0402/FinResAr
    Institut: S212  = banker, totalt
    Konton  : E00133 (B5 Prov.intäkter) och E00134 (B6 Prov.kostnader)
    Innehåll: FM0402B1 (mnkr)
    """
    print("PROXY 1 – SCB: Bankernas provisionsintäkter (netto)...")
    table = "FM/FM0402/FinResAr"
    # FM0402/FinResAr publiceras med ~1 års fördröjning – max tillgängligt är 2024
    years = [str(y) for y in range(START_YEAR, min(END_YEAR, 2024) + 1)]

    results = {}
    for kod, label in [("E00133", "prov_intakter"), ("E00134", "prov_kostnader")]:
        query = {
            "query": [
                {
                    "code": "Finansinstitut",
                    "selection": {"filter": "item", "values": ["S212"]}   # banker, totalt
                },
                {
                    "code": "Kontopost",
                    "selection": {"filter": "item", "values": [kod]}
                },
                {
                    "code": "ContentsCode",
                    "selection": {"filter": "item", "values": ["FM0402B1"]}
                },
                {
                    "code": "Tid",
                    "selection": {"filter": "item", "values": years}
                }
            ],
            "response": {"format": "json"}
        }
        url = f"{SCB_BASE}/{table}"
        try:
            r = requests.post(url, json=query, timeout=30)
            r.raise_for_status()
            data = r.json()
            rows = [row["key"] + row["values"] for row in data["data"]]
            # Svaret innehåller: finansinstitut, kontopost, år, värde (4 kol, ej ContentsCode)
            df = pd.DataFrame(rows, columns=["finansinstitut", "kontopost", "tid", "varde"])
            df["varde"] = pd.to_numeric(df["varde"], errors="coerce")
            df["ar"] = df["tid"].astype(int)
            results[label] = df[["ar", "varde"]].rename(columns={"varde": label})
            print(f"  ✓ {label}: {len(df)} år hämtade")
        except requests.HTTPError as e:
            print(f"  ✗ HTTP-fel {e.response.status_code} för {kod}: {e.response.text[:200]}")
        except Exception as e:
            print(f"  ✗ Misslyckades för {kod}: {e}")

    if len(results) < 2:
        print("  → Returnerar tom DataFrame (kunde ej hämta båda kontoposterna)")
        return pd.DataFrame(columns=["period", "provision_netto_mnkr"])

    # Beräkna netto = intäkter − kostnader
    df_netto = results["prov_intakter"].merge(results["prov_kostnader"], on="ar", how="inner")
    df_netto["provision_netto_mnkr"] = df_netto["prov_intakter"] - df_netto["prov_kostnader"]

    # ── Interpolera årsdata → kvartal ────────────────────────────────────────
    # Skapa ett kvartalsindex och linjärinterpolera mellan årsmittpunkter (Q2)
    df_netto = df_netto[["ar", "provision_netto_mnkr"]].dropna().sort_values("ar")

    quarter_rows = []
    for _, row in df_netto.iterrows():
        for q in range(1, 5):
            quarter_rows.append({"period": f"{int(row['ar'])}Q{q}",
                                  "provision_netto_mnkr": row["provision_netto_mnkr"]})
    df_q = pd.DataFrame(quarter_rows)

    # Linjär interpolering inom varje år (fördela jämnt Q1–Q4)
    all_q = pd.period_range(f"{START_YEAR}Q1", f"{END_YEAR}Q4", freq="Q").astype(str)
    df_q = (pd.DataFrame({"period": all_q})
              .merge(df_q.drop_duplicates("period"), on="period", how="left"))
    df_q["provision_netto_mnkr"] = (df_q["provision_netto_mnkr"]
                                     .interpolate(method="linear", limit_direction="both"))

    print(f"  ✓ Interpolerat till {len(df_q)} kvartal (årsdata × 4, linjär interpolering)")
    print("  ⚠  OBS: Årsdata upprepas per kvartal – notera detta i metodavsnittet")
    return df_q[["period", "provision_netto_mnkr"]]


# ══════════════════════════════════════════════════════════════════════════════
# PROXY 2 – ECB SDMX: Betalningsvolymer Sverige (kredittransferingar)
# Dataset: PSS  |  Årsdata
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ecb_payments_sweden() -> pd.DataFrame:
    """
    Hämtar svenska betalningsstatistik från ECB SDMX 2.1 REST API.
    Dataset: PSS (Payment and Securities Settlement Statistics)

    ECB PSS har 10 dimensioner (årsdata):
      FREQ.REF_AREA.PSS_INFO_TYPE.PSS_INSTRUMENT.PSS_SYSTEM.
      COUNT_AREA.COUNT_SECTOR.CURRENCY_TRANS.SERIES_DENOM.DATA_TYPE_PSS

    Provade nycklar för Sverige kredittransferingar:
      A.SE.T0201.Z0.P21.Z0.Z0.Z0.Z0.NR  (antal, miljoner)
      A.SE.T0201.Z0.P21.Z0.Z0.Z0.Z0.WT  (värde, MEUR)
    """
    print("PROXY 2 – ECB SDMX: Betalningsvolymer Sverige...")

    # ECB PSS – 10 dimensioner (verifierade via live API-sökning 2025):
    #   FREQ.REF_AREA.PSS_INFO_TYPE.PSS_INSTRUMENT.PSS_SYSTEM.
    #   DATA_TYPE_PSS.COUNT_AREA.COUNT_SECTOR.CURRENCY_TRANS.SERIES_DENOM
    # F000 = alla betalningssystem, I31 = kreditöverföringar
    # NT = antal transaktioner (milj), VT = totalt värde (MEUR, .N = nominellt)
    ecb_series = {
        "ct_antal_milj": "A.SE.F000.I31.Z00Z.NT.X0.20.Z0Z.Z",
        "ct_varde_meur":  "A.SE.F000.I31.Z00Z.VT.X0.20.Z01.N",
    }

    results = {}
    for col, key in ecb_series.items():
        url = (
            f"{ECB_BASE}/PSS/{key}"
            f"?startPeriod={START_YEAR}&endPeriod={END_YEAR}"
            f"&format=csvdata&detail=dataonly"
        )
        try:
            r = requests.get(url, timeout=60)  # Inget Accept-huvud – låt ECB välja format
            if r.status_code == 200 and len(r.text) > 100:
                df_tmp = pd.read_csv(StringIO(r.text))
                if "TIME_PERIOD" in df_tmp.columns and "OBS_VALUE" in df_tmp.columns:
                    df_tmp = df_tmp[["TIME_PERIOD", "OBS_VALUE"]].rename(
                        columns={"TIME_PERIOD": "ar", "OBS_VALUE": col}
                    )
                    df_tmp[col] = pd.to_numeric(df_tmp[col], errors="coerce")
                    results[col] = df_tmp
                    print(f"  ✓ {col}: {len(df_tmp)} år hämtade")
                else:
                    print(f"  ✗ {col}: Oväntade kolumner – {df_tmp.columns.tolist()}")
            else:
                print(f"  ✗ {col}: HTTP {r.status_code} – kontrollera serie-nyckeln manuellt på")
                print(f"       https://data.ecb.europa.eu/data/datasets/PSS")
        except Exception as e:
            print(f"  ✗ {col}: {e}")
        time.sleep(0.5)

    if not results:
        print("  → ECB-data ej tillgänglig. Returnerar tom DataFrame.")
        print("  → Alternativ: ladda ned manuellt från https://data.ecb.europa.eu/data/datasets/PSS")
        return pd.DataFrame(columns=["period", "ct_antal_milj", "ct_varde_meur"])

    # Slå ihop serier på år
    df = list(results.values())[0][["ar"]].copy()
    for col, df_s in results.items():
        df = df.merge(df_s[["ar", col]], on="ar", how="outer")

    # Expandera årsdata till kvartal (samma värde Q1–Q4 för varje år)
    rows = []
    for _, row in df.iterrows():
        try:
            yr = int(row["ar"])
        except (ValueError, TypeError):
            continue
        for q in range(1, 5):
            rows.append({
                "period": f"{yr}Q{q}",
                **{c: row[c] for c in df.columns if c != "ar"}
            })
    df_q = pd.DataFrame(rows)
    df_q["ecb_data_frekvens"] = "årsvis (upprepat per kvartal)"
    print(f"  ✓ ECB-data konverterad till {len(df_q)} kvartalrader")
    return df_q


# ══════════════════════════════════════════════════════════════════════════════
# PROXY 3 – SCB: Företagsinlåning hos MFI (icke-finansiella företag, S11)
# Tabell: FM/FM0401/FM0401X/MFIM1  |  Månadsdata → kvartal
# ══════════════════════════════════════════════════════════════════════════════

def fetch_corporate_deposits() -> pd.DataFrame:
    """
    Hämtar inlåning från icke-finansiella företag i svenska MFI:er, månadsvis.
    Aggregeras till kvartal som medelvärde av utestående belopp.

    Tabell  : FM/FM0401/FM0401X/MFIM1
    Institut: S21     = Monetära finansinstitut (MFI), totalt
    Konto   : 201014  = In-/upplåning, Sv, Icke-finansiella företag
    Valuta  : v2      = Svenska kronor
    Innehåll: FM0401XX (mnkr)
    """
    print("PROXY 3 – SCB: Företagsinlåning hos MFI (icke-finansiella företag)...")
    table = "FM/FM0401/FM0401X/MFIM1"

    months = [
        f"{y}M{str(m).zfill(2)}"
        for y in range(START_YEAR, END_YEAR + 1)
        for m in range(1, 13)
    ]

    query = {
        "query": [
            {
                "code": "Institut",
                "selection": {"filter": "item", "values": ["S21"]}       # MFI totalt
            },
            {
                "code": "Kontopost",
                # K22100 = "201014 In-/upplåning, Sv, Icke-finansiella företag"
                # OBS: "201014" är textbeskrivningen – API-koden är K22100
                "selection": {"filter": "item", "values": ["K22100"]}
            },
            {
                "code": "Valuta",
                "selection": {"filter": "item", "values": ["v2"]}        # Svenska kronor
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["FM0401XX"]}
            },
            {
                "code": "Tid",
                "selection": {"filter": "item", "values": months}
            }
        ],
        "response": {"format": "json"}
    }

    url = f"{SCB_BASE}/{table}"
    try:
        r = requests.post(url, json=query, timeout=30)
        r.raise_for_status()
        data = r.json()
        rows_data = [row["key"] + row["values"] for row in data["data"]]
        df = pd.DataFrame(rows_data, columns=["institut", "kontopost", "valuta", "tid", "foretag_inlaning_mnkr"])
        df["foretag_inlaning_mnkr"] = pd.to_numeric(df["foretag_inlaning_mnkr"], errors="coerce")
        df["datum"]  = pd.to_datetime(df["tid"], format="%YM%m")
        df["period"] = df["datum"].dt.to_period("Q").astype(str)
        df_q = df.groupby("period")["foretag_inlaning_mnkr"].mean().reset_index()
        print(f"  ✓ {len(df_q)} kvartal hämtade (medelvärde inlåning per kvartal, SEK)")
        return df_q
    except requests.HTTPError as e:
        print(f"  ✗ HTTP-fel {e.response.status_code}: {e.response.text[:300]}")
        return pd.DataFrame(columns=["period", "foretag_inlaning_mnkr"])
    except Exception as e:
        print(f"  ✗ Misslyckades: {e}")
        return pd.DataFrame(columns=["period", "foretag_inlaning_mnkr"])


# ══════════════════════════════════════════════════════════════════════════════
# PROXY 4 – SCB: Cash/Asset-kvot för icke-finansiella bolag (S11)
# Tabell: FM/FM0103/FM0103A/FirENS2010ofKv  |  Kvartalsdata
# ══════════════════════════════════════════════════════════════════════════════

def fetch_corporate_cash_ratio() -> pd.DataFrame:
    """
    Hämtar kassa+inlåning (F.2) och totala finansiella tillgångar (FA0100)
    för icke-finansiella bolag (S11) från SCB Finansräkenskaper.

    Beräknar: cash_asset_kvot = (FA2100 + FA2200 + FA2900) / FA0100

    OBS: Täljaren är kassa/inlåning relativt FINANSIELLA tillgångar – inte
         total balansomslutning (reala tillgångar saknas i finansräkenskaperna).

    Tabell     : FM/FM0103/FM0103A/FirENS2010ofKv
    Sektor     : S11  (icke-finansiella bolag)
    Motsektor  : S0   (alla motparter)
    Innehåll   : FM0103AS (ställningsvärden, MNKR)
    Frekvens   : Kvartal (1996K1–2025K4)
    """
    print("PROXY 4 – SCB Finansräkenskaper: Cash/Asset-kvot (S11)...")
    table = "FM/FM0103/FM0103A/FirENS2010ofKv"

    quarters = [
        f"{y}K{q}"
        for y in range(START_YEAR, END_YEAR + 1)
        for q in range(1, 5)
    ]

    # Hämta FA0100, FA2100, FA2200, FA2900 i ett anrop
    kontopostkoder = ["FA0100", "FA2100", "FA2200", "FA2900"]

    query = {
        "query": [
            {
                "code": "Sektor",
                "selection": {"filter": "item", "values": ["S11"]}
            },
            {
                "code": "Kontopost",
                "selection": {"filter": "item", "values": kontopostkoder}
            },
            {
                "code": "Motsektor",
                "selection": {"filter": "item", "values": ["S0"]}
            },
            {
                "code": "ContentsCode",
                "selection": {"filter": "item", "values": ["FM0103AS"]}
            },
            {
                "code": "Tid",
                "selection": {"filter": "item", "values": quarters}
            }
        ],
        "response": {"format": "json"}
    }

    url = f"{SCB_BASE}/{table}"
    try:
        r = requests.post(url, json=query, timeout=30)
        r.raise_for_status()
        data = r.json()

        rows_data = [row["key"] + row["values"] for row in data["data"]]
        df = pd.DataFrame(
            rows_data,
            columns=["sektor", "kontopost", "motsektor", "tid", "varde"]
        )
        df["varde"] = pd.to_numeric(df["varde"], errors="coerce")

        # Konvertera SCB-kvartal (2022K1) → standard (2022Q1)
        df["period"] = df["tid"].str.replace("K", "Q", regex=False)

        # Pivotera så varje kontopost blir en kolumn
        df_wide = df.pivot_table(
            index="period", columns="kontopost", values="varde", aggfunc="first"
        ).reset_index()

        # Beräkna cash = sedlar + transfererbar inlåning + övrig inlåning
        kassa_cols = [c for c in ["FA2100", "FA2200", "FA2900"] if c in df_wide.columns]
        df_wide["kassa_inlaning_mnkr"] = df_wide[kassa_cols].sum(axis=1)

        if "FA0100" in df_wide.columns:
            df_wide["cash_asset_kvot"] = df_wide["kassa_inlaning_mnkr"] / df_wide["FA0100"]
        else:
            df_wide["cash_asset_kvot"] = np.nan
            print("  ! FA0100 (tot finansiella tillgångar) saknas – kvot ej beräknad")

        result = df_wide[["period", "kassa_inlaning_mnkr", "cash_asset_kvot"]].copy()
        print(f"  ✓ {len(result)} kvartal hämtade")
        print(f"  ✓ cash_asset_kvot: kassa+inlåning / totala finansiella tillgångar (S11)")
        return result

    except requests.HTTPError as e:
        print(f"  ✗ HTTP-fel {e.response.status_code}: {e.response.text[:300]}")
        return pd.DataFrame(columns=["period", "kassa_inlaning_mnkr", "cash_asset_kvot"])
    except Exception as e:
        print(f"  ✗ Misslyckades: {e}")
        return pd.DataFrame(columns=["period", "kassa_inlaning_mnkr", "cash_asset_kvot"])


# ══════════════════════════════════════════════════════════════════════════════
# Slå ihop alla proxies till ett dataset
# ══════════════════════════════════════════════════════════════════════════════

def build_sector_dataset() -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("  HÄMTAR SEKTORDATA – CASH MANAGEMENT PROXIES")
    print("=" * 60 + "\n")

    df_prov  = fetch_scb_bank_commissions()
    df_ecb   = fetch_ecb_payments_sweden()
    df_depos = fetch_corporate_deposits()
    df_cash  = fetch_corporate_cash_ratio()

    all_quarters = pd.period_range(
        start=f"{START_YEAR}Q1",
        end=f"{END_YEAR}Q4",
        freq="Q"
    ).astype(str)
    master = pd.DataFrame({"period": all_quarters})

    for df, name in [
        (df_prov,  "Provisionsintäkter netto (SCB FM0402)"),
        (df_ecb,   "Betalningsvolymer (ECB PSS)"),
        (df_depos, "Företagsinlåning MFI (SCB FM0401)"),
        (df_cash,  "Cash/Asset-kvot S11 (SCB FM0103)"),
    ]:
        if not df.empty and "period" in df.columns:
            num_cols = ["period"] + [
                c for c in df.columns
                if c != "period" and pd.api.types.is_numeric_dtype(df[c])
            ]
            merge_cols = [c for c in num_cols if c in df.columns]
            master = master.merge(
                df[merge_cols].drop_duplicates("period"),
                on="period", how="left"
            )
            print(f"  ✓ {name} inlagd")

    print(f"\nSektordataset: {master.shape[0]} rader × {master.shape[1]} kolumner")
    print(f"Saknade värden:\n{master.isnull().sum()}\n")

    print("─" * 60)
    print("TOLKNING AV PROXIES:")
    print("  provision_netto_mnkr  → Bankernas nettoprovisioner (B5−B6), årsdata interpolerat")
    print("  ct_antal_milj         → Antal kredittransferingar i Sverige (ECB, årsvis)")
    print("  ct_varde_meur         → Värde av kredittransferingar (MEUR, ECB, årsvis)")
    print("  foretag_inlaning_mnkr → Företagsinlåning MFI, SEK (månadsdata → kvartal)")
    print("  kassa_inlaning_mnkr   → Kassa+inlåning för S11-bolag (finansräkenskaper, kvartal)")
    print("  cash_asset_kvot       → Kassa / tot finansiella tillgångar S11 (kvartal)")
    print("─" * 60)

    return master


# ══════════════════════════════════════════════════════════════════════════════
# Kör
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    df = build_sector_dataset()
    df.to_csv(OUTPUT_CSV, index=False, sep=";", decimal=",")
    print(f"\nData sparad till: {OUTPUT_CSV}")
    print("\nFörhandsgranskning (rader med data):")
    data_cols = [c for c in df.columns if c != "period"]
    print(df.dropna(how="all", subset=data_cols).head(12).to_string())
