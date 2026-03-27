"""
Kombinerade tidsserieplottar – normaliserade för jämförbarhet
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# ── Läs och slå ihop data ─────────────────────────────────────────────────────
macro  = pd.read_csv("data/macro_data_sweden.csv",  sep=";", decimal=",")
sector = pd.read_csv("data/sector_proxy_data.csv",  sep=";", decimal=",")
df = macro.merge(sector, on="period", how="outer").sort_values("period").reset_index(drop=True)
df["date"] = pd.PeriodIndex(df["period"], freq="Q").to_timestamp()

def normalize(s):
    """Min-max normalisering till 0–1"""
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx != mn else s * 0

# ── Färger ────────────────────────────────────────────────────────────────────
colors = {
    "bnp_tillvaxt_pct":      "#2196F3",
    "inflation_pct":         "#FF9800",
    "arbetsloshet_pct":      "#F44336",
    "investeringar_pct":     "#9C27B0",
    "reporate_pct":          "#00BCD4",
    "provision_netto_mnkr":  "#4CAF50",
    "ct_antal_milj":         "#FF5722",
    "foretag_inlaning_mnkr": "#607D8B",
    "kassa_inlaning_mnkr":   "#E91E63",
    "cash_asset_kvot":       "#009688",
}

labels = {
    "bnp_tillvaxt_pct":      "BNP-tillväxt (%)",
    "inflation_pct":         "Inflation (%)",
    "arbetsloshet_pct":      "Arbetslöshet (%)",
    "investeringar_pct":     "Investeringar (%)",
    "reporate_pct":          "Styrränta (%)",
    "provision_netto_mnkr":  "Bankprovisioner netto (MNKR)",
    "ct_antal_milj":         "Kredittransf. antal (milj)*",
    "foretag_inlaning_mnkr": "Företagsinlåning (MNKR)",
    "kassa_inlaning_mnkr":   "Kassa+inlåning S11 (MNKR)",
    "cash_asset_kvot":       "Cash/Asset-kvot",
}

def shade(ax, df):
    ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"),
               alpha=0.12, color="gray", zorder=0)
    ax.set_xlim(df["date"].min(), df["date"].max())
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="both", labelsize=8)

fig = plt.figure(figsize=(16, 22))
fig.suptitle("Tidsserie – Sverige 2010–2025\n(grå = COVID-19)",
             fontsize=14, fontweight="bold", y=0.99)
gs = gridspec.GridSpec(3, 1, hspace=0.45)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 – Makrovariabler (faktiska värden, dubbla y-axlar)
# ══════════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0])
ax1r = ax1.twinx()

rate_vars  = ["reporate_pct", "inflation_pct", "bnp_tillvaxt_pct", "investeringar_pct"]
pct_vars   = ["arbetsloshet_pct"]

for col in rate_vars:
    if col in df.columns:
        ax1.plot(df["date"], df[col], color=colors[col], linewidth=1.8,
                 label=labels[col])

for col in pct_vars:
    if col in df.columns:
        ax1r.plot(df["date"], df[col], color=colors[col], linewidth=1.8,
                  linestyle="--", label=labels[col])

ax1.set_title("1. Makroekonomiska variabler", fontsize=11, fontweight="bold", loc="left")
ax1.set_ylabel("Procentuell förändring / nivå (%)", fontsize=8)
ax1r.set_ylabel("Arbetslöshet (%)", fontsize=8, color=colors["arbetsloshet_pct"])
ax1r.tick_params(axis="y", labelcolor=colors["arbetsloshet_pct"], labelsize=8)

lines1, lab1 = ax1.get_legend_handles_labels()
lines2, lab2 = ax1r.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper right", ncol=2)
shade(ax1, df)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 – CM-proxies volymer (normaliserade 0–1)
# ══════════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1])

vol_vars = ["provision_netto_mnkr", "ct_antal_milj",
            "foretag_inlaning_mnkr", "kassa_inlaning_mnkr"]

for col in vol_vars:
    if col in df.columns:
        normed = normalize(df[col])
        ax2.plot(df["date"], normed, color=colors[col], linewidth=1.8,
                 label=labels[col])

ax2.set_title("2. CM-proxies – volymer (normaliserade 0–1 för jämförbarhet)",
              fontsize=11, fontweight="bold", loc="left")
ax2.set_ylabel("Normaliserat värde", fontsize=8)
ax2.legend(fontsize=8, loc="upper left", ncol=2)
shade(ax2, df)

# Notera ECB-gap
ax2.annotate("*ECB-data saknas\n 2022–2025",
             xy=(pd.Timestamp("2022-01-01"), 0.75),
             fontsize=7, color=colors["ct_antal_milj"],
             arrowprops=dict(arrowstyle="->", color=colors["ct_antal_milj"]),
             xytext=(pd.Timestamp("2019-06-01"), 0.88))

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 – Cash/Asset-kvot vs styrränta (direkt jämförelse)
# ══════════════════════════════════════════════════════════════════════════════
ax3  = fig.add_subplot(gs[2])
ax3r = ax3.twinx()

ax3.plot(df["date"], df["cash_asset_kvot"], color=colors["cash_asset_kvot"],
         linewidth=2.2, label="Cash/Asset-kvot (vänster)")
ax3.fill_between(df["date"], df["cash_asset_kvot"], alpha=0.15,
                 color=colors["cash_asset_kvot"])

ax3r.plot(df["date"], df["reporate_pct"], color=colors["reporate_pct"],
          linewidth=1.8, linestyle="--", label="Styrränta % (höger)")
ax3r.plot(df["date"], df["inflation_pct"], color=colors["inflation_pct"],
          linewidth=1.5, linestyle=":", label="Inflation % (höger)")

ax3.set_title("3. Cash/Asset-kvot vs styrränta & inflation",
              fontsize=11, fontweight="bold", loc="left")
ax3.set_ylabel("Cash/Asset-kvot (andel)", fontsize=8, color=colors["cash_asset_kvot"])
ax3r.set_ylabel("Procent (%)", fontsize=8)
ax3.tick_params(axis="y", labelcolor=colors["cash_asset_kvot"], labelsize=8)

lines3, lab3 = ax3.get_legend_handles_labels()
lines4, lab4 = ax3r.get_legend_handles_labels()
ax3.legend(lines3 + lines4, lab3 + lab4, fontsize=8, loc="upper left", ncol=2)
shade(ax3, df)

os.makedirs("figures", exist_ok=True)
out = "figures/timeseries_combined.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Sparad: {out}")
plt.show()
