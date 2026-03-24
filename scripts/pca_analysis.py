"""
PCA-analys: Makroekonomiska variabler vs. Cash Management-försäljning
======================================================================
Förväntar sig:
  - macro_data_sweden.csv    (genereras av fetch_macro_data.py)
  - sales_data.csv           (Nordea/bankdata – läggs till manuellt)
                              Kolumner: period (YYYYQX), sales (volym/värde)

Steg:
  1. Ladda och rensa data
  2. Standardisera variabler
  3. PCA på makrovariabler
  4. Korrelera huvudkomponenter med försäljning
  5. Visualisera (scree plot, biplot, korrelationsmatris, tidsserier)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Icke-interaktivt – sparar figurer utan att öppna fönster
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# ── Konfiguration ──────────────────────────────────────────────────────────────
MACRO_FILE = "data/master_dataset.csv"   # Slår ihop makro + sektorproxies
SALES_FILE = "data/sales_data.csv"       # Nordea-data – lägg till manuellt
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# Alla tillgängliga variabler med korrekta kolumnnamn och etiketter
MACRO_VARS = {
    # ── Makroekonomiska variabler (fetch_macro_data.py) ──
    "bnp_tillvaxt_pct":       "BNP-tillväxt (%)",
    "inflation_pct":          "Inflation KPI (%)",
    "arbetsloshet_pct":       "Arbetslöshet (%)",
    "investeringar_pct":      "Fasta bruttoinvest. (%)",   # var: _pct, inte _mnkr
    "reporate_pct":           "Reporänta (%)",
    # ── Sektorproxies för CM-efterfrågan (fetch_sector_data.py) ──
    "provision_netto_mnkr":   "Bankprov. netto (mnkr)",
    "ct_antal_milj":          "Kredittransf. antal (milj)",
    "ct_varde_meur":          "Kredittransf. värde (MEUR)",
    "foretag_inlaning_mnkr":  "Företagsinlåning (mnkr)",
    "kassa_inlaning_mnkr":    "Kassa+inlåning S11 (mnkr)",
    "cash_asset_kvot":        "Cash/Asset-kvot S11",
}

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})


# ══════════════════════════════════════════════════════════════════════════════
# 1. Ladda data
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    macro = pd.read_csv(MACRO_FILE, sep=";", decimal=",")
    print(f"Master-dataset: {macro.shape[0]} perioder, {macro.shape[1]-1} variabler")
    print(f"  Kolumner: {[c for c in macro.columns if c != 'period']}")

    # Försäljningsdata (valfritt – hoppa över om filen saknas)
    if Path(SALES_FILE).exists():
        sales = pd.read_csv(SALES_FILE, sep=";", decimal=",")
        df = macro.merge(sales, on="period", how="inner")
        print(f"Försäljningsdata inlagd: {df.shape[0]} matchande perioder")
    else:
        df = macro.copy()
        df["sales"] = np.nan
        print(f"  OBS: {SALES_FILE} saknas – sales-kolumn satt till NaN")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Förbered data för PCA
# ══════════════════════════════════════════════════════════════════════════════

def prepare_features(df: pd.DataFrame):
    available = [col for col in MACRO_VARS if col in df.columns]
    X = df[available].copy()

    # Imputera saknade värden med linjär interpolation, sedan median
    X = X.interpolate(method="linear", limit_direction="both")
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=available, index=X.index)

    # Standardisera (Z-score) – nödvändigt för PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    labels = [MACRO_VARS[c] for c in available]
    return X_scaled, available, labels, X_imp


# ══════════════════════════════════════════════════════════════════════════════
# 3. PCA
# ══════════════════════════════════════════════════════════════════════════════

def run_pca(X_scaled, feature_names, labels):
    pca = PCA()
    components = pca.fit_transform(X_scaled)
    explained  = pca.explained_variance_ratio_

    print("\n── PCA Förklarad varians ──────────────────────────────")
    cumulative = 0
    for i, ev in enumerate(explained):
        cumulative += ev
        print(f"  PC{i+1}: {ev*100:.1f}%  (kumulativ: {cumulative*100:.1f}%)")

    # Loadings: hur mycket varje original-variabel bidrar till varje PC
    loadings = pd.DataFrame(
        pca.components_.T,
        index=labels,
        columns=[f"PC{i+1}" for i in range(len(explained))]
    )
    print("\n── Loadings (de 2 första PC:erna) ───────────────────")
    print(loadings[["PC1", "PC2"]].round(3).to_string())

    return pca, components, loadings


# ══════════════════════════════════════════════════════════════════════════════
# 4. Visualiseringar
# ══════════════════════════════════════════════════════════════════════════════

def plot_scree(pca, save=True):
    """Scree plot – visar hur mycket varians varje PC förklarar."""
    ev = pca.explained_variance_ratio_ * 100
    n  = len(ev)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(1, n+1), ev, color="#2166ac", alpha=0.8, label="Enskild PC")
    ax.plot(range(1, n+1), ev.cumsum(), "o-", color="#d6604d", label="Kumulativ")
    ax.axhline(80, color="gray", linestyle="--", linewidth=0.8, label="80%-gräns")
    ax.set_xlabel("Huvudkomponent")
    ax.set_ylabel("Förklarad varians (%)")
    ax.set_title("Scree Plot – PCA på makroekonomiska variabler")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "scree_plot.png")
        print("  Sparad: figures/scree_plot.png")
    plt.show()


def plot_loadings_heatmap(loadings, n_pc=4, save=True):
    """Heatmap av loadings – vilka variabler driver vilka PC:er."""
    subset = loadings.iloc[:, :min(n_pc, loadings.shape[1])]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        subset,
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax
    )
    ax.set_title("PCA Loadings – bidrag per variabel och PC")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "loadings_heatmap.png")
        print("  Sparad: figures/loadings_heatmap.png")
    plt.show()


def plot_biplot(components, loadings, labels, save=True):
    """Biplot – PC1 vs PC2, med pilarna för originalsvariabler."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(components[:, 0], components[:, 1],
               alpha=0.5, s=30, color="#2166ac", label="Kvartal")

    scale = max(abs(components[:, :2]).max() * 0.4, 1)
    for i, label in enumerate(labels):
        x = loadings.iloc[i, 0] * scale
        y = loadings.iloc[i, 1] * scale
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#d6604d", lw=1.5))
        ax.text(x * 1.08, y * 1.08, label, fontsize=8, color="#d6604d")

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel(f"PC1 ({loadings.columns[0]})")
    ax.set_ylabel(f"PC2 ({loadings.columns[1]})")
    ax.set_title("Biplot – PC1 vs PC2")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "biplot.png")
        print("  Sparad: figures/biplot.png")
    plt.show()


def plot_correlation_matrix(X_imp, labels, save=True):
    """Korrelationsmatris mellan originalsvariabler."""
    corr = pd.DataFrame(X_imp.values, columns=labels).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.5, ax=ax)
    ax.set_title("Korrelationsmatris – Makroekonomiska variabler")
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "correlation_matrix.png")
        print("  Sparad: figures/correlation_matrix.png")
    plt.show()


def plot_pc_vs_sales(df, components, save=True):
    """Tidsserie: PC1, PC2 och försäljning (om tillgänglig)."""
    has_sales = "sales" in df.columns and df["sales"].notna().any()
    n_plots   = 3 if has_sales else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 3 * n_plots), sharex=True)

    x = range(len(df))
    xlabels = df["period"].iloc[::4].values
    xticks  = list(range(0, len(df), 4))

    axes[0].plot(x, components[:, 0], color="#2166ac")
    axes[0].set_ylabel("PC1")
    axes[0].set_title("Huvudkomponent 1 över tid")
    axes[0].axhline(0, color="gray", linewidth=0.5)

    axes[1].plot(x, components[:, 1], color="#4dac26")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("Huvudkomponent 2 över tid")
    axes[1].axhline(0, color="gray", linewidth=0.5)

    if has_sales:
        axes[2].plot(x, df["sales"].values, color="#d6604d")
        axes[2].set_ylabel("Försäljning")
        axes[2].set_title("Cash Management Försäljning")

    for ax in axes:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=45, ha="right")

    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "pc_timeseries.png")
        print("  Sparad: figures/pc_timeseries.png")
    plt.show()


def print_correlation_with_sales(df, components):
    """Beräknar korrelation mellan PC:er och försäljning."""
    if "sales" not in df.columns or df["sales"].isna().all():
        print("\n  OBS: Ingen försäljningsdata – hoppar över korrelationsanalys")
        return

    print("\n── Korrelation PC:er ↔ Försäljning ──────────────────")
    sales = df["sales"].values
    for i in range(components.shape[1]):
        # Använd bara rader där båda värden finns
        mask = ~np.isnan(sales)
        r = np.corrcoef(components[mask, i], sales[mask])[0, 1]
        print(f"  PC{i+1} ↔ sales: r = {r:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# Huvud
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*60)
    print("  PCA-ANALYS – CASH MANAGEMENT & MAKROEKONOMI")
    print("="*60)

    df = load_data()
    X_scaled, feature_names, labels, X_imp = prepare_features(df)
    pca, components, loadings = run_pca(X_scaled, feature_names, labels)

    print("\nGenererar figurer...")
    plot_scree(pca)
    plot_loadings_heatmap(loadings)
    plot_biplot(components, loadings, labels)
    plot_correlation_matrix(X_imp, labels)
    plot_pc_vs_sales(df, components)
    print_correlation_with_sales(df, components)

    # Spara PC-scores till CSV (för vidare regressionsanalys)
    pc_df = pd.DataFrame(
        components,
        columns=[f"PC{i+1}" for i in range(components.shape[1])]
    )
    pc_df.insert(0, "period", df["period"].values)
    pc_df.to_csv("data/pca_scores.csv", index=False, sep=";", decimal=",")
    print("\nPC-scores sparade till: data/pca_scores.csv")
    print("\nKlar!")
