# AI Coding Guidelines for Cash Management Thesis Project

## Project Overview
This is a Swedish economics thesis analyzing cash management service demand using macroeconomic indicators and sector proxies. The project fetches data from Swedish statistical agencies (SCB, Riksbanken) and ECB, performs PCA analysis, and correlates with sales data.

## Architecture
- **fetch_macro_data.py**: Fetches macroeconomic variables (GDP, inflation, unemployment, investments, interest rates) from SCB and Riksbanken APIs
- **fetch_sector_data.py**: Fetches sector proxies for cash management demand (bank commissions, payment volumes, corporate deposits) from SCB and ECB APIs
- **pca_analysis.py**: Performs PCA on macro variables, generates visualizations, and correlates with sales data

## Key Conventions
- **Language**: Swedish variable names and comments (e.g., `bnp_tillvaxt_pct`, `provision_netto_mnkr`)
- **Data Format**: Quarterly periods as "YYYYQX" (e.g., "2010Q1")
- **CSV Output**: Semicolon-separated with comma decimals (`sep=";", decimal=","`)
- **API Queries**: Complex nested JSON structures for SCB API with specific ContentsCode values
- **Error Handling**: Graceful fallbacks with empty DataFrames when APIs fail
- **Rate Limiting**: `time.sleep(0.3-0.5)` between API calls

## Data Flow
1. Run `fetch_macro_data.py` → generates `macro_data_sweden.csv`
2. Run `fetch_sector_data.py` → generates `sector_proxy_data.csv`
3. Manually add `sales_data.csv` with columns: `period`, `sales`
4. Run `pca_analysis.py` → generates PCA scores, visualizations in `figures/`

## Dependencies
Install from `requirements.txt`:
```
pip install -r requirements.txt
```

## Common Patterns
- **SCB API**: Use `scb_post()` helper with table paths like "NR/NR0103/NR0103A/NR0103ENS2010T01Kv"
- **Quarterly Aggregation**: Convert monthly data to quarters using `pd.to_period("Q")`
- **Data Merging**: Left join on "period" column with master DataFrame
- **Visualization**: Matplotlib/seaborn with Swedish labels, saved to `figures/` directory
- **PCA Workflow**: Standardize → impute missing values → fit PCA → analyze loadings

## API Endpoints
- SCB: `https://api.scb.se/OV0104/v1/doris/sv/ssd`
- Riksbanken: `https://api.riksbank.se/swea/v1`
- ECB: `https://data-api.ecb.europa.eu/service/data`

## File Structure
- Data files: `*.csv` with Swedish formatting
- Figures: `figures/*.png` (auto-created)
- No config files - all constants defined in script headers

## Development Notes
- Scripts are standalone and can run independently
- Missing data handled with interpolation then median imputation
- Focus on quarterly analysis for economic modeling
- All code includes detailed Swedish docstrings explaining data sources and relevance</content>
<parameter name="filePath">/Users/isactrollsas/Downloads/Kandidatuppsats/.github/copilot-instructions.md