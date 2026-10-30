# Input data (not included)

Morningstar and LSEG data are licensed and cannot be published. Place the following files in this folder, or point `FUNDML_DATA_DIR` to them. File names are set in `src/fundml/config.py`. `tests/synthetic_data.py` writes fake files in exactly these formats, which can serve as a template.

| File | Source | Format |
|---|---|---|
| `morningstar_static.xlsx` | Morningstar Direct | One row per share class. Columns used: `SecId`, `FundId`, `Name`, `Institutional` (Yes/No), `Oldest Share Class`, `Turnover Ratio %`, `Manager Tenure (Average)`, `Asset Alloc Equity % (Long Rescaled)`, `Inception Date`, `Domicile`, `Investment Area` |
| `morningstar_monthly_returns.xlsx` | Morningstar Direct | `SecId` + one column per month, header containing `Monthly Return YYYY-MM`, values in %, EUR |
| `morningstar_net_assets.xlsx` | Morningstar Direct | `SecId` + `Net Assets - share class (Monthly) YYYY-MM`, EUR |
| `morningstar_category.xlsx` | Morningstar Direct | `SecId` + `Morningstar Category YYYY-MM` |
| `morningstar_expense_ratio.xlsx` | Morningstar Direct | `SecId` + `Annual Report Net Expense Ratio YearYYYY`, % p.a. |
| `selected_categories.xlsx` | own selection | one column `category` listing the Morningstar categories in the sample |
| `Europe_5_Factors.txt`, `Europe_3_Factors.txt`, `Europe_MOM_Factor.txt` | Kenneth French data library | whitespace-separated, monthly block only, first column `date` (YYYYMM), values in %; columns `Mkt-RF SMB HML RMW CMA RF` / `Mkt-RF SMB HML RF` / `WML` |
| `usd_eur_exchange_rate.xlsx` | LSEG Datastream | columns `date` (month-start observation) and `USD TO EUR (LSEG DS) - EXCHANGE RATE` (USD per EUR) |
| `euribor_1m_bundesbank.csv` | Deutsche Bundesbank | 1-month EURIBOR, monthly, % p.a.; Bundesbank csv layout (header + 8 metadata rows) |
| `eonia_monthly.xlsx` | ECB / Bundesbank | columns `date`, `value` (% p.a.) |
| `estr_monthly.csv` | ECB | €STR monthly average, % p.a.; same csv layout as EURIBOR |
