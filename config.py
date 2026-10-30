"""Central configuration: file names, sample filters and experiment settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("FUNDML_DATA_DIR", ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("FUNDML_OUTPUT_DIR", ROOT / "outputs"))

# Quick mode shrinks model grids for smoke tests (set FUNDML_QUICK=1).
QUICK = os.environ.get("FUNDML_QUICK", "0") == "1"

# ---------------------------------------------------------------- input files
FILES = {
    "static": "morningstar_static.xlsx",
    "returns": "morningstar_monthly_returns.xlsx",
    "expense_ratio": "morningstar_expense_ratio.xlsx",
    "net_assets": "morningstar_net_assets.xlsx",
    "category": "morningstar_category.xlsx",
    "selected_categories": "selected_categories.xlsx",
    "ff5": "Europe_5_Factors.txt",
    "ff3": "Europe_3_Factors.txt",
    "mom": "Europe_MOM_Factor.txt",
    "fx": "usd_eur_exchange_rate.xlsx",
    "euribor": "euribor_1m_bundesbank.csv",
    "eonia": "eonia_monthly.xlsx",
    "estr": "estr_monthly.csv",
}
FX_COLUMN = "USD TO EUR (LSEG DS) - EXCHANGE RATE"
MONEY_MARKET_CSV_SKIPROWS = 8      # metadata rows below the header in Bundesbank/ECB csv exports
OIS_SWITCH = "2021-12-31"          # EONIA up to this month, €STR afterwards

# ---------------------------------------------------------------- sample construction
MIN_LAGGED_TNA_FOR_FLOWS = 1e5     # EUR; flows undefined below this lagged size
ENTRY_MIN_TNA_MLN = 5.0            # share class / fund enters once TNA >= EUR 5m ...
ENTRY_MIN_AGE_MONTHS = 36          # ... and age >= 36 months
MIN_EQUITY_ALLOCATION = 70.0       # % equity (Morningstar snapshot, see README)
TURNOVER_MAX = 10.0                # turnover (fraction) above 1000% p.a. treated as invalid

EEA_INVESTMENT_AREAS = [
    "Europe", "Europe (North)", "Europe ex UK", "Scandinavia", "Baltics", "Euroland",
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark",
    "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands", "Poland",
    "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden", "Iceland",
    "Liechtenstein", "Norway",
]

PASSIVE_NAME_PATTERN = (
    r"\b(?:"
    r"index|idx|tracker|etf|passive|"
    r"benchmark|replica|replication|"
    r"s&p|sp500|msci|ftse|stoxx|"
    r"nasdaq|dow|russell|"
    r"all[- ]?world|acwi|eafe|"
    r"core index|smart beta|factor index"
    r")\b"
)

# Unit of analysis: "fund" (share classes aggregated by FundId, weighted by lagged TNA)
# or "share_class" (thesis version).
UNIT = os.environ.get("FUNDML_UNIT", "fund")

# ---------------------------------------------------------------- rolling alphas
ALPHA_WINDOW_MONTHS = 36
ALPHA_MIN_MONTHS = 30
ALPHA_HAC_LAGS = 3

# ---------------------------------------------------------------- annual / ML panel
MIN_MONTHS_PER_YEAR = 10
TARGET = "Monthly6Alpha_ann"

PREDICTOR_BASE = [
    "Monthly6Alpha_ann",
    "alpha_tstat",
    "net_assets_mln",
    "expense_ratio",
    "class_age_months",
    "net_flow_winsor_ann",
    "manager_tenure",
    "turnover_ratio_frac_w",
    "flow_volatility_annual",
    "value_added_ann",
    "market_beta_tstat",
    "profitability_tstat",
    "investment_tstat",
    "size_beta_tstat",
    "value_beta_tstat",
    "momentum_beta_tstat",
    "Monthly6Factor_R2",
]
# Only available as one Morningstar snapshot (download date). Applied to every
# historical year they carry future information, so they are excluded by default.
SNAPSHOT_PREDICTORS = ["manager_tenure", "turnover_ratio_frac_w"]
PREDICTORS = [p for p in PREDICTOR_BASE if p not in SNAPSHOT_PREDICTORS]

# Unstandardised t-1 values carried into the ML panel (benchmarks, misallocation analysis)
RAW_LAG_COLUMNS = ["net_assets_mln", "expense_ratio"]

# ---------------------------------------------------------------- experiment
PANEL_LENGTH = 10                  # initial training window in years
THRESHOLDS = [0.80, 0.90]          # top 20% / top 10%
CV_FOLDS = 5
RANDOM_SEED = 1011

# ---------------------------------------------------------------- evaluation
NW_LAGS_PORTFOLIO = 12
MIN_OBS_PORTFOLIO = 60
FACTOR_MODELS = {
    "FF3_MOM": ["mkt-rf_eur", "smb_3ff_eur", "hml_3ff_eur", "wml_eur"],
    "FF5": ["mkt-rf_eur", "smb_5ff_eur", "hml_5ff_eur", "rmw_5ff_eur", "cma_5ff_eur"],
    "FF5_MOM": ["mkt-rf_eur", "smb_5ff_eur", "hml_5ff_eur", "rmw_5ff_eur", "cma_5ff_eur", "wml_eur"],
}
RF_COL = "rf_eur"
