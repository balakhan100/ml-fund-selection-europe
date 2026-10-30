"""Monthly -> annual fund panel and the lagged, standardised ML panel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .alphas import ALPHA_OUTPUT_COLS


def attach_alphas(panel: pd.DataFrame, alphas: pd.DataFrame, id_col: str = "secid") -> pd.DataFrame:
    """Left-join rolling-regression outputs (estimated on the full history) onto the
    screened panel and compute monthly value added = (alpha_6F + ER) x lagged TNA (EUR m)."""
    a = alphas[[id_col, "date"] + ALPHA_OUTPUT_COLS]
    out = panel.drop(columns=[c for c in ALPHA_OUTPUT_COLS if c in panel], errors="ignore")
    out = out.merge(a, on=[id_col, "date"], how="left", validate="one_to_one")
    out["value_added"] = (out["Monthly6Alpha"] + out["expense_ratio"]) * out["net_assets_l1_mln"]
    return out


DECEMBER_VARS = ["expense_ratio", "turnover_ratio_frac_w", "class_age_months", "net_assets_mln",
                 "alpha_tstat", "market_beta_tstat", "size_beta_tstat", "value_beta_tstat",
                 "profitability_tstat", "investment_tstat", "momentum_beta_tstat",
                 "manager_tenure", "Monthly6Factor_R2"]


def build_annual_panel(m: pd.DataFrame, id_col: str = "secid") -> pd.DataFrame:
    """
    Fund-year panel (no minimum-month filter applied; 'n_months' = months with a 6F alpha).
    Annualised (12 x mean) alpha, value added and flows; flow volatility (sd x sqrt 12);
    year-end characteristics from the December observation.
    """
    df = m.copy()
    df["year"] = df["date"].dt.year
    g = df.groupby([id_col, "year"])
    ann = pd.DataFrame({
        "Monthly6Alpha_ann": g["Monthly6Alpha"].mean() * 12,
        "value_added_ann": g["value_added"].mean() * 12,
        "net_flow_winsor_ann": g["net_flow_winsor"].mean() * 12,
        "flow_volatility_annual": g["net_flow_winsor"].std(ddof=1) * np.sqrt(12),
        "n_months": g["Monthly6Alpha"].count(),
    }).reset_index()
    dec = (df.loc[df["date"].dt.month == 12, [id_col, "year"] + DECEMBER_VARS]
             .drop_duplicates([id_col, "year"], keep="last"))
    return ann.merge(dec, on=[id_col, "year"], how="left", validate="one_to_one")


def zscore_by_year(df: pd.DataFrame, cols: list[str], year_col: str = "year") -> pd.DataFrame:
    """Cross-sectional z-scores per year; undefined / missing -> 0 (cross-sectional mean)."""
    out = df.copy()
    for c in cols:
        g = out.groupby(year_col)[c]
        z = (out[c] - g.transform("mean")) / g.transform(lambda s: s.std(ddof=1))
        out[c + "_z"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def build_ml_panel(annual: pd.DataFrame, predictor_base: list[str] = C.PREDICTOR_BASE,
                   target: str = C.TARGET, raw_lag_cols: list[str] = C.RAW_LAG_COLUMNS,
                   id_col: str = "secid", min_months: int = C.MIN_MONTHS_PER_YEAR):
    """
    Row (i, t): characteristics of year t-1, z-scored within t-1, and realized alpha of t.

    * Characteristics exist only if fund i has >= min_months alpha months in t-1, so
      lags never jump over missing years and no fund enters with all-zero predictors.
    * The year-t universe is every fund with t-1 characteristics. Funds that die during t
      stay in it (target NaN): they were investable on 1 January, so excluding them would
      condition selection on future survival.
    * Raw (unstandardised) t-1 values of raw_lag_cols are kept as '<col>_lag1' for the
      asset-weighted benchmark and the capital-misallocation analysis (not predictors).
    """
    a = annual.copy()
    a[id_col] = a[id_col].astype(str)
    a["year"] = a["year"].astype(int)

    chars = a.loc[a["n_months"] >= min_months].copy()
    chars = zscore_by_year(chars, predictor_base)
    z_cols = [c + "_z" for c in predictor_base]
    lagged = chars[[id_col, "year"] + z_cols + raw_lag_cols].rename(
        columns={**{c + "_z": c + "_z_lag1" for c in predictor_base},
                 **{c: c + "_lag1" for c in raw_lag_cols}})
    lagged["year"] += 1

    tgt = a[[id_col, "year", target, "n_months"]].copy()
    tgt.loc[tgt["n_months"] < min_months, target] = np.nan

    panel = lagged.merge(tgt[[id_col, "year", target]], on=[id_col, "year"], how="left",
                         validate="one_to_one")
    panel = panel.loc[panel["year"] <= a["year"].max()].reset_index(drop=True)
    return panel


def predictor_columns(base: list[str]) -> list[str]:
    return [c + "_z_lag1" for c in base]


def add_gross_returns(m: pd.DataFrame, id_col: str = "secid") -> pd.DataFrame:
    """gross = net + monthly expense ratio. Missing ERs are carried forward (then backward)
    within the fund; if a fund never reports an ER the net return is used. A missing gross
    return would otherwise be treated as a liquidated position in the portfolio code."""
    out = m.sort_values([id_col, "date"]).copy()
    er = out.groupby(id_col)["expense_ratio"].ffill()
    er = er.groupby(out[id_col]).bfill()
    out["gross_return"] = (out["monthly_return"] + er).fillna(out["monthly_return"])
    return out
