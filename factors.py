"""Fama-French European factors, EUR risk-free rate and USD -> EUR conversion."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .data import month_end


def _read_ff(path: Path, rename: dict) -> pd.DataFrame:
    f = pd.read_csv(path, sep=r"\s+", na_values=["-99.99"])
    f.index = month_end(pd.to_datetime(f["date"].astype(str) + "01", format="%Y%m%d")).values
    f = f.drop(columns="date").astype(float).replace(-99.99, np.nan) / 100   # % -> decimal
    return f.rename(columns=rename)


def _monthly_from_pa_pct(s: pd.Series) -> pd.Series:
    """Annualised money-market rate in % (ACT/360) -> compounded monthly return."""
    days = s.index.to_period("M").days_in_month
    out = (1 + (s / 100) / 360) ** np.asarray(days) - 1
    out.index = month_end(s.index).values
    return out[~out.index.duplicated(keep="last")]


def _read_money_market_csv(path: Path) -> pd.Series:
    raw = pd.read_csv(path)
    x = raw.iloc[C.MONEY_MARKET_CSV_SKIPROWS:, :2].copy()
    x.columns = ["date", "rate"]
    x["date"] = pd.to_datetime(x["date"])
    x["rate"] = pd.to_numeric(x["rate"].astype(str).str.replace(",", "."), errors="coerce")
    return x.set_index("date").sort_index()["rate"]


def build_rf_eur(data_dir: Path | str = C.DATA_DIR) -> pd.Series:
    """EUR risk-free rate = min(1M EURIBOR, OIS proxy); OIS = EONIA until 2021-12, €STR after."""
    d = Path(data_dir)
    euribor = _monthly_from_pa_pct(_read_money_market_csv(d / C.FILES["euribor"]))
    estr = _monthly_from_pa_pct(_read_money_market_csv(d / C.FILES["estr"]))
    eo = pd.read_excel(d / C.FILES["eonia"])
    eonia = _monthly_from_pa_pct(eo.assign(date=pd.to_datetime(eo["date"])).set_index("date")
                                 .sort_index()["value"])
    ois = eonia.loc[eonia.index <= pd.Timestamp(C.OIS_SWITCH)].combine_first(estr)
    rf = pd.concat([euribor.rename("e"), ois.rename("o")], axis=1).dropna().min(axis=1)
    full = pd.date_range(rf.index.min(), rf.index.max(), freq="ME")
    missing = full.difference(rf.index)
    if len(missing):
        print(f"WARNING: rf_eur missing {len(missing)} months")
    return rf.rename("rf_eur")


def load_fx_return(data_dir: Path | str = C.DATA_DIR) -> pd.Series:
    """
    Monthly change in USD per EUR. Observations are month-start rates, so the return
    of month m is rate(m+1)/rate(m) - 1. Check with fx_alignment_check(): converting the
    market factor to EUR must LOWER its volatility; a one-month misalignment raises it.
    """
    fx = pd.read_excel(Path(data_dir) / C.FILES["fx"])
    fx["date"] = pd.to_datetime(fx["date"])
    fx = fx.set_index("date").sort_index()
    rate = pd.to_numeric(fx[C.FX_COLUMN].astype(str).str.replace(",", "."), errors="coerce")
    rate.index = month_end(rate.index).values
    rate = rate[~rate.index.duplicated(keep="last")]
    return (rate.shift(-1) / rate - 1).dropna().rename("r_fx_usd_per_eur")


def build_factor_panel(data_dir: Path | str = C.DATA_DIR) -> pd.DataFrame:
    """Monthly panel (month-end index) with USD factors and their EUR-converted versions."""
    d = Path(data_dir)
    ff5 = _read_ff(d / C.FILES["ff5"], {"Mkt-RF": "mkt-rf_usd", "SMB": "smb_5ff_usd",
                                        "HML": "hml_5ff_usd", "RMW": "rmw_5ff_usd",
                                        "CMA": "cma_5ff_usd", "RF": "rf_usd"})
    ff3 = _read_ff(d / C.FILES["ff3"], {"SMB": "smb_3ff_usd", "HML": "hml_3ff_usd"})
    ff3 = ff3[["smb_3ff_usd", "hml_3ff_usd"]]
    mom = _read_ff(d / C.FILES["mom"], {"WML": "wml_usd"})[["wml_usd"]]
    fac = pd.concat([ff5, ff3, mom, build_rf_eur(d), load_fx_return(d)], axis=1, join="inner")
    fac = fac.loc[:, ~fac.columns.duplicated()].sort_index()

    fx = fac["r_fx_usd_per_eur"]
    # long-short factors: LS_EUR = LS_USD / (1 + fx)   (Glück et al.)
    for c in ["smb_5ff", "hml_5ff", "rmw_5ff", "cma_5ff", "wml", "smb_3ff", "hml_3ff"]:
        fac[f"{c}_eur"] = fac[f"{c}_usd"] / (1 + fx)
    # market: gross USD return -> EUR -> minus EUR risk-free
    fac["rm_eur"] = (1 + fac["mkt-rf_usd"] + fac["rf_usd"]) / (1 + fx) - 1
    fac["mkt-rf_eur"] = fac["rm_eur"] - fac["rf_eur"]
    fac.index.name = "date"
    if fac[["mkt-rf_eur", "rf_eur", "mkt-rf_usd", "rf_usd"]].isna().any().any():
        raise ValueError("missing values in core factor columns")
    return fac


def fx_alignment_check(fac: pd.DataFrame) -> pd.Series:
    """Volatility of the market factor in USD vs EUR (EUR should be lower)."""
    out = fac[["mkt-rf_usd", "mkt-rf_eur"]].std().rename("monthly_sd")
    if out["mkt-rf_eur"] >= out["mkt-rf_usd"]:
        print("WARNING: EUR market factor is not less volatile than USD - check FX timing.")
    return out
