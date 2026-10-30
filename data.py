"""Load Morningstar exports and build the monthly share-class panel."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# ---------------------------------------------------------------- helpers
def month_end(s) -> pd.Series:
    """Any date-like Series -> month-end Timestamp (00:00)."""
    d = pd.to_datetime(pd.Series(s), errors="coerce")
    return d.dt.to_period("M").dt.to_timestamp() + pd.offsets.MonthEnd(0)


def assert_unique(df: pd.DataFrame, key: list[str], name: str) -> None:
    n = df.duplicated(key, keep=False).sum()
    if n:
        raise AssertionError(f"{name}: {n} duplicate rows on {key}")


def winsorize_by(df: pd.DataFrame, col: str, by: str, lo=0.01, hi=0.99) -> pd.Series:
    return df.groupby(by)[col].transform(lambda x: x.clip(x.quantile(lo), x.quantile(hi)))


# ---------------------------------------------------------------- loading
def load_morningstar(data_dir: Path | str = C.DATA_DIR) -> dict[str, pd.DataFrame]:
    d = Path(data_dir)
    return {k: pd.read_excel(d / C.FILES[k])
            for k in ["static", "returns", "expense_ratio", "net_assets", "category"]}


def _melt_monthly(df, prefix, value_name):
    cols = df.columns[df.columns.astype(str).str.startswith(prefix)]
    out = df.melt(id_vars=["SecId"], value_vars=cols, var_name="_c", value_name=value_name)
    out["date"] = month_end(out["_c"].astype(str).str.extract(r"(\d{4}-\d{2})", expand=False))
    return out.drop(columns="_c").dropna(subset=["date"])


def reshape_morningstar(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Wide Morningstar exports -> long (SecId, date) / (SecId, year) tables."""
    ret = _melt_monthly(raw["returns"], "Monthly Return ", "monthly_return")
    ret["monthly_return"] = pd.to_numeric(ret["monthly_return"], errors="coerce") / 100

    na = _melt_monthly(raw["net_assets"], "Net Assets - share class (Monthly) ", "net_assets")
    na["net_assets"] = pd.to_numeric(na["net_assets"], errors="coerce")

    cat = _melt_monthly(raw["category"], "Morningstar Category ", "morningstar_category")

    e = raw["expense_ratio"]
    er_cols = e.columns[e.columns.astype(str).str.startswith("Annual Report Net Expense Ratio Year")]
    er = e.melt(id_vars=["SecId"], value_vars=er_cols, var_name="_c", value_name="expense_ratio")
    er["year"] = er["_c"].str.extract(r"Year(\d{4})", expand=False).astype("int64")
    # annual % -> monthly decimal, comparable with monthly returns
    er["expense_ratio"] = pd.to_numeric(er["expense_ratio"], errors="coerce") / 1200
    er = er.drop(columns="_c")

    out = {"returns": ret, "net_assets": na, "category": cat, "expense_ratio": er}
    for k, key in [("returns", ["SecId", "date"]), ("net_assets", ["SecId", "date"]),
                   ("category", ["SecId", "date"]), ("expense_ratio", ["SecId", "year"])]:
        assert_unique(out[k], key, k)
        out[k] = out[k].sort_values(key).reset_index(drop=True)
    return out


# ---------------------------------------------------------------- monthly panel
def build_monthly_panel(raw: dict[str, pd.DataFrame], long: dict[str, pd.DataFrame],
                        selected_categories: list[str], verbose: bool = True) -> pd.DataFrame:
    """
    Share-class x month panel with returns, size, flows, costs and characteristics,
    restricted to active, EEA-focused equity funds in the selected categories.
    The entry screen (size/age) is NOT applied here: rolling betas use the full history.
    """
    static = raw["static"]
    log = (lambda *a: print(*a)) if verbose else (lambda *a: None)

    # returns x net assets (both must be present)
    df = long["returns"].merge(long["net_assets"], on=["SecId", "date"], how="inner",
                               validate="one_to_one").sort_values(["SecId", "date"])

    # one-month lag of TNA, only if the previous row is exactly one calendar month earlier
    prev = df.groupby("SecId")["date"].shift(1)
    gap = (df["date"].dt.year * 12 + df["date"].dt.month) - (prev.dt.year * 12 + prev.dt.month)
    df["net_assets_l1"] = df.groupby("SecId")["net_assets"].shift(1).where(gap == 1)
    log("month gaps between consecutive rows:", gap.value_counts(dropna=False).head(5).to_dict())

    # annual expense ratio by calendar year
    df["year"] = df["date"].dt.year.astype("int64")
    df = df.merge(long["expense_ratio"], on=["SecId", "year"], how="left", validate="many_to_one")
    df["expense_ratio"] = df["expense_ratio"].mask(df["expense_ratio"].le(0))

    # net flows (Sirri-Tufano), only with a meaningful lagged size
    valid = (df["net_assets"].notna() & df["net_assets_l1"].notna()
             & df["monthly_return"].notna() & (df["net_assets_l1"] >= C.MIN_LAGGED_TNA_FOR_FLOWS))
    df["net_flow"] = np.where(
        valid,
        (df["net_assets"] - df["net_assets_l1"] * (1 + df["monthly_return"])) / df["net_assets_l1"],
        np.nan)
    df["net_flow"] = df["net_flow"].replace([np.inf, -np.inf], np.nan)
    df["net_flow_winsor"] = winsorize_by(df, "net_flow", "date")
    log(f"flow-valid rows: {valid.sum():,} of {len(df):,} ({valid.mean():.2%})")

    # static characteristics + time-varying category
    static_cols = ["SecId", "FundId", "Name", "Institutional", "Oldest Share Class",
                   "Turnover Ratio %", "Manager Tenure (Average)",
                   "Asset Alloc Equity % (Long Rescaled)", "Inception Date",
                   "Domicile", "Investment Area"]
    assert_unique(static, ["SecId"], "static")
    df = df.merge(static[static_cols], on="SecId", how="inner", validate="many_to_one")
    df = df.merge(long["category"], on=["SecId", "date"], how="inner", validate="one_to_one")
    df["Institutional"] = df["Institutional"].map({"Yes": 1, "No": 0})

    # active funds: name-based passive screen
    name = df["Name"].astype("string").str.lower()
    df = df.loc[~name.str.contains(C.PASSIVE_NAME_PATTERN, na=False, regex=True)]
    df = df.loc[df["FundId"].notna()]

    df = df.rename(columns={
        "SecId": "secid", "FundId": "fundid", "Name": "name", "Institutional": "institutional",
        "Oldest Share Class": "oldest_share_class", "Turnover Ratio %": "turnover_ratio",
        "Manager Tenure (Average)": "manager_tenure",
        "Asset Alloc Equity % (Long Rescaled)": "asset_alloc_equity",
        "Inception Date": "inception_date", "Domicile": "domicile",
        "Investment Area": "investment_area"})

    # share-class age in months (negative = bad record)
    inc = pd.to_datetime(df["inception_date"], errors="coerce")
    age = (df["date"].dt.year - inc.dt.year) * 12 + (df["date"].dt.month - inc.dt.month)
    df["class_age_months"] = age.where(age >= 0)

    df = df.loc[df["monthly_return"].notna() & df["net_assets"].notna()]

    # investment universe
    eea = {a.lower() for a in C.EEA_INVESTMENT_AREAS}
    df = df.loc[df["investment_area"].astype(str).str.strip().str.lower().isin(eea)]
    df = df.loc[df["morningstar_category"].isin(selected_categories)]
    df["asset_alloc_equity"] = pd.to_numeric(df["asset_alloc_equity"], errors="coerce")
    df = df.loc[df["asset_alloc_equity"] > C.MIN_EQUITY_ALLOCATION].copy()

    # turnover: % -> fraction, invalid values -> NaN, then monthly 1/99 winsorisation
    t = pd.to_numeric(df["turnover_ratio"], errors="coerce") / 100
    df["turnover_ratio_frac"] = t.mask((t > C.TURNOVER_MAX) | (t < 0))
    df["turnover_ratio_frac_w"] = winsorize_by(df, "turnover_ratio_frac", "date")

    df["manager_tenure"] = pd.to_numeric(df["manager_tenure"], errors="coerce")
    df["net_assets_mln"] = df["net_assets"] / 1e6
    df["net_assets_l1_mln"] = df["net_assets_l1"] / 1e6

    keep = ["secid", "fundid", "date", "year", "monthly_return", "net_assets", "net_assets_l1",
            "net_assets_mln", "net_assets_l1_mln", "net_flow_winsor", "expense_ratio",
            "class_age_months", "turnover_ratio_frac_w", "manager_tenure", "institutional",
            "oldest_share_class", "morningstar_category", "domicile", "investment_area"]
    df = df[keep].sort_values(["secid", "date"]).reset_index(drop=True)
    assert_unique(df, ["secid", "date"], "monthly panel")
    log(f"monthly panel: {len(df):,} rows, {df['secid'].nunique():,} share classes, "
        f"{df['fundid'].nunique():,} funds")
    return df


# ---------------------------------------------------------------- fund-level aggregation
def aggregate_to_fund_level(m: pd.DataFrame) -> pd.DataFrame:
    """
    Share-class panel -> fund panel (one row per FundId and month).
    Returns, expense ratio and turnover are weighted by lagged class TNA (equal weights
    where no lagged TNA exists); TNA is summed; age is the oldest class's age; flows
    are recomputed from fund TNA. Output uses 'secid' = FundId so downstream code is
    unit-agnostic.
    """
    m = m.copy()
    key = ["fundid", "date"]
    w = m["net_assets_l1"].where(m["net_assets_l1"] > 0)
    g = m.groupby(key)
    f = pd.DataFrame({
        "net_assets": g["net_assets"].sum(min_count=1),
        "net_assets_l1": g["net_assets_l1"].sum(min_count=1),
        "class_age_months": g["class_age_months"].max(),
        "n_share_classes": g["secid"].nunique(),
        "manager_tenure": g["manager_tenure"].first(),
        "morningstar_category": g["morningstar_category"].first(),
        "domicile": g["domicile"].first(),
        "investment_area": g["investment_area"].first(),
    })
    for v in ["monthly_return", "expense_ratio", "turnover_ratio_frac_w"]:
        wv = w.where(m[v].notna())
        num = (m[v] * wv).groupby([m["fundid"], m["date"]]).sum(min_count=1)
        den = wv.groupby([m["fundid"], m["date"]]).sum(min_count=1)
        f[v] = (num / den).fillna(g[v].mean())
    f = f.reset_index()
    f["year"] = f["date"].dt.year.astype("int64")
    f["net_assets_mln"] = f["net_assets"] / 1e6
    f["net_assets_l1_mln"] = f["net_assets_l1"] / 1e6
    valid = (f["net_assets_l1"] >= C.MIN_LAGGED_TNA_FOR_FLOWS) & f["monthly_return"].notna()
    flow = (f["net_assets"] - f["net_assets_l1"] * (1 + f["monthly_return"])) / f["net_assets_l1"]
    f["net_flow"] = flow.where(valid).replace([np.inf, -np.inf], np.nan)
    f["net_flow_winsor"] = winsorize_by(f, "net_flow", "date")
    f = f.drop(columns="net_flow").rename(columns={"fundid": "secid"})
    f["fundid"] = f["secid"]
    return f.sort_values(["secid", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------- entry screen
def apply_entry_screen(df: pd.DataFrame, id_col: str = "secid", verbose: bool = True) -> pd.DataFrame:
    """Keep each unit from the first month in which it has reached BOTH
    TNA >= EUR 5m (at some point so far) and age >= 36 months (incubation-bias filter)."""
    first_tna = df.loc[df["net_assets_mln"] >= C.ENTRY_MIN_TNA_MLN].groupby(id_col)["date"].min()
    first_age = df.loc[df["class_age_months"] >= C.ENTRY_MIN_AGE_MONTHS].groupby(id_col)["date"].min()
    start = pd.concat([first_tna, first_age], axis=1, join="inner").max(axis=1).rename("include_from")
    out = df.merge(start, left_on=id_col, right_index=True, how="inner")
    out = out.loc[out["date"] >= out["include_from"]].drop(columns="include_from")
    if verbose:
        print(f"entry screen: {df[id_col].nunique():,} -> {out[id_col].nunique():,} units, "
              f"{len(df):,} -> {len(out):,} rows")
    return out.sort_values([id_col, "date"]).reset_index(drop=True)
