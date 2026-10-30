"""Out-of-sample portfolio evaluation (DeMiguel et al., 2023, Tables 3-5 style)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C
from .alphas import ols_hac


def combine_rw_ew(results: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Monthly returns of all strategies with _RW / _EW suffixes, and average turnover
    (first test year excluded: no previous portfolio)."""
    rets = pd.concat([results["returns_rw"].add_suffix("_RW"),
                      results["returns_ew"].add_suffix("_EW")], axis=1)
    to = pd.concat([results["turnover_rw"].add_suffix("_RW"),
                    results["turnover_ew"].add_suffix("_EW")], axis=1)
    return rets, to.iloc[1:].mean()


def nw_alpha(excess: pd.Series, X: pd.DataFrame, lags: int = C.NW_LAGS_PORTFOLIO,
             min_obs: int = C.MIN_OBS_PORTFOLIO) -> dict:
    d = pd.concat([excess.rename("_y"), X], axis=1, join="inner").dropna()
    if len(d) < min_obs:
        return {"alpha": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "n": len(d), "resid": None}
    Xm = np.column_stack([np.ones(len(d)), d.drop(columns="_y").to_numpy(float)])
    b, se, t, _, u = ols_hac(Xm, d["_y"].to_numpy(float), lags)
    return {"alpha": b[0], "se": se[0], "t": t[0], "p": 2 * stats.norm.sf(abs(t[0])),
            "n": len(d), "resid": u}


def _stars(p):
    return "" if not np.isfinite(p) else "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def alpha_table(returns: pd.DataFrame, factors: pd.DataFrame, models=C.FACTOR_MODELS,
                rf_col: str = C.RF_COL, benchmark: str | None = None) -> pd.DataFrame:
    """Monthly alpha (%) with Newey-West SEs per strategy and factor model.
    With `benchmark`, every strategy is evaluated as the long-short return strategy - benchmark."""
    rows = []
    for s in returns.columns:
        if s == benchmark:
            continue
        r = returns[s] - returns[benchmark] if benchmark else returns[s] - factors[rf_col]
        row = {"strategy": s if not benchmark else f"{s} - {benchmark}"}
        for m, cols in models.items():
            res = nw_alpha(r, factors[cols])
            row.update({f"{m}_alpha_pct": res["alpha"] * 100, f"{m}_se_pct": res["se"] * 100,
                        f"{m}_t": res["t"], f"{m}_sig": _stars(res["p"]), f"{m}_n": res["n"]})
        rows.append(row)
    return pd.DataFrame(rows).set_index("strategy")


def _max_drawdown(r):
    w = (1 + r.dropna()).cumprod()
    return float(-(w / w.cummax() - 1).min()) if len(w) else np.nan


def risk_table(returns: pd.DataFrame, factors: pd.DataFrame, turnover: pd.Series | None = None,
               rf_col: str = C.RF_COL, ff_cols=None) -> pd.DataFrame:
    ff_cols = ff_cols or C.FACTOR_MODELS["FF5_MOM"]
    rows = []
    for s in returns.columns:
        d = pd.concat([returns[s].rename("r"), factors[rf_col].rename("rf")], axis=1,
                      join="inner").dropna()
        ex = d["r"] - d["rf"]
        sd = d["r"].std(ddof=1)
        down = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
        a = nw_alpha(ex, factors[ff_cols])
        ir = a["alpha"] / np.std(a["resid"], ddof=1) if a["resid"] is not None else np.nan
        rows.append({"strategy": s, "mean_excess_pct": ex.mean() * 100, "sd_pct": sd * 100,
                     "sharpe": ex.mean() / sd if sd > 0 else np.nan,
                     "sortino": ex.mean() / down if down > 0 else np.nan,
                     "info_ratio": ir, "max_drawdown_pct": _max_drawdown(d["r"]) * 100,
                     "VaR99_pct": -np.quantile(d["r"], 0.01) * 100,
                     "turnover": turnover.get(s, np.nan) if turnover is not None else np.nan,
                     "n": len(d)})
    return pd.DataFrame(rows).set_index("strategy")


def format_alpha_table(tbl: pd.DataFrame, models=C.FACTOR_MODELS) -> pd.DataFrame:
    """'0.123** (0.045)' strings for display / LaTeX."""
    out = pd.DataFrame(index=tbl.index)
    for m in models:
        out[m] = [f"{a:.3f}{s} ({e:.3f})" if np.isfinite(a) else "n/a"
                  for a, s, e in zip(tbl[f"{m}_alpha_pct"], tbl[f"{m}_sig"], tbl[f"{m}_se_pct"])]
    return out


def evaluate(results: dict, factors: pd.DataFrame, ols_strategy: str = "ols_top_10_EW",
             verbose: bool = True) -> dict:
    rets, to = combine_rw_ew(results)
    t3 = alpha_table(rets, factors)
    t4 = alpha_table(rets, factors, benchmark=ols_strategy) if ols_strategy in rets else None
    t5 = risk_table(rets, factors, to)
    ic = ic_summary(results)
    if verbose:
        print("Out-of-sample alpha (monthly %, Newey-West SE, 12 lags)")
        print(format_alpha_table(t3).to_string())
        if t4 is not None:
            print(f"\nAlpha relative to {ols_strategy}")
            print(format_alpha_table(t4).to_string())
        print("\nRisk and turnover")
        print(t5.round(3).to_string())
        print("\nSpearman IC by model")
        print(ic.round(4).to_string())
    return {"returns": rets, "turnover": to, "alpha": t3, "alpha_vs_ols": t4, "risk": t5, "ic": ic}


def ic_summary(results: dict) -> pd.DataFrame:
    ic = results["ic"].pivot_table(index="year", columns="model", values="ic")
    out = pd.DataFrame({"mean_ic": ic.mean(), "sd_ic": ic.std(ddof=1),
                        "t_stat": ic.mean() / (ic.std(ddof=1) / np.sqrt(ic.count())),
                        "share_positive": (ic > 0).mean(), "n_years": ic.count()})
    return out


def importance_summary(results: dict) -> pd.DataFrame:
    """Across test years: mean/sd of signed coefficients (linear models) or importances
    (trees), sign consistency and selection frequency."""
    imp = results["importances"]
    if imp.empty:
        return pd.DataFrame()
    g = imp.groupby(["model", "kind", "feature"])["value"]
    out = pd.DataFrame({
        "mean": g.mean(), "sd": g.std(ddof=1),
        "share_positive": g.apply(lambda v: (v > 0).mean()),
        "share_nonzero": g.apply(lambda v: (v.abs() > 1e-12).mean()),
        "n_years": g.count()}).reset_index()
    return out.sort_values(["model", "mean"], ascending=[True, False])


def capital_misallocation(results: dict, panel: pd.DataFrame, model: str = "ensemble",
                          id_col: str = "secid", size_col: str = "net_assets_mln_lag1",
                          er_col: str = "expense_ratio_lag1") -> pd.DataFrame:
    """Predicted-alpha deciles each test year: size and fees measured at formation (end of
    t-1), realised alpha in t; averaged over years."""
    p = results["predictions"].query("model == @model")
    p = p.merge(panel[[id_col, "year", size_col, er_col]], on=[id_col, "year"], how="left")
    p["decile"] = p.groupby("year")["y_hat"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    by_year = p.groupby(["year", "decile"]).agg(
        mean_size=(size_col, "mean"), median_size=(size_col, "median"),
        mean_er_annual_pct=(er_col, lambda s: s.mean() * 1200),
        realised_alpha_pct=("y_true", lambda s: s.mean() * 100),
        n=("y_hat", "size"))
    return by_year.groupby("decile").mean()


def cumulative_alpha(returns: pd.DataFrame, factors: pd.DataFrame, factor_cols: list[str],
                     rf_col: str = C.RF_COL) -> pd.DataFrame:
    """Monthly realised alpha r_ex - beta' f with full-sample betas (DeMiguel et al., fn. 27),
    cumulated."""
    out = {}
    for s in returns.columns:
        d = pd.concat([returns[s].rename("_r"), factors[[rf_col] + factor_cols]], axis=1,
                      join="inner").dropna()
        if len(d) < 12:
            continue
        y = (d["_r"] - d[rf_col]).to_numpy()
        F = d[factor_cols].to_numpy()
        b = np.linalg.lstsq(np.column_stack([np.ones(len(d)), F]), y, rcond=None)[0]
        out[s] = pd.Series(y - F @ b[1:], index=d.index)
    return pd.DataFrame(out).fillna(0.0).cumsum()


def summary_across_specs(results: dict[str, dict], factors: pd.DataFrame,
                         strategies=("ols_top_10_EW", "ensemble_top_10_EW", "ridge_top_10_EW",
                                     "rf_top_10_EW"),
                         factor_model: str = "FF5_MOM") -> pd.DataFrame:
    """One row per specification: FF5+MOM alpha (monthly %) and t-stat of selected
    strategies, plus mean IC of OLS and the ensemble."""
    rows = []
    cols = C.FACTOR_MODELS[factor_model]
    for spec, res in results.items():
        rets, _ = combine_rw_ew(res)
        row = {"spec": spec}
        for s in strategies:
            if s not in rets:
                continue
            a = nw_alpha(rets[s] - factors[C.RF_COL], factors[cols])
            row[f"{s}_alpha_pct"] = a["alpha"] * 100
            row[f"{s}_t"] = a["t"]
        ic = res["ic"].groupby("model")["ic"].mean()
        row["mean_ic_ols"] = ic.get("ols", np.nan)
        row["mean_ic_ensemble"] = ic.get("ensemble", np.nan)
        rows.append(row)
    return pd.DataFrame(rows).set_index("spec")
