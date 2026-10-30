"""Rolling realized alphas (1F/3F/4F/6F) and the EUR-vs-USD factor comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import config as C

TSTAT_NAMES = ["alpha_tstat", "market_beta_tstat", "size_beta_tstat", "value_beta_tstat",
               "profitability_tstat", "investment_tstat", "momentum_beta_tstat"]
ALPHA_COLS = ["Monthly1Alpha", "Monthly3Alpha", "Monthly4Alpha", "Monthly6Alpha"]
ALPHA_OUTPUT_COLS = ALPHA_COLS + ["Monthly6Factor_R2"] + TSTAT_NAMES


def factor_sets(currency: str) -> dict[str, list[str]]:
    c = currency.lower()
    return {
        "1F": [f"mkt-rf_{c}"],
        "3F": [f"mkt-rf_{c}", f"smb_3ff_{c}", f"hml_3ff_{c}"],
        "4F": [f"mkt-rf_{c}", f"smb_3ff_{c}", f"hml_3ff_{c}", f"wml_{c}"],
        "6F": [f"mkt-rf_{c}", f"smb_5ff_{c}", f"hml_5ff_{c}", f"rmw_5ff_{c}",
               f"cma_5ff_{c}", f"wml_{c}"],
    }


def ols_hac(X: np.ndarray, y: np.ndarray, maxlags: int, use_correction: bool = False):
    """
    OLS (X includes the constant) with Newey-West/HAC covariance, same formula as
    statsmodels OLS.fit(cov_type="HAC", cov_kwds={"maxlags": L, "use_correction": ...}):
    Bartlett weights, S = sum_l w_l (G_l + G_l'), cov = (X'X)^-1 S (X'X)^-1 [* n/(n-k)].
    Returns params, standard errors, t-values, R^2, residuals.
    """
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    b = xtx_inv @ (X.T @ y)
    u = y - X @ b
    xu = X * u[:, None]
    S = xu.T @ xu
    for lag in range(1, maxlags + 1):
        s = xu[lag:].T @ xu[:-lag]
        S += (1.0 - lag / (maxlags + 1.0)) * (s + s.T)
    cov = xtx_inv @ S @ xtx_inv
    if use_correction:
        cov *= n / float(n - k)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t = b / se
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (u @ u) / tss if tss > 0 else np.nan
    return b, se, t, r2, u


def rolling_alpha(
    df: pd.DataFrame,
    currency: str = "eur",
    id_col: str = "secid",
    date_col: str = "date",
    ret_col: str = "monthly_return",
    window_months: int = C.ALPHA_WINDOW_MONTHS,
    min_months: int = C.ALPHA_MIN_MONTHS,
    hac_lags: int = C.ALPHA_HAC_LAGS,
    models=("1F", "3F", "4F", "6F"),
) -> pd.DataFrame:
    """
    Realized alpha in month m: alpha_{i,m} = excess_{i,m} - beta_hat' f_m, with betas from
    an OLS (with intercept) on the calendar months [m - window, m - 1]; the intercept is not
    subtracted (DeMiguel et al., 2023). A window needs >= min_months observations.
    For 6F the R^2 and HAC t-statistics (maxlags=hac_lags, small-sample correction) of the
    window regression are stored as fund characteristics.

    Point estimates come from cumulative X'X / X'y sums (exact OLS for every window at once);
    only the 6F t-statistics need a per-window loop.
    """
    fs = factor_sets(currency)
    x = df.sort_values([id_col, date_col]).reset_index(drop=True).copy()
    x["excess"] = (pd.to_numeric(x[ret_col], errors="coerce")
                   - pd.to_numeric(x[f"rf_{currency.lower()}"], errors="coerce"))
    per = pd.to_datetime(x[date_col]).dt.to_period("M")
    mon = (per.dt.year * 12 + per.dt.month).to_numpy(np.int64)

    out = {f"Monthly{m[0]}Alpha": np.full(len(x), np.nan) for m in models}
    if "6F" in models:
        out["Monthly6Factor_R2"] = np.full(len(x), np.nan)
        for nme in TSTAT_NAMES:
            out[nme] = np.full(len(x), np.nan)

    all_f = sorted({f for m in models for f in fs[m]})
    F = x[all_f].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    fpos = {f: i for i, f in enumerate(all_f)}
    exc = x["excess"].to_numpy(float)
    ids = x[id_col].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    bounds = np.r_[starts, len(x)]

    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = np.arange(a, b)
        pos = mon[rows] - mon[a]
        T = pos[-1] + 1
        y_cal = np.full(T, np.nan)
        y_cal[pos] = exc[rows]
        f_cal = np.full((T, F.shape[1]), np.nan)
        f_cal[pos] = F[rows]
        present = np.zeros(T, bool)
        present[pos] = True
        c_n = np.r_[0, np.cumsum(present)]

        for mdl in models:
            idx = [fpos[f] for f in fs[mdl]]
            Xc = np.column_stack([np.ones(T), f_cal[:, idx]])
            ok = present & np.isfinite(y_cal) & np.isfinite(Xc).all(1)
            Xz = np.where(ok[:, None], Xc, 0.0)
            yz = np.where(ok, y_cal, 0.0)
            k = Xz.shape[1]
            c_xx = np.concatenate([np.zeros((1, k, k)), np.cumsum(Xz[:, :, None] * Xz[:, None, :], 0)])
            c_xy = np.concatenate([np.zeros((1, k)), np.cumsum(Xz * yz[:, None], 0)])
            c_ok = np.r_[0, np.cumsum(ok)]
            name = f"Monthly{mdl[0]}Alpha"
            for r_i, p in zip(rows, pos):
                lo, hi = max(p - window_months, 0), p
                if c_n[hi] - c_n[lo] < min_months or c_ok[hi] - c_ok[lo] <= k:
                    continue
                beta = np.linalg.lstsq(c_xx[hi] - c_xx[lo], c_xy[hi] - c_xy[lo], rcond=None)[0]
                out[name][r_i] = y_cal[p] - f_cal[p, idx] @ beta[1:]
                if mdl == "6F":
                    w = ok[lo:hi]
                    _, _, t, r2, _ = ols_hac(Xc[lo:hi][w], y_cal[lo:hi][w], hac_lags, True)
                    out["Monthly6Factor_R2"][r_i] = r2
                    for nme, tv in zip(TSTAT_NAMES, t):
                        out[nme][r_i] = tv

    for k_, v in out.items():
        x[k_] = v
    return x


# ---------------------------------------------------------------- EUR vs USD
def newey_west_mean(x, lags: int = 12):
    """Mean of a time series with a Newey-West t-statistic."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    X = np.ones((len(x), 1))
    b, se, t, _, _ = ols_hac(X, x, lags)
    p = 2 * stats.norm.sf(abs(t[0]))
    return float(b[0]), float(t[0]), float(p), len(x)


def compare_currencies(alphas_eur: pd.DataFrame, alphas_usd: pd.DataFrame,
                       id_col: str = "secid", date_col: str = "date",
                       nw_lags: int = 12) -> pd.DataFrame:
    """
    EUR-adjusted minus USD-factor alphas per model. Inference uses the monthly
    cross-sectional average difference (one observation per month) with Newey-West
    standard errors; fund-month observations are strongly cross-correlated, so
    observation-level paired t-tests overstate significance.
    """
    cols = ALPHA_COLS + ["Monthly6Factor_R2"]
    m = alphas_eur[[id_col, date_col] + cols].merge(
        alphas_usd[[id_col, date_col] + cols], on=[id_col, date_col],
        suffixes=("_eur", "_usd"), validate="one_to_one")
    rows = []
    for c in ALPHA_COLS:
        p = m.dropna(subset=[f"{c}_eur", f"{c}_usd"])
        d = p[f"{c}_eur"] - p[f"{c}_usd"]
        ts = d.groupby(p[date_col]).mean()
        mean, t, pval, n_months = newey_west_mean(ts.values, nw_lags)
        rows.append({
            "model": c.replace("Monthly", "").replace("Alpha", "F"),
            "mean_alpha_usd_pct": p[f"{c}_usd"].mean() * 100,
            "mean_alpha_eur_pct": p[f"{c}_eur"].mean() * 100,
            "median_alpha_usd_pct": p[f"{c}_usd"].median() * 100,
            "median_alpha_eur_pct": p[f"{c}_eur"].median() * 100,
            "diff_monthly_pct": mean * 100,
            "diff_annual_pct": mean * 1200,
            "nw_t": t, "p_value": pval,
            "share_eur_below_usd": (d < 0).mean(),
            "n_obs": len(p), "n_months": n_months,
        })
    out = pd.DataFrame(rows).set_index("model")
    r2 = m[["Monthly6Factor_R2_eur", "Monthly6Factor_R2_usd"]].mean()
    out.attrs["mean_R2_6F"] = r2.to_dict()
    return out
