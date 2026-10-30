"""Expanding-window out-of-sample fund selection (DeMiguel et al., 2023 design)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import BayesianRidge, ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from . import config as C

try:
    from xgboost import XGBRegressor
except ImportError:          # pragma: no cover
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except ImportError:          # pragma: no cover
    LGBMRegressor = None


# ================================================================ scoring & CV
def spearman_ic(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    # fewer than 30 funds, or a constant prediction (e.g. Lasso shrunk to zero): no ranking
    if ok.sum() < 30 or np.unique(y_pred[ok]).size < 2 or np.unique(y_true[ok]).size < 2:
        return 0.0
    r = spearmanr(y_true[ok], y_pred[ok])[0]
    return float(r) if np.isfinite(r) else 0.0


SCORERS = {"ic": make_scorer(spearman_ic), "rmse": "neg_root_mean_squared_error"}


class ExpandingWindowCV:
    """Validate on each of the last n_splits training years, training on all earlier years."""

    def __init__(self, years, n_splits: int = C.CV_FOLDS):
        self.years = np.asarray(years)
        self.n_splits = n_splits

    def split(self, X=None, y=None, groups=None):
        for vy in sorted(set(self.years))[-self.n_splits:]:
            tr, va = np.flatnonzero(self.years < vy), np.flatnonzero(self.years == vy)
            if len(tr) and len(va):
                yield tr, va

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def make_cv(kind: str, years):
    if kind == "expanding":
        return ExpandingWindowCV(years, C.CV_FOLDS)
    if kind == "kfold":
        return KFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_SEED)
    raise ValueError(f"unknown cv '{kind}'")


# ================================================================ models
def get_model_specs(n_predictors: int, quick: bool = C.QUICK) -> list[tuple]:
    """(name, pipeline, param_grid or None). Grids are fixed ex ante for every specification."""
    def pipe(est):
        return Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                         ("model", est)])

    s, p = C.RANDOM_SEED, n_predictors
    alphas_l1 = [1e-6, 1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.5, 1.0]
    specs = [
        ("ols", pipe(LinearRegression()), None),
        ("ridge", pipe(Ridge(random_state=s)),
         {"model__alpha": [1e-4, 1e-3, 0.01, 0.1, 1.0, 10, 100, 1000, 10000]}),
        ("lasso", pipe(Lasso(random_state=s, max_iter=100_000, tol=1e-4)),
         {"model__alpha": alphas_l1}),
        ("bayridge", pipe(BayesianRidge()), None),
        ("enet", pipe(ElasticNet(random_state=s, max_iter=100_000, tol=1e-4)),
         {"model__alpha": alphas_l1,
          "model__l1_ratio": [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 1.0]}),
    ]
    boost_grid = {"model__n_estimators": [200, 500, 1000], "model__learning_rate": [0.05, 0.1]}
    if XGBRegressor is not None:
        specs.append(("xgb", pipe(XGBRegressor(
            objective="reg:squarederror", random_state=s, tree_method="hist", n_jobs=-1,
            verbosity=0, max_depth=1, min_child_weight=20)),
            {**boost_grid, "model__reg_lambda": [1, 10, 100]}))
    else:
        specs.append(("xgb", pipe(GradientBoostingRegressor(
            random_state=s, max_depth=1, min_samples_leaf=20)), boost_grid))
    if LGBMRegressor is not None:
        specs.append(("lgbm", pipe(LGBMRegressor(
            random_state=s, n_jobs=-1, verbose=-1, force_col_wise=True, num_leaves=2,
            min_child_samples=20)), {**boost_grid, "model__reg_lambda": [1, 10, 100]}))
    specs.append(("rf", pipe(RandomForestRegressor(n_estimators=1000, random_state=s, n_jobs=-1)),
                  {"model__max_features": sorted({max(1, int(np.sqrt(p))), max(1, p // 2), p}),
                   "model__max_depth": [1, 2, 3],
                   "model__min_samples_leaf": [20, 50, 100]}))

    if quick:   # smoke tests only: grids too small for CV/scoring choices to matter
        small = []
        for name, pl, grid in specs:
            if name in ("xgb", "lgbm"):
                pl.set_params(model__n_estimators=50)
                grid = None
            elif name == "rf":
                pl.set_params(model__n_estimators=30, model__max_depth=2, model__min_samples_leaf=20)
                grid = None
            elif grid:
                grid = {k: v[:2] for k, v in grid.items()}
            small.append((name, pl, grid))
        specs = small
    return specs


def _fit(pipeline, grid, X, y, scoring, cv):
    if grid is None:
        est = clone(pipeline).fit(X, y)
        return est, {}, np.nan
    gs = GridSearchCV(clone(pipeline), grid, scoring=scoring, cv=cv, n_jobs=-1,
                      error_score=np.nan)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_, gs.best_score_


def _importance(est, predictors):
    m = est.named_steps["model"]
    if hasattr(m, "coef_"):
        return "coef", np.asarray(m.coef_, float).ravel()
    if hasattr(m, "feature_importances_"):
        return "importance", np.asarray(m.feature_importances_, float).ravel()
    return None, None


# ================================================================ portfolios
def weights_in_top_funds(pred, thres: float) -> np.ndarray:
    """Prediction-weighted portfolio of funds above the `thres` quantile of predicted alpha
    (port of DeMiguel et al.'s R code: predictions rescaled to [0, 1] when the cutoff is < 0)."""
    x = np.asarray(pred, float).ravel()
    w = np.zeros(len(x))
    ok = np.isfinite(x)
    if not ok.any():
        return w
    cutoff = np.nanquantile(np.where(ok, x, np.nan), thres)
    if cutoff < 0:
        lo, hi = np.nanmin(x[ok]), np.nanmax(x[ok])
        if hi == lo:
            return w
        x = (x - lo) / (hi - lo)
        cutoff = np.nanquantile(x[ok], thres)
    sel = ok & (x > cutoff)
    if not sel.any():
        return w
    vals = np.where(sel, x, 0.0)
    if vals.sum() > 0:
        return vals / vals.sum()
    w[sel] = 1.0 / sel.sum()
    return w


def equal_weights_like(w: np.ndarray) -> np.ndarray:
    sel = np.asarray(w) > 0
    out = np.zeros(len(sel))
    if sel.any():
        out[sel] = 1.0 / sel.sum()
    return out


def portfolio_returns(w0, R) -> tuple[np.ndarray, np.ndarray]:
    """
    Buy-and-hold over the 12 months of R (funds x 12), weights drifting with returns.
    A fund without a return in a month (merged/liquidated) is sold and its weight
    reinvested pro rata in the remaining holdings. Returns monthly portfolio returns and
    the drifted end-of-year weights (used for turnover).
    """
    w = np.asarray(w0, float).ravel().copy()
    R = np.asarray(R, float)
    port = np.zeros(R.shape[1])
    for r in range(R.shape[1]):
        alive = np.isfinite(R[:, r]) & (w > 0)
        if not alive.any():
            w = np.zeros_like(w)
            continue                          # nothing left: cash at 0
        w = np.where(alive, w, 0.0)
        w /= w.sum()
        rr = np.where(alive, R[:, r], 0.0)
        port[r] = w @ rr
        w = w * (1 + rr)
        w = w / w.sum() if w.sum() > 0 else w
    return port, w


def _turnover(new: dict, prev_end: dict | None) -> float:
    if prev_end is None:
        return np.nan
    ids = set(new) | set(prev_end)
    return float(sum(abs(new.get(i, 0.0) - prev_end.get(i, 0.0)) for i in ids))


def returns_wide(monthly: pd.DataFrame, ret_col: str, id_col: str = "secid") -> pd.DataFrame:
    """Monthly returns as (fund x 'YYYYMM') matrix."""
    x = monthly[[id_col, "date", ret_col]].copy()
    x[id_col] = x[id_col].astype(str)
    x["ym"] = pd.to_datetime(x["date"]).dt.strftime("%Y%m")
    return x.pivot_table(index=id_col, columns="ym", values=ret_col, aggfunc="first")


def year_matrix(wide: pd.DataFrame, ids: list[str], year: int):
    months = [f"{year}{m:02d}" for m in range(1, 13)]
    return wide.reindex(index=ids, columns=months).to_numpy(float), months


# ================================================================ experiment
def run_oos_experiment(
    panel: pd.DataFrame,
    monthly: pd.DataFrame,
    predictors: list[str],
    target: str = C.TARGET,
    ret_col: str = "monthly_return",
    scoring: str = "ic",
    cv: str = "expanding",
    panel_length: int = C.PANEL_LENGTH,
    thresholds=tuple(C.THRESHOLDS),
    id_col: str = "secid",
    model_specs: list | None = None,
    ensemble: bool = True,
    verbose: bool = True,
) -> dict:
    """
    For each test year t: train on all rows with year < t (non-missing target), tune with
    `cv` and `scoring`, predict alpha for every fund in the year-t universe, and hold the
    top-quantile portfolios (prediction-weighted 'RW' and equal-weighted 'EW') for 12 months.
    The ensemble is the equal-weight average of all regularised / ML models (no selection
    on performance).

    Returns dict with: returns_rw, returns_ew (monthly, columns '<model>_top_<pct>'),
    turnover_rw, turnover_ew (per year), ic (per model-year), predictions, importances.
    """
    df = panel.copy()
    df[id_col] = df[id_col].astype(str)
    years = sorted(df["year"].unique())
    if len(years) <= panel_length:
        raise ValueError(f"need more than {panel_length} years, have {len(years)}")
    specs = model_specs or get_model_specs(len(predictors))
    names = [s[0] for s in specs] + (["ensemble"] if ensemble else [])
    wide = returns_wide(monthly, ret_col, id_col)

    rets_rw, rets_ew, dates, ic_rows, pred_rows, imp_rows = [], [], [], [], [], []
    to_rw, to_ew = [], []
    prev_end_rw, prev_end_ew = {}, {}
    t_start = time.time()

    for test_year in years[panel_length:]:
        tr = df.loc[df["year"] < test_year].dropna(subset=[target])
        te = df.loc[df["year"] == test_year]
        ids = te[id_col].tolist()
        R, months = year_matrix(wide, ids, test_year)
        if not np.isfinite(R).any():
            if verbose:
                print(f"{test_year}: no returns, skipped")
            continue
        X_tr = tr[predictors].to_numpy(np.float64)
        y_tr = tr[target].to_numpy(np.float64)
        X_te = te[predictors].to_numpy(np.float64)
        y_te = te[target].to_numpy(np.float64)
        cv_obj = make_cv(cv, tr["year"].to_numpy())

        preds = {}
        for name, pl, grid in specs:
            est, best, _ = _fit(pl, grid, X_tr, y_tr, SCORERS[scoring], cv_obj)
            preds[name] = np.nan_to_num(est.predict(X_te), nan=0.0)
            kind, vals = _importance(est, predictors)
            if kind is not None:
                imp_rows += [{"year": test_year, "model": name, "kind": kind,
                              "feature": f, "value": v} for f, v in zip(predictors, vals)]
            if verbose and best:
                print(f"  {test_year} {name}: {best}")
        if ensemble:
            ml = [n for n in preds if n != "ols"]
            preds["ensemble"] = np.mean([preds[n] for n in ml], axis=0)

        yr_rw = np.zeros((12, len(names) * len(thresholds)))
        yr_ew = np.zeros_like(yr_rw)
        yr_to_rw, yr_to_ew = {}, {}
        col = 0
        for name in names:
            ic_rows.append({"year": test_year, "model": name,
                            "ic": spearman_ic(y_te, preds[name]),
                            "n_funds": len(ids), "n_with_target": int(np.isfinite(y_te).sum())})
            pred_rows.append(pd.DataFrame({"year": test_year, "model": name, id_col: ids,
                                           "y_hat": preds[name], "y_true": y_te}))
            for th in thresholds:
                key = f"{name}_top_{int(round((1 - th) * 100))}"
                w_rw = weights_in_top_funds(preds[name], th)
                w_ew = equal_weights_like(w_rw)
                r_rw, end_rw = portfolio_returns(w_rw, R)
                r_ew, end_ew = portfolio_returns(w_ew, R)
                yr_rw[:, col], yr_ew[:, col] = r_rw, r_ew
                yr_to_rw[key] = _turnover(dict(zip(ids, w_rw)), prev_end_rw.get(key))
                yr_to_ew[key] = _turnover(dict(zip(ids, w_ew)), prev_end_ew.get(key))
                prev_end_rw[key] = dict(zip(ids, end_rw))
                prev_end_ew[key] = dict(zip(ids, end_ew))
                col += 1
        rets_rw.append(yr_rw)
        rets_ew.append(yr_ew)
        dates += months
        to_rw.append(pd.Series(yr_to_rw, name=test_year))
        to_ew.append(pd.Series(yr_to_ew, name=test_year))
        if verbose:
            ics = ", ".join(f"{r['model']}={r['ic']:.3f}" for r in ic_rows[-len(names):])
            print(f"{test_year}: train {len(tr):,} | test {len(ids):,} | IC {ics} "
                  f"[{(time.time() - t_start) / 60:.1f} min]")

    cols = [f"{n}_top_{int(round((1 - th) * 100))}" for n in names for th in thresholds]
    idx = pd.to_datetime(dates, format="%Y%m") + pd.offsets.MonthEnd(0)
    return {
        "returns_rw": pd.DataFrame(np.vstack(rets_rw), index=idx, columns=cols),
        "returns_ew": pd.DataFrame(np.vstack(rets_ew), index=idx, columns=cols),
        "turnover_rw": pd.DataFrame(to_rw)[cols],
        "turnover_ew": pd.DataFrame(to_ew)[cols],
        "ic": pd.DataFrame(ic_rows),
        "predictions": pd.concat(pred_rows, ignore_index=True),
        "importances": pd.DataFrame(imp_rows),
        "settings": {"scoring": scoring, "cv": cv, "ret_col": ret_col,
                     "predictors": list(predictors), "panel_length": panel_length},
    }


def run_benchmarks(panel: pd.DataFrame, monthly: pd.DataFrame, ret_col: str = "monthly_return",
                   panel_length: int = C.PANEL_LENGTH, id_col: str = "secid",
                   size_col: str = "net_assets_mln_lag1") -> pd.DataFrame:
    """Equal-weighted and asset-weighted (t-1 TNA) portfolios of the whole year-t universe."""
    df = panel.copy()
    df[id_col] = df[id_col].astype(str)
    wide = returns_wide(monthly, ret_col, id_col)
    out, dates = [], []
    for test_year in sorted(df["year"].unique())[panel_length:]:
        te = df.loc[df["year"] == test_year]
        ids = te[id_col].tolist()
        R, months = year_matrix(wide, ids, test_year)
        if not np.isfinite(R).any():
            continue
        size = te[size_col].to_numpy(float)
        size = np.where(np.isfinite(size) & (size > 0), size, 0.0)
        ew, _ = portfolio_returns(np.full(len(ids), 1 / len(ids)), R)
        aw, _ = portfolio_returns(size / size.sum(), R)
        out.append(np.column_stack([ew, aw]))
        dates += months
    idx = pd.to_datetime(dates, format="%Y%m") + pd.offsets.MonthEnd(0)
    return pd.DataFrame(np.vstack(out), index=idx, columns=["EW_all_funds", "AW_all_funds"])
