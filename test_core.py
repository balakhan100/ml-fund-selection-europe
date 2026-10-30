"""Unit tests. Run with `pytest` or `python tests/test_core.py`."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fundml import alphas, data, experiment, panel  # noqa: E402
from fundml.alphas import ols_hac  # noqa: E402

FAC6 = ["mkt-rf_eur", "smb_5ff_eur", "hml_5ff_eur", "rmw_5ff_eur", "cma_5ff_eur", "wml_eur"]
FAC_ALL = FAC6 + ["smb_3ff_eur", "hml_3ff_eur"]


def _synthetic_panel(n_funds=25, seed=0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2000-01", "2012-12", freq="M")
    fac = pd.DataFrame({c: rng.normal(0, 0.03, len(months)) for c in FAC_ALL})
    fac["rf_eur"] = 0.001
    fac["date"] = months.to_timestamp() + pd.offsets.MonthEnd(0)
    rows = []
    for i in range(n_funds):
        s = rng.integers(0, 60)
        e = rng.integers(s + 25, len(months))
        idx = np.arange(s, e)
        idx = idx[rng.random(len(idx)) > 0.07]            # random gaps in the history
        b = rng.normal(1, 0.2)
        for j in idx:
            r = 0.001 + b * fac.at[j, "mkt-rf_eur"] + 0.3 * fac.at[j, "wml_eur"] + rng.normal(0, 0.01)
            rows.append((f"F{i}", fac.at[j, "date"], r + 0.001))
    df = pd.DataFrame(rows, columns=["secid", "date", "monthly_return"])
    return df.merge(fac, on="date")


def _reference_rolling_alpha(df, window=36, min_months=30):
    """Straight port of the thesis notebook's loop (pandas window mask, missing='drop')."""
    sets = alphas.factor_sets("eur")
    x = df.sort_values(["secid", "date"]).reset_index(drop=True).copy()
    x["per"] = x["date"].dt.to_period("M")
    x["excess"] = x["monthly_return"] - x["rf_eur"]
    out = {k: np.full(len(x), np.nan) for k in ["Monthly1Alpha", "Monthly3Alpha", "Monthly4Alpha",
                                                "Monthly6Alpha", "Monthly6Factor_R2",
                                                "alpha_tstat", "momentum_beta_tstat"]}
    for _, g in x.groupby("secid"):
        for i in g.index:
            cm = g.at[i, "per"]
            w = g[(g["per"] >= cm - window) & (g["per"] <= cm - 1)]
            if len(w) < min_months:
                continue
            for k, cols in sets.items():
                X = np.column_stack([np.ones(len(w)), w[cols].to_numpy()])
                b, _, t, r2, _ = ols_hac(X, w["excess"].to_numpy(), 3, True)
                out[f"Monthly{k[0]}Alpha"][i] = g.at[i, "excess"] - g.loc[i, cols].to_numpy(float) @ b[1:]
                if k == "6F":
                    out["Monthly6Factor_R2"][i], out["alpha_tstat"][i] = r2, t[0]
                    out["momentum_beta_tstat"][i] = t[6]
    return x.assign(**out)


def test_rolling_alpha_matches_reference_loop():
    df = _synthetic_panel()
    fast = alphas.rolling_alpha(df, "eur")
    ref = _reference_rolling_alpha(df)
    for c in ["Monthly1Alpha", "Monthly3Alpha", "Monthly4Alpha", "Monthly6Alpha",
              "Monthly6Factor_R2", "alpha_tstat", "momentum_beta_tstat"]:
        a, b = fast[c].to_numpy(), ref[c].to_numpy()
        assert (np.isnan(a) == np.isnan(b)).all(), c
        assert np.nanmax(np.abs(a - b)) < 1e-10, c
    assert np.isfinite(fast["Monthly6Alpha"]).sum() > 500


def test_hac_matches_statsmodels():
    try:
        import statsmodels.api as sm
    except ImportError:
        print("  (statsmodels not installed - skipped)")
        return
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(120), rng.normal(size=(120, 3))])
    y = X @ [0.1, 1, -0.5, 0.2] + rng.normal(size=120)
    for corr in (False, True):
        res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3, "use_correction": corr})
        _, se, t, r2, _ = ols_hac(X, y, 3, corr)
        assert np.allclose(se, res.bse) and np.allclose(t, res.tvalues) and np.isclose(r2, res.rsquared)


def test_ml_panel_lags_and_universe():
    ann = pd.DataFrame({
        "secid": ["A"] * 4 + ["B"] * 3,
        "year": [2010, 2011, 2013, 2014, 2010, 2011, 2012],
        "x": [1.0, 2, 3, 4, 5, 6, 7],
        "net_assets_mln": [10.0, 11, 12, 13, 20, 21, 22],
        "expense_ratio": [0.001] * 7,
        "Monthly6Alpha_ann": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        "n_months": [12, 12, 12, 12, 12, 12, 5],
    })
    p = panel.build_ml_panel(ann, ["x"]).set_index(["secid", "year"])
    # A has no 2012 row -> no 2013 row (no stale lag across the gap)
    assert ("A", 2013) not in p.index and ("A", 2014) in p.index
    # B 2012 has only 5 months: in the universe (investable), but no target
    assert ("B", 2012) in p.index and np.isnan(p.loc[("B", 2012), "Monthly6Alpha_ann"])
    # first year of each fund never appears with all-zero predictors
    assert ("A", 2010) not in p.index and ("B", 2010) not in p.index
    # raw lagged size is the previous year's value
    assert p.loc[("A", 2011), "net_assets_mln_lag1"] == 10.0
    # nothing beyond the last data year
    assert p.index.get_level_values("year").max() == 2014


def test_portfolio_reinvests_dead_fund():
    R = np.array([[0.01] * 12, [0.03, 0.03] + [np.nan] * 10])
    ret, end_w = experiment.portfolio_returns([0.5, 0.5], R)
    assert np.isclose(ret[0], 0.02) and np.isclose(ret[2], 0.01)   # month 3: all in fund 1
    assert np.isclose(end_w.sum(), 1) and end_w[1] == 0


def test_top_fund_weights():
    pred = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6.0])
    w = experiment.weights_in_top_funds(pred, 0.8)
    assert np.isclose(w.sum(), 1) and (w > 0).sum() == 2 and w[-1] > w[-2]
    w_neg = experiment.weights_in_top_funds(-np.arange(1, 11.0), 0.8)   # all negative
    assert np.isclose(w_neg.sum(), 1) and (w_neg > 0).sum() == 2


def test_gross_returns_carry_er_forward():
    m = pd.DataFrame({"secid": ["A"] * 4, "date": pd.date_range("2020-01-31", periods=4, freq="ME"),
                      "monthly_return": [0.01] * 4, "expense_ratio": [np.nan, 0.001, np.nan, np.nan]})
    g = panel.add_gross_returns(m)["gross_return"].to_numpy()
    assert np.allclose(g, 0.011)


def test_fund_aggregation():
    m = pd.DataFrame({
        "fundid": ["F"] * 4, "secid": ["a", "b", "a", "b"],
        "date": pd.to_datetime(["2020-01-31"] * 2 + ["2020-02-29"] * 2),
        "net_assets": [110.0e6, 22e6, 120e6, 20e6], "net_assets_l1": [100.0e6, 20e6, 110e6, 22e6],
        "monthly_return": [0.05, 0.04, 0.02, np.nan], "expense_ratio": [0.001, 0.002, 0.001, 0.002],
        "turnover_ratio_frac_w": [0.5] * 4, "class_age_months": [50, 20, 51, 21],
        "manager_tenure": [3.0] * 4, "morningstar_category": ["c"] * 4,
        "domicile": ["LU"] * 4, "investment_area": ["Europe"] * 4})
    f = data.aggregate_to_fund_level(m)
    assert np.isclose(f.loc[0, "monthly_return"], (100 * 0.05 + 20 * 0.04) / 120)
    assert np.isclose(f.loc[1, "monthly_return"], 0.02)              # only class a reported
    assert f.loc[0, "net_assets"] == 132e6 and f.loc[0, "class_age_months"] == 50
    assert (f["secid"] == "F").all()


def test_entry_screen():
    d = pd.DataFrame({"secid": "A", "date": pd.date_range("2010-01-31", periods=6, freq="ME"),
                      "net_assets_mln": [1, 6, 2, 3, 4, 5.0], "class_age_months": [33, 34, 35, 36, 37, 38]})
    s = data.apply_entry_screen(d, verbose=False)
    assert s["date"].min() == pd.Timestamp("2010-04-30") and len(s) == 3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
