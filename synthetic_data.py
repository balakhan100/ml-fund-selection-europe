"""
Write a small synthetic data set in exactly the raw formats the pipeline expects
(wide Morningstar exports, Ken French text files, Bundesbank/ECB csv, LSEG FX xlsx).
Used by the smoke test; it also documents the expected input formats.

    python tests/synthetic_data.py <target_dir>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fundml import config as C  # noqa: E402


def _write_money_market_csv(path, dates, values):
    lines = ["date,value"] + [f"meta{i},x" for i in range(C.MONEY_MARKET_CSV_SKIPROWS)]
    lines += [f"{d:%Y-%m},{v:.3f}" for d, v in zip(dates, values)]
    path.write_text("\n".join(lines) + "\n")


def make(target: Path, n_funds: int = 90, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------ factors (USD, in %)
    fdates = pd.period_range("1990-07", "2025-11", freq="M")
    n = len(fdates)
    ff = pd.DataFrame({"date": fdates.strftime("%Y%m")})
    ff["Mkt-RF"] = rng.normal(0.5, 5.0, n)
    for c in ["SMB", "HML", "RMW", "CMA"]:
        ff[c] = rng.normal(0.1, 2.0, n)
    ff["RF"] = np.clip(rng.normal(0.2, 0.1, n), 0, None)
    ff.round(2).to_csv(target / C.FILES["ff5"], sep=" ", index=False)
    ff3 = ff[["date", "Mkt-RF", "SMB", "HML", "RF"]].copy()
    ff3["SMB"] += rng.normal(0, 0.3, n)
    ff3["HML"] += rng.normal(0, 0.3, n)
    ff3.round(2).to_csv(target / C.FILES["ff3"], sep=" ", index=False)
    pd.DataFrame({"date": ff["date"], "WML": rng.normal(0.6, 3.5, n)}).round(2).to_csv(
        target / C.FILES["mom"], sep=" ", index=False)

    # ------------------------------------------------ FX (USD per EUR, month-start obs)
    mdates = pd.date_range("1999-01-01", "2025-11-01", freq="MS")
    rate = 1.17 * np.exp(np.cumsum(rng.normal(0, 0.025, len(mdates))))
    pd.DataFrame({"date": mdates, C.FX_COLUMN: [f"{v:.4f}".replace(".", ",") for v in rate]}
                 ).to_excel(target / C.FILES["fx"], index=False)

    # ------------------------------------------------ money market (% p.a.)
    eur_rate = np.clip(3 + np.cumsum(rng.normal(0, 0.15, len(mdates))), -0.6, 5)
    _write_money_market_csv(target / C.FILES["euribor"], mdates, eur_rate + 0.1)
    eon = mdates[mdates <= "2021-12-01"]
    pd.DataFrame({"date": eon, "value": eur_rate[: len(eon)]}).to_excel(
        target / C.FILES["eonia"], index=False)
    est = mdates[mdates >= "2019-10-01"]
    _write_money_market_csv(target / C.FILES["estr"], est, eur_rate[-len(est):] - 0.05)

    # ------------------------------------------------ fund returns from the EUR factors
    from fundml.factors import build_factor_panel
    fac = build_factor_panel(target)
    months = pd.period_range("1999-01", "2025-12", freq="M")
    cats = ["EAA Fund Europe Large-Cap Blend Equity", "EAA Fund Germany Equity",
            "EAA Fund Global Large-Cap Blend Equity"]
    areas = ["Europe", "Germany", "Euroland", "Global"]

    static, wide_ret, wide_na, wide_cat, wide_er = [], {}, {}, {}, {}
    sec = 0
    for f in range(n_funds):
        start = int(rng.integers(0, 150))
        end = int(min(len(months), start + rng.integers(60, 330)))
        skill = rng.normal(0, 0.0015)
        beta = rng.normal(1.0, 0.15)
        tilt = rng.normal(0, 0.3, 5)
        passive = f < 4
        area = areas[3] if f % 23 == 5 else areas[f % 3]
        cat = cats[2] if f % 19 == 3 else cats[f % 2]
        n_cls = int(rng.integers(1, 4))
        base_tna = rng.uniform(2e6, 80e6)
        fund_ret = np.full(len(months), np.nan)
        for j in range(start, end):
            d = months[j].to_timestamp() + pd.offsets.MonthEnd(0)
            if d in fac.index:
                fr = fac.loc[d]
                fund_ret[j] = (fr["rf_eur"] + beta * fr["mkt-rf_eur"] + skill
                               + tilt @ fr[["smb_5ff_eur", "hml_5ff_eur", "rmw_5ff_eur",
                                            "cma_5ff_eur", "wml_eur"]].to_numpy()
                               + rng.normal(0, 0.012))
            else:
                fund_ret[j] = rng.normal(0.006, 0.04)
        for k in range(n_cls):
            sec += 1
            sid = f"F0000{sec:05d}"
            er = rng.uniform(0.4, 2.0) - 0.3 * k
            r = fund_ret - er / 1200 + rng.normal(0, 0.0005, len(months))
            tna = np.full(len(months), np.nan)
            level = base_tna * (0.3 if k else 1.0)
            for j in range(start, end):
                level *= (1 + r[j]) * (1 + rng.normal(0.003 + 5 * skill, 0.03))
                tna[j] = level
            inc = months[start] - int(rng.integers(0, 40))
            static.append({
                "SecId": sid, "FundId": f"FS{f:05d}",
                "Name": ("MSCI Europe Index Fund" if passive else f"Active Fund {f}") + f" Class {k}",
                "Institutional": "Yes" if k == 1 else "No",
                "Oldest Share Class": "Yes" if k == 0 else "No",
                "Turnover Ratio %": float(rng.choice([rng.uniform(5, 300), 59575.0], p=[0.97, 0.03])),
                "Manager Tenure (Average)": round(rng.uniform(1, 15), 2),
                "Asset Alloc Equity % (Long Rescaled)": 55.0 if f % 17 == 2 else rng.uniform(85, 100),
                "Inception Date": inc.to_timestamp(),
                "Domicile": "Luxembourg", "Investment Area": area,
                "Management Approach - Active": "Yes",
            })
            wide_ret[sid] = np.round(r * 100, 4)
            wide_na[sid] = tna
            wide_cat[sid] = np.where(np.isfinite(tna), cat, None)
            er_years = {y: (er if rng.random() > 0.1 else np.nan) for y in range(1999, 2026)}
            wide_er[sid] = er_years

    ids = list(wide_ret)
    labels = [m.strftime("%Y-%m") for m in months]

    def wide(prefix, data):
        df = pd.DataFrame(np.array([data[i] for i in ids], dtype=object),
                          columns=[f"{prefix}{l}" for l in labels])
        df.insert(0, "SecId", ids)
        return df

    pd.DataFrame(static).to_excel(target / C.FILES["static"], index=False)
    wide("Monthly Return ", wide_ret).to_excel(target / C.FILES["returns"], index=False)
    wide("Net Assets - share class (Monthly) ", wide_na).to_excel(target / C.FILES["net_assets"], index=False)
    wide("Morningstar Category ", wide_cat).to_excel(target / C.FILES["category"], index=False)
    er = pd.DataFrame({f"Annual Report Net Expense Ratio Year{y}": [wide_er[i][y] for i in ids]
                       for y in range(1999, 2026)})
    er.insert(0, "SecId", ids)
    er.to_excel(target / C.FILES["expense_ratio"], index=False)
    pd.DataFrame({"category": cats[:2]}).to_excel(target / C.FILES["selected_categories"], index=False)
    return target


if __name__ == "__main__":
    print(make(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "tests" / "_synthetic_data"))
