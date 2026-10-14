"""Valuation Anchor Engine - generic demo on SYNTHETIC data.

Pipeline: multiple -> log -> rolling percentile -> buckets -> forward returns
-> Spearman IC, HAC (Newey-West) t-stat, FDR across tests, AR(1) half-life,
and a simple expanding walk-forward IC check.

No client data is used. Replace `make_synthetic_panel` with your own loader
(date, ticker, price, one column per valuation multiple).
"""
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests

CONFIG = {
    "tickers": ["AAA", "BBB", "CCC", "DDD"],
    "metrics": ["pe", "ev_rev", "ev_ebitda"],
    "signal_window": 252,                         # rolling window (trading days)
    "bucket_cuts": [0, .10, .25, .75, .90, 1.0],  # D1 cheap ... D5 expensive
    "horizons": [30, 60, 90, 120],
    "min_obs": 60,
}


def make_synthetic_panel(n_days=1000, seed=7):
    """Random-walk price; multiples = price / slowly-moving fundamentals."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    frames = []
    for t in CONFIG["tickers"]:
        ret = rng.normal(0.0004, 0.017, n_days)
        price = 100 * np.exp(np.cumsum(ret))
        row = {"date": dates, "ticker": t, "price": price}
        for m in CONFIG["metrics"]:
            fundamentals = np.exp(np.cumsum(rng.normal(0.0002, 0.004, n_days)))
            row[m] = price / fundamentals * rng.uniform(0.15, 0.3)
        frames.append(pd.DataFrame(row))
    return pd.concat(frames, ignore_index=True)


def add_signals(df, metric):
    """Rolling percentile of log multiple. Uses only data up to each date."""
    out = []
    for t, g in df.groupby("ticker"):
        g = g.sort_values("date").copy()
        x = np.log(g[metric].where(g[metric] > 0))
        g["pctile"] = x.rolling(CONFIG["signal_window"], min_periods=CONFIG["signal_window"]).apply(
            lambda w: (w[:-1] < w[-1]).mean() + 0.5 * (w[:-1] == w[-1]).mean(), raw=True)
        g["bucket"] = pd.cut(g["pctile"], CONFIG["bucket_cuts"], labels=["D1", "D2", "D3", "D4", "D5"],
                             include_lowest=True)
        for h in CONFIG["horizons"]:
            g[f"fwd_{h}"] = g["price"].shift(-h) / g["price"] - 1  # forward-looking target only
        out.append(g)
    return pd.concat(out)


def hac_slope_t(y, x, h):
    """OLS of forward return on percentile with Newey-West errors (lag = horizon)."""
    X = sm.add_constant(x.values)
    res = sm.OLS(y.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": h})
    return res.params[1], res.tvalues[1], res.pvalues[1]


def evaluate(df):
    rows = []
    for metric in CONFIG["metrics"]:
        sig = add_signals(df, metric)
        for t, g in sig.groupby("ticker"):
            for h in CONFIG["horizons"]:
                d = g.dropna(subset=["pctile", f"fwd_{h}"])
                if len(d) < CONFIG["min_obs"]:
                    continue
                ic = spearmanr(d["pctile"], d[f"fwd_{h}"])[0]
                slope, tstat, p = hac_slope_t(d[f"fwd_{h}"], d["pctile"], h)
                cheap = d.loc[d["bucket"] == "D1", f"fwd_{h}"]
                exp_ = d.loc[d["bucket"] == "D5", f"fwd_{h}"]
                rows.append({"ticker": t, "metric": metric, "horizon": h, "n": len(d),
                             "ic_spearman": ic, "hac_slope": slope, "hac_t": tstat, "p_value": p,
                             "n_cheap": len(cheap), "cheap_median": cheap.median(),
                             "n_expensive": len(exp_), "expensive_median": exp_.median(),
                             "spread_cheap_minus_exp": cheap.median() - exp_.median()})
    res = pd.DataFrame(rows)
    res["p_fdr"] = multipletests(res["p_value"], method="fdr_bh")[1]
    res["significant_fdr_5pct"] = res["p_fdr"] < 0.05
    return res


def half_life(df):
    """AR(1) on the log multiple: half-life = ln(.5)/ln(phi) when 0<phi<1."""
    rows = []
    for metric in CONFIG["metrics"]:
        for t, g in df.groupby("ticker"):
            x = np.log(g.sort_values("date")[metric].where(lambda s: s > 0)).dropna()
            phi = sm.OLS(x.values[1:], sm.add_constant(x.values[:-1])).fit().params[1]
            hl = np.log(0.5) / np.log(phi) if 0 < phi < 1 else np.nan
            rows.append({"ticker": t, "metric": metric, "phi": phi, "half_life_days": hl})
    return pd.DataFrame(rows)


def walk_forward_ic(df, metric, h=60, first_train=504, step=63):
    """Expanding window: IC measured only on each unseen test block."""
    sig = add_signals(df, metric).dropna(subset=["pctile", f"fwd_{h}"]).sort_values("date")
    dates = sorted(sig["date"].unique())
    ics = []
    for start in range(first_train, len(dates) - h, step):
        test = sig[(sig["date"] >= dates[start]) & (sig["date"] < dates[min(start + step, len(dates) - 1)])]
        # Signals are rolling percentiles (no parameters fit on train), so this checks
        # stability of IC across time blocks. Fitted models would also need purging
        # of train rows whose forward window overlaps the test block.
        if len(test) > 30:
            ics.append(spearmanr(test["pctile"], test[f"fwd_{h}"])[0])
    return pd.Series(ics, name=f"{metric}_{h}d_oos_ic")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_demo.xlsx")
    a = ap.parse_args()
    panel = make_synthetic_panel()
    res = evaluate(panel)
    hl = half_life(panel)
    wf = pd.concat([walk_forward_ic(panel, m) for m in CONFIG["metrics"]], axis=1)
    with pd.ExcelWriter(a.out) as w:
        res.to_excel(w, sheet_name="signal_tests", index=False)
        hl.to_excel(w, sheet_name="half_life", index=False)
        wf.describe().T.to_excel(w, sheet_name="walk_forward_ic")
    print(res.round(3).head(12).to_string(index=False))
    print(f"\n{res['significant_fdr_5pct'].sum()} of {len(res)} tests significant after FDR "
          "(synthetic random-walk data: expect ~0)")
    print(wf.describe().T.round(3))
