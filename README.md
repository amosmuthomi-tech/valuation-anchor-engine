# Valuation Anchor Engine
### Rolling valuation signals and forward-return calibration

When a stock looks cheap or expensive versus its own recent history, what have the next 30, 60, 90 and 120 trading days typically looked like, and which valuation multiple is the most useful anchor for each company?

I built this for a US quantitative research firm (Google Sheets v0.5, then a Python/Colab v1). The client agreed to publish the code; their data is confidential and not included.

- `notebooks/valuation_anchor_v1.ipynb` — the delivered Colab v1 (outputs cleared, set `SOURCE_FILE` and `SHEET_URL` to your own)
- `valuation_anchor_demo.py` — a runnable demo on **synthetic data** that also adds HAC errors, FDR correction and a walk-forward check, which the v1 notebook does not implement

## What the method does

1. **Signal:** log of the valuation multiple, then a rolling percentile over a trailing window (252 trading days in the delivered v1). Only past data is used at each date.
2. **Buckets:** percentile cut into five states, D1 (cheap) to D5 (expensive). Delivered v1 used 10/25/75/90 cuts.
3. **Forward returns:** 30/60/90/120-day price returns per bucket (median, spread cheap minus expensive).
4. **Rank quality:** Spearman information coefficient of percentile vs forward return.
5. **Mean reversion speed:** AR(1) half-life of the log multiple.
6. **Data conventions:** forward-fill only (no backfill), negative or missing denominators excluded.

## What the demo adds on top (`valuation_anchor_demo.py`)

- Newey-West (HAC) standard errors for the slope of forward return on percentile, since forward windows overlap
- Benjamini-Hochberg FDR correction across all ticker x metric x horizon tests
- A simple expanding walk-forward check of IC stability across time blocks

## Run it

```bash
pip install -r requirements.txt
python valuation_anchor_demo.py --out results_demo.xlsx
```

On the synthetic random-walk data the demo finds **0 of 48 tests significant after FDR**, which is the correct answer when no real signal exists. Several cheap/expensive buckets are also thin (single-digit counts in the expensive bucket), a reminder that headline hit rates from small samples should not be trusted.

## Limitations (read before using on real data)

- Demo data is synthetic; no real-world performance claims are made here
- Overlapping forward windows reduce the effective sample size; HAC helps but does not remove the problem
- Walk-forward here checks IC stability only; a model with fitted parameters needs purged, embargoed splits
- Four to a handful of tickers is a small cross-section
- FY1-based multiples are affected by fiscal-year roll effects

## Related

[Tech Regime Indicators](https://github.com/amosmuthomi-tech/tech-regime-indicators) for the macro data layer.

`Python` `pandas` `numpy` `scipy` `statsmodels` `Time Series`
