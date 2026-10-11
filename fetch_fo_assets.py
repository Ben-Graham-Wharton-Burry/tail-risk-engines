#!/usr/bin/env python3
"""Fetch the two family-office sleeves (gold + REIT) with LONG daily history.

Run from anywhere (paths resolve relative to this file):

    python fetch_fo_assets.py            # normal run, fails loud if history is short
    python fetch_fo_assets.py --allow-short   # accept a short series (debug only)

Why this rewrite
----------------
The previous version preferred Stooq (gold) and Vanguard funds (REIT) and, when
Stooq silently failed, fell back to the COMEX gold *futures* series (yfinance
``GC=F``) which only starts 2000-08-30. That single fallback -- not REIT --
truncated the whole FO sample to 2000 and cost us the dot-com bubble.

Two fixes:
  1. Source both sleeves from long, free, keyless FRED daily series, so the
     sample can begin in 1990 and cover the 2000-02 dot-com crash with a full
     5-year estimation window.
       * GOLD : LBMA Gold Price PM fix (FRED 'GOLDPMGBD228NLBM'), daily since 1968.
                A spot fix -> a clean "holding gold" return, not a roll-affected
                futures contract.
       * REIT : Wilshire US REIT index (FRED 'WILLREITIND'), daily since 1978.
                NOTE: this is a PRICE index (ex-dividends). REIT dividends are
                large (~3-4%/yr), so it understates total return. For this
                project's risk focus (vol / VaR / drawdown) that is acceptable;
                if you need total return, drop a TR CSV in data/raw/ (below) or
                splice VGSIX-TR from 1996. Flagged, not hidden.
  2. FAIL LOUD: if the chosen series does not reach REQUIRE_START, the script
     raises instead of silently shipping a short history. No more silent 2000.

Source order per sleeve (longest history first); a local CSV always wins so you
can substitute an official download (LBMA from goldhub, Nareit/Wilshire TR):
  * data/raw/gold.csv  / data/raw/reit.csv   (Date,Close ; optional, highest priority)
  * FRED long series
  * yfinance / Stooq    (shorter; only used with --allow-short)

Writes:
  * data/raw/gold_raw.csv, data/raw/reit_raw.csv   (cached native level series)
  * data/fo_assets.csv                              (GLD_ret, REIT_ret on the
                                                     equity/bond trading calendar)
"""

import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import requests

try:
    import yfinance as yf
except Exception as e:                                       # pragma: no cover
    yf = None
    print("yfinance unavailable (only needed for short fallbacks):", e)

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(_HERE, "data")
RAW = os.path.join(DATA, "raw")
PRICES_CSV = os.path.join(DATA, "prices.csv")   # equity/bond calendar (1990+), built by NB01
OUT_CSV = os.path.join(DATA, "fo_assets.csv")

_HEADERS = {"User-Agent": "Mozilla/5.0 (research script)"}

# Hard floor: if the selected series starts after this, the long source failed.
REQUIRE_START = pd.Timestamp("1991-01-01")
# Bridge equity-vs-asset holiday calendar mismatches by at most this many days.
FFILL_LIMIT = 3


# --------------------------------------------------------------------------- #
# Source helpers (each returns a DataFrame with a 'Close' level column)
# --------------------------------------------------------------------------- #
def _strip_tz(df):
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df


def _local_csv(path):
    """A user-supplied CSV with columns Date,Close (highest-priority source)."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = pd.read_csv(path)
    low = {c.lower(): c for c in d.columns}
    if "close" not in low:
        raise ValueError(f"{path} needs a 'Close' column; got {list(d.columns)}")
    datecol = low.get("date", d.columns[0])   # unnamed first column = the date index
    d = d.rename(columns={datecol: "Date", low["close"]: "Close"})
    d["Date"] = pd.to_datetime(d["Date"])
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    return d.dropna(subset=["Close"]).set_index("Date").sort_index()[["Close"]]


def _fred(series):
    """FRED daily series via the keyless fredgraph CSV endpoint."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    d = pd.read_csv(io.StringIO(r.text), index_col=0, parse_dates=True)
    d.columns = ["Close"]
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")   # FRED uses '.' for NA
    return d.dropna().sort_index()


def _stooq(symbol):
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    r = requests.get(url, headers=_HEADERS, timeout=30)
    r.raise_for_status()
    if not r.text.lstrip().lower().startswith("date"):
        raise ValueError(f"Stooq non-CSV (rate-limited?): {r.text[:80]!r}")
    d = pd.read_csv(io.StringIO(r.text))
    if "Date" not in d.columns or "Close" not in d.columns:
        raise ValueError(f"unexpected Stooq columns for {symbol!r}: {list(d.columns)}")
    d["Date"] = pd.to_datetime(d["Date"])
    return d.set_index("Date").sort_index()[["Close"]]


def _yf(ticker):
    if yf is None:
        raise RuntimeError("yfinance not available")
    t = yf.Ticker(ticker).history(period="max", auto_adjust=True, actions=False)
    if isinstance(t.columns, pd.MultiIndex):
        t.columns = t.columns.get_level_values(0)
    if "Close" not in t.columns and "Adj Close" in t.columns:
        t = t.rename(columns={"Adj Close": "Close"})
    return _strip_tz(t)[["Close"]]


def _kenfrench_realestate():
    """Ken French 'Real Estate' industry (49-ind daily): TOTAL-return daily since 1926.

    Free, keyless, stable URL. Broader than a pure REIT index (SIC real estate,
    incl. operating cos) but long and consistent. Returns a synthetic price level
    so it fits the same pct_change pipeline as the other sources.
    """
    import zipfile
    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "49_Industry_Portfolios_daily_CSV.zip")
    r = requests.get(url, headers=_HEADERS, timeout=60)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    lines = z.read(z.namelist()[0]).decode("latin-1").splitlines()
    hdr = next(i for i, l in enumerate(lines) if "RlEst" in l)   # first = VW daily table
    ci = [c.strip() for c in lines[hdr].split(",")].index("RlEst")
    dates, vals = [], []
    for l in lines[hdr + 1:]:
        p = l.split(",")
        d = p[0].strip()
        if not (len(d) == 8 and d.isdigit()):
            break
        dates.append(pd.to_datetime(d, format="%Y%m%d"))
        vals.append(float(p[ci]))
    ret = pd.Series(vals, index=dates) / 100.0
    ret = ret[ret > -0.99]                       # drop -99.99 / -999 missing flags
    return (1.0 + ret).cumprod().to_frame("Close")


def fetch_longest(label, sources, allow_short):
    """Try sources in order; keep the one with the EARLIEST start. Fail loud."""
    best_name = best = None
    for name, fn in sources:
        try:
            s = fn()["Close"].dropna()
            if len(s) == 0:
                raise ValueError("empty series")
            print(f"  [{label}] {name:32s}: {len(s):6d} rows | "
                  f"{s.index[0].date()} -> {s.index[-1].date()}")
            if best is None or s.index[0] < best.index[0]:
                best_name, best = name, s
        except Exception as e:
            print(f"  [{label}] {name:32s}: FAILED ({type(e).__name__}: {e})")
    if best is None:
        raise SystemExit(
            f"\nABORT: every source failed for {label}. Save a daily {label} CSV "
            f"(columns Date,Close) to data/raw/{label}.csv and re-run.\n"
            f"  gold: open https://stooq.com/q/d/l/?s=xauusd&i=d in a BROWSER (the script "
            f"gets rate-limited, the browser usually does not), or LBMA goldhub.com.\n"
            f"  reit: FRED 'WILLREITIND' CSV export, or a Nareit/Wilshire download.")
    print(f"  [{label}] -> using {best_name} (start {best.index[0].date()})")
    if best.index[0] > REQUIRE_START and not allow_short:
        raise SystemExit(
            f"\nABORT: {label} only reaches {best.index[0].date()}, after the "
            f"{REQUIRE_START.date()} floor.\nThe long source failed above -- fix it "
            f"(or drop data/raw/{label.lower()}.csv) rather than shipping a short\n"
            f"history. Re-run with --allow-short only for debugging.")
    return best_name, best


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-short", action="store_true",
                    help="accept a series that starts after the REQUIRE_START floor")
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)

    print("GOLD (real-assets sleeve):")
    # NOTE: FRED no longer hosts the LBMA gold fix (ICE licensing). The reliable
    # long-history route is a local CSV in data/raw/gold.csv; Stooq is a flaky
    # best-effort automated attempt.
    gold_src, gold = fetch_longest("gold", [
        ("local data/raw/gold.csv",       lambda: _local_csv(os.path.join(RAW, "gold.csv"))),
        ("Stooq XAUUSD",                  lambda: _stooq("xauusd")),
    ], args.allow_short)

    print("\nREIT (real-estate sleeve):")
    reit_src, reit = fetch_longest("reit", [
        ("local data/raw/reit.csv",       lambda: _local_csv(os.path.join(RAW, "reit.csv"))),
        ("Ken French Real Estate (49ind)", _kenfrench_realestate),
        ("yfinance VGSIX (short, TR)",    lambda: _yf("VGSIX")),
    ], args.allow_short)

    # cache native level series
    gold.to_frame("Close").to_csv(os.path.join(RAW, "gold_raw.csv"))
    reit.to_frame("Close").to_csv(os.path.join(RAW, "reit_raw.csv"))

    # align to the equity/bond trading calendar if available, else gold's own
    if os.path.exists(PRICES_CSV):
        cal = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True).index
        print(f"\nAligning to equity/bond calendar from prices.csv: "
              f"{cal[0].date()} -> {cal[-1].date()} ({len(cal)} days)")
    else:
        cal = gold.index.union(reit.index)
        print("\nprices.csv not found -- aligning to the gold/REIT union calendar. "
              "Re-run NB01 first for a clean equity/bond calendar.")

    def to_cal_returns(level, name):
        lv = level.reindex(cal.union(level.index)).sort_index()
        lv = lv.ffill(limit=FFILL_LIMIT).reindex(cal)
        return lv.pct_change().rename(name)

    fo = pd.concat([to_cal_returns(gold, "GLD_ret"),
                    to_cal_returns(reit, "REIT_ret")], axis=1).dropna()
    fo.to_csv(OUT_CSV)

    print(f"\nSaved {OUT_CSV}: {fo.shape[0]} rows | "
          f"{fo.index[0].date()} -> {fo.index[-1].date()}")
    print(f"  gold source: {gold_src}")
    print(f"  reit source: {reit_src}")
    if reit_src.startswith("Ken French"):
        print("  NOTE: REIT sleeve = Ken French 'Real Estate' industry (value-weighted, "
              "TOTAL return incl. dividends, daily since 1926). Broader than a pure REIT "
              "index but long, free and consistent.")
    print(">>> FO sleeves now start:", fo.index[0].date(),
          "(common start is set by the later of gold/REIT vs the equity calendar)")
    print(">>> Re-run NB01 (delete cached data/returns.csv & data/fo_assets.csv "
          "first) so the 1990 start propagates through the pipeline.")


if __name__ == "__main__":
    main()
