"""Order-book metrics (notebook 06): spread, depth within a band around the mid, truncation.

Input: the long table of ``src.data.order_book.SNAPSHOT_COLUMNS`` (one row per price level).
A snapshot is identified by ``(market, local_request_utc)``.

Definitions (per snapshot):
- ``best_bid`` = highest buy price, ``best_ask`` = lowest sell price, ``mid = (bid + ask) / 2``.
- ``spread_usdt = ask - bid``; ``spread_bps = spread / mid * 10,000`` (1 bp = 0.01%).
- Depth within ±x%: on the bid side, the levels priced at or above ``mid * (1 - x)``; on the
  ask side, at or below ``mid * (1 + x)``. Reported in base units (``amount``) and in USDT
  (``price * amount``, summed).
- ``reach_<side>_pct``: how far from the mid the deepest level we received is, in %.
  If ``reach < x``, the snapshot ends **inside** the band, so the depth we see is only a
  lower bound: ``complete_<band>_<side>`` is False ("truncated"). Nothing is filled in.
- ``covers_1m_<side>``: the levels we received add up to at least ``target_notional`` USDT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ±0.2% added post-inspection: the narrowest band of Binance's public `bookDepth` files (USD-M).
BANDS = {"0.1pct": 0.001, "0.2pct": 0.002, "1pct": 0.01}
TARGET_NOTIONAL = 1_000_000.0
SNAPSHOT_KEY = ["market", "local_request_utc"]

# --- choosing the "less liquid" altcoin ------------------------------------------------------

# Not crypto-native price risk, so not a fair "altcoin" (classified by hand, see notebook 06).
NON_CRYPTO_BASES = {
    "fiat": {"EUR", "EURI", "AEUR", "TRY", "BRL"},
    "wrapped": {"WBTC", "WBETH", "BETH"},
    "commodity": {"PAXG", "XAUT"},
    "tokenized equity / ETF": {
        "NVDAB", "SOXLB", "SPCXB", "CRCLB", "MUB", "GOOGLB", "MSTRB", "SOXSB", "SNDKB", "KORUB",
        "INTCB", "SKHYB", "QQQB", "SNXXB", "TQQQB", "ANTB", "AMDB", "MRVLB", "NBISB", "AMZNB",
        "TSLAB", "EWYB", "TSMB", "DRAMB", "BEB", "COINB", "AVGOB"},
}


def rank_usdt_spot_pairs(tickers, markets):
    """Active spot pairs quoted in USDT, ranked by 24 h quote volume (1 = largest).

    ``tickers``/``markets``: ccxt ``fetch_tickers()`` / ``markets``. Returns a DataFrame with
    rank, symbol, base, quote_volume_24h (USDT), last, high, low.
    """
    rows = []
    for symbol, t in tickers.items():
        m = markets.get(symbol, {})
        if not (m.get("spot") and m.get("active") and m.get("quote") == "USDT"):
            continue
        rows.append({"symbol": symbol, "base": m["base"],
                     "quote_volume_24h": float(t.get("quoteVolume") or 0.0),
                     "last": t.get("last"), "high": t.get("high"), "low": t.get("low")})
    df = pd.DataFrame(rows).sort_values("quote_volume_24h", ascending=False, kind="stable")
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df.reset_index(drop=True)


def classify_pair(row, stable_tol=0.02):
    """Why a pair is excluded ('stablecoin', 'fiat', ...) or '' if it is an eligible altcoin.

    Stablecoin rule: last price within ``stable_tol`` of 1 USDT **and** 24 h range
    (high / low - 1) below ``stable_tol``. It catches stablecoins we did not list by name.
    """
    for reason, bases in NON_CRYPTO_BASES.items():
        if row["base"] in bases:
            return reason
    last, high, low = row["last"], row["high"], row["low"]
    if last and high and low and abs(last - 1) < stable_tol and high / low - 1 < stable_tol:
        return "stablecoin"
    return ""


def pick_altcoin(ranking, start_rank=75, stable_tol=0.02):
    """First eligible pair at rank >= ``start_rank``; returns (symbol, ranking with reasons)."""
    ranking = ranking.copy()
    ranking["excluded_as"] = ranking.apply(classify_pair, axis=1, stable_tol=stable_tol)
    eligible = ranking[(ranking["rank"] >= start_rank) & (ranking["excluded_as"] == "")]
    if eligible.empty:
        raise ValueError(f"No eligible pair at rank >= {start_rank}")
    return eligible.iloc[0]["symbol"], ranking


# --- metrics per snapshot --------------------------------------------------------------------

def snapshot_metrics(levels, bands=BANDS, target_notional=TARGET_NOTIONAL):
    """One row per snapshot (market, local_request_utc) with the metrics of the module docstring.

    Raises if a snapshot is crossed (best bid >= best ask) or has an empty side (no bids or no
    asks at all): both would be data errors, and silent NaNs would hide them.
    """
    df = levels[SNAPSHOT_KEY + ["side", "level", "price", "amount"]].copy()
    for col in ("market", "side"):  # plain strings: categoricals add empty groups in groupby
        df[col] = df[col].astype(str)
    df["quote"] = df["price"] * df["amount"]
    best = (df[df["level"] == 0].pivot_table(index=SNAPSHOT_KEY, columns="side",
                                              values="price", aggfunc="first"))
    out = pd.DataFrame({"best_bid": best.get("bid"), "best_ask": best.get("ask")},
                       index=best.index)
    empty = out[["best_bid", "best_ask"]].isna().any(axis=1)
    if empty.any():
        raise ValueError(f"{int(empty.sum())} snapshot(s) with an empty side: no best bid or ask")
    crossed = out["best_bid"] >= out["best_ask"]
    if crossed.any():
        raise ValueError(f"{int(crossed.sum())} crossed snapshot(s): best bid >= best ask")
    out["mid"] = (out["best_bid"] + out["best_ask"]) / 2
    out["spread_usdt"] = out["best_ask"] - out["best_bid"]
    out["spread_bps"] = out["spread_usdt"] / out["mid"] * 1e4

    df = df.join(out["mid"], on=SNAPSHOT_KEY)
    df["dist"] = (df["price"] / df["mid"] - 1).abs()   # distance from the mid, as a fraction
    grouped = df.groupby(SNAPSHOT_KEY + ["side"])
    per_side = pd.DataFrame({"n_levels": grouped.size(), "reach": grouped["dist"].max(),
                             "notional_usdt": grouped["quote"].sum()}).unstack("side")
    for side in ("bid", "ask"):
        out[f"n_levels_{side}"] = per_side[("n_levels", side)]
        out[f"reach_{side}_pct"] = per_side[("reach", side)] * 100
        out[f"notional_{side}_usdt"] = per_side[("notional_usdt", side)]
        out[f"covers_1m_{side}"] = out[f"notional_{side}_usdt"] >= target_notional

    for tag, width in bands.items():
        inside = df[df["dist"] <= width * (1 + 1e-12)]   # tiny tolerance for float rounding
        sums = inside.groupby(SNAPSHOT_KEY + ["side"])[["amount", "quote"]].sum().unstack("side")
        for side in ("bid", "ask"):
            out[f"depth_{tag}_{side}_base"] = sums.get(("amount", side), pd.Series(dtype=float)) \
                .reindex(out.index).fillna(0.0)
            out[f"depth_{tag}_{side}_usdt"] = sums.get(("quote", side), pd.Series(dtype=float)) \
                .reindex(out.index).fillna(0.0)
            out[f"complete_{tag}_{side}"] = out[f"reach_{side}_pct"] >= width * 100
    return out.reset_index()


def quantile_summary(values, quantiles=(0.1, 0.5, 0.9)):
    """p10 / median / p90 and the count of a Series (NaN dropped)."""
    values = values.dropna()
    res = {f"p{int(q * 100)}": values.quantile(q) if len(values) else np.nan for q in quantiles}
    res["n"] = len(values)
    return res


def consecutive_changes(metrics, column):
    """Change of ``column`` between consecutive snapshots of the same market.

    Returns a DataFrame with the absolute change and the relative change (change / previous),
    plus the seconds between the two snapshots, so gaps from failed requests are visible.
    """
    m = metrics.sort_values(SNAPSHOT_KEY)
    g = m.groupby("market")
    out = pd.DataFrame({
        "market": m["market"],
        "local_request_utc": m["local_request_utc"],
        "gap_s": g["local_request_utc"].diff().dt.total_seconds(),
        "change": g[column].diff(),
    })
    out["rel_change"] = out["change"] / g[column].shift()
    return out.dropna(subset=["change"]).reset_index(drop=True)
