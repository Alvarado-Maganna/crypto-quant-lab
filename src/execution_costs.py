"""Execution costs from order-book snapshots (notebook 07): market orders vs limit orders.

Input: the long table of ``src.data.order_book.SNAPSHOT_COLUMNS`` (one row per price level).
A snapshot is identified by ``(market, local_request_utc)``; it is placed in time by
``local_receive_utc`` (the first moment we knew it), never by the request time.

**Order size** is a notional in the quote currency (USDT) on both sides: a buy of size N spends
N USDT walking the asks; a sell of size N sells coins until it has received N USDT walking the
bids. So entry and exit trade the same notional (the price does not move between them).

**Market (taker) order walking the book**, per snapshot, side and size N:
- Levels are taken best first. A level holds ``price * amount`` USDT; the last level used is
  taken in part. ``base_filled`` = coins bought or sold; ``avg_price = N / base_filled``.
- Costs in USDT, measured against the mid (``mid = (best_bid + best_ask) / 2``):
  ``half_spread_usdt = base_filled * |best - mid|`` and ``impact_usdt = base_filled *
  |avg_price - best|``, where ``best`` is the best ask for a buy and the best bid for a sell.
  Their sum is ``base_filled * |avg_price - mid|``, the cost of the trade versus the mid.
- Every cost in **bps** is ``cost_usdt / N * 10,000``: a share of the notional traded, so
  spread, impact and fee (``fee_rate * 10,000``) add up exactly.
- If the levels received add up to less than N, the order is **not fillable** with what we
  saw, and no cost is reported (no extrapolation). Two cases (notebook 06, section 6):
  ``STATUS_TRUNCATED``: the side came back with the endpoint's maximum number of levels, so the
  book goes on beyond what the API returned; ``STATUS_EXHAUSTED``: the side came back with fewer
  levels than asked for, so we saw the whole book and it does not hold N.

**Limit (maker) order at the best bid / ask**: cost = maker fee only, spread and impact 0
(conservative: we price it at the mid, though a fill at the best bid/ask would be half a spread
*better* than the mid). Nothing here models whether it gets filled (fill probability) or
whether fills come when the price is about to move against us (adverse selection).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SIZES_USDT = (1_000.0, 10_000.0, 100_000.0, 1_000_000.0)
SNAPSHOT_KEY = ["market", "local_request_utc"]
STATUS_FILLED = "filled"
STATUS_TRUNCATED = "book truncated by the API"
STATUS_EXHAUSTED = "book exhausted"
SIDE_OF_ACTION = {"buy": "ask", "sell": "bid"}     # a buy walks the asks, a sell the bids


def market_order_legs(levels, sizes=SIZES_USDT):
    """Cost of a market buy and a market sell of each size, in every snapshot.

    Returns one row per (market, local_request_utc, action, size_usdt) with
    ``local_receive_utc``, best bid/ask, mid, ``best`` (the price on the side walked),
    ``base_filled``, ``avg_price``, ``levels_used``, ``status``, and the costs
    ``half_spread_usdt / impact_usdt / slippage_usdt`` (the sum) and the same in bps of the
    notional. Costs are NaN when ``status`` is not ``STATUS_FILLED``.
    Raises on a crossed book or an empty side (data errors).
    """
    df = levels[SNAPSHOT_KEY + ["local_receive_utc", "side", "level", "price", "amount",
                                "requested_limit"]].copy()
    for col in ("market", "side"):  # plain strings: categoricals add empty groups in groupby
        df[col] = df[col].astype(str)
    df = df.sort_values(SNAPSHOT_KEY + ["side", "level"], kind="stable").reset_index(drop=True)

    best = df[df["level"] == 0].pivot_table(index=SNAPSHOT_KEY, columns="side", values="price",
                                            aggfunc="first")
    snap = pd.DataFrame({"best_bid": best.get("bid"), "best_ask": best.get("ask")},
                        index=best.index)
    if snap.isna().any(axis=1).any():
        raise ValueError("snapshot(s) with an empty side: no best bid or ask")
    if (snap["best_bid"] >= snap["best_ask"]).any():
        raise ValueError("crossed snapshot(s): best bid >= best ask")
    snap["mid"] = (snap["best_bid"] + snap["best_ask"]) / 2
    snap["local_receive_utc"] = df.groupby(SNAPSHOT_KEY)["local_receive_utc"].first()

    group_key = SNAPSHOT_KEY + ["side"]
    df["quote"] = df["price"] * df["amount"]
    df["cum_before"] = df.groupby(group_key)["quote"].cumsum() - df["quote"]
    side_info = df.groupby(group_key).agg(n_levels=("level", "size"),
                                          total_quote=("quote", "sum"),
                                          requested_limit=("requested_limit", "first"))

    out = []
    for size in sizes:
        take = (size - df["cum_before"]).clip(lower=0.0).clip(upper=df["quote"])
        used = df.assign(take_quote=take, take_base=take / df["price"], used=take > 0)
        agg = used.groupby(group_key).agg(base_filled=("take_base", "sum"),
                                          levels_used=("used", "sum"))
        agg = agg.join(side_info)
        agg["size_usdt"] = size
        out.append(agg.reset_index())
    legs = pd.concat(out, ignore_index=True)

    legs["action"] = legs["side"].map({v: k for k, v in SIDE_OF_ACTION.items()})
    legs = legs.join(snap, on=SNAPSHOT_KEY)
    legs["best"] = np.where(legs["side"] == "ask", legs["best_ask"], legs["best_bid"])
    filled = legs["total_quote"] >= legs["size_usdt"] * (1 - 1e-12)  # tolerance for rounding
    truncated = legs["n_levels"] >= legs["requested_limit"]
    legs["status"] = np.select([filled, truncated], [STATUS_FILLED, STATUS_TRUNCATED],
                               default=STATUS_EXHAUSTED)
    ok = legs["status"] == STATUS_FILLED
    legs["avg_price"] = (legs["size_usdt"] / legs["base_filled"]).where(ok)
    legs["half_spread_usdt"] = (legs["base_filled"] * (legs["best"] - legs["mid"]).abs()).where(ok)
    legs["impact_usdt"] = (legs["base_filled"] * (legs["avg_price"] - legs["best"]).abs()).where(ok)
    legs["slippage_usdt"] = legs["half_spread_usdt"] + legs["impact_usdt"]
    for part in ("half_spread", "impact", "slippage"):
        legs[f"{part}_bps"] = legs[f"{part}_usdt"] / legs["size_usdt"] * 1e4
    cols = SNAPSHOT_KEY + ["local_receive_utc", "action", "side", "size_usdt", "best_bid",
                           "best_ask", "mid", "best", "n_levels", "requested_limit",
                           "total_quote", "levels_used", "base_filled", "avg_price", "status",
                           "half_spread_usdt", "impact_usdt", "slippage_usdt",
                           "half_spread_bps", "impact_bps", "slippage_bps"]
    return legs[cols].sort_values(["market", "local_receive_utc", "action", "size_usdt"],
                                  kind="stable").reset_index(drop=True)


# --- round trips ------------------------------------------------------------------------------

# (entry order type, exit order type); "taker" = market order, "maker" = limit at the best price
COMBOS = {"taker -> taker": ("taker", "taker"), "maker -> maker": ("maker", "maker"),
          "maker -> taker": ("maker", "taker")}
# A long enters with a buy and exits with a sell; a short does the opposite.
DIRECTIONS = {"long": ("buy", "sell"), "short": ("sell", "buy")}


def _leg(legs_by_action, action, order_type, fee):
    """Spread / impact / fee (bps) and status of one leg, for every (snapshot, size)."""
    leg = legs_by_action[action]
    if order_type == "taker":
        return pd.DataFrame({"spread_bps": leg["half_spread_bps"], "impact_bps": leg["impact_bps"],
                             "fee_bps": fee["taker"] * 1e4, "status": leg["status"]})
    return pd.DataFrame({"spread_bps": 0.0, "impact_bps": 0.0, "fee_bps": fee["maker"] * 1e4,
                         "status": STATUS_FILLED}, index=leg.index)   # fill not modeled


def round_trips(legs, fees, combos=COMBOS, directions=DIRECTIONS):
    """Entry + exit cost in every snapshot, for each combo, direction and size.

    ``legs``: output of ``market_order_legs``. ``fees``: dict market -> {"maker": rate,
    "taker": rate} as fractions (0.001 = 0.10%). Both legs use the same snapshot and the same
    notional (``size_usdt``). Returns one row per (market, local_request_utc, size_usdt, combo,
    direction) with ``spread_bps``, ``impact_bps``, ``fee_bps``, ``total_bps``, ``total_usdt``
    and ``status``: ``STATUS_FILLED`` if every taker leg was fillable, else the status of the
    first unfillable leg (entry first); the costs are NaN then.
    """
    key = SNAPSHOT_KEY + ["local_receive_utc", "size_usdt"]
    by_action = {a: g.set_index(key) for a, g in legs.groupby("action")}
    rows = []
    for combo, (entry_type, exit_type) in combos.items():
        for direction, (entry_action, exit_action) in directions.items():
            parts = []
            for action, order_type in ((entry_action, entry_type), (exit_action, exit_type)):
                market = by_action[action].index.get_level_values("market")
                fee = pd.DataFrame([fees[m] for m in market], index=by_action[action].index)
                parts.append(_leg(by_action, action, order_type, fee))
            entry = parts[0]
            exit_ = parts[1].reindex(entry.index)
            total = pd.DataFrame({c: entry[c] + exit_[c]
                                  for c in ("spread_bps", "impact_bps", "fee_bps")})
            status = entry["status"].where(entry["status"] != STATUS_FILLED, exit_["status"])
            bad = status != STATUS_FILLED
            total.loc[bad, ["spread_bps", "impact_bps"]] = np.nan
            total.loc[bad, "fee_bps"] = np.nan
            total["total_bps"] = total["spread_bps"] + total["impact_bps"] + total["fee_bps"]
            total["status"] = status
            total["combo"], total["direction"] = combo, direction
            rows.append(total.reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["total_usdt"] = out["total_bps"] * out["size_usdt"] / 1e4
    return out


def summarize_round_trips(rt, by=("market", "combo", "direction", "size_usdt")):
    """Median and p90 of the round-trip cost across snapshots, plus how many were fillable.

    The quantiles use **only fillable snapshots**. ``n_unfillable`` > 0 means the reported
    cost leaves out the snapshots where the book was too thin to fill, so it is biased low;
    when no snapshot is fillable the cost is NaN.
    """
    by = list(by)
    g = rt.groupby(by, sort=False)
    out = g.agg(n_snapshots=("status", "size"),
                n_fillable=("status", lambda s: int((s == STATUS_FILLED).sum())),
                n_truncated=("status", lambda s: int((s == STATUS_TRUNCATED).sum())),
                n_exhausted=("status", lambda s: int((s == STATUS_EXHAUSTED).sum())),
                median_bps=("total_bps", "median"),
                p90_bps=("total_bps", lambda s: s.quantile(0.9)),
                median_spread_bps=("spread_bps", "median"),
                median_impact_bps=("impact_bps", "median"),
                median_fee_bps=("fee_bps", "median"))
    out["n_unfillable"] = out["n_snapshots"] - out["n_fillable"]
    size = out.index.get_level_values("size_usdt")
    out["median_usdt"] = out["median_bps"] * size / 1e4
    out["p90_usdt"] = out["p90_bps"] * size / 1e4
    return out


def cost_shares(summary):
    """Share (%) of spread, impact and fee in the round-trip cost.

    Computed from the medians of each component, divided by the sum of those medians (the
    median of a sum is not the sum of the medians, so the sum is shown next to the total).
    """
    parts = summary[["median_spread_bps", "median_impact_bps", "median_fee_bps"]]
    total = parts.sum(axis=1, min_count=3)
    out = parts.div(total, axis=0) * 100
    out.columns = ["spread (%)", "impact (%)", "fee (%)"]
    out.insert(0, "sum of medians (bps)", total)
    out.insert(1, "median total (bps)", summary["median_bps"])
    return out
