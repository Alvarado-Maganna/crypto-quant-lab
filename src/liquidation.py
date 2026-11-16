"""Liquidation price of a linear, isolated-margin perpetual position (one-way mode).

"Linear" means the contract is quoted and margined in USDT (e.g. BTC/USDT:USDT): profit and
loss is ``qty * (price - entry)`` in USDT. "Isolated" means only the margin put into this one
position can be lost; the rest of the wallet is not used to keep it alive.

The exchange liquidates when the position's equity falls to its **maintenance margin**:

    equity             = margin + unrealized PnL
    maintenance margin = notional * MMR - cum          (notional = qty * price)

where MMR is the maintenance margin rate of the notional tier the position sits in, and
``cum`` is the "maintenance amount" that tier subtracts (0 in tier 1). Setting
equity == maintenance margin and solving for the price gives Binance's formula for isolated
mode (wallet balance = isolated margin, other positions' maintenance and PnL = 0):

    long:  LP = (margin + cum - qty * entry) / (qty * MMR - qty)
    short: LP = (margin + cum + qty * entry) / (qty * MMR + qty)

Sources (Binance support FAQ):
- Liquidation price formula: https://www.binance.com/en/support/faq/b3c689c1f50a44cabb3a84e663b81d93
- Maintenance margin = notional * MMR - cum: https://www.binance.com/en/support/faq/360033162192

This is an approximation: it ignores trading fees, funding payments and the liquidation
clearance fee, all of which move the real liquidation price closer to the entry.
"""

from __future__ import annotations

SIDES = ("long", "short")


def _check_inputs(side, entry_price, mmr):
    if side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    if entry_price <= 0:
        raise ValueError("entry_price must be > 0")
    if not 0 <= mmr < 1:
        raise ValueError("mmr is a fraction (0.004 = 0.40%) and must be in [0, 1)")


def liquidation_price_isolated(side, entry_price, qty, margin, mmr, cum=0.0):
    """Price at which an isolated linear position is liquidated (Binance formula).

    Parameters
    ----------
    side : "long" or "short"
    entry_price : entry price in quote currency (USDT per BTC).
    qty : position size in base currency (BTC), > 0 for both sides.
    margin : isolated margin posted, in USDT.
    mmr : maintenance margin rate as a fraction (0.004 means 0.40%).
    cum : maintenance amount of the tier, in USDT (0 for tier 1).

    A long with margin >= notional (leverage <= ~1x) gets LP <= 0: it cannot be liquidated by
    a price move, because the price cannot go below 0. The value is returned as is.
    """
    _check_inputs(side, entry_price, mmr)
    if qty <= 0:
        raise ValueError("qty must be > 0 (use side to say long or short)")
    if margin <= 0:
        raise ValueError("margin must be > 0")
    if side == "long":
        return (margin + cum - qty * entry_price) / (qty * mmr - qty)
    return (margin + cum + qty * entry_price) / (qty * mmr + qty)


def approx_liquidation_price(side, entry_price, leverage, mmr):
    """Rule-of-thumb liquidation price: the "1/leverage" shortcut, corrected by MMR.

    long:  entry * (1 - 1/leverage + mmr)
    short: entry * (1 + 1/leverage - mmr)

    Intuition: with leverage L the margin is 1/L of the notional, so a move of about 1/L wipes
    it out; the exchange steps in a bit earlier, when only MMR of the notional is left. The
    shortcut measures that MMR cushion at the entry notional instead of at the (smaller, for a
    long) notional at the liquidation price, so for a long it sits slightly above the exact LP.
    """
    _check_inputs(side, entry_price, mmr)
    if leverage <= 0:
        raise ValueError("leverage must be > 0")
    if side == "long":
        return entry_price * (1 - 1 / leverage + mmr)
    return entry_price * (1 + 1 / leverage - mmr)


def equity_and_maintenance(side, price, entry_price, qty, margin, mmr, cum=0.0):
    """Return ``(equity, maintenance_margin)`` of the position if the price were ``price``.

    equity = margin + PnL, with PnL = qty * (price - entry) for a long and
    qty * (entry - price) for a short. maintenance = qty * price * mmr - cum.
    At the liquidation price the two are equal (used to check the formula).
    """
    _check_inputs(side, entry_price, mmr)
    pnl = qty * (price - entry_price) if side == "long" else qty * (entry_price - price)
    return margin + pnl, qty * price * mmr - cum


# --------------------------------------------------------------------------------------
# Simulation on candles (notebook 04)
# --------------------------------------------------------------------------------------


def first_touch_at_or_below(low, entry_times, levels):
    """For each entry, the first candle **after** the entry candle whose low is <= its level.

    ``low``: Series of candle lows indexed by open time (sorted, unique).
    ``entry_times``: open times of the entry candles (each must be in ``low.index``).
    ``levels``: one price level per entry (e.g. its liquidation price).

    The entry candle itself is skipped on purpose: the position is opened at that candle's
    close, so the low of the same candle happened **before** the position existed. Checking it
    would be look-ahead in reverse (using a price from before the entry). The search runs to the
    last candle of ``low``. Returns a Series of open times (``NaT`` if never touched), indexed by
    ``entry_times``.
    """
    import numpy as np
    import pandas as pd

    if not (low.index.is_unique and low.index.is_monotonic_increasing):
        raise ValueError("low must have a sorted, unique index")
    entry_times = pd.DatetimeIndex(entry_times)
    levels = np.asarray(levels, dtype=float)
    if len(levels) != len(entry_times):
        raise ValueError("one level per entry time is needed")
    positions = low.index.get_indexer(entry_times)
    if (positions < 0).any():
        raise ValueError("every entry time must be a candle in low.index")
    values = low.to_numpy(dtype=float)
    touched = []
    for pos, level in zip(positions, levels):
        hits = np.flatnonzero(values[pos + 1:] <= level)
        touched.append(low.index[pos + 1 + hits[0]] if len(hits) else pd.NaT)
    # dtype=low.index.dtype keeps the UTC timezone even when every value is NaT (never touched)
    return pd.Series(pd.DatetimeIndex(touched, dtype=low.index.dtype), index=entry_times,
                     name="touched_at")


def simulate_isolated_longs(entry_close, lows, leverages, margin, mmr, cum=0.0):
    """One hypothetical isolated long per entry candle and leverage; when is each liquidated?

    ``entry_close``: Series of entry prices (close of each entry candle), indexed by open time.
    ``lows``: dict ``{name: Series of lows}``, e.g. ``{"mark": ..., "last": ...}``; each series
    must contain the entry candles and the candles after them.
    Each position has ``margin`` USDT and size ``qty = margin * leverage / entry``.

    Returns one row per (leverage, entry time) with entry, qty, notional, the exact liquidation
    price ``lp``, the drop from entry to ``lp`` in %, and ``liquidated_<name>`` (open time of
    the first candle after entry whose low <= lp, or NaT) for every series in ``lows``.
    """
    import pandas as pd

    frames = []
    for leverage in leverages:
        entry = entry_close.astype(float)
        qty = margin * leverage / entry
        lp = pd.Series([liquidation_price_isolated("long", e, q, margin, mmr, cum)
                        for e, q in zip(entry, qty)], index=entry.index)
        out = pd.DataFrame({"leverage": leverage, "entry": entry, "qty": qty,
                            "notional": qty * entry, "lp": lp,
                            "drop_to_lp_pct": 100 * (1 - lp / entry)})
        for name, low in lows.items():
            # set_axis keeps the tz-aware datetime dtype (to_numpy would turn it into objects)
            out[f"liquidated_{name}"] = first_touch_at_or_below(low, entry.index, lp).set_axis(out.index)
        frames.append(out)
    result = pd.concat(frames)
    result.index.name = "entry_time"
    return result.reset_index().set_index(["leverage", "entry_time"])
