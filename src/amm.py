"""A constant-product AMM (automated market maker): the pool rule ``x * y = k``.

Glossary:
- **Pool**: a smart contract holding two tokens, ``x`` units of the base (e.g. ETH) and ``y``
  units of the quote (e.g. USDC). There is no order book: the formula sets the price.
- **Spot price** of the pool: ``y / x`` (quote per base), the price of a tiny trade.
- **Swap**: you put ``dy`` quote in and take ``dx`` base out, so that ``(x - dx) * (y + dy) = k``
  still holds. Solving: ``dx = x * dy / (y + dy)``, which is always < ``x``.
- **Fee** ``f`` (e.g. 0.003 = 0.30%, as in Uniswap v2): only ``dy * (1 - f)`` is used in the
  formula; the whole ``dy`` stays in the pool, so ``k`` grows a little with every swap.
- **Average price** you pay: ``dy / dx``.
- **Price impact** (here): how much worse the average price is than the spot price before the
  trade, ``avg_price / spot_before - 1``. Without fee it equals exactly ``dy / y``.
- **Pool price change**: how much the spot price moved, ``new_price / spot_before - 1``. It is
  larger than the impact (about twice for small trades): the *next* trader pays it.
- **Slippage** is something else: the difference between the price you *expected* when you sent
  the order and the one you got, because the pool moved in between. This module does not model it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SwapResult:
    """Outcome of one swap (quote in, base out). Percentages are in % (10.0 means 10%)."""

    dx_out: float               # base received (e.g. ETH)
    avg_price: float            # quote paid per base received: dy / dx
    spot_before: float          # y / x before the trade
    new_price: float            # pool spot price after the trade
    impact_pct: float           # (avg_price / spot_before - 1) * 100, fee included if f > 0
    pool_price_change_pct: float  # (new_price / spot_before - 1) * 100
    x_after: float
    y_after: float


def _check_pool(x, y, fee):
    if x <= 0 or y <= 0:
        raise ValueError("Pool reserves must be positive")
    if not 0 <= fee < 1:
        raise ValueError("fee must be in [0, 1)")


def swap_quote_for_base(x, y, dy, fee=0.0):
    """Swap ``dy`` units of quote (e.g. USDC) into a pool with reserves ``x`` base, ``y`` quote.

    Example (no fee): pool 100 ETH / 300,000 USDC, dy = 30,000 USDC ->
    dx = 100 * 30,000 / 330,000 = 9.0909 ETH, avg price 3,300 (+10% = dy / y),
    new price 330,000 / 90.909 = 3,630 (+21%).
    """
    _check_pool(x, y, fee)
    if dy <= 0:
        raise ValueError("dy must be positive")
    dy_effective = dy * (1 - fee)
    dx = x * dy_effective / (y + dy_effective)
    x_after, y_after = x - dx, y + dy
    spot = y / x
    avg_price = dy / dx
    new_price = y_after / x_after
    return SwapResult(
        dx_out=dx,
        avg_price=avg_price,
        spot_before=spot,
        new_price=new_price,
        impact_pct=(avg_price / spot - 1) * 100,
        pool_price_change_pct=(new_price / spot - 1) * 100,
        x_after=x_after,
        y_after=y_after,
    )


def cost_to_buy_base(x, y, dx, fee=0.0):
    """Quote you must pay to take exactly ``dx`` base out of the pool (``dx`` must be < ``x``).

    Example: a vending machine with 10 apples and $100 (k = 1,000). Taking 1 apple leaves 9,
    so the cash box must hold 1,000 / 9 = $111.11: the apple costs $11.11, not $10.
    """
    _check_pool(x, y, fee)
    if not 0 < dx < x:
        raise ValueError("dx must be between 0 and x (a pool can never be emptied)")
    dy_effective = y * dx / (x - dx)
    return dy_effective / (1 - fee)


def quote_reserve_for_impact(dy, impact):
    """Quote reserve ``y`` a no-fee pool needs so that a ``dy`` order has impact ``impact``.

    From impact = dy / y (no fee): y = dy / impact. ``impact`` is a fraction (0.002 = 20 bps).
    The pool also holds the same value in base, so its total value is about ``2 * y``.
    """
    if dy <= 0 or impact <= 0:
        raise ValueError("dy and impact must be positive")
    return dy / impact
