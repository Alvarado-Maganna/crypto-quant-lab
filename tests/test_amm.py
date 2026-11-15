"""Tests for src/amm.py, with numbers computed by hand."""

import pytest

from src.amm import cost_to_buy_base, quote_reserve_for_impact, swap_quote_for_base


def test_textbook_pool_100_eth_300k_usdc():
    r = swap_quote_for_base(x=100, y=300_000, dy=30_000)
    assert r.dx_out == pytest.approx(9.0909, abs=1e-4)      # 100 * 30k / 330k
    assert r.avg_price == pytest.approx(3_300)                # 30k / 9.0909
    assert r.impact_pct == pytest.approx(10.0)                # = dy / y
    assert r.new_price == pytest.approx(3_630)                # 330k / 90.909
    assert r.pool_price_change_pct == pytest.approx(21.0)     # (1.1)^2 - 1


def test_vending_machine_apple_costs_11_11():
    # 10 apples, $100 -> k = 1,000. After 1 apple: 9 apples, so 1,000 / 9 = $111.11 in the box.
    assert cost_to_buy_base(x=10, y=100, dx=1) == pytest.approx(11.11, abs=0.01)


def test_buying_back_the_same_amount_is_consistent():
    r = swap_quote_for_base(x=10, y=100, dy=cost_to_buy_base(10, 100, 1))
    assert r.dx_out == pytest.approx(1.0)


def test_k_is_unchanged_without_fee():
    r = swap_quote_for_base(x=100, y=300_000, dy=12_345)
    assert r.x_after * r.y_after == pytest.approx(100 * 300_000)


def test_k_grows_with_fee():
    r = swap_quote_for_base(x=100, y=300_000, dy=12_345, fee=0.003)
    assert r.x_after * r.y_after > 100 * 300_000


@pytest.mark.parametrize("dy", [1e-6, 1, 30_000, 3e6, 3e9, 3e15])
def test_never_drains_the_pool(dy):
    assert swap_quote_for_base(x=100, y=300_000, dy=dy).dx_out < 100


def test_fee_gives_less_base():
    no_fee = swap_quote_for_base(100, 300_000, 30_000)
    fee = swap_quote_for_base(100, 300_000, 30_000, fee=0.003)
    assert fee.dx_out < no_fee.dx_out
    assert fee.impact_pct > no_fee.impact_pct


def test_tiny_trade_has_almost_no_impact_but_still_pays_the_fee():
    assert swap_quote_for_base(100, 300_000, 0.01).impact_pct == pytest.approx(0, abs=1e-5)
    # with fee f, a tiny trade costs f / (1 - f) above spot: 0.3009% for f = 0.3%
    tiny_fee = swap_quote_for_base(100, 300_000, 0.01, fee=0.003)
    assert tiny_fee.impact_pct == pytest.approx(100 * 0.003 / 0.997, abs=1e-5)


def test_deeper_pool_means_proportionally_less_impact():
    impacts = [swap_quote_for_base(100 * m, 300_000 * m, 30_000).impact_pct for m in (1, 2, 10)]
    assert impacts == pytest.approx([10.0, 5.0, 1.0])


def test_reserve_for_20_bps_impact():
    y = quote_reserve_for_impact(dy=1_000_000, impact=0.002)
    assert y == pytest.approx(500_000_000)
    assert swap_quote_for_base(y / 100_000, y, 1_000_000).impact_pct == pytest.approx(0.20)


@pytest.mark.parametrize("args", [(0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, -5)])
def test_rejects_bad_inputs(args):
    with pytest.raises(ValueError):
        swap_quote_for_base(*args)


def test_cannot_buy_the_whole_pool():
    with pytest.raises(ValueError):
        cost_to_buy_base(x=10, y=100, dx=10)
