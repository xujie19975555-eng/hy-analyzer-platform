"""Tests for backtest service"""

import pytest

from app.services.backtest import (
    EquityPoint,
    _compute_drawdowns,
    period_to_time_range_ms,
    run_backtest,
)


class TestPeriodToTimeRange:
    """Test period parsing"""

    def test_7d_period(self):
        now_ms = 1700000000000
        start_ms, end_ms = period_to_time_range_ms("7d", now_ms=now_ms)
        assert end_ms == now_ms
        assert start_ms == now_ms - (7 * 86_400_000)

    def test_30d_period(self):
        now_ms = 1700000000000
        start_ms, end_ms = period_to_time_range_ms("30d", now_ms=now_ms)
        assert start_ms == now_ms - (30 * 86_400_000)

    def test_90d_period(self):
        now_ms = 1700000000000
        start_ms, end_ms = period_to_time_range_ms("90d", now_ms=now_ms)
        assert start_ms == now_ms - (90 * 86_400_000)

    def test_180d_period(self):
        now_ms = 1700000000000
        start_ms, end_ms = period_to_time_range_ms("180d", now_ms=now_ms)
        assert start_ms == now_ms - (180 * 86_400_000)

    def test_all_period(self):
        now_ms = 1700000000000
        start_ms, end_ms = period_to_time_range_ms("all", now_ms=now_ms)
        # Hyperliquid genesis: 2023-01-01 00:00:00 UTC
        HYPERLIQUID_GENESIS_MS = 1672531200000
        assert start_ms == HYPERLIQUID_GENESIS_MS
        assert end_ms == now_ms


class TestRunBacktest:
    """Test backtest engine"""

    def test_basic_backtest(self):
        """Basic backtest with profitable trades"""
        fills = [
            {
                "time": 1700000000000,
                "px": "100",
                "sz": "1",
                "closedPnl": "50",
                "fee": "1",
                "side": "B",
            },
            {
                "time": 1700000001000,
                "px": "100",
                "sz": "1",
                "closedPnl": "30",
                "fee": "1",
                "side": "A",
            },
        ]
        result = run_backtest(
            address="0xtest",
            period="7d",
            capital=1000.0,
            account_value=1000.0,
            fills=fills,
        )
        assert result.trade_count == 2
        assert result.simulated_pnl == 78.0  # (50-1) + (30-1) = 78
        assert result.scale == 1.0

    def test_scaling(self):
        """Test scaling with different capital"""
        fills = [
            {
                "time": 1700000000000,
                "px": "100",
                "sz": "1",
                "closedPnl": "100",
                "fee": "0",
                "side": "B",
            },
        ]
        result = run_backtest(
            address="0xtest",
            period="7d",
            capital=500.0,
            account_value=1000.0,
            fills=fills,
        )
        assert result.scale == 0.5
        assert result.simulated_pnl == 50.0  # 100 * 0.5

    def test_min_notional_filter(self):
        """Test that trades below 11 USDT are skipped"""
        fills = [
            # notional = 1 * 5 = 5 USDT, below 11 -> skipped
            {
                "time": 1700000000000,
                "px": "1",
                "sz": "5",
                "closedPnl": "10",
                "fee": "0",
                "side": "B",
            },
            # notional = 100 * 1 = 100 USDT, above 11 -> counted
            {
                "time": 1700000001000,
                "px": "100",
                "sz": "1",
                "closedPnl": "50",
                "fee": "0",
                "side": "A",
            },
        ]
        result = run_backtest(
            address="0xtest",
            period="7d",
            capital=1000.0,
            account_value=1000.0,
            fills=fills,
        )
        assert result.trade_count == 1
        assert result.skipped_trade_count == 1
        # scale = 1.0 (capital == account_value), pnl = 50 * 1.0
        assert result.simulated_pnl == 50.0

    def test_empty_fills(self):
        """Test with no fills"""
        result = run_backtest(
            address="0xtest",
            period="7d",
            capital=1000.0,
            account_value=1000.0,
            fills=[],
        )
        assert result.trade_count == 0
        assert result.simulated_pnl == 0.0
        assert result.simulated_roe == 0.0

    def test_invalid_capital(self):
        """Test with invalid capital"""
        with pytest.raises(ValueError, match="capital must be > 0"):
            run_backtest(
                address="0xtest",
                period="7d",
                capital=0,
                account_value=1000.0,
                fills=[],
            )

    def test_invalid_account_value(self):
        """Test with invalid account value"""
        with pytest.raises(ValueError, match="account_value must be > 0"):
            run_backtest(
                address="0xtest",
                period="7d",
                capital=1000.0,
                account_value=0,
                fills=[],
            )


class TestDrawdownCalculation:
    """Test drawdown computation"""

    def test_no_drawdown(self):
        """Test with only gains"""
        points = [
            EquityPoint(time_ms=1000, equity=100.0),
            EquityPoint(time_ms=2000, equity=110.0),
            EquityPoint(time_ms=3000, equity=120.0),
        ]
        max_dd, max_dd_pct, periods = _compute_drawdowns(points)
        assert max_dd == 0.0
        assert max_dd_pct == 0.0
        assert len(periods) == 0

    def test_single_drawdown(self):
        """Test with one drawdown period"""
        points = [
            EquityPoint(time_ms=1000, equity=100.0),
            EquityPoint(time_ms=2000, equity=120.0),
            EquityPoint(time_ms=3000, equity=100.0),
            EquityPoint(time_ms=4000, equity=130.0),
        ]
        max_dd, max_dd_pct, periods = _compute_drawdowns(points)
        assert max_dd == -20.0
        assert len(periods) == 1

    def test_multiple_drawdowns(self):
        """Test with multiple drawdown periods"""
        points = [
            EquityPoint(time_ms=1000, equity=100.0),
            EquityPoint(time_ms=2000, equity=120.0),
            EquityPoint(time_ms=3000, equity=110.0),
            EquityPoint(time_ms=4000, equity=130.0),
            EquityPoint(time_ms=5000, equity=100.0),
            EquityPoint(time_ms=6000, equity=140.0),
        ]
        max_dd, max_dd_pct, periods = _compute_drawdowns(points)
        assert max_dd == -30.0
        assert len(periods) == 2

    def test_ongoing_drawdown(self):
        """Test with drawdown at end of series"""
        points = [
            EquityPoint(time_ms=1000, equity=100.0),
            EquityPoint(time_ms=2000, equity=120.0),
            EquityPoint(time_ms=3000, equity=90.0),
        ]
        max_dd, max_dd_pct, periods = _compute_drawdowns(points)
        assert max_dd == -30.0
        assert len(periods) == 1
