"""Tests for analyzer service"""
import pytest
from app.services.analyzer import parse_superx_stats, analyze_trades


class TestParseSuperxStats:
    """Test SuperX data parsing"""

    def test_parse_full_data(self):
        """Should parse complete SuperX response"""
        data = {
            "roeAllTime": 1650.29,
            "roe7d": 15.5,
            "roe30d": 45.2,
            "roe90d": 120.3,
            "pnlAllTime": 1682095.47,
            "pnl7d": 15000.0,
            "pnl30d": 45000.0,
            "pnl90d": 120000.0,
            "winRate": 77.63,
            "profitFactorAll": 2.29,
            "maxDrawdownPercentAll": 46.63,
            "tradeCount": 1500
        }
        stats = parse_superx_stats("0xtest", data)

        assert stats.roe_all_time == 1650.29
        assert stats.pnl_all_time == 1682095.47
        assert stats.win_rate == 77.63
        assert stats.profit_factor == 2.29
        assert stats.max_drawdown_pct == 46.63

    def test_parse_partial_data(self):
        """Should handle missing fields gracefully"""
        data = {"roeAllTime": 100.0}
        stats = parse_superx_stats("0xtest", data)

        assert stats.roe_all_time == 100.0
        assert stats.pnl_7d is None
        assert stats.win_rate is None


class TestAnalyzeTrades:
    """Test trade analysis"""

    def test_analyze_empty_trades(self):
        """Should raise error for empty trades"""
        with pytest.raises(ValueError, match="No trade data"):
            analyze_trades([])

    def test_analyze_single_trade(self):
        """Should analyze single trade correctly"""
        fills = [{
            "time": 1700000000000,
            "coin": "BTC",
            "side": "B",
            "px": "50000",
            "sz": "0.1",
            "closedPnl": "100.0",
            "fee": "5.0",
            "user": "0xtest"
        }]
        result = analyze_trades(fills)

        assert result.total_trades == 1
        assert result.total_pnl == 100.0
        assert result.total_fees == 5.0
        assert result.net_pnl == 95.0
        assert result.win_rate == 100.0

    def test_analyze_mixed_trades(self):
        """Should calculate win rate correctly for mixed trades"""
        fills = [
            {"time": 1700000000000, "closedPnl": "100.0", "fee": "1.0", "user": "0xtest"},
            {"time": 1700000001000, "closedPnl": "-50.0", "fee": "1.0", "user": "0xtest"},
            {"time": 1700000002000, "closedPnl": "200.0", "fee": "1.0", "user": "0xtest"},
            {"time": 1700000003000, "closedPnl": "-30.0", "fee": "1.0", "user": "0xtest"},
        ]
        result = analyze_trades(fills)

        assert result.total_trades == 4
        assert result.winning_trades == 2
        assert result.losing_trades == 2
        assert result.win_rate == 50.0

    def test_max_drawdown_calculation(self):
        """Should calculate max drawdown correctly"""
        fills = [
            {"time": 1700000000000, "closedPnl": "100.0", "fee": "0", "user": "0xtest"},
            {"time": 1700000001000, "closedPnl": "100.0", "fee": "0", "user": "0xtest"},
            {"time": 1700000002000, "closedPnl": "-150.0", "fee": "0", "user": "0xtest"},
            {"time": 1700000003000, "closedPnl": "50.0", "fee": "0", "user": "0xtest"},
        ]
        result = analyze_trades(fills)

        # Peak was 200, dropped to 50, drawdown = -150
        assert result.max_drawdown == -150.0
