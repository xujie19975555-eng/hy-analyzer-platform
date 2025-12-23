"""Tests for schemas validation"""
import pytest
from app.models.schemas import WalletAddress, TraderStats


class TestWalletAddress:
    """Test wallet address validation"""

    def test_valid_address(self):
        """Valid Ethereum address should pass"""
        addr = WalletAddress(address="0x6c06bfd51ea8032ddaeea8b69009417b54f3587b")
        assert addr.address == "0x6c06bfd51ea8032ddaeea8b69009417b54f3587b"

    def test_valid_address_uppercase(self):
        """Uppercase address should be normalized to lowercase"""
        addr = WalletAddress(address="0x6C06BFD51EA8032DDAEEA8B69009417B54F3587B")
        assert addr.address == "0x6c06bfd51ea8032ddaeea8b69009417b54f3587b"

    def test_invalid_address_short(self):
        """Short address should fail"""
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            WalletAddress(address="0x6c06bfd51ea8032")

    def test_invalid_address_no_prefix(self):
        """Address without 0x prefix should fail"""
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            WalletAddress(address="6c06bfd51ea8032ddaeea8b69009417b54f3587b")

    def test_invalid_address_wrong_chars(self):
        """Address with invalid characters should fail"""
        with pytest.raises(ValueError, match="Invalid Ethereum address"):
            WalletAddress(address="0xZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ")


class TestTraderStats:
    """Test TraderStats model"""

    def test_create_trader_stats(self):
        """Should create TraderStats with valid data"""
        stats = TraderStats(
            address="0x6c06bfd51ea8032ddaeea8b69009417b54f3587b",
            roe_all_time=1650.29,
            pnl_all_time=1682095.47,
            win_rate=77.63
        )
        assert stats.roe_all_time == 1650.29
        assert stats.pnl_all_time == 1682095.47

    def test_trader_stats_optional_fields(self):
        """Optional fields should default to None"""
        stats = TraderStats(address="0x6c06bfd51ea8032ddaeea8b69009417b54f3587b")
        assert stats.roe_7d is None
        assert stats.profit_factor is None
