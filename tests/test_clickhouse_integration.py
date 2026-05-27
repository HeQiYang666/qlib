import pytest
import pandas as pd
import qlib
from qlib.data import D


@pytest.fixture(scope="module")
def init_qlib():
    qlib.init(
        provider_uri="clickhouse://hqy:hqy888@localhost:9000/stock_l2",
        region="cn",
        auto_mount=False,
    )
    yield


class TestClickHouseIntegration:
    def test_calendar_available(self, init_qlib):
        """Verify calendar loads from ClickHouse."""
        cal = D.calendar(start_time="2025-12-01", end_time="2025-12-05")
        assert len(cal) > 0
        assert cal[0].strftime("%Y-%m-%d") == "2025-12-01"

    def test_instruments_available(self, init_qlib):
        """Verify instrument list loads from ClickHouse."""
        config = D.instruments(market="all")
        stocks = D.list_instruments(config, freq="day", as_list=True)
        assert "sz000001" in stocks
        assert len(stocks) > 100

    def test_features_loading(self, init_qlib):
        """Verify loading features from ClickHouse."""
        df = D.features(
            ["sz000001"],
            ["$close"],
            start_time="2025-12-01",
            end_time="2025-12-05",
            freq="day",
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "$close" in df.columns

    def test_list_instruments_count(self, init_qlib):
        """Verify list_instruments returns reasonable stock count."""
        config = D.instruments(market="all")
        stocks = D.list_instruments(config, freq="day", as_list=True)
        assert len(stocks) > 5000
