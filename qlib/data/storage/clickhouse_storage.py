# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd

from qlib.data.storage import CalendarStorage, CalVT, InstrumentStorage, InstKT, InstVT
from qlib.data.storage.clickhouse_utils import ch_symbol_to_qlib, get_client
from qlib.log import get_module_logger

logger = get_module_logger("clickhouse_storage")


class ClickHouseCalendarStorage(CalendarStorage):
    """Read trading calendar from ClickHouse snapshot_ticks table."""

    def __init__(self, freq: str, future: bool, **kwargs):
        super().__init__(freq, future, **kwargs)
        self._data_cache = None

    @property
    def data(self) -> List[CalVT]:
        if self.future:
            raise NotImplementedError(
                "ClickHouseCalendarStorage does not support future calendar data"
            )

        if self._data_cache is not None:
            return self._data_cache

        try:
            client = get_client()
            if self.freq == "day":
                query = (
                    "SELECT DISTINCT trade_date "
                    "FROM snapshot_ticks "
                    "ORDER BY trade_date"
                )
                rows = client.query(query).result_rows
                result = [pd.Timestamp(r[0]).strftime("%Y-%m-%d") for r in rows]
            else:
                # Minute-level calendar: distinct minute bars
                query = (
                    "SELECT DISTINCT toStartOfMinute(ts) AS bar "
                    "FROM snapshot_ticks "
                    "ORDER BY bar"
                )
                rows = client.query(query).result_rows
                result = [pd.Timestamp(r[0]).strftime("%Y-%m-%d %H:%M:%S") for r in rows]
        except Exception as e:
            raise ValueError(f"Failed to query calendar data from ClickHouse: {e}") from e

        self._data_cache = result
        return result

    def clear(self) -> None:
        self._data_cache = None

    def extend(self, iterable: Iterable[CalVT]) -> None:
        raise NotImplementedError("ClickHouseCalendarStorage is read-only")

    def index(self, value: CalVT) -> int:
        return self.data.index(value)

    def insert(self, index: int, value: CalVT) -> None:
        raise NotImplementedError("ClickHouseCalendarStorage is read-only")

    def remove(self, value: CalVT) -> None:
        raise NotImplementedError("ClickHouseCalendarStorage is read-only")

    def __setitem__(self, i, value) -> None:
        raise NotImplementedError("ClickHouseCalendarStorage is read-only")

    def __delitem__(self, i) -> None:
        raise NotImplementedError("ClickHouseCalendarStorage is read-only")

    def __getitem__(self, i: Union[int, slice]) -> Union[CalVT, List[CalVT]]:
        return self.data[i]

    def __len__(self) -> int:
        return len(self.data)


class ClickHouseInstrumentStorage(InstrumentStorage):
    """Read instrument (stock) list from ClickHouse."""

    def __init__(self, market: str, freq: str, **kwargs):
        super().__init__(market, freq, **kwargs)
        self._data_cache = None

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        if self._data_cache is not None:
            return self._data_cache

        try:
            client = get_client()
            query = (
                "SELECT symbol, "
                "  min(trade_date) AS start_date, "
                "  max(trade_date) AS end_date "
                "FROM snapshot_ticks "
                "GROUP BY symbol "
                "ORDER BY symbol"
            )
            rows = client.query(query).result_rows

            result: Dict[InstKT, InstVT] = {}
            for symbol, start_date, end_date in rows:
                qlib_sym = ch_symbol_to_qlib(symbol)
                start_str = pd.Timestamp(start_date).strftime("%Y-%m-%d")
                end_str = pd.Timestamp(end_date).strftime("%Y-%m-%d")
                result[qlib_sym] = [(start_str, end_str)]

            self._data_cache = result
            return result
        except Exception as e:
            raise ValueError(f"Failed to query instrument data from ClickHouse: {e}") from e

    def clear(self) -> None:
        self._data_cache = None

    def update(self, *args, **kwargs) -> None:
        raise NotImplementedError("ClickHouseInstrumentStorage is read-only")

    def __setitem__(self, k: InstKT, v: InstVT) -> None:
        raise NotImplementedError("ClickHouseInstrumentStorage is read-only")

    def __delitem__(self, k: InstKT) -> None:
        raise NotImplementedError("ClickHouseInstrumentStorage is read-only")

    def __getitem__(self, k: InstKT) -> InstVT:
        return self.data[k]

    def __len__(self) -> int:
        return len(self.data)
