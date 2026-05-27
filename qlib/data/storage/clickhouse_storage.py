# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Iterable, List, Union

import numpy as np
import pandas as pd

from qlib.data.storage import CalendarStorage, CalVT
from qlib.data.storage.clickhouse_utils import get_client
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
