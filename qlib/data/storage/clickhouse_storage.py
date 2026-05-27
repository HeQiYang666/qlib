# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, Iterable, List, Union

import numpy as np
import pandas as pd

from qlib.data.storage import CalendarStorage, CalVT, FeatureStorage, InstrumentStorage, InstKT, InstVT
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


class ClickHouseFeatureStorage(FeatureStorage):
    """Read feature data from ClickHouse snapshot_ticks table."""

    def __init__(self, instrument: str, field: str, freq: str, **kwargs):
        super().__init__(instrument, field, freq, **kwargs)
        self._calendar = None

    def _get_calendar(self) -> list:
        """Get the calendar as a list of date strings for index-to-date conversion."""
        if self._calendar is not None:
            return self._calendar
        from qlib.data.storage.clickhouse_storage import ClickHouseCalendarStorage

        cal_storage = ClickHouseCalendarStorage(freq=self.freq, future=False)
        self._calendar = cal_storage.data
        return self._calendar

    def _query_feature(self, start_dt: str, end_dt: str) -> pd.Series:
        """Query a single instrument/field from ClickHouse and return a Series."""
        from qlib.data.storage.clickhouse_utils import (
            FIELD_MAP, get_client, price_to_float, qlib_symbol_to_ch,
        )

        client = get_client()
        ch_symbol = qlib_symbol_to_ch(self.instrument)

        if self.field not in FIELD_MAP:
            col_name = self.field
            agg_func = "argMax"
            agg_on = "ts"
            price_type = "price"
        else:
            col_name, agg_func, agg_on, price_type = FIELD_MAP[self.field]

        if self.freq == "day":
            time_col = "trade_date"
            time_label = "trade_date"
        else:
            time_col = "toStartOfMinute(ts)"
            time_label = "bar"

        if self.field == "vwap":
            query = (
                f"SELECT {time_col} AS {time_label}, "
                f"  sum(amount) / nullIf(sum(volume), 0) AS val "
                f"FROM snapshot_ticks "
                f"WHERE symbol = %(symbol)s "
                f"  AND trade_date BETWEEN %(start)s AND %(end)s "
                f"GROUP BY {time_label} "
                f"ORDER BY {time_label}"
            )
        elif agg_func == "argMax":
            query = (
                f"SELECT {time_col} AS {time_label}, "
                f"  argMax({col_name}, {agg_on}) AS val "
                f"FROM snapshot_ticks "
                f"WHERE symbol = %(symbol)s "
                f"  AND trade_date BETWEEN %(start)s AND %(end)s "
                f"GROUP BY {time_label} "
                f"ORDER BY {time_label}"
            )
        else:
            query = (
                f"SELECT {time_col} AS {time_label}, "
                f"  {agg_func}({col_name}) AS val "
                f"FROM snapshot_ticks "
                f"WHERE symbol = %(symbol)s "
                f"  AND trade_date BETWEEN %(start)s AND %(end)s "
                f"GROUP BY {time_label} "
                f"ORDER BY {time_label}"
            )

        params = {
            "symbol": ch_symbol,
            "start": str(pd.Timestamp(start_dt).date()),
            "end": str(pd.Timestamp(end_dt).date()),
        }
        try:
            result = client.query(query, parameters=params)
            if result.result_rows:
                idx = [pd.Timestamp(r[0]) for r in result.result_rows]
                vals = [price_to_float(r[1], price_type) for r in result.result_rows]
                return pd.Series(vals, index=idx)
            return pd.Series(dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to query feature data from ClickHouse: {e}") from e

    def clear(self) -> None:
        pass  # Read-only storage

    def write(self, data_array, index=None) -> None:
        raise NotImplementedError("ClickHouseFeatureStorage is read-only")

    @property
    def start_index(self) -> int:
        try:
            return 0
        except Exception:
            return None

    @property
    def end_index(self) -> int:
        try:
            cal = self._get_calendar()
            return len(cal) - 1
        except Exception:
            return None

    def __getitem__(self, i):
        cal = self._get_calendar()

        if isinstance(i, int):
            if i >= len(cal):
                return (None, None)
            start_dt = cal[i]
            end_dt = cal[i]
            series = self._query_feature(start_dt, end_dt)
            if len(series) > 0:
                return (i, float(series.iloc[0]))
            return (i, np.nan)

        elif isinstance(i, slice):
            start = i.start if i.start is not None else 0
            stop = i.stop if i.stop is not None else len(cal)
            step = i.step if i.step is not None else 1

            if start >= len(cal):
                return pd.Series(dtype=np.float32)

            stop = min(stop, len(cal))
            start_dt = cal[start]
            end_dt = cal[stop - 1]

            series = self._query_feature(start_dt, end_dt)

            if len(series) == 0:
                return pd.Series(dtype=np.float32)

            cal_subset = cal[start:stop:step]
            cal_subset_dt = [pd.Timestamp(c) for c in cal_subset]
            result = series.reindex(cal_subset_dt)
            result.index = pd.RangeIndex(start, start + len(result) * step, step)
            return result

        else:
            raise TypeError(f"type(i) = {type(i)}")

    @property
    def data(self) -> pd.Series:
        return self[:]

    def __len__(self) -> int:
        return len(self._get_calendar())
