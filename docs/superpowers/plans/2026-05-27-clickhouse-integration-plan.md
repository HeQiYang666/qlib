# ClickHouse 数据接入 Qlib — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 ClickHouse 数据存储层，使 Qlib 可以从本地 ClickHouse `stock_l2` 数据库直接读取行情数据用于因子挖掘和回测。

**Architecture:** 在 `qlib/data/storage/` 下新增 3 个 ClickHouse Storage 类 + 1 个工具模块，修改 `config.py` 识别 `clickhouse://` URI，修改 `data.py` 的 `ProviderBackendMixin` 自动路由。不改动 Expression/Dataset/Model/Backtest/Workflow 任何上层代码。

**Tech Stack:** Python, ClickHouse (clickhouse-connect driver), pandas, numpy

**Data flow reminder:**
```
Expression.load(instrument, start_index, end_index, freq)
  → Feature._load_internal(instrument, start_index, end_index, freq)
  → FeatureD.feature(instrument, field, start_index, end_index, freq)
  → LocalFeatureProvider.feature(instrument, field, start_index, end_index, freq)
  → backend_obj(instrument, field, freq)[start_index : end_index + 1]   # <-- slice with calendar indices
  → FeatureStorage.__getitem__(slice)
```

---

### Task 1: Install clickhouse-connect dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `clickhouse-connect` to project dependencies**

In `pyproject.toml`, add to the `dependencies` list:

```toml
dependencies = [
  ...
  "clickhouse-connect",
  ...
]
```

Exact location: after `"pydantic-settings"` line.

- [ ] **Step 2: Install the dependency**

```bash
pip install clickhouse-connect
```

- [ ] **Step 3: Verify import works**

```bash
python -c "import clickhouse_connect; print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add clickhouse-connect dependency"
```

---

### Task 2: Create clickhouse_utils.py — 连接/URI/符号/价格转换

**Files:**
- Create: `qlib/data/storage/clickhouse_utils.py`

- [ ] **Step 1: Create the module with connection manager and helpers**

```python
# qlib/data/storage/clickhouse_utils.py

import re
from urllib.parse import urlparse, unquote
from typing import Optional

import clickhouse_connect
import pandas as pd

from qlib.config import C
from qlib.log import get_module_logger

logger = get_module_logger("clickhouse_storage")

# Single shared client instance, lazily initialized
_client: Optional[clickhouse_connect.driver.Client] = None


def get_client():
    """Return a shared clickhouse-connect client. Connects lazily on first call."""
    global _client
    if _client is not None:
        return _client

    ch_cfg = C.get("clickhouse_cache", {})
    if not ch_cfg:
        ch_cfg = _parse_clickhouse_uri(C["provider_uri"])

    _client = clickhouse_connect.get_client(
        host=ch_cfg["host"],
        port=ch_cfg["port"],
        username=ch_cfg.get("username", "default"),
        password=ch_cfg.get("password", ""),
        database=ch_cfg.get("database", "stock_l2"),
        connect_timeout=ch_cfg.get("connect_timeout", 30),
    )
    return _client


def _parse_clickhouse_uri(provider_uri) -> dict:
    """Parse clickhouse://user:pass@host:port/db into connection dict.
    
    If provider_uri is a dict (multi-freq), use the value of the first key
    that starts with 'clickhouse://'.
    """
    uri = provider_uri
    if isinstance(provider_uri, dict):
        for v in provider_uri.values():
            if isinstance(v, str) and v.startswith("clickhouse://"):
                uri = v
                break

    parsed = urlparse(uri)
    cfg = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 8123,
        "username": unquote(parsed.username) if parsed.username else "default",
        "password": unquote(parsed.password) if parsed.password else "",
        "database": parsed.path.lstrip("/") or "stock_l2",
        "connect_timeout": 30,
    }
    C["clickhouse_cache"] = cfg
    return cfg


def ch_symbol_to_qlib(ch_symbol: str) -> str:
    """Convert '000001.SZ' → 'sz000001'."""
    code, market = ch_symbol.split(".")
    return f"{market.lower()}{code}"


def qlib_symbol_to_ch(qlib_symbol: str) -> str:
    """Convert 'sz000001' → '000001.SZ'."""
    market = qlib_symbol[:2].upper()
    code = qlib_symbol[2:]
    return f"{code}.{market}"


def price_to_float(ch_value: int, field_type: str = "price") -> float:
    """Convert ClickHouse Int64 price/amount to float in yuan.
    
    Prices: divided by 10000
    Amounts: divided by 10000
    Volumes: no conversion
    """
    multiplier = C.get("price_multiplier", {}).get(field_type, 10000)
    if multiplier == 1:
        return float(ch_value)
    return ch_value / multiplier


# Field name mapping: Qlib expression name → ClickHouse column + aggregation
FIELD_MAP = {
    "open": ("open_price", "argMax", "ts", "price"),
    "high": ("high_price", "max", None, "price"),
    "low": ("low_price", "min", None, "price"),
    "close": ("last_price", "argMax", "ts", "price"),
    "volume": ("volume", "sum", None, "volume"),
    "amount": ("amount", "sum", None, "amount"),
    "vwap": ("amount", "sum", None, "amount"),  # handled specially: sum(amount)/sum(volume)
}
```

- [ ] **Step 2: Commit**

```bash
git add qlib/data/storage/clickhouse_utils.py
git commit -m "feat: add ClickHouse connection manager and helpers"
```

---

### Task 3: Create ClickHouseCalendarStorage

**Files:**
- Modify: `qlib/data/storage/clickhouse_utils.py` (add CalendarStorage class)
  - Or better: Create the storage file first, then we'll put all 3 classes into `clickhouse_storage.py`
- Create: `qlib/data/storage/clickhouse_storage.py`

**Note:** The plan puts all 3 storage classes in `clickhouse_storage.py` as the spec says. But we'll build them incrementally.

- [ ] **Step 1: Create `clickhouse_storage.py` with ClickHouseCalendarStorage**

```python
# qlib/data/storage/clickhouse_storage.py

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
        if self._data_cache is not None:
            return self._data_cache

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
```

- [ ] **Step 2: Verify the CalendarStorage interface is correct**

Check that the class properly inherits from `CalendarStorage` and implements all required methods by comparing with `file_storage.py:FileCalendarStorage`.

- [ ] **Step 3: Quick smoke test**

```bash
python -c "
from qlib.data.storage.clickhouse_storage import ClickHouseCalendarStorage
cs = ClickHouseCalendarStorage(freq='day', future=False)
print(f'Calendar days: {len(cs.data)}')
print(f'First 3: {cs.data[:3]}')
print(f'Last 3: {cs.data[-3:]}')
"
```

- [ ] **Step 4: Commit**

```bash
git add qlib/data/storage/clickhouse_storage.py
git commit -m "feat: add ClickHouseCalendarStorage"
```

---

### Task 4: Create ClickHouseInstrumentStorage

**Files:**
- Modify: `qlib/data/storage/clickhouse_storage.py` (append class)

- [ ] **Step 1: Add ClickHouseInstrumentStorage class**

Append to `clickhouse_storage.py`:

```python
from typing import Dict

from qlib.data.storage import InstrumentStorage, InstKT, InstVT
from qlib.data.storage.clickhouse_utils import ch_symbol_to_qlib, get_client


class ClickHouseInstrumentStorage(InstrumentStorage):
    """Read instrument (stock) list from ClickHouse."""

    def __init__(self, market: str, freq: str, **kwargs):
        super().__init__(market, freq, **kwargs)
        self._data_cache = None

    @property
    def data(self) -> Dict[InstKT, InstVT]:
        if self._data_cache is not None:
            return self._data_cache

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
```

- [ ] **Step 2: Smoke test**

```bash
python -c "
from qlib.data.storage.clickhouse_storage import ClickHouseInstrumentStorage
is_ = ClickHouseInstrumentStorage(market='all', freq='day')
data = is_.data
print(f'Instruments count: {len(data)}')
print(f'First 3: {list(data.items())[:3]}')
"
```

- [ ] **Step 3: Commit**

```bash
git add qlib/data/storage/clickhouse_storage.py
git commit -m "feat: add ClickHouseInstrumentStorage"
```

---

### Task 5: Create ClickHouseFeatureStorage (core)

**Files:**
- Modify: `qlib/data/storage/clickhouse_storage.py` (append class)

- [ ] **Step 1: Add ClickHouseFeatureStorage class**

```python
import numpy as np
import pandas as pd
from qlib.data.storage import FeatureStorage


class ClickHouseFeatureStorage(FeatureStorage):
    """Read feature data from ClickHouse snapshot_ticks table."""

    def __init__(self, instrument: str, field: str, freq: str, **kwargs):
        super().__init__(instrument, field, freq, **kwargs)
        self._calendar = None

    def _get_calendar(self) -> list:
        """Get the calendar as a list of date strings for index→date conversion."""
        if self._calendar is not None:
            return self._calendar
        # Reuse ClickHouseCalendarStorage to fetch the calendar
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
            # Fallback: try to treat field as a direct column name
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
            # VWAP = sum(amount) / sum(volume), handle specially
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
        result = client.query(query, parameters=params)
        if result.result_rows:
            idx = [pd.Timestamp(r[0]) for r in result.result_rows]
            vals = [price_to_float(r[1], price_type) for r in result.result_rows]
            return pd.Series(vals, index=idx)
        return pd.Series(dtype=np.float32)

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

    def __getitem__(self, i: Union[int, slice]) -> Union[tuple, pd.Series]:
        cal = self._get_calendar()

        if isinstance(i, int):
            # Return (index, value) tuple, matching FileFeatureStorage protocol
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

            # Reindex to match the requested calendar index range
            cal_subset = cal[start:stop:step]
            cal_subset_dt = [pd.Timestamp(c) for c in cal_subset]
            result = series.reindex(cal_subset_dt)
            result.index = pd.RangeIndex(start, start + len(result), step)
            return result

        else:
            raise TypeError(f"type(i) = {type(i)}")

    @property
    def data(self) -> pd.Series:
        return self[:]

    def __len__(self) -> int:
        return len(self._get_calendar())
```

- [ ] **Step 2: Smoke test — fetch single feature**

```bash
python -c "
import qlib
qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')  # needed for global C init only
from qlib.data.storage.clickhouse_storage import ClickHouseFeatureStorage
from qlib.data.storage.clickhouse_utils import get_client

# Test get_client first
client = get_client()
print(f'Client: {client}')

# Test feature storage
fs = ClickHouseFeatureStorage(instrument='sz000001', field='close', freq='day')
data = fs[0:5]
print(f'First 5 data points:\n{data}')
"
```

- [ ] **Step 3: Commit**

```bash
git add qlib/data/storage/clickhouse_storage.py
git commit -m "feat: add ClickHouseFeatureStorage"
```

---

### Task 6: Modify config.py + __init__.py — support clickhouse:// URI

**Files:**
- Modify: `qlib/config.py`
- Modify: `qlib/__init__.py`

- [ ] **Step 1: Add CLICKHOUSE_URI constant and update get_uri_type**

In `QlibConfig` class, `LOCAL_URI = "local"` and `NFS_URI = "nfs"` block — add one line:

```python
CLICKHOUSE_URI = "clickhouse"
```

In `DataPathManager.get_uri_type()`, add a check at the start of the method:

```python
@staticmethod
def get_uri_type(uri: Union[str, Path]):
    uri = uri if isinstance(uri, str) else str(uri.expanduser().resolve())
    if isinstance(uri, str) and uri.startswith("clickhouse://"):
        return QlibConfig.CLICKHOUSE_URI
    # ... rest of existing method unchanged
```

- [ ] **Step 2: Handle CLICKHOUSE_URI in get_data_uri, resolve_path, and init**

In `DataPathManager.get_data_uri()`, add an early return:

```python
def get_data_uri(self, freq: Optional[Union[str, Freq]] = None) -> Path:
    # ... existing freq handling ...
    _provider_uri = self.provider_uri[freq]
    if self.get_uri_type(_provider_uri) == QlibConfig.CLICKHOUSE_URI:
        # ClickHouse doesn't have a filesystem path; return a virtual path
        return Path(_provider_uri)
    # ... rest of existing method
```

In `QlibConfig.resolve_path()`, skip path resolution for ClickHouse URIs:

```python
def resolve_path(self):
    _mount_path = self["mount_path"]
    _provider_uri = self.DataPathManager.format_provider_uri(self["provider_uri"])
    if not isinstance(_mount_path, dict):
        _mount_path = {_freq: _mount_path for _freq in _provider_uri.keys()}

    for _freq in _provider_uri.keys():
        if self.DataPathManager.get_uri_type(_provider_uri[_freq]) == self.CLICKHOUSE_URI:
            # ClickHouse URIs don't need path resolution
            continue
        _mount_path[_freq] = (
            _mount_path[_freq] if _mount_path[_freq] is None else str(Path(_mount_path[_freq]).expanduser())
        )
    self["provider_uri"] = _provider_uri
    self["mount_path"] = _mount_path
```

- [ ] **Step 3: Add CLICKHOUSE_URI handling in qlib/__init__.py**

In `qlib/__init__.py` function `init()`, the URI-type check loop currently raises `NotImplementedError` for unknown types. Add a ClickHouse branch:

```python
# In the for _freq, provider_uri loop, add after NFS_URI branch:
elif uri_type == C.CLICKHOUSE_URI:
    pass  # ClickHouse doesn't need filesystem mounts
```

- [ ] **Step 4: Commit**

```bash
git add qlib/config.py
git commit -m "feat: add clickhouse:// URI type support in config"
```

---

### Task 7: Modify data.py — auto-route to ClickHouse storage

**Files:**
- Modify: `qlib/data/data.py`

- [ ] **Step 1: Update ProviderBackendMixin to route ClickHouse storage**

Replace `get_default_backend` in `ProviderBackendMixin`:

```python
class ProviderBackendMixin:
    def get_default_backend(self):
        backend = {}
        provider_name: str = re.findall("[A-Z][^A-Z]*", self.__class__.__name__)[-2]
        uri_type = self._get_uri_type()

        if uri_type == C.CLICKHOUSE_URI:
            backend.setdefault("class", f"ClickHouse{provider_name}Storage")
            backend.setdefault("module_path", "qlib.data.storage.clickhouse_storage")
        else:
            backend.setdefault("class", f"File{provider_name}Storage")
            backend.setdefault("module_path", "qlib.data.storage.file_storage")
        return backend

    def _get_uri_type(self):
        """Determine the URI type from provider_uri config."""
        provider_uri = C.get("provider_uri", "")
        if isinstance(provider_uri, str):
            return C.DataPathManager.get_uri_type(provider_uri)
        elif isinstance(provider_uri, dict):
            for v in provider_uri.values():
                if isinstance(v, str) and v.startswith("clickhouse://"):
                    return C.CLICKHOUSE_URI
        return None

    def backend_obj(self, **kwargs):
        backend = self.backend if self.backend else self.get_default_backend()
        backend = copy.deepcopy(backend)
        backend.setdefault("kwargs", {}).update(**kwargs)
        return init_instance_by_config(backend)
```

- [ ] **Step 2: Commit**

```bash
git add qlib/data/data.py
git commit -m "feat: auto-route to ClickHouse storage via provider_uri"
```

---

### Task 8: Integration test — qlib.init + D.features()

**Files:**
- Create: `tests/test_clickhouse_integration.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_clickhouse_integration.py
import pytest
import pandas as pd
import qlib
from qlib.data import D


@pytest.fixture(scope="module")
def init_qlib():
    qlib.init(
        provider_uri="clickhouse://hqy:hqy888@localhost:8123/stock_l2",
        region="cn",
        auto_mount=False,
    )
    yield
    # No explicit cleanup needed


class TestClickHouseIntegration:
    def test_calendar_available(self, init_qlib):
        """Verify calendar loads from ClickHouse."""
        cal = D.calendar(start_time="2025-12-01", end_time="2025-12-05")
        assert len(cal) > 0
        assert cal[0].strftime("%Y-%m-%d") == "2025-12-01"

    def test_instruments_available(self, init_qlib):
        """Verify instrument list loads from ClickHouse."""
        inst = D.instruments(market="all")
        assert "sz000001" in inst
        assert len(inst) > 100

    def test_feature_loading(self, init_qlib):
        """Verify loading a single feature field."""
        series = D.feature(
            instrument="sz000001",
            field="$close",
            start_time="2025-12-01",
            end_time="2025-12-05",
            freq="day",
        )
        assert isinstance(series, pd.Series)
        assert len(series) > 0

    def test_list_instruments(self, init_qlib):
        """Verify list_instruments returns stock codes."""
        stocks = D.list_instruments(
            instruments="all",
            start_time="2025-12-01",
            end_time="2025-12-05",
            freq="day",
            as_list=True,
        )
        assert len(stocks) > 100
        assert "sz000001" in stocks
```

- [ ] **Step 2: Run the integration tests**

```bash
pytest tests/test_clickhouse_integration.py -v
```

Expected: All 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_clickhouse_integration.py
git commit -m "test: add ClickHouse integration tests"
```

---

### Task 9: End-to-end test — qrun with config.yaml

**Files:**
- Create: `examples/clickhouse_workflow_example.yaml`

- [ ] **Step 1: Create a minimal qrun config using ClickHouse data**

```yaml
# examples/clickhouse_workflow_example.yaml
qlib_init:
  provider_uri: "clickhouse://hqy:hqy888@localhost:8123/stock_l2"
  region: cn

task:
  model:
    class: LGBModel
    module_path: qlib.contrib.model.gbdt
    kwargs:
      loss: mse
      early_stopping_rounds: 5
  dataset:
    class: DatasetH
    module_path: qlib.data.dataset
    kwargs:
      handler:
        class: Alpha158
        module_path: qlib.contrib.data.handler
        kwargs:
          start_time: 2025-12-01
          end_time: 2025-12-31
          fit_start_time: 2025-12-01
          fit_end_time: 2025-12-20
          instruments: csi300
          infer_processors: []
          learn_processors: []
      segments:
        train: [2025-12-01, 2025-12-20]
        valid: [2025-12-21, 2025-12-25]
        test: [2025-12-26, 2025-12-31]
```

- [ ] **Step 2: Run the workflow**

```bash
qrun examples/clickhouse_workflow_example.yaml
```

Expected: Training completes successfully, MLflow experiment logged.

- [ ] **Step 3: Commit**

```bash
git add examples/clickhouse_workflow_example.yaml
git commit -m "example: add ClickHouse qrun workflow config"
```

---

### Task 10: Final verification and edge cases

**Files:**
- No new files. Verify everything works together.

- [ ] **Step 1: Run all integration tests together**

```bash
pytest tests/test_clickhouse_integration.py -v
```

- [ ] **Step 2: Test edge case — empty date range returns empty Series**

```bash
python -c "
import qlib
qlib.init(provider_uri='clickhouse://hqy:hqy888@localhost:8123/stock_l2', region='cn')
from qlib.data import D
series = D.feature('sz000001', '\$close', start_time='2010-01-01', end_time='2010-01-05', freq='day')
print(f'Empty range result: {series}')
assert len(series) == 0, 'Expected empty series for out-of-range dates'
print('OK')
"
```

- [ ] **Step 3: Test edge case — invalid instrument returns empty**

```bash
python -c "
import qlib
qlib.init(provider_uri='clickhouse://hqy:hqy888@localhost:8123/stock_l2', region='cn')
from qlib.data import D
series = D.feature('sz999999', '\$close', start_time='2025-12-01', end_time='2025-12-05', freq='day')
print(f'Invalid instrument result: {series}')
print('OK (empty or NaN series expected)')
"
```

- [ ] **Step 4: Test 1min frequency support**

```bash
python -c "
import qlib
qlib.init(provider_uri={'1min': 'clickhouse://hqy:hqy888@localhost:8123/stock_l2'}, region='cn')
from qlib.data import D
cal = D.calendar(start_time='2025-12-01', end_time='2025-12-01', freq='1min')
print(f'1min calendar entries on one day: {len(cal)}')
print(f'First 3: {cal[:3]}')
print(f'Last 3: {cal[-3:]}')
"
```

- [ ] **Step 5: Run all tests one final time**

```bash
pytest tests/test_clickhouse_integration.py -v
```

- [ ] **Step 6: Commit any remaining changes**

```bash
git status
git commit -m "test: add edge case verification for ClickHouse integration"
```
