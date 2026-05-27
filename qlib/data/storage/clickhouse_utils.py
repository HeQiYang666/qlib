# qlib/data/storage/clickhouse_utils.py

import os
import threading
from urllib.parse import urlparse, unquote
from typing import Dict, Optional

from clickhouse_driver import Client

from qlib.config import C
from qlib.log import get_module_logger

logger = get_module_logger("clickhouse_storage")

# Per-process client instances to survive fork() in multiprocessing
_clients: Dict[int, Client] = {}
_lock = threading.Lock()


def _create_client() -> Client:
    """Create a new ClickHouse native-protocol client."""
    ch_cfg = C.get("clickhouse_cache", {})
    if not ch_cfg:
        ch_cfg = _parse_clickhouse_uri(C["provider_uri"])
        C["clickhouse_cache"] = ch_cfg

    return Client(
        host=ch_cfg["host"],
        port=ch_cfg["port"],
        user=ch_cfg.get("username", "default"),
        password=ch_cfg.get("password", ""),
        database=ch_cfg.get("database", "stock_l2"),
        connect_timeout=ch_cfg.get("connect_timeout", 30),
        send_receive_timeout=ch_cfg.get("send_receive_timeout", 300),
    )


def get_client():
    """Return a per-process ClickHouse Client. Creates one lazily per PID."""
    pid = os.getpid()
    if pid in _clients:
        return _clients[pid]

    with _lock:
        if pid in _clients:
            return _clients[pid]
        _clients[pid] = _create_client()
    return _clients[pid]


def _parse_clickhouse_uri(provider_uri) -> dict:
    """Parse clickhouse://user:pass@host:port/db into connection dict.

    If provider_uri is a dict (multi-freq), use the value of the first key
    that starts with 'clickhouse://'.
    """
    uri = provider_uri
    if isinstance(provider_uri, dict):
        uri = None
        for v in provider_uri.values():
            if isinstance(v, str) and v.startswith("clickhouse://"):
                uri = v
                break
        if not isinstance(uri, str):
            raise ValueError(
                f"provider_uri dict contains no 'clickhouse://' entry; got {provider_uri}"
            )

    if not isinstance(uri, str):
        raise ValueError(
            f"provider_uri must be a dict with a clickhouse:// entry or a clickhouse:// string; got {type(uri).__name__}: {uri}"
        )

    parsed = urlparse(uri)
    cfg = {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 9000,
        "username": unquote(parsed.username) if parsed.username else "default",
        "password": unquote(parsed.password) if parsed.password else "",
        "database": parsed.path.lstrip("/") or "stock_l2",
        "connect_timeout": 30,
    }
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


_DEFAULT_MULTIPLIERS = {"price": 10000, "amount": 1, "volume": 1}


def price_to_float(ch_value: int, field_type: str = "price") -> float:
    """Convert ClickHouse Int64 price/amount to float in yuan.

    Prices (open/high/low/close): stored × 10000
    Amounts: stored in yuan directly (no multiplier)
    Volumes: stored in shares (no multiplier)
    """
    default = _DEFAULT_MULTIPLIERS.get(field_type, 10000)
    multiplier = C.get("price_multiplier", {}).get(field_type, default)
    if multiplier == 1:
        return float(ch_value)
    return ch_value / multiplier


# Field name mapping: Qlib expression name → ClickHouse column + aggregation
FIELD_MAP = {
    "open": ("open_price", "argMax", "ts", "price"),
    "high": ("high_price", "max", None, "price"),
    "low": ("low_price", "minIf", "low_price > 0", "price"),
    "close": ("last_price", "argMax", "ts", "price"),
    "volume": ("volume", "sum", None, "volume"),
    "amount": ("amount", "sum", None, "amount"),
    "vwap": ("amount", "sum", None, "volume"),  # sum(amount)/sum(volume) gives price in yuan directly
}
