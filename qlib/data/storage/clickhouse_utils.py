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
