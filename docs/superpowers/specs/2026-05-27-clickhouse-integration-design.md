# ClickHouse 数据接入 Qlib 因子挖掘 — 方案设计

## 目标

将本地 ClickHouse `stock_l2` 数据库的 L2 行情数据接入 Qlib，实现从本地数据源直接进行因子挖掘、模型训练和回测的完整流水线。

## 数据现状

- **数据库**: ClickHouse `stock_l2`，账户 hqy / hqy888
- **数据量**: ~470 GB，覆盖 8743 只股票，2025-12 至 2026-01
- **三张表**:
  - `snapshot_ticks` (253 GB, 12亿行) — 3秒快照，10档深度，OHLC，累计成交
  - `trade_ticks` (105 GB, 126亿行) — 逐笔成交，含 BS 标记、订单序号
  - `order_ticks` (111 GB, 194亿行) — 逐笔委托，含委托类型/方向
- **价格精度**: Int64，price × 10000，amount × 10000

## 技术路线：方案三 — 分层计算

只动数据存储层 (Storage)，不动表达式引擎 (Expression) 及以上。

```
Expression 引擎 ─→ FeatureStorage (ClickHouse) ─→ ClickHouse snapshot_ticks
Expression 引擎 ─→ FeatureStorage (file)        ─→ 磁盘二进制文件 (保留)
```

## 架构变更

### 新增文件

| 文件 | 内容 |
|------|------|
| `qlib/data/storage/clickhouse_storage.py` | `ClickHouseCalendarStorage`, `ClickHouseInstrumentStorage`, `ClickHouseFeatureStorage` |
| `qlib/data/storage/clickhouse_utils.py` | 连接池、URI 解析、价格精度转换、SQL 构建辅助 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `qlib/config.py` | `DataPathManager` 识别 `clickhouse://` URI 类型；新增 `clickhouse` 配置项 |
| `qlib/data/data.py` | `ProviderBackendMixin` 根据 URI 类型自动选择 ClickHouse/File storage |

### 不改动

Expression 引擎、Dataset 层、Model 层、Backtest 层、Workflow/CLI 层。

## 各 Storage 设计

### ClickHouseCalendarStorage

- `calendar(start_time, end_time, freq)` → `pd.Series`
- **day**: `SELECT DISTINCT trade_date FROM snapshot_ticks ORDER BY trade_date`
- **1min**: `SELECT DISTINCT toStartOfMinute(ts) AS bar FROM snapshot_ticks ORDER BY bar`
- 结果缓存到内存

### ClickHouseInstrumentStorage

- `instruments()` → 所有股票列表
- 从三张表的 `symbol` 字段 de-duplicate 得到
- 上市/退市时间：按 symbol 的 min/max trade_date 推断

### ClickHouseFeatureStorage (核心)

- 实现 `feature(instrument, field, start_index, end_index, freq)` → `pd.Series`
- **批量查询**: 同一次请求内，多个 instrument + field 合并为一条 SQL
- **日频**:

```sql
SELECT symbol, trade_date AS bar,
  argMax(open_price, ts) AS open,
  max(high_price) AS high,
  min(low_price) AS low,
  argMax(last_price, ts) AS close,
  sum(volume) AS volume,
  sum(amount) AS amount
FROM snapshot_ticks
WHERE symbol IN (...) AND trade_date BETWEEN '...' AND '...'
GROUP BY symbol, trade_date
ORDER BY symbol, trade_date
```

- **分钟频**: 用 `toStartOfMinute(ts)` 做聚合，取 OHLC
- **字段映射** (Int64 → Python float，除以 10000):

| Qlib 字段 | ClickHouse 来源 |
|-----------|----------------|
| `$open` | `open_price / 10000` |
| `$high` | `high_price / 10000` |
| `$low` | `low_price / 10000` |
| `$close` | `last_price / 10000` |
| `$volume` | `volume` |
| `$amount` | `amount / 10000` |
| `$vwap` | `amount / volume` |

### Symbol 格式转换

- ClickHouse: `000001.SZ`
- Qlib: `sz000001`
- 转换逻辑在 ClickHouse utils 中统一处理

### 连接管理

- 使用 `clickhouse-connect` Python 驱动
- 连接池单例，线程安全
- 连接参数从 provider_uri 解析

## 配置变更

### provider_uri 格式

```python
qlib.init(provider_uri={
    "day": "clickhouse://hqy:hqy888@localhost:8123/stock_l2",
    "1min": "clickhouse://hqy:hqy888@localhost:8123/stock_l2",
})
```

`DataPathManager.get_uri_type()` 新增 `CLICKHOUSE_URI` 返回类型。

### 新增配置项

```python
"clickhouse": {
    "host": "localhost",
    "port": 8123,
    "username": "hqy",
    "password": "hqy888",
    "database": "stock_l2",
    "connect_timeout": 30,
}

"price_multiplier": {
    "price": 10000,    # Int64 价格 → 元
    "amount": 10000,   # Int64 金额 → 元
    "volume": 1,       # 股数，不需要转换
}
```

## L2 因子扩展（第二版规划）

第一版只用 `snapshot_ticks` 提供 OHLC 基础因子。后续新增 L2 算子只需：

1. 创建 `ExpressionOps` 子类
2. 实现 `_load_internal()` — 调用 ClickHouse 从 trade_ticks/order_ticks 聚合
3. 实现 `get_longest_back_rolling()` / `get_extended_window_size()` — 声明窗口

代码结构在第一版设计时留好扩展点（`ClickHouseFeatureStorage` 接受 table 参数，算子可以在 `_load_internal` 中使用 storage 实例直接查 ClickHouse）。

## 依赖

- 新增 Python 依赖: `clickhouse-connect`（或 `clickhouse-driver`）

## 测试

- 单元测试: `ClickHouseCalendarStorage` / `ClickHouseFeatureStorage` 独立测试
- 集成测试: `qlib.init(provider_uri="clickhouse://...")` → `D.features()` 跑通
- 端到端: 用 `qrun config.yaml` 跑一个完整 training pipeline
