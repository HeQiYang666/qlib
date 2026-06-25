# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build / Dev Commands

```bash
# Install with Cython extensions compiled (first-time setup)
make install

# Install all optional deps + dev tools
make dev

# Install specific optional dependency groups
make develop   # base dev tools (pytest, statsmodels)
make lint      # linting tools (black, pylint, flake8, mypy, nbqa)
make test      # test dependencies (yahooquery, baostock)
make rl        # RL dependencies (torch, tianshou)

# Compile Cython extensions only (if .so files missing)
make prerequisite

# Build wheel
make build
```

## Running Tests

```bash
# Run all tests (requires test deps: `make test`)
pytest tests/

# Run all tests except slow ones
pytest tests/ -m "not slow"

# Run a single test file
pytest tests/model/test_xxx.py

# Run a single test function
pytest tests/model/test_xxx.py::test_function_name
```

## Linting

```bash
# Individual linters (all require `make lint` first)
make black      # black -l 120 --check --diff
make pylint     # pylint on qlib/ and scripts/
make flake8     # flake8 on qlib/
make mypy       # mypy on qlib/

# All linters
make lint
```

Line length is 120 chars for black.

## High-Level Architecture

Qlib is an AI-oriented quantitative investment platform. The entry point is `qlib.init()` which sets up config, data paths, caching, and registers all components into the global singleton `C` (`qlib.config.C`).

### Data Layer (`qlib/data/`)

The data layer uses a **provider pattern** — `CalendarProvider`, `InstrumentProvider`, `FeatureProvider`, `ExpressionProvider`, `DatasetProvider`, `PITProvider` each have `Local*` and `Client*` implementations. The `LocalProvider` bundles them together.

The **expression engine** (`qlib/data/base.py`) is the core abstraction for feature engineering:
- `Expression` — abstract base with overloaded operators (`+`, `-`, `>`, `&`, etc.) that build expression trees
- `Feature` (`$name`) — loads raw features from storage
- `PFeature` (`$$name`) — loads point-in-time features
- `ExpressionOps` — operators like `Ref`, `Mean`, `Std`, `Delta`, etc. (defined in `qlib/data/ops.py`)
- `Expression.load()` caches results in `H["f"]` (global memory cache)

**Storage** (`qlib/data/storage/`): file-based calendar/instrument/feature storage. `_libs/` contains two Cython extensions (`rolling`, `expanding`) for performance-critical windowed calculations.

**Cache** (`qlib/data/cache.py`): multi-layer — in-memory (`H`), disk (`DiskDatasetCache`, `DiskExpressionCache`), Redis-backed, and `DatasetURICache`.

### Model Layer (`qlib/model/`)

- `BaseModel` — abstract, has `predict()`
- `Model(BaseModel)` — adds `fit(dataset, reweighter)` and `predict(dataset, segment)`
- `ModelFT(Model)` — adds `finetune(dataset)` for fine-tunable models

`qlib/contrib/model/` contains concrete models (LightGBM, neural nets, etc.). Model configs use `class` + `module_path` keys resolved via `init_instance_by_config()`.

### Strategy & Backtest (`qlib/backtest/`, `qlib/strategy/`)

Backtesting uses a **nested decision execution** framework:
- `Exchange` — simulates the market (prices, costs, limits)
- `Account`/`Position` — tracks cash and holdings
- `BaseStrategy.generate_trade_decision()` — produces `BaseTradeDecision` objects containing `Order` lists
- `BaseExecutor` — executes decisions, supports multi-level nesting (e.g., daily strategy with minutely execution)
- `RLStrategy`/`RLIntStrategy` — RL variants with state/action interpreters

`qlib.backtest.backtest()` is the high-level entry point. `qlib.backtest.collect_data()` yields decisions for RL training.

### Workflow & Experiment Management (`qlib/workflow/`)

The global `R` (`QlibRecorder`) wraps MLflow for experiment tracking:
```python
with R.start(experiment_name="exp", recorder_name="run"):
    R.log_params(lr=0.01)
    R.log_metrics(loss=0.5, step=1)
    R.save_objects(model=model, artifact_path="models")
```

`qlib/workflow/task/` provides a task-based workflow system for defining end-to-end pipelines via YAML configs, runnable via the CLI:
```bash
qrun config.yaml
```

### CLI (`qlib/cli/`)

`qrun` (`qlib.cli.run:run`) reads a YAML config and executes a workflow task or model training. The config specifies `task` or `model` sections with `class`/`module_path` keys.

### Global Configuration (`qlib/config.py`)

`C` is the global `QlibConfig` instance. It has `client` and `server` modes, supports region configs (CN/US/TW for trading rules), and manages `provider_uri` (local paths or NFS mounts). It is initialized once via `qlib.init()`.

### Contrib (`qlib/contrib/`)

Community/advanced modules organized by domain: `contrib/model/`, `contrib/strategy/`, `contrib/rolling/`, `contrib/tuner/`, `contrib/report/`, `contrib/workflow/`, `contrib/online/`, `contrib/meta/`.
