# RD-Agent 因子按 Feature-Importance 剔除低贡献因子 实现计划

> **For agentic workers:** 用 superpowers:subagent-driven-development 或 superpowers:executing-plans 逐任务实现。步骤用 `- [ ]` 复选框跟踪。
> **重要约束（来自项目）:** 任何 `git commit` 在执行前必须先与用户确认；绝不用 `git add -A`/`git add .`，只按文件名 add；改动主体在 RD-Agent repo。

**Goal:** 让 RD-Agent 的 fin_factor loop 在每轮组合回测后,把对模型贡献过低(归一化 gain importance `==0` 或 `<1%`)的**本轮新挖因子**标记为"不入库",使其不再进入后续轮次的 SOTA 因子组合,从而实现因子级别的优胜劣汰,缓解"1 好 2 坏整批绑定"的问题。

**运行环境(已核实,docker/conda 均可):** `workspace.execute`(workspace.py:18-40)对 docker 与 conda 走**同一套流程**(都 `qtde.check_output(local_path=workspace_path, entry="qrun ...")` 再 `entry="python read_exp_res.py"`),仅 `qtde` 不同。conda 路径 `QlibCondaEnv(LocalEnv)`(env.py:839)用 `subprocess.Popen(cwd=workspace_path, env={**os.environ, **run_env})`(env.py:668-674)执行:**cwd=workspace_path**、**run_env 的 `PYTHONPATH=./`(factor_runner.py:135)注入子进程**、用 **conda env 的 PATH 跑 qrun**(env.py:647-653)。因此随 template 注入的 `feature_importance_record` 模块在 conda 下同样可被 `module_path` import,本计划逻辑无需为 conda 改动。

**Architecture:** 现有链路里,SOTA 库因子在每轮被使用时是通过遍历历史 experiment 的 `sub_workspace_list` 重新执行因子代码生成的(`utils.py:process_factor_data` → `_build_execute_calls`,过滤条件 `if implementation and feedback`)。因此**只需新增一个 per-factor "drop 名单"开关**:回测拿到 importance 后,把低贡献因子的 `factor_name` 加入 `exp.dropped_factors`;`_build_execute_calls` 跳过名单内因子。importance 的获取依赖一条新的透传链:运行环境(docker/conda)内自定义 qlib `Record` 算 importance 并存盘 → `read_exp_res.py` 导出 csv → `workspace.execute` 读回 → `factor_runner.develop` 消费。两个必须处理的坑:**列名映射**(LGBM 用 `lgb.Dataset(x.values)` 丢了列名,Booster `feature_name()` 是 `Column_i`,要按训练 feature 列顺序映射回因子名)和 **base_features 豁免**(Alpha158 那 20 个基础特征不论 importance 高低都不剔)。本轮回测仍用全量因子(decision 基于全量),剔除只影响"入库供未来用"。

**Tech Stack:** Python, qlib (`RecordTemp`/`LGBModel`/`LightGBMFInt`), LightGBM Booster, pandas, RD-Agent (CoSTEER factor loop, Jinja2 conf 模板, qrun 经 docker 或 conda 运行环境)。

---

## File Structure

改动文件(全部在 `/home/hqy/RD-Agent/` 下,除非特别说明):

| 文件 | 类型 | 职责 |
|---|---|---|
| `rdagent/scenarios/qlib/experiment/factor_template/feature_importance_record.py` | **新建** | 运行环境(docker/conda)内自定义 qlib Record:训练后取 gain importance、把 `Column_i` 映射回真实因子名、存 `feature_importance.pkl`。含可单测的纯映射函数。 |
| `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml` | 改 | record 列表追加 `FeatureImportanceRecord`(fin_factor 实际用的回测模板)。 |
| `rdagent/scenarios/qlib/experiment/factor_template/read_exp_res.py` | 改 | 从 recorder 把 `feature_importance.pkl` 导出为 `feature_importance.csv`(搭 `ret.pkl` 便车,try/except)。 |
| `rdagent/scenarios/qlib/experiment/workspace.py` | 改 | `execute()` 读回 `feature_importance.csv` → 挂到 workspace 实例属性 `self.feature_importance`(不改返回值签名)。 |
| `rdagent/scenarios/qlib/experiment/factor_experiment.py` | 改 | `QlibFactorExperiment.__init__` 增加 `self.dropped_factors: set[str] = set()`。 |
| `rdagent/scenarios/qlib/developer/factor_runner.py` | 改 | 回测后:读 importance、归一化、按阈值挑出本轮要剔的因子名(base 豁免)、写入 `exp.dropped_factors`。含可单测纯函数 `select_dropped_factors`。 |
| `rdagent/scenarios/qlib/developer/utils.py` | 改 | `_build_execute_calls` 跳过 `exp.dropped_factors` 中的因子,使其不进入后续 SOTA 组合。 |
| `rdagent/scenarios/qlib/developer/conf.py`(或 factor_runner 顶部常量) | 改 | 新增阈值常量 `IMPORTANCE_DROP_THRESHOLD = 0.01`。 |
| `test/scenarios/qlib/test_factor_importance_pruning.py`(RD-Agent 测试目录,路径按其约定) | **新建** | 纯函数单测:列名映射 + 阈值筛选 + base 豁免。 |

**实现顺序**:Task 1(record+映射纯函数)→ 2(conf)→ 3(read_exp_res)→ 4(workspace)→ 5(experiment 属性)→ 6(runner 筛选)→ 7(utils 跳过)→ 8(端到端验证)。1 和 6 含可 TDD 的纯函数,先写测试。

---

## Task 1: 自定义 FeatureImportanceRecord + 列名映射纯函数

**Files:**
- Create: `rdagent/scenarios/qlib/experiment/factor_template/feature_importance_record.py`
- Test: `test/scenarios/qlib/test_factor_importance_pruning.py`

**关键事实(已核实):**
- `LGBModel`(qlib `contrib/model/gbdt.py:16`)继承 `LightGBMFInt`,底层 `self.model` 是原生 lgb Booster。
- `LightGBMFInt.get_feature_importance()`(`qlib/model/interpret/base.py:33-45`)会 `.sort_values()`,**index 被按值排序、不再是训练列顺序** → 不能按位置 zip。改用底层 `booster.feature_importance(importance_type="gain")`(原生返回 numpy,**按训练列顺序**)。
- 训练列顺序 == `dataset.prepare("train", col_set="feature").columns` 顺序(因为 `gbdt.py:37-54` 就是 `df["feature"].values` 喂给 `lgb.Dataset`)。
- `RecordTemp` 模板见 `SignalRecord`(`qlib/workflow/record_temp.py:161-209`):`__init__(self, model, dataset, recorder)` + `generate()` 里 `self.save(**{"x.pkl": obj})`。

- [ ] **Step 1: 写映射纯函数的失败测试**

新建 `test/scenarios/qlib/test_factor_importance_pruning.py`:

```python
import numpy as np
import pandas as pd
import pytest

from rdagent.scenarios.qlib.experiment.factor_template.feature_importance_record import (
    map_importance_to_names,
)


def test_map_importance_basic_single_level():
    raw = np.array([10.0, 0.0, 5.0])
    cols = pd.Index(["f_a", "f_b", "f_c"])
    s = map_importance_to_names(raw, cols)
    assert isinstance(s, pd.Series)
    assert list(s.index) == ["f_a", "f_b", "f_c"]
    assert s["f_a"] == 10.0 and s["f_b"] == 0.0 and s["f_c"] == 5.0


def test_map_importance_multiindex_takes_last_level():
    raw = np.array([1.0, 2.0])
    cols = pd.MultiIndex.from_tuples([("feature", "f_a"), ("feature", "f_b")])
    s = map_importance_to_names(raw, cols)
    assert list(s.index) == ["f_a", "f_b"]
    assert s["f_b"] == 2.0


def test_map_importance_length_mismatch_raises():
    with pytest.raises(ValueError):
        map_importance_to_names(np.array([1.0, 2.0]), pd.Index(["only_one"]))
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `cd /home/hqy/RD-Agent && python -m pytest test/scenarios/qlib/test_factor_importance_pruning.py -v`
Expected: FAIL — ImportError(模块/函数不存在)。

- [ ] **Step 3: 实现 record 文件(含映射纯函数)**

新建 `rdagent/scenarios/qlib/experiment/factor_template/feature_importance_record.py`:

```python
"""自定义 qlib Record: 训练后取 LightGBM gain importance, 映射回真实因子名并存盘。
随 factor_template 注入 workspace(docker/conda 均适用); conf 里以 module_path=feature_importance_record 引用。"""
import numpy as np
import pandas as pd

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow.record_temp import RecordTemp


def map_importance_to_names(raw_values, feature_columns) -> pd.Series:
    """把按训练列顺序排列的 importance 数组映射成以因子名为 index 的 Series。

    raw_values: np.ndarray, 长度 == 特征数, 顺序 == 训练 feature 列顺序。
    feature_columns: pd.Index 或 pd.MultiIndex(取最后一层为因子名)。
    """
    raw_values = np.asarray(raw_values)
    if isinstance(feature_columns, pd.MultiIndex):
        names = list(feature_columns.get_level_values(-1))
    else:
        names = list(feature_columns)
    if len(raw_values) != len(names):
        raise ValueError(
            f"importance length {len(raw_values)} != feature columns {len(names)}; "
            "训练列顺序与映射列名不一致, 无法安全映射。"
        )
    return pd.Series(raw_values, index=names)


class FeatureImportanceRecord(RecordTemp):
    """生成并保存每个特征的 gain importance(以真实因子名为 index)。"""

    def __init__(self, model=None, dataset=None, recorder=None):
        super().__init__(recorder=recorder)
        self.model = model
        self.dataset = dataset

    def generate(self, **kwargs):
        booster = getattr(self.model, "model", None)
        if booster is None:
            raise ValueError("FeatureImportanceRecord: model 未训练或不含底层 booster。")
        # 原生 importance: 按训练列顺序的 numpy 数组(不要用 get_feature_importance, 它会 sort)
        raw = booster.feature_importance(importance_type="gain")
        # 训练 feature 列顺序: 与喂给 lgb.Dataset 的 df["feature"].values 列序一致
        feat_df = self.dataset.prepare("train", col_set="feature", data_key=DataHandlerLP.DK_L)
        importance = map_importance_to_names(raw, feat_df.columns)
        self.save(**{"feature_importance.pkl": importance})

    def list(self):
        return ["feature_importance.pkl"]
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `cd /home/hqy/RD-Agent && python -m pytest test/scenarios/qlib/test_factor_importance_pruning.py -v`
Expected: PASS(3 个映射相关用例)。

- [ ] **Step 5: 暂存(提交前与用户确认)**

```bash
git add rdagent/scenarios/qlib/experiment/factor_template/feature_importance_record.py test/scenarios/qlib/test_factor_importance_pruning.py
# 与用户确认后再: git commit -m "feat(factor): add FeatureImportanceRecord with gain-importance name mapping"
```

---

## Task 2: 把 FeatureImportanceRecord 加进回测 conf

**Files:**
- Modify: `rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml`(record 段,现 line 99-113 起)

- [ ] **Step 1: 在 record 列表追加 importance record**

在 `record:` 列表末尾(`PortAnaRecord` 之后)追加:

```yaml
        - class: FeatureImportanceRecord
          module_path: feature_importance_record
          kwargs:
            model: <MODEL>
            dataset: <DATASET>
```

说明:`module_path: feature_importance_record` 是相对模块名;qrun 工作目录为 workspace_path 且 run_env 含 `PYTHONPATH=./`(见 `factor_runner.py:135`),docker 与 conda 均把该 env 注入子进程(conda 见 env.py:668-674),故该文件随 template 注入后可 import。`<MODEL>`/`<DATASET>` 由 qlib workflow 占位替换(与 SignalRecord 一致)。

- [ ] **Step 2: 验证 YAML 渲染合法**

Run: `cd /home/hqy/RD-Agent && python -c "import jinja2, yaml; t=open('rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml').read(); print('OK has record' if 'FeatureImportanceRecord' in t else 'MISSING')"`
Expected: 打印 `OK has record`。(完整渲染需 Jinja 变量,此处仅确认追加成功且文件可读。)

- [ ] **Step 3: 暂存(提交前确认)**

```bash
git add rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml
```

---

## Task 3: read_exp_res.py 导出 feature_importance.csv

**Files:**
- Modify: `rdagent/scenarios/qlib/experiment/factor_template/read_exp_res.py`(现 line 54-67 有 ret.pkl/pred/label 导出)

- [ ] **Step 1: 仿照 ret.pkl 段追加 importance 导出**

在 `label.pkl` 的 try/except 块之后(line 67 后)追加:

```python
    try:
        feature_importance = latest_recorder.load_object("feature_importance.pkl")
        feature_importance.to_csv(Path(__file__).resolve().parent / "feature_importance.csv")
    except Exception:
        pass
```

说明:try/except 保证 baseline(无此 artifact)或旧 recorder 不报错。`Path` 已在文件顶部 import。

- [ ] **Step 2: 语法检查**

Run: `cd /home/hqy/RD-Agent && python -m py_compile rdagent/scenarios/qlib/experiment/factor_template/read_exp_res.py && echo OK`
Expected: `OK`。

- [ ] **Step 3: 暂存(提交前确认)**

```bash
git add rdagent/scenarios/qlib/experiment/factor_template/read_exp_res.py
```

---

## Task 4: workspace.execute 读回 importance

**Files:**
- Modify: `rdagent/scenarios/qlib/experiment/workspace.py`(`QlibFBWorkspace.execute`, line 18-59)

- [ ] **Step 1: 在 execute 末尾读回 csv 挂到实例属性**

在 `execute` 方法内,读 `qlib_res.csv` 成功返回**之前**,加入 importance 读回(默认 None 兜底)。把 line 50-59 区段改为:

```python
        # 读回 feature importance(可能不存在, 如 baseline)
        self.feature_importance = None
        fi_path = self.workspace_path / "feature_importance.csv"
        if fi_path.exists():
            try:
                self.feature_importance = pd.read_csv(fi_path, index_col=0).iloc[:, 0]
            except Exception as e:
                logger.warning(f"Failed to read feature_importance.csv: {e}")

        qlib_res_path = self.workspace_path / "qlib_res.csv"
        if qlib_res_path.exists():
            # Here, we ensure that the qlib experiment has run successfully before extracting information from execute_qlib_log using regex; otherwise, we keep the original experiment stdout.
            pattern = r"(Epoch\d+: train -[0-9\.]+, valid -[0-9\.]+|best score: -[0-9\.]+ @ \d+ epoch)"
            matches = re.findall(pattern, execute_qlib_log)
            execute_qlib_log = "\n".join(matches)
            return pd.read_csv(qlib_res_path, index_col=0).iloc[:, 0], execute_qlib_log
        else:
            logger.error(f"File {qlib_res_path} does not exist.")
            return None, execute_qlib_log
```

说明:不改 `execute` 返回值签名(避免改动所有调用点);importance 经 `self.feature_importance` 暴露,由 develop 取用。每次 execute 先置 None,避免读到上一次回测的残留。

- [ ] **Step 2: 语法检查**

Run: `cd /home/hqy/RD-Agent && python -m py_compile rdagent/scenarios/qlib/experiment/workspace.py && echo OK`
Expected: `OK`。

- [ ] **Step 3: 暂存(提交前确认)**

```bash
git add rdagent/scenarios/qlib/experiment/workspace.py
```

---

## Task 5: QlibFactorExperiment 增加 dropped_factors 属性

**Files:**
- Modify: `rdagent/scenarios/qlib/experiment/factor_experiment.py`(`QlibFactorExperiment.__init__`)

- [ ] **Step 1: 定位 __init__**

Run: `cd /home/hqy/RD-Agent && grep -n "class QlibFactorExperiment\|def __init__" rdagent/scenarios/qlib/experiment/factor_experiment.py`
Expected: 输出 `QlibFactorExperiment` 类与其 `__init__` 行号。

- [ ] **Step 2: 在 __init__ 末尾添加属性**

在 `QlibFactorExperiment.__init__` 的 `super().__init__(...)` 之后添加:

```python
        # 因 importance 过低而不入 SOTA 库的本轮因子名(回测后由 factor_runner 填充)
        self.dropped_factors: set[str] = set()
```

说明:集中声明,避免各处 `getattr(exp, "dropped_factors", set())` 兜底。若该类无显式 `__init__`,则新增一个调用 `super().__init__(*args, **kwargs)` 后设此属性。

- [ ] **Step 3: 语法检查**

Run: `cd /home/hqy/RD-Agent && python -m py_compile rdagent/scenarios/qlib/experiment/factor_experiment.py && echo OK`
Expected: `OK`。

- [ ] **Step 4: 暂存(提交前确认)**

```bash
git add rdagent/scenarios/qlib/experiment/factor_experiment.py
```

---

## Task 6: factor_runner 按阈值挑出要剔的因子(纯函数 + 接线)

**Files:**
- Modify: `rdagent/scenarios/qlib/developer/factor_runner.py`(`develop`, 回测分支 line 218-273 之后)
- Test: `test/scenarios/qlib/test_factor_importance_pruning.py`(追加)

- [ ] **Step 1: 写阈值筛选纯函数的失败测试**

向 `test/scenarios/qlib/test_factor_importance_pruning.py` 追加:

```python
from rdagent.scenarios.qlib.developer.factor_runner import select_dropped_factors


def test_select_dropped_zero_and_below_threshold():
    # 总 gain = 1000; 阈值 1% => 10
    imp = pd.Series({
        "base_KLEN": 500.0,   # base feature, 必须豁免
        "good_factor": 480.0,
        "weak_factor": 5.0,   # 0.5% < 1% => drop
        "dead_factor": 0.0,   # 0 => drop
    })
    candidate_factor_names = ["good_factor", "weak_factor", "dead_factor"]  # 不含 base
    dropped = select_dropped_factors(imp, candidate_factor_names, threshold=0.01)
    assert dropped == {"weak_factor", "dead_factor"}


def test_select_dropped_base_features_exempt():
    imp = pd.Series({"base_x": 0.0, "factor_a": 1000.0})
    # base_x 不在候选名单 => 不被剔, 即便它 importance 为 0
    dropped = select_dropped_factors(imp, ["factor_a"], threshold=0.01)
    assert dropped == set()


def test_select_dropped_missing_factor_treated_as_zero():
    # 某候选因子不在 importance index(模型从未用到/对不上) => 视为 0 => drop
    imp = pd.Series({"factor_a": 1000.0})
    dropped = select_dropped_factors(imp, ["factor_a", "factor_ghost"], threshold=0.01)
    assert dropped == {"factor_ghost"}
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `cd /home/hqy/RD-Agent && python -m pytest test/scenarios/qlib/test_factor_importance_pruning.py -k select_dropped -v`
Expected: FAIL — ImportError(`select_dropped_factors` 不存在)。

- [ ] **Step 3: 在 factor_runner.py 顶部实现纯函数 + 阈值常量**

在 `factor_runner.py` 顶部(import 之后)添加:

```python
IMPORTANCE_DROP_THRESHOLD = 0.01  # 归一化 gain importance 低于此占比的本轮因子不入 SOTA 库


def select_dropped_factors(importance, candidate_factor_names, threshold=IMPORTANCE_DROP_THRESHOLD):
    """挑出本轮要剔除(不入库)的因子名。

    importance: pd.Series, index=特征名(含 base+因子), value=gain importance。
    candidate_factor_names: 本轮新挖因子名(已豁免 base_features, 调用方只传 sub_task 因子名)。
    规则: 归一化 gain(占全部特征 gain 之和) == 0 或 < threshold 即剔; index 缺失的因子视为 0。
    """
    import pandas as pd

    if importance is None or len(importance) == 0:
        return set()
    total = float(importance.sum())
    if total <= 0:
        return set(candidate_factor_names)
    dropped = set()
    for name in candidate_factor_names:
        val = float(importance.get(name, 0.0))
        if val / total < threshold:
            dropped.add(name)
    return dropped
```

说明:分母用**全部特征**(含 base)的 gain 之和,得到"该因子在整个组合中的贡献占比";`==0` 自然落入 `< threshold`。base_features 不在 `candidate_factor_names` 内,天然豁免。

- [ ] **Step 4: 运行测试,确认通过**

Run: `cd /home/hqy/RD-Agent && python -m pytest test/scenarios/qlib/test_factor_importance_pruning.py -v`
Expected: PASS(全部 6 个用例)。

- [ ] **Step 5: 在 develop 回测后接线写入 exp.dropped_factors**

在 `develop` 中,回测取得 `result, stdout`(`factor_runner.py:242-250` 的 sota_model/LGBM 两分支之后、`if result is None` 检查 line 275 之前)插入:

```python
        # 按 feature importance 标记低贡献的本轮因子为"不入库"
        importance = getattr(exp.experiment_workspace, "feature_importance", None)
        if importance is not None and exp.sub_tasks:
            candidate_names = [t.factor_name for t in exp.sub_tasks]
            dropped = select_dropped_factors(importance, candidate_names)
            if dropped:
                exp.dropped_factors |= dropped
                logger.info(
                    f"Feature-importance pruning: dropping {len(dropped)}/{len(candidate_names)} "
                    f"low-contribution factors from SOTA library: {sorted(dropped)}"
                )
```

说明:只对本轮 `exp.sub_tasks`(新挖因子)判断;`exp.dropped_factors` 已在 Task 5 初始化。本轮回测已用全量完成,此标记只影响该 exp 未来作为 SOTA 成员时的因子加载(Task 7)。

- [ ] **Step 6: 语法检查 + 复跑单测**

Run: `cd /home/hqy/RD-Agent && python -m py_compile rdagent/scenarios/qlib/developer/factor_runner.py && python -m pytest test/scenarios/qlib/test_factor_importance_pruning.py -v`
Expected: 编译 OK + 单测全 PASS。

- [ ] **Step 7: 暂存(提交前确认)**

```bash
git add rdagent/scenarios/qlib/developer/factor_runner.py test/scenarios/qlib/test_factor_importance_pruning.py
```

---

## Task 7: _build_execute_calls 跳过 dropped_factors

**Files:**
- Modify: `rdagent/scenarios/qlib/developer/utils.py`(`_build_execute_calls`, line 29-41)

- [ ] **Step 1: 改 _build_execute_calls 跳过被剔因子**

把 `_build_execute_calls`(line 29-41)的 sub_tasks 分支改为(用 enumerate 拿 factor_name):

```python
def _build_execute_calls(exp: QlibFactorExperiment, base_feature_workspaces: list[FactorFBWorkspace]) -> list[tuple]:
    execute_calls = []

    if exp.sub_tasks:
        assert isinstance(exp.prop_dev_feedback, CoSTEERMultiFeedback)
        dropped = getattr(exp, "dropped_factors", set())
        execute_calls.extend(
            (implementation.execute, ("All",))
            for sub_task, implementation, feedback in zip(
                exp.sub_tasks, exp.sub_workspace_list, exp.prop_dev_feedback
            )
            if implementation and feedback and sub_task.factor_name not in dropped
        )

    execute_calls.extend((workspace.execute, ("All",)) for workspace in base_feature_workspaces)
    return execute_calls
```

说明:新增条件 `sub_task.factor_name not in dropped`。当这个历史 exp 在未来轮次被 `process_factor_data(sota_factor_experiments_list)` 处理时,被剔因子的代码不再执行 → 不进入 SOTA 组合。base feature workspaces 不受影响(豁免)。`getattr(..., set())` 兜底兼容尚未设属性的旧对象。

- [ ] **Step 2: 语法检查**

Run: `cd /home/hqy/RD-Agent && python -m py_compile rdagent/scenarios/qlib/developer/utils.py && echo OK`
Expected: `OK`。

- [ ] **Step 3: 暂存(提交前确认)**

```bash
git add rdagent/scenarios/qlib/developer/utils.py
```

---

## Task 8: 端到端验证

**Files:** 无改动,仅运行验证。

- [ ] **Step 1: 跑一轮 fin_factor(后台,按 background-task-monitoring 习惯)**

Run: `cd /home/hqy/RD-Agent && rdagent fin_factor --loop-n 1 --no-checkout 2>&1`
(后台运行 + 10 分钟间隔检查;完成后再继续验证。)

- [ ] **Step 2: 确认 importance 透传成功**

在本轮 workspace 目录(`git_ignore_folder/RD-Agent_workspace/...` 或日志中 runner result 的 workspace_path)检查:
```bash
find /home/hqy/RD-Agent -name feature_importance.csv -newermt "-1 hour" 2>/dev/null | head
```
Expected: 找到至少一个 `feature_importance.csv`,内容为 `因子名,gain值` 两列,且 index 是**真实因子名**(非 `Column_i`)。

- [ ] **Step 3: 确认 drop 逻辑触发(若有低贡献因子)**

在运行日志里搜索:
```bash
grep -i "Feature-importance pruning" <本轮日志文件>
```
Expected: 若本轮存在 importance 占比 `<1%` 的新因子,出现 `dropping N/M low-contribution factors` 日志,且列出的是因子名(不含 base 特征名)。若本轮因子都强,可能无此行(正常)。

- [ ] **Step 4: 跑第二轮,确认被剔因子不进 SOTA 组合**

前置:Step 3 至少剔除过 1 个因子(否则人为把 `IMPORTANCE_DROP_THRESHOLD` 临时调高如 `0.5` 重跑一轮制造 drop,验证后改回 `0.01`)。
Run: `cd /home/hqy/RD-Agent && rdagent fin_factor --loop-n 2 --no-checkout 2>&1`
检查第二轮 workspace 的 `combined_factors_df.parquet` 列名**不含**第一轮被 drop 的因子名:
```bash
python -c "import pandas as pd; df=pd.read_parquet('<第二轮workspace>/combined_factors_df.parquet'); print([c[-1] if isinstance(c, tuple) else c for c in df.columns])"
```
Expected: 被 drop 的因子名不出现在列里;未被 drop 的第一轮因子仍在。

- [ ] **Step 5: 汇总验证结果并暂存(提交前确认)**

确认:① importance 文件生成且因子名正确;② 低贡献因子被标记;③ 下一轮 SOTA 组合不含被剔因子。三者通过即功能完成。

```bash
# 全部改动一次性提交(提交前与用户确认):
git add rdagent/scenarios/qlib/experiment/factor_template/feature_importance_record.py \
        rdagent/scenarios/qlib/experiment/factor_template/conf_combined_factors.yaml \
        rdagent/scenarios/qlib/experiment/factor_template/read_exp_res.py \
        rdagent/scenarios/qlib/experiment/workspace.py \
        rdagent/scenarios/qlib/experiment/factor_experiment.py \
        rdagent/scenarios/qlib/developer/factor_runner.py \
        rdagent/scenarios/qlib/developer/utils.py \
        test/scenarios/qlib/test_factor_importance_pruning.py
# git commit -m "feat(factor): prune low feature-importance factors from SOTA library"
```

---

## 风险与边界(执行时留意)

1. **列名映射对齐**:若 `dataset.prepare("train", col_set="feature").columns` 的列数与 booster importance 长度不等,`map_importance_to_names` 会抛 `ValueError`。这是**有意的快速失败**——映射对不上时绝不能静默错配因子。出现时需排查 NestedDataLoader 列结构(base 20 + 组合因子)。
2. **importance 单次方差**:受 `seed`/`subsample=0.879`/`colsample=0.888` 影响,单次低 importance 未必稳定。阈值 1% 保守即为此;暂不做多 seed 平均(贵)。
3. **阈值起步保守**:`0.01` 仅剔几乎零贡献者。入库不可逆,宁少剔。跑几轮观察库质量后再决定是否降到 0.5% 或改用相对中位数 `< median × 0.1`。
4. **sota_model / baseline conf**:本计划只改 `conf_combined_factors.yaml`(fin_factor 路径)。`conf_combined_factors_sota_model.yaml`(fin_quant)如需同款能力,按 Task 2 同样追加;baseline 无新因子无需。
5. **不改 qlib 本体**:自定义 record 放 RD-Agent 的 factor_template 并随 workspace 注入运行环境(docker/conda),`LGBModel`/`RecordTemp` 仅作为依赖使用。

## Self-Review(已对照)

- Spec 覆盖:importance 获取(T1)→ 存盘(T1/T2)→ 导出(T3)→ 读回(T4)→ 消费筛选(T5/T6)→ 入库过滤(T7)→ 验证(T8),链路闭合。两个坑(列名映射 T1、base 豁免 T6)均有显式处理与测试。
- 无 placeholder:所有改动均给出完整代码与确切位置。
- 类型/名称一致:`select_dropped_factors`、`map_importance_to_names`、`FeatureImportanceRecord`、`exp.dropped_factors`、`self.feature_importance` 在各任务间命名一致。
