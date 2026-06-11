# RD-Agent + Qlib Pipeline Visualization Design

**Goal:** Single-file interactive HTML page showing the RD-Agent + Qlib factor mining pipeline with three-level zoom drill-down.

**Architecture:** Pure vanilla HTML/CSS/JS + inline SVG. Radial layout with 6 clickable nodes around a center detail area. Click drills down: node → sub-steps → inline detail cards. Click center blank returns to default overview.

**Tech Stack:** HTML5, CSS3 (transitions, flexbox), inline SVG, vanilla JS. Zero dependencies.

---

## Layout

```
                    [1.假设]
        [6.反馈]               [2.方案]
              \               /
               [ 中心区域  ]
              /               \
        [5.回测]               [3.编码]
                    [4.运行]
```

- **Ring:** 6 circular nodes (r=24px) on a circle (r=120px) around center (x=0, y=0 in SVG viewBox)
- **Center:** Rounded rect (280x200 or fill available), default shows Qlib backtest 7-step flow
- **SVG viewBox:** responsive, centered in page

## Three Levels

### Level 1 — Ring Overview
- 6 nodes on circle + center Qlib backtest preview
- Each node: colored circle + icon + label
- Click node → Level 2
- Arrows between nodes showing cycle direction

### Level 2 — Center Sub-steps
- Ring shrinks to edges (opacity 0.4), clicked node highlights
- Center clears and shows numbered sub-step list (4-7 items per stage)
- Each sub-step: clickable row with ▸ indicator
- Click sub-step → Level 3
- Click center blank space → return to Level 1

### Level 3 — Inline Detail Card
- Sub-step expands inline (accordion-style)
- Card shows: 📥 Input, 📤 Output, ⚙️ Context, 📋 Code snippet (where relevant)
- Click ▾ to collapse back
- Other sub-steps remain visible but collapsed

## Data Structure

All content stored as JS object:

```js
const PIPELINE = {
  center: { title: "Qlib 回测核心", steps: [...] },
  stages: [
    {
      id: "hypothesis",
      icon: "🤔",
      label: "提出假设",
      color: "#a6e3a1",
      substeps: [
        {
          title: "分析历史反馈",
          detail: { input: "...", output: "...", context: "..." }
        },
        ...
      ]
    },
    ...
  ]
}
```

## Content (6 Stages, code-verified)

| # | Stage | Sub-steps | Key Detail |
|---|-------|-----------|------------|
| 1 | 🤔 提出假设 | 组装trace.hist → RAG策略 → LLM生成 → 解析JSON | Prompt: factor_hypothesis_specification |
| 2 | 📝 实验方案 | 发送LLM → 生成因子规格 → 去重检查 → FactorTask | 输出: name/description/formulation/variables |
| 3 | 💻 代码生成 | 解析规格 → LLM生成 → CoSTEER多轮改进 → 安全扫描 | 接口: daily_pv.h5 → result.h5, "$"前缀 |
| 4 | ▶️ 因子运行 | 并行执行factor.py → 收集result.h5 → IC去重 → 合并保存 | 去重: IC>=0.99剔除, SOTA因子累积 |
| 5 | 📊 回测评估 | 加载特征 → DataHandler对齐 → LGB训练 → 策略选股 → 执行 → 指标 | 输出: qlib_res.csv, ret.pkl, IC/ICIR/Sharpe |
| 6 | 💡 反馈学习 | 提取指标 → 对比SOTA → LLM分析 → 决定替换 → 写入trace | 关键: IC/年化收益/最大回撤, 年化提升→替换 |

Center default: 数据加载 → 特征计算 → DataHandler → LGB训练 → 策略 → 模拟执行 → 报告

## Interaction Spec

1. Page loads → Level 1 ring visible, center shows Qlib backtest overview
2. Click stage node → transition: ring shrinks, center fades to sub-steps (CSS `opacity` + `transform`, 300ms ease)
3. Click sub-step row → accordion expand (max-height transition, 250ms)
4. Click center background (not on any sub-step) → return to Level 1
5. Click different node while Level 2 is open → switch to that stage's sub-steps
6. Breadcrumb bar at top of center shows current path: `总览 > 代码生成 > LLM生成代码`

## Implementation Notes

- Single file: `docs/pipeline_viz.html`
- All CSS in `<style>`, all JS in `<script>`, all content in JS data object
- SVG is the outer frame (ring + connectors + center area)
- Center detail area is HTML overlay positioned over SVG center rect
- Use `pointer-events` to distinguish node clicks from background clicks
- Ring uses SVG `<circle>` + `<text>`, connectors use `<path>` with arrow markers
- Color scheme: Catppuccin Mocha palette
