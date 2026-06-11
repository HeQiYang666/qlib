# Pipeline Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Single-file interactive HTML page showing RD-Agent + Qlib pipeline with three-level radial zoom navigation.

**Architecture:** Pure vanilla HTML/CSS/JS + inline SVG. Single file at `docs/pipeline_viz.html`. CSS transitions for animations. JS data object holds all pipeline content. Three render functions produce Level 1 (ring), Level 2 (sub-steps), and Level 3 (detail card) views.

**Tech Stack:** HTML5, CSS3 (flexbox, transitions), inline SVG, vanilla JS. Zero dependencies.

**Verification:** Open in browser after each task, verify visual output and interactions. No automated tests — this is a visualization page.

---

## File Map

| File | Purpose |
|------|---------|
| `docs/pipeline_viz.html` | Single file: all HTML structure, CSS, SVG, JS, and pipeline content data |

---

### Task 1: HTML Skeleton + Content Data Object

**Files:**
- Create: `docs/pipeline_viz.html`

- [ ] **Step 1: Create the HTML file with basic structure**

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RD-Agent × Qlib 因子挖掘流程</title>
<style>
  :root {
    --base: #1e1e2e; --surface0: #313244; --surface1: #45475a;
    --text: #cdd6f4; --subtext: #a6adc8; --blue: #89b4fa;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--base); color: var(--text);
    font-family: -apple-system, 'Noto Sans SC', sans-serif;
    display: flex; justify-content: center; align-items: center;
    min-height: 100vh; overflow: hidden;
  }
  #app { position: relative; width: 900px; height: 650px; }
  #ring-svg { width: 100%; height: 100%; }
</style>
</head>
<body>
<div id="app">
  <svg id="ring-svg" viewBox="-450 -325 900 650" xmlns="http://www.w3.org/2000/svg">
    <defs id="svg-defs"></defs>
    <g id="ring-group"></g>
    <g id="center-group"></g>
  </svg>
</div>
<script>
// Content data and rendering logic will go here
</script>
</body>
</html>
```

- [ ] **Step 2: Add the content data object**

Add inside `<script>`:

```js
const CENTER = {
  title: "Qlib 回测核心",
  steps: [
    { label: "数据加载", detail: "D.features() 加载市场数据" },
    { label: "特征计算", detail: "20 Alpha158 + N 自定义因子" },
    { label: "DataHandler", detail: "按字母序对齐特征列，去除 NaN" },
    { label: "LGB 训练", detail: "LightGBM 模型训练 (train/valid/test)" },
    { label: "策略选股", detail: "Top-K 多头选股策略" },
    { label: "模拟执行", detail: "Account/Exchange 模拟交易" },
    { label: "指标报告", detail: "IC, ICIR, Sharpe, MaxDD" }
  ]
};

const STAGES = [
  {
    id: "hypothesis", icon: "🤔", label: "提出假设", color: "#a6e3a1",
    code: "rd_loop.py:_propose\nfactor_proposal.py:19-58",
    substeps: [
      { title: "组装历史上下文", detail: { input: "trace.hist（每轮的 hypothesis + feedback + result）", output: "格式化后的 hypothesis_and_feedback 文本", context: "prompts.yaml: hypothesis_and_feedback 模板" } },
      { title: "RAG 策略注入", detail: { input: "当前轮次编号", output: "策略提示文本", context: "前 15 轮：简单因子优先。之后：ML/复杂因子优先" } },
      { title: "LLM 生成假设", detail: { input: "上下文 + RAG + hypothesis_output_format", output: "{hypothesis, reason} JSON", context: "prompts.yaml: factor_hypothesis_specification — 每轮1-5个因子，渐进复杂" } },
      { title: "解析输出", detail: { input: "LLM 返回的 JSON 字符串", output: "QlibFactorHypothesis 对象", context: "QlibFactorHypothesisGen.convert_response()" } }
    ]
  },
  {
    id: "exp_gen", icon: "📝", label: "实验方案", color: "#f9e2af",
    code: "rd_loop.py:_exp_gen\nfactor_proposal.py:61-132",
    substeps: [
      { title: "组装 LLM 上下文", detail: { input: "hypothesis + scenario 描述 + 历史实验", output: "完整的 prompt 上下文", context: "scenario: 数据格式、接口说明、回测设置" } },
      { title: "LLM 生成因子规格", detail: { input: "上下文 + experiment_output_format", output: "{\"因子名\": {description, formulation, variables}} JSON", context: "prompts.yaml: factor_experiment_output_format" } },
      { title: "去重检查", detail: { input: "新生成的因子名列表 + based_experiments 中历史因子名", output: "去重后的因子列表", context: "同名因子自动跳过，不重复生成" } },
      { title: "输出 FactorTask", detail: { input: "去重后的因子规格", output: "FactorTask 列表 (name, description, formulation, variables)", context: "每个因子一个 FactorTask 对象" } }
    ]
  },
  {
    id: "coding", icon: "💻", label: "代码生成", color: "#fab387",
    code: "rd_loop.py:coding\nFactorCoSTEER (父类)",
    substeps: [
      { title: "解析因子规格", detail: { input: "FactorTask (name, formula, variables)", output: "代码生成的 prompt 参数", context: "factor_template/ 目录提供代码骨架" } },
      { title: "LLM 生成 factor.py", detail: { input: "规格 + 接口说明 + 数据格式", output: "factor.py (含 calculate_{name} 函数)", context: "接口: 读 daily_pv.h5 → 计算 → 写 result.h5；列名用 '$' 前缀" } },
      { title: "CoSTEER 多轮改进", detail: { input: "生成的代码 + 执行结果/错误", output: "改进后的 factor.py", context: "生成→执行→捕获错误→反馈给 LLM→重新生成→重试（最多 N 轮）" } },
      { title: "保存代码", detail: { input: "最终通过的 factor.py", output: "workspace.file_dict['factor.py']", context: "代码存入 workspace，供 runner 阶段执行" } }
    ]
  },
  {
    id: "running", icon: "▶️", label: "因子运行", color: "#cba6f7",
    code: "factor_runner.py:119-277\nutils.py:131-177",
    substeps: [
      { title: "多进程执行 factor.py", detail: { input: "各 workspace 的 factor.py", output: "每个因子生成 result.h5", context: "multiprocessing_wrapper 并行执行；每个 factor.py 独立读取 daily_pv.h5" } },
      { title: "收集并归一化索引", detail: { input: "所有 result.h5 文件", output: "归一化后的 DataFrame 列表", context: "统一索引为 (datetime, instrument)；过滤分钟级数据" } },
      { title: "IC 相关度去重", detail: { input: "SOTA 因子 + 新因子 DataFrame", output: "剔除高相关（IC≥0.99）的新因子", context: "逐对计算 IC，max(IC) ≥ 0.99 则剔除该新因子" } },
      { title: "合并并保存", detail: { input: "SOTA 因子 + 去重后的新因子", output: "combined_factors_df.parquet + factor_info.json", context: "列按字母序排列；MultiIndex 列 ('feature', factor_name)" } }
    ]
  },
  {
    id: "backtest", icon: "📊", label: "回测评估", color: "#89dceb",
    code: "factor_runner.py:237-268\nworkspace.py:execute()",
    substeps: [
      { title: "加载特征数据", detail: { input: "combined_factors_df.parquet + Alpha158 表达式", output: "完整特征矩阵", context: "D.features() 表达式引擎计算 20 个基础 Alpha158；合并自定义因子" } },
      { title: "DataHandler 处理", detail: { input: "特征矩阵 (含 NaN)", output: "对齐后的特征 + Label", context: "DataHandlerLP: 按字母序排列特征列，dropna，对齐标签（Ref($close,-2)/$close -1）" } },
      { title: "模型训练", detail: { input: "train/valid/test 数据", output: "训练好的 LGBModel", context: "LightGBM booster；参数由 conf.yaml 指定；有 SOTA 模型时可替换为 PyTorch" } },
      { title: "生成预测信号", detail: { input: "测试集特征", output: "每只股票每天的 score", context: "booster.predict()；score 越高预期收益越好" } },
      { title: "策略执行", detail: { input: "score + 价格数据", output: "交易信号 (买入/卖出)", context: "Top-K 多头：选 score 最高的 K 只股票，等权配置" } },
      { title: "模拟交易 & 报告", detail: { input: "交易信号 + Exchange 数据", output: "qlib_res.csv（IC/ICIR/Sharpe/MaxDD 等）", context: "Account/Position 模拟资金和持仓；计算含/不含成本的超额收益" } }
    ]
  },
  {
    id: "feedback", icon: "💡", label: "反馈学习", color: "#f38ba8",
    code: "rd_loop.py:feedback\nfeedback.py:54-118",
    substeps: [
      { title: "提取关键指标", detail: { input: "qlib_res.csv", output: "IC / 年化收益 / 最大回撤", context: "IMPORTANT_METRICS: IC, annualized_return, max_drawdown" } },
      { title: "对比 SOTA 结果", detail: { input: "current_result + sota_result", output: "指标对比文本", context: "格式化：'IC of Current is X, of SOTA is Y'" } },
      { title: "LLM 分析反馈", detail: { input: "指标对比 + hypothesis + task_details", output: "{Observations, Feedback, New Hypothesis, Reasoning, Replace Best Result}", context: "LLM 判断假设是否成立，分析成败原因" } },
      { title: "替换判断", detail: { input: "年化收益对比", output: "decision: true/false", context: "年化收益提升 → 替换 SOTA；微小波动可接受" } },
      { title: "写入 trace", detail: { input: "(experiment, feedback) 元组", output: "追加到 trace.hist", context: "下一轮 _propose 会读取 trace.hist 作为上下文" } }
    ]
  }
];
```

- [ ] **Step 3: Verify** — Open `docs/pipeline_viz.html` in browser, check that page loads with dark background and no JS errors in console.

---

### Task 2: SVG Ring Layout + Center Area

**Files:**
- Modify: `docs/pipeline_viz.html`

- [ ] **Step 1: Add arrow marker defs and ring rendering function**

Add inside `<defs id="svg-defs">`:
```html
<marker id="arrow" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
  <path d="M0,0 L8,3 L0,6" fill="#6c7086"/>
</marker>
```

Add JS function to draw the ring:
```js
const RING_R = 180, NODE_R = 26;
const CX = 0, CY = 0;

function renderRing(activeId) {
  const g = document.getElementById("ring-group");
  g.innerHTML = "";
  const n = STAGES.length;
  STAGES.forEach((s, i) => {
    const angle = -Math.PI/2 + (2*Math.PI * i / n); // start from top
    const x = CX + RING_R * Math.cos(angle);
    const y = CY + RING_R * Math.sin(angle);
    const isActive = activeId === s.id;
    const opacity = (activeId && !isActive) ? 0.35 : 1;

    // connector line from this node to next
    const nextAngle = -Math.PI/2 + (2*Math.PI * ((i+1) % n) / n);
    const nx = CX + RING_R * Math.cos(nextAngle);
    const ny = CY + RING_R * Math.sin(nextAngle);

    const gNode = document.createElementNS("http://www.w3.org/2000/svg", "g");
    gNode.setAttribute("transform", `translate(${x},${y})`);
    gNode.setAttribute("opacity", opacity);
    gNode.style.cursor = "pointer";
    gNode.style.transition = "opacity 0.3s";
    gNode.addEventListener("click", (e) => {
      e.stopPropagation();
      showLevel2(s.id);
    });

    // outer glow when active
    if (isActive) {
      const glow = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      glow.setAttribute("r", NODE_R + 5);
      glow.setAttribute("fill", "none");
      glow.setAttribute("stroke", s.color);
      glow.setAttribute("stroke-width", "2.5");
      glow.setAttribute("opacity", "0.5");
      gNode.appendChild(glow);
    }

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("r", NODE_R);
    circle.setAttribute("fill", s.color);
    circle.setAttribute("opacity", isActive ? 1 : 0.85);
    gNode.appendChild(circle);

    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("dy", "0.35em");
    text.setAttribute("fill", "#1e1e2e");
    text.setAttribute("font-size", "14");
    text.setAttribute("font-family", "sans-serif");
    text.textContent = s.icon;
    gNode.appendChild(text);

    // label below node
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("y", NODE_R + 18);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("fill", isActive ? s.color : "#6c7086");
    label.setAttribute("font-size", "12");
    label.textContent = s.label;
    gNode.appendChild(label);

    g.appendChild(gNode);

    // arrow line between nodes (drawn on ring group too)
    const midAngle = angle + Math.PI/n;
    // draw small arc
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const ax = CX + (RING_R - NODE_R - 8) * Math.cos(angle);
    const ay = CY + (RING_R - NODE_R - 8) * Math.sin(angle);
    const bnx = CX + (RING_R - NODE_R - 8) * Math.cos(nextAngle);
    const bny = CY + (RING_R - NODE_R - 8) * Math.sin(nextAngle);
    path.setAttribute("d", `M${ax},${ay} A${RING_R-NODE_R-8},${RING_R-NODE_R-8} 0 0,1 ${bnx},${bny}`);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", "#6c7086");
    path.setAttribute("stroke-width", "1.2");
    path.setAttribute("marker-end", "url(#arrow)");
    path.setAttribute("opacity", opacity);
    g.appendChild(path);
  });
}

function renderCenterDefault() {
  const g = document.getElementById("center-group");
  g.innerHTML = "";
  g.style.cursor = "default";

  // background
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("x", -200); bg.setAttribute("y", -120);
  bg.setAttribute("width", 400); bg.setAttribute("height", 200);
  bg.setAttribute("rx", 14);
  bg.setAttribute("fill", "#313244");
  bg.setAttribute("stroke", "#89b4fa");
  bg.setAttribute("stroke-width", "1.5");
  bg.addEventListener("click", () => showLevel1());
  g.appendChild(bg);

  // title
  const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
  title.setAttribute("x", 0); title.setAttribute("y", -85);
  title.setAttribute("text-anchor", "middle");
  title.setAttribute("fill", "#89b4fa");
  title.setAttribute("font-size", "16");
  title.setAttribute("font-weight", "bold");
  title.textContent = CENTER.title;
  g.appendChild(title);

  // steps in a horizontal flow
  const steps = CENTER.steps;
  const stepW = 48, gap = 4, totalW = steps.length * stepW + (steps.length-1) * gap;
  const startX = -totalW/2;

  steps.forEach((s, i) => {
    const sx = startX + i * (stepW + gap);
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", sx); rect.setAttribute("y", -35);
    rect.setAttribute("width", stepW); rect.setAttribute("height", 70);
    rect.setAttribute("rx", 6);
    rect.setAttribute("fill", "#45475a");
    g.appendChild(rect);

    const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
    t.setAttribute("x", sx + stepW/2); t.setAttribute("y", -8);
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", "#cdd6f4");
    t.setAttribute("font-size", "9");
    t.textContent = s.label;
    g.appendChild(t);

    // arrow between steps
    if (i < steps.length - 1) {
      const ax = sx + stepW + 2;
      const arrow = document.createElementNS("http://www.w3.org/2000/svg", "text");
      arrow.setAttribute("x", ax + 90); arrow.setAttribute("y", 5);
      arrow.setAttribute("text-anchor", "middle");
      arrow.setAttribute("fill", "#6c7086");
      arrow.setAttribute("font-size", "10");
      // Simple → between steps (small gap)
    }
  });

  // tip text
  const tip = document.createElementNS("http://www.w3.org/2000/svg", "text");
  tip.setAttribute("x", 0); tip.setAttribute("y", 80);
  tip.setAttribute("text-anchor", "middle");
  tip.setAttribute("fill", "#6c7086");
  tip.setAttribute("font-size", "11");
  tip.textContent = "点击外环节点查看详情";
  g.appendChild(tip);
}

function showLevel1() {
  renderRing(null);
  renderCenterDefault();
}
```

- [ ] **Step 2: Add initial render call at end of script**

```js
showLevel1();
```

- [ ] **Step 3: Verify** — Open in browser. 6 colored nodes on circle, center shows 7-step Qlib flow. Arrows connect nodes clockwise.

---

### Task 3: Level 2 — Click Node → Center Sub-steps

**Files:**
- Modify: `docs/pipeline_viz.html`

- [ ] **Step 1: Add renderCenterSubSteps function**

Replace the placeholder `showLevel2` (called from ring click) with:

```js
let currentStage = null;
let expandedSubstep = null;

function showLevel2(stageId) {
  currentStage = STAGES.find(s => s.id === stageId);
  expandedSubstep = null;
  renderRing(stageId);
  renderCenterSubSteps(currentStage);
}

function renderCenterSubSteps(stage) {
  const g = document.getElementById("center-group");
  g.innerHTML = "";

  // background
  const bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  bg.setAttribute("x", -210); bg.setAttribute("y", -130);
  bg.setAttribute("width", 420); bg.setAttribute("height", 260);
  bg.setAttribute("rx", 14);
  bg.setAttribute("fill", "#313244");
  bg.setAttribute("stroke", stage.color);
  bg.setAttribute("stroke-width", "1.5");
  bg.addEventListener("click", (e) => {
    // only go back if clicked on background (not on a sub-step row)
    if (e.target === bg) showLevel1();
  });
  g.appendChild(bg);

  // breadcrumb
  const bc = document.createElementNS("http://www.w3.org/2000/svg", "text");
  bc.setAttribute("x", -195); bc.setAttribute("y", -105);
  bc.setAttribute("fill", "#6c7086");
  bc.setAttribute("font-size", "10");
  bc.innerHTML = `🏠 总览 &gt; <tspan fill="${stage.color}">${stage.icon} ${stage.label}</tspan>`;
  g.appendChild(bc);

  // title
  const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
  title.setAttribute("x", -195); title.setAttribute("y", -80);
  title.setAttribute("fill", stage.color);
  title.setAttribute("font-size", "16");
  title.setAttribute("font-weight", "bold");
  title.textContent = `${stage.icon} ${stage.label}`;
  g.appendChild(title);

  // code ref
  const ref = document.createElementNS("http://www.w3.org/2000/svg", "text");
  ref.setAttribute("x", 195); ref.setAttribute("y", -80);
  ref.setAttribute("text-anchor", "end");
  ref.setAttribute("fill", "#585b70");
  ref.setAttribute("font-size", "8");
  ref.setAttribute("font-family", "monospace");
  ref.textContent = stage.code;
  g.appendChild(ref);

  // sub-steps
  const startY = -55;
  const rowH = 32;
  stage.substeps.forEach((ss, i) => {
    const y = startY + i * rowH;
    const isExpanded = expandedSubstep === i;

    // row background
    const row = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    row.setAttribute("x", -195); row.setAttribute("y", y);
    row.setAttribute("width", 390); row.setAttribute("height", rowH);
    row.setAttribute("rx", 5);
    row.setAttribute("fill", isExpanded ? "#45475a" : "transparent");
    row.style.cursor = "pointer";
    row.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleSubstep(stage, i);
    });
    g.appendChild(row);

    // number
    const num = document.createElementNS("http://www.w3.org/2000/svg", "text");
    num.setAttribute("x", -185); num.setAttribute("y", y + 21);
    num.setAttribute("fill", stage.color);
    num.setAttribute("font-size", "12");
    num.textContent = (i + 1) + ".";
    g.appendChild(num);

    // title
    const st = document.createElementNS("http://www.w3.org/2000/svg", "text");
    st.setAttribute("x", -170); st.setAttribute("y", y + 21);
    st.setAttribute("fill", "#cdd6f4");
    st.setAttribute("font-size", "13");
    st.textContent = ss.title;
    g.appendChild(st);

    // expand indicator
    const indicator = document.createElementNS("http://www.w3.org/2000/svg", "text");
    indicator.setAttribute("x", 195); indicator.setAttribute("y", y + 21);
    indicator.setAttribute("text-anchor", "end");
    indicator.setAttribute("fill", "#6c7086");
    indicator.setAttribute("font-size", "12");
    indicator.textContent = isExpanded ? "▾" : "▸";
    g.appendChild(indicator);

    // divider
    if (i < stage.substeps.length - 1) {
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("x1", -185); line.setAttribute("x2", 195);
      line.setAttribute("y1", y + rowH + 1); line.setAttribute("y2", y + rowH + 1);
      line.setAttribute("stroke", "#45475a");
      line.setAttribute("stroke-width", "0.5");
      g.appendChild(line);
    }
  });

  // tip
  const tip = document.createElementNS("http://www.w3.org/2000/svg", "text");
  tip.setAttribute("x", 0); tip.setAttribute("y", 110);
  tip.setAttribute("text-anchor", "middle");
  tip.setAttribute("fill", "#6c7086");
  tip.setAttribute("font-size", "10");
  tip.textContent = "点击子步骤展开详情  ·  点击空白处返回总览";
  g.appendChild(tip);
}
```

- [ ] **Step 2: Verify** — Open in browser. Click a ring node → center shows sub-steps with breadcrumb. Click center blank → returns to Level 1.

---

### Task 4: Level 3 — Click Sub-step → Inline Detail Card

**Files:**
- Modify: `docs/pipeline_viz.html`

- [ ] **Step 1: Add toggleSubstep function and modify renderCenterSubSteps to show detail card**

Add the toggle function and detail card rendering:

```js
function toggleSubstep(stage, idx) {
  if (expandedSubstep === idx) {
    expandedSubstep = null;
  } else {
    expandedSubstep = idx;
  }
  renderCenterSubSteps(stage);
}

function renderDetailCard(stage, substep, idx, startY) {
  const g = document.getElementById("center-group");
  const detail = substep.detail;
  const cardY = startY + (stage.substeps.length) * 32 + 8;

  // card background
  const card = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  card.setAttribute("x", -195); card.setAttribute("y", cardY);
  card.setAttribute("width", 390); card.setAttribute("height", 110);
  card.setAttribute("rx", 8);
  card.setAttribute("fill", "#1e1e2e");
  card.setAttribute("stroke", stage.color);
  card.setAttribute("stroke-width", "1");
  card.setAttribute("opacity", "0.95");
  g.appendChild(card);

  const fields = [
    { icon: "📥", label: "输入", value: detail.input },
    { icon: "📤", label: "输出", value: detail.output },
    { icon: "⚙️", label: "上下文", value: detail.context },
  ];

  const colW = 130, colGap = 10;
  const totalW = fields.length * colW + (fields.length - 1) * colGap;
  const startX = -totalW / 2;

  fields.forEach((f, fi) => {
    const fx = startX + fi * (colW + colGap);
    // field icon + label
    const fl = document.createElementNS("http://www.w3.org/2000/svg", "text");
    fl.setAttribute("x", fx); fl.setAttribute("y", cardY + 20);
    fl.setAttribute("fill", stage.color);
    fl.setAttribute("font-size", "10");
    fl.textContent = `${f.icon} ${f.label}`;
    g.appendChild(fl);

    // field value (wrapped)
    const words = f.value.split(" ");
    let line = "";
    let lineY = cardY + 40;
    const maxChars = 18;
    words.forEach(w => {
      if ((line + " " + w).length > maxChars) {
        const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
        t.setAttribute("x", fx); t.setAttribute("y", lineY);
        t.setAttribute("fill", "#a6adc8"); t.setAttribute("font-size", "9");
        t.textContent = line.trim();
        g.appendChild(t);
        line = w; lineY += 14;
      } else {
        line += " " + w;
      }
    });
    if (line.trim()) {
      const t = document.createElementNS("http://www.w3.org/2000/svg", "text");
      t.setAttribute("x", fx); t.setAttribute("y", lineY);
      t.setAttribute("fill", "#a6adc8"); t.setAttribute("font-size", "9");
      t.textContent = line.trim();
      g.appendChild(t);
    }
  });
}
```

- [ ] **Step 2: Call renderDetailCard from renderCenterSubSteps when expandedSubstep is set**

Add at the end of `renderCenterSubSteps`, before the tip text creation:
```js
  if (expandedSubstep !== null && stage.substeps[expandedSubstep]) {
    renderDetailCard(stage, stage.substeps[expandedSubstep], expandedSubstep, startY);
  }
```

- [ ] **Step 3: Verify** — Click node → click sub-step → detail card appears below. Click same sub-step again → card collapses. Click background → returns to Level 1.

---

### Task 5: Transitions, Polish & Responsive

**Files:**
- Modify: `docs/pipeline_viz.html`

- [ ] **Step 1: Add CSS transitions for smooth fade between levels**

Add to `<style>`:
```css
#ring-group * { transition: opacity 0.3s ease, transform 0.3s ease; }
#center-group > rect:first-child { transition: stroke 0.3s ease, opacity 0.2s ease; }
```

- [ ] **Step 2: Add hover effects on ring nodes**

Add to `<style>`:
```css
#ring-group g[style*="cursor: pointer"]:hover circle:first-of-type {
  filter: brightness(1.15);
}
```

- [ ] **Step 3: Add header title and legend**

Add before `<div id="app">`:
```html
<div style="text-align:center;padding:16px 0 0 0">
  <h1 style="font-size:22px;font-weight:700;color:#cdd6f4;margin:0">
    RD-Agent <span style="color:#89b4fa">×</span> Qlib
  </h1>
  <p style="font-size:12px;color:#6c7086;margin:4px 0 0 0">量化因子自动挖掘流水线</p>
</div>
```

- [ ] **Step 4: Add responsive scaling**

Add to `<style>`:
```css
@media (max-width: 960px) {
  #app { width: 100vw; height: auto; }
}
```

- [ ] **Step 5: Add round counter visualization**

Add a subtle counter next to the title showing the iterative nature:
```html
<div style="text-align:center;margin-top:4px">
  <span style="display:inline-flex;gap:6px;align-items:center;font-size:10px;color:#585b70">
    <span style="color:#a6e3a1">Round 1</span> →
    <span style="color:#f9e2af">Round 2</span> →
    <span style="color:#fab387">Round 3</span> →
    <span style="color:#cba6f7">Round 4</span> →
    <span style="color:#89dceb">Round 5</span> →
    <span style="color:#f38ba8">Round 6</span> →
    <span style="color:#a6adc8">...</span>
  </span>
</div>
```

- [ ] **Step 6: Verify** — Full page looks polished: title, round counter, smooth transitions, hover effects.

---

### Task 6: Final Verification & Commit

**Files:**
- Modify: `docs/pipeline_viz.html`

- [ ] **Step 1: End-to-end walkthrough**

Open in browser and verify:
1. Page loads with ring + center default view
2. Click each of 6 ring nodes → center updates correctly
3. Click sub-step → detail card appears with correct content
4. Click sub-step again → card collapses
5. Click center background → returns to Level 1
6. Click different node while Level 2 is open → switches stages
7. No JS console errors

- [ ] **Step 2: Fix any visual issues**

Check: text overflow, alignment, color contrast, font sizes.

- [ ] **Step 3: Commit**

```bash
git add docs/pipeline_viz.html
git commit -m "feat: add RD-Agent × Qlib pipeline visualization page

Interactive SVG-based radial diagram showing the 6-stage factor mining
pipeline with three-level zoom drill-down navigation.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```
