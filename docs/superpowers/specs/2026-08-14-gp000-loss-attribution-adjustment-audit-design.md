# gp_000 亏损归因与复权全链路审计设计

## 目标

在正式训练窗口内，对正式基线因子 `gp_000` 完成可复现的亏损归因与四层复权审计，输出
`docs/risk/gp000_loss_attribution.md`。专项必须区分工程 bug、参数配置问题和因子 alpha
不足，并以证据确定优先级，不因已有报告结论而跳过重新计算。

## 固定研究对象

- 数据集：`data/raw/argus_quant_working.parquet`
- 正式因子库：`data/artifacts/argus/event_factors.json`
- 正式因子：库中唯一的 `gp_000`
- 当前表达式：
  `add(add(stock_intra_amp_d1d3_mean, div(stock_vwap_dev_d1, vol_burst_count_20d)), stock_intra_amp_d0)`
- 因子方向：库内 `sign=+1`，分数越高越优先做多
- 名义训练窗口：2022-01-04 至 2024-09-04，共 649 个交易日
- D+2 完整决策窗口：2022-01-04 至 2024-09-02，共 647 个 D0
- 生产组合：Top4，D+1 开盘建仓，D+2 收盘退出，资金重叠系数 2
- 净收益成本：配置中的佣金、过户费、按日期切换的印花税和单边滑点，使用乘法记账

报告元数据记录输入文件 SHA-256、因子表达式、因子方向、训练日历 SHA-256 和运行命令。
若正式库不再是单因子事件库、因子名不是 `gp_000`、方向不是 `+1`，脚本直接失败，禁止
静默切换到其他同名实验因子。

## 方案选择

### 采用：独立专项脚本复用稳定内核

新增 `scripts/gp000_loss_attribution.py`，复用以下已验证代码：

- `helix.gp.library.compute_factors`：正式表达式回放
- `helix.eval.ic.daily_ic`：逐日截面 IC
- `helix.eval.backtest._cost_rates/_net_returns`：统一成本
- `helix.eval.backtest.summarize_portfolio_returns`：CAGR、Sharpe、回撤
- `helix.eval.style_neutralize.style_residualize`：逐日风格中性化
- `scripts/g3_style_ablation.py` 中的行业区间对齐逻辑与风格缓存合同

专项脚本负责边界、复权对照、分位分析、期限分析、月度归因和报告渲染；不复制 GP、IC、
成本或中性化数学实现。

### 未采用：扩展 `objective_pnl_alignment.py`

该脚本验证“训练目标与生产 PnL 排序是否对齐”，而本专项还要审计数据源、复权因子和
除权样本。继续扩展会让目标治理与行情口径审计耦合，并增加既有报告回归风险。

### 未采用：拼接既有风险报告

既有报告使用过 Top10 和覆盖受限的 D+1 至 D+10 缓存，不能满足本专项的 Top4、完整训练
窗口和逐层复权证据要求。

## 输入与输出

### 输入

1. 事件表：因子输入、原始标签价、原始 `label_d2_return`。
2. 正式因子库：表达式、方向和字段列表。
3. `data/raw/d2_exit_cache/*.parquet`：逐交易日原始开高收、成交量、当日
   `adj_factor`、涨跌停价。
4. `data/artifacts/g3_style_market.parquet`：总市值、换手率和风格回溯所需市场数据。
5. `data/artifacts/g3_sw2021_members.parquet`：申万一级行业历史区间。
6. `configs/default.yaml`：生产 TopK 与交易成本配置。

### 输出

- `docs/risk/gp000_loss_attribution.md`：完整报告。
- `data/artifacts/gp000_loss_attribution.json`：机器可读指标和审计元数据。
- `data/artifacts/gp000_loss_attribution_daily.parquet`：各期限、原始/中性、毛/净日收益。
- `docs/risk/assets/gp000_loss_attribution_equity.svg`：毛收益、净收益和中性净收益累计曲线。
- `docs/risk/assets/gp000_loss_attribution_decay.svg`：D+1 至 D+10 IC 与 Top4 终值/曲线摘要。

SVG 只表达报告表格中的同一批数值，不包含额外筛选或交互状态。

## 训练边界合同

所有日期先标准化为 `YYYY-MM-DD`，并基于缓存中的真实交易日历映射：

1. 名义 D0 必须位于 `[2022-01-04, 2024-09-04]`。
2. D+h 分析仅保留 `exit_date(D0, h) <= 2024-09-04` 的 D0。
3. D+2 主分析因此排除 2024-09-03、2024-09-04 两个 D0。
4. D+10 使用更早的末端 D0，不允许读取训练结束后的行情再补齐收益。
5. 标签、价格或风格缺失保持 NaN；被选中但结果不可观测时，固定仓位留现金，不以更深
   排名替补。
6. 报告同时列名义 D0 数、各期限可用 D0 数、边界剔除 D0 和剔除行数。

边界校验在加载后和每项期限计算前各执行一次；越界不是告警，而是异常。

## 第一部分：复权全链路审计

### 数据源层

审计两条数据通路：

- 面板通路：Tushare `daily` 为原始价，`adj_factor` 为各交易日观测到的累计因子；
  `raw_price * adj_factor` 为后复权价格。因子不使用训练窗末日因子回刷历史，因此是
  point-in-time 变换。
- 正式事件表通路：`label_px_d1_open/high/close` 为原始价，文件不携带 `adj_factor`；
  因子输入是上游预计算的无量纲字段，仓库内没有其价格复权血缘。

脚本用逐日缓存独立重建原始价与后复权价，并报告事件表价格对原始价、后复权价的匹配
误差。事件特征缺少上游复权声明作为工程合同缺口记录，不能把“未观察到异常”写成“已
证明无未来函数”。

### 因子计算层

- 面板基础字段逐项静态分类：跨日价格比较应使用 `*_hfq`，同日蜡烛形状及涨跌停比较
  可以使用原始价。
- 正式 `gp_000` 由三个事件特征计算，Helix 仅回放表达式，不再次调整价格。
- 对所有 D0 除权样本，计算因子值、截面分位、截面 robust z-score、与同股票最近一次
  可比事件值的跳变；同时与非除权样本的尾部率对比。由于事件池不是完整日频面板，最近
  一次事件跳变只作诊断，不作为单独 bug 判据。

### 标签计算层

对每个 D0 重新映射 D+1 与 D+2：

```text
raw_return = raw_close[D+2] / raw_open[D+1] - 1
hfq_return = raw_close[D+2] * adj[D+2]
             / (raw_open[D+1] * adj[D+1]) - 1
```

同时用原始和后复权 D+2 high 重算 8% 触达标签。报告全样本与除权子集中的收益误差、
标签翻转数、方向和极值。

### 回测引擎层

并列核查：

- `helix.eval.backtest` 面板路径使用 `LabelSet.entry_price/exit_price`，来自后复权价格。
- 事件研究脚本使用事件表 `label_px_*` 和 `label_d2_return`，当前为原始价。
- 成交性与涨跌停比较必须继续使用原始价，不能因统一收益口径而改为后复权价。

### 除权日专项

除权事件定义为同股票相邻交易日 `adj_factor` 发生变化。输出：

- D0、D+1、D+2 分别落在除权日的事件数和股票数；
- 原始/后复权收益差的均值、中位数、p95、最大绝对值；
- 原始/后复权触达标签翻转数；
- `gp_000` 除权样本的分位、robust z-score 尾部率和跳变诊断；
- Top4 中除权交易的数量、收益贡献及将原始收益替换为后复权收益后的组合指标变化。

只有当修复后 Top4 毛收益或净收益符号翻转，或变化量足以覆盖原亏损，复权 bug 才能被
判为核心根因。否则将其列为真实工程缺陷或合同缺口，但明确不是亏损主因。

## 第二部分：gp_000 亏损归因

### 五分位单调性

每天按 `gp_000` 分数稳定排序并等频分成五组，Q1 最低、Q5 最高。每组输出：

- D+2 后复权毛收益/笔；
- 使用当日成本后的净收益/笔；
- 样本量、日期覆盖和触达率；
- Q5-Q1 收益差与 Spearman 分位单调系数。

分组只使用当天候选池，不跨日池化分位边界。

### 成本拆分

同一固定 Top4 排名分别计算：

- 无任何费用/滑点的毛收益组合；
- 生产佣金、过户费、印花税和滑点下的净收益组合。

两臂都输出累计收益、CAGR、Sharpe、最大回撤、单笔收益、执行率和日数。成本拖累以净值
和单笔收益的差额表示；若毛收益已经为负，成本只能是放大项，不能列为方向翻转原因。

### D+1 至 D+10 收益衰减

对每个 h：

- 用后复权 `close[D+h] / open[D+1] - 1` 作为目标；
- 输出逐日 IC 均值、ICIR、Top4 毛/净单笔收益、CAGR、Sharpe、终值和可用 D0 数；
- 组合资金占用采用 `overlap=h`，保持每个重叠 tranche 的资金分母一致；
- 绘制各期限累计净收益曲线，并在报告中明确不同期限使用不同的边界截断样本。

### 时间分布

以 D+2 Top4 生产组合为主，输出逐月毛收益、净收益、月末累计净值、交易日数和月胜率。
报告列出亏损贡献最大的月份以及亏损是否集中于单一区间。累计曲线从 1.0 开始，避免首日
亏损被最大回撤基准遗漏。

### 风格中性收益

在同一风格完整样本交集上对比原始和风格中性因子：

- 风格：对数总市值、申万一级行业、20 日动量、20 日波动率、20 日平均换手率；
- 每个 D0 只使用 D0 及此前 19 个交易日；
- 每日独立残差化，不跨日估计系数；
- 输出 Top4 净收益/笔、CAGR、Sharpe、最大回撤、终值、IC/ICIR 和覆盖率；
- 输出残差对设计矩阵的最大标准化暴露作为正交性校验。

若原始和中性组合均亏损，且中性后收益没有改善到正值，则风格暴露不是亏损的充分解释；
剩余部分定义为纯 alpha 表现，而不是自动命名为正 alpha。

## 根因优先级与修复建议

报告严格按以下层级排序：

1. 工程 bug：复权错配、未来因子、边界越界、标签/回测价格错配。
2. 参数配置：因子方向、TopK、持有期、成本、退出规则与目标错配。
3. 因子 alpha：分位不单调、收益 IC 为负、风格剥离后仍无净收益。

每项根因包含：证据、严重度、是否主导亏损、修复文件/接口、回归测试、预期指标变化和
不能承诺的效果。修复预期使用审计中的反事实差值，不给未经测量的收益承诺。

## 代码结构

### `scripts/gp000_loss_attribution.py`

职责分为可单测的纯函数：

- `validate_training_calendar` / `outcome_complete_dates`
- `validate_formal_factor`
- `load_market_cache`
- `build_price_lookup`
- `replay_formal_factor`
- `audit_adjustment_chain`
- `evaluate_ex_right_samples`
- `evaluate_quintiles`
- `evaluate_top_k_book`
- `evaluate_horizon_decay`
- `evaluate_monthly_returns`
- `evaluate_style_neutral_book`
- `rank_root_causes`
- `render_report`
- `write_outputs`

数据加载与文件写入只在顶层编排函数发生；统计函数接收 DataFrame/ndarray 并返回
DataFrame/dict，测试不依赖 3.4GB 本地数据。

### `tests/test_gp000_loss_attribution.py`

使用小型合成面板覆盖：

- D+2 与 D+10 边界严格截断；
- `raw * adj_factor` 后复权收益消除除权跳空；
- 事件标签原始价与后复权价错配可被识别；
- 五分位 Q1 至 Q5 分组及样本计数；
- 稳定 Top4、缺失退出留现金、不向下替补；
- 毛/净成本差与 2023-08-28 印花税切换；
- 月度复利与累计净值；
- 风格残差正交及两臂使用共同样本；
- 根因排序始终为工程 bug、参数配置、alpha；
- 报告包含交付要求中的全部章节和表格标题。

## 错误处理

以下情形立即失败，不生成部分报告：

- 输入文件或正式因子缺失；
- 正式库身份、方向或表达式合同变化；
- 训练日历起止、数量或摘要变化；
- 行情缓存缺少训练窗任一所需交易日；
- 同一日期股票重复且价格冲突；
- D+h 退出越过训练边界；
- 原始价格无法复现事件表标签价且误差超过容差；
- 报告关键指标非有限或必需章节缺失。

风格/行业缺失只允许通过共同样本掩码剔除，并在报告中披露覆盖率；不得插补未来或全样本
统计量。

## 验收标准

1. 报告完整覆盖用户指定的四层审计、除权专项和五项亏损归因。
2. 所有主结论只使用训练窗口和相应 D+h 完整结果。
3. 表格与 SVG 使用同一机器可读结果生成。
4. 根因优先级满足工程 bug > 参数配置 > 因子 alpha。
5. 新增测试先红后绿，专项测试可重复运行。
6. `PYTHONPATH=. /Users/aochong/code/helix/.venv/bin/pytest` 全量通过。
7. `PYTHONPATH=. /Users/aochong/code/helix/.venv/bin/ruff check .` 零违规。
