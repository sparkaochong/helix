# 后复权基线统一设计

**日期：** 2026-08-14  
**分支：** `fix/adjustment-unification`  
**目标：** 在不改写历史实验结论的前提下，为数据、特征、标签和回测建立唯一的点时后复权价格契约；任何新因子链路缺少可验证血缘时必须 fail-closed。

## 1. 决策与边界

采用“双通道强契约迁移”。

- `data/raw/argus_quant_working.parquet`、正式 `gp_000` 成品特征和既有报告保留为 legacy 修复前基线。其价格特征没有生成公式、复权声明或 as-of 血缘，因此不得补写“已后复权”等无证据结论，也不得进入新因子训练入口。
- 新链路只接受带四元血缘的点时后复权价格。缺字段、日期错位、口径不是 HFQ、因子版本不一致或无法重建时直接报错，不允许警告后继续、使用原始价替代或填充相邻交易日因子。
- 原始 OHLC 只允许用于与当日原始涨跌停价比较，以及停牌/可成交性校验。它不得参与因子值、标签值、成交计价或持仓收益核算。
- 本次只建立合规复权基线，不重新训练 `gp_000`，不改写既有审计报告数值，也不承诺修复其盈利能力。后续新一代 GP 因子必须从新基线生成。

## 2. 价格血缘合同

### 2.1 四元元数据

每个进入因子、标签或收益计算的价格字段必须绑定以下信息：

| 字段 | 语义 | 强制规则 |
| --- | --- | --- |
| `source_date` | 该价格和复权因子对应的实际交易日 | 必须等于价格所在 panel 行；不允许 shift、ffill 或 bfill |
| `as_of_time` | 该观测按业务时钟可见的时间 | 因子输入不得晚于 D0 收盘；标签/收益只能位于声明的 D+1/D+2 结果日，且不得越过训练窗末 |
| `price_basis` | 价格口径 | 计算链路只接受固定值 `hfq` |
| `adj_factor_version` | 本次复权因子数据集及算法的不可变版本 | 使用因子分区内容摘要与算法版本生成；同一次计算所用字段必须版本一致 |

血缘以受类型约束、可序列化的 field-level 对象保存。日期和 as-of 数组与价格矩阵严格对齐；`price_basis` 和 `adj_factor_version` 为不可变字段属性。panel 缓存保存并恢复该对象，旧缓存缺少血缘时自动失效重建，而不是补默认值。

### 2.2 点时后复权构建

数据层通过唯一构建器完成：

1. 对 `daily` 和 `adj_factor` 按 `(trade_date, ts_code)` 去重并拒绝冲突键。
2. 只做同股票、同交易日一对一匹配；缺失因子直接报错，不允许邻日填充。
3. 计算 `price_hfq[t, s] = price_raw[t, s] * adj_factor[t, s]`。
4. 为 `open/high/low/close/pre_close_hfq` 固化血缘，并校验有限原始价格必须对应有限、正值因子。
5. `adj_factor_version` 包含输入内容摘要和构建算法版本，因此缓存或数据源变化可被反向识别。

这里的“后复权”不使用训练窗末因子统一缩放历史序列。每一格只读取该股票当日因子，从结构上排除前复权式未来回刷。

## 3. 四节点统一

### 3.1 数据源层

`Panel` 增加价格血缘容器及 `require_adjusted_prices(...)` 校验入口。`build_panel` 是合规 HFQ 的唯一生产构建器；缓存加载必须同时验证字段、血缘形状、口径和版本。

legacy event 表不修改原文件。正式 event 加载器要求显式的特征血缘清单；任何价格衍生特征缺少 `source_date/as_of_time/price_basis/adj_factor_version` 声明都拒绝加载。现有 `gp_000` 四个上游字段没有该证据，只能由专项 baseline adapter 只读回放。

### 3.2 特征计算端

`compute_base_fields` 进入计算前要求 `open/high/low/close_hfq` 的合规血缘。

- `ret1/ret5/ret20`、均线偏离、RSV、波动率和相关性继续或改为 HFQ。
- `gap` 改为 `open_hfq[t] / close_hfq[t-1] - 1`，不再使用会保留除权跳空的 raw `pre_close`。
- `intraday`、振幅、收盘位置、上下影线统一由同日 HFQ OHLC 计算。
- `to_up_limit` 和 `limitup_cnt20` 仍使用 raw price 与 raw limit，因为它们属于涨跌停状态判断；不得把结果反向用作跨日成交收益。

新 event GP 入口不推断成品特征名的含义，也不因“同日比例理论上复权不变”而豁免血缘。

### 3.3 标签计算端

`build_touch_label` 只从合规 `open_hfq[D+1]`、`high_hfq[D+2]`、`close_hfq[D+2]` 生成 entry、touch 和 exit。`LabelSet` 携带所用价格口径和因子版本，供回测再次校验。

D+1 是否开盘封板、D+1/D+2 是否交易仍通过原始 OHLC、涨跌停价和交易状态判断。D+2 标签只有在出场日不晚于训练窗末时可用；最后两个未完成 D0 必须保持无标签。

### 3.4 回测与收益端

`run_backtest` 在收益计算前验证 `LabelSet` 的 HFQ 合同；realistic-exit 路径同时验证 panel 血缘。开仓、D+2 收盘退出及延期退出只用 HFQ 价格计价，原始价仅决定是否能成交。

手续费、印花税、过户费和滑点继续按名义金额比例计算：

`net = (1 + gross_hfq) * (1 - sell_cost) / (1 + buy_cost) - 1`

不会对复权价额外缩放费率，也不会把固定价差滑点混入本次修改。

## 4. Legacy 基线与修复后指标

新增专项、只读的 baseline 对照入口。它是唯一允许消费无血缘 event 成品特征和 raw event 标签的路径，并必须在输出中标记 `legacy_unverified_lineage=true`。

对照遵循以下固定口径：

- 因子：正式库唯一的 `gp_000`，表达式、方向和历史分数不变，不重新训练。
- 窗口：`2022-01-04` 至 `2024-09-04` 名义训练窗；D0 只保留 D+2 出场仍在窗内的结果完整日期。
- before：event 表既有 raw D+1→D+2 outcome。
- after：同一批 D0、股票和固定分数，使用同日精确匹配的 HFQ D+1 open、D+2 high/close。
- 组合：固定 Top4、固定 shortlist、固定成本与滑点配置，不因修复后的未来结果补位或改选。

修复文档至少输出：

| 指标 | before | after | 变化 |
| --- | --- | --- | --- |
| D+2 close IC | raw outcome IC | HFQ outcome IC | after-before |
| Top4 单笔净收益 | raw | HFQ | 百分点变化 |
| Top4 组合净收益/CAGR | raw | HFQ | 百分点变化 |
| 年化 Sharpe | raw | HFQ | 变化 |

验收要求是与既有审计影响量级一致：Top4 单笔净收益约改善 `0.022226%`，修复后约 `-0.5233%`，亏损方向不翻转。若重算超出数值容差，脚本必须失败并要求重新审计，不得自动更新历史报告。

该表仅量化 outcome 复权修正，不把 legacy `gp_000` 认证为合规新因子。

## 5. 错误处理

以下情况使用明确异常并终止：

- 参与因子、标签或收益的价格字段缺少任一四元元数据；
- `price_basis != "hfq"`；
- `source_date` 与 panel 行日期不相等，或出现前移/后移匹配；
- 同一计算中的 `adj_factor_version` 不一致；
- raw price 有效但当日因子缺失、非有限或非正；
- 因子输入 `as_of_time` 晚于 D0 收盘；
- D+2 outcome 越过训练窗末；
- 新 event GP 尝试加载未认证的 legacy 特征。

异常信息必须包含用途、字段名、首个失败日期/股票和失败规则，支持反向排查。legacy baseline adapter 不能被正式 pipeline、mine 或 backtest 入口调用。

## 6. 测试与验收

测试按层覆盖：

1. 数据层：同日精确匹配、缺因子拒绝、邻日因子不能填充、缓存血缘往返、版本不一致拒绝。
2. 特征层：构造除权样本，验证 raw 跳空不会进入 `gap/ret/volatility`；传入无血缘或 raw-basis 特征时 fail-closed。
3. 标签层：D+1→D+2 复权收益与触达正确，最后两个 D0 被 D+2 截断，成交性仍看 raw limit。
4. 回测层：entry/exit/gross/net 全部来自 HFQ；改变 raw 价格只能改变可成交性，不能改变已成交收益。
5. event 层：现有无血缘 event 表被正式加载器拒绝；仅显式 legacy adapter 可用于 before 对照。
6. 除权专项：同股票当日 `adj_factor` 变化能平滑价格跳空，且不存在跨股票、跨日期匹配。
7. 指标回归：before/after 表与既有审计的影响量级一致，且亏损结论不翻转。

最终执行完整 `pytest` 和 `ruff check .`，要求零失败、零违规。

## 7. 文档与治理

- 新增 `docs/risk/adjustment_unification_fix.md`，记录合同、修改面、测试、before/after 指标和限制。
- 更新 `docs/factor-governance.md` 的 D10 为修复完成，链接本修复说明与既有专项审计；明确旧基线不翻案、新 `gp_000` 未因此获得盈利能力。
- 既有 `docs/risk/gp000_loss_attribution.md` 及历史 artifacts 不重算、不覆盖。

## 8. 非目标

- 不回溯重建 `gp_000` 四个上游特征。
- 不为 legacy event 表补造血缘元数据。
- 不重新挖掘、训练、调参或改变正式因子方向。
- 不修改涨跌停制度、停牌处理、选股补位或交易成本模型。
- 不合并 PR；本次交付止于分支提交、验证通过和创建 PR。
