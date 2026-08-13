# 安慰剂 IC/Gini 校准与因子显著性筛查设计

**日期：** 2026-08-13

**状态：** 已批准，待实现

**适用数据集：** `data/raw/argus_quant_working.parquet`

## 1. 目标与边界

本改动为事件表通路建立可重复的安慰剂校准基准，用训练集内的随机标签经验分布
替代拍脑袋的 IC、ICIR 和 gini 门槛，并用同一基准筛查当前正式因子库。

本改动只增加校准、筛查、配置基准和治理文档，不改变 GP 搜索、因子生成、因子排序、
G2 去重或 G3 消融流程，也不自动删除或改写任何已有因子库。

必须满足以下红线：

1. 阈值计算只使用正式因子挖掘时可见的训练日期，不允许样本外行进入任何指标数组。
2. 随机标签只在同一交易日的可观测标签之间置换，每组置换逐日保持精确的正负样本数。
3. 同一组、同一日的置换对全部因子共用，沿用 `make_cs_columns.py` 安慰剂臂的共享行置换契约。
4. 阈值只作为 G1 入库参照；已有因子库和主挖掘流程保持不变。

## 2. 数据范围

### 2.1 正式校准范围

正式因子库固定为：

```text
data/artifacts/argus/event_factors.json
```

当前该库包含一个正式因子 `gp_000`。只有该库参与安慰剂经验分布和 p95、p99、
p99.9 阈值计算。

### 2.2 补充对照范围

以下两组已被 G3 否决的实验因子只做补充筛查：

```text
data/artifacts/argus_n40/event_factors.json   # 12 个因子
data/artifacts/argus_multi/event_factors.json # 17 个因子
```

它们使用正式阈值标注显著性，但不得进入正式经验分布，不得影响配置值，也不得获得正式
入库资格。筛查结果必须带 `scope` 字段，避免三个库中重复的 `gp_NNN` 名称混淆。

### 2.3 训练窗口

正式库的导出脚本记录 `SEARCH_END = '2024-09-04'`。与现有正式因子的原始挖掘窗口
一致，校准样本固定为：

```text
train_start = 2022-01-04
train_end   = 2024-09-04
n_dates     = 649
```

校准命令必须显式传入 `--train-end 2024-09-04`，不提供“默认使用全历史”的降级路径。
读取 Parquet 时即使用 `trade_date <= train_end` 谓词，只选择日期、代码、二分类标签和三个
因子库所需的特征列。加载后再次断言最大日期不晚于 `train_end`，报告和 Parquet 产物同时
记录训练起止日期。样本外日期不会参与因子计算、随机标签生成、分位数或报告统计。

## 3. 指标定义

三个指标统一使用二分类目标 `label_d2_hit_8pct`，并严格按交易日先计算、再跨日汇总：

1. **IC mean**：每日因子与二分类标签的 Spearman 截面相关系数均值。
2. **ICIR**：每日 IC 均值除以每日 IC 的样本标准差（`ddof=1`）。
3. **gini**：每日 `2 × AUC − 1` 的均值。

每日有效样本门槛使用 `gp.min_daily_samples = 50`。因子计算应用 `FactorSpec.sign`，使真实
因子的方向与现有因子库一致。

GP 在训练集上允许选择正负方向，因此安慰剂阈值和显著性比较使用方向选择后的统计量：

```text
abs(IC mean), abs(ICIR), abs(mean daily gini)
```

产物中同时保留真实因子的有符号值和用于分级的绝对值，避免审计时丢失方向信息。

## 4. 随机标签与向量化计算

### 4.1 置换契约

使用 `numpy.random.default_rng(seed)` 生成 1000 组可复现随机标签。对每个交易日：

1. 取该日可观测的原始二分类标签，共 `n` 个，其中正样本 `k` 个。
2. 为该日一次生成形状为 `(1000, n)` 的随机键。
3. 每行选择随机键最小的 `k` 个位置作为正样本，其余位置为负样本。
4. 因为连续随机键的所有 `k` 元子集等概率，该过程等价于对原二分类标签做均匀随机置换。

这保证每组随机标签逐日精确保留 `k` 和 `n-k`，不会跨日移动标签，也不会改变每日可观测
标签数。随机标签只依赖当日标签和固定种子，不读取任何未来日期。

### 4.2 计算布局

性能关键路径不对 1000 组随机标签逐组循环，也不对因子逐个计算指标：

- 随机键、正样本掩码和每日指标均以 NumPy 二维矩阵批量计算。
- 正样本数与秩和通过矩阵运算同时得到全部置换、全部因子的结果。
- 只保留一个按交易日的边界循环，以适配每日不同的截面宽度并限制峰值内存。
- 因子表达式仍通过现有 `FactorLibrary`/`compute_factors` 计算，避免复制表达式语义。

每日结果累计为 `(n_permutations, n_factors)` 的和、平方和与有效日计数，最后一次性得到
全周期 IC mean、ICIR 和 gini。该结构的内存量与“最大单日截面 × 1000”成正比，而不是
与“全部训练行 × 1000”成正比。

### 4.3 多正式因子时的规则

当前正式库只有一个因子，所以每组随机标签自然产生一行三指标。若以后正式库包含多个
因子，每组置换分别对三项指标取正式库内最大绝对值，再形成该组的阈值样本。这使全局
入库线控制正式库内的多重尝试，不会因为库规模增加而放松。

补充库永远不参与这个最大值。

## 5. 经验分布与阈值

正式经验分布保存到：

```text
data/artifacts/placebo_ic_distribution.parquet
```

每行对应一个置换编号，至少包含：

```text
permutation_id
ic_mean
icir
gini
seed
train_start
train_end
n_train_dates
formal_factor_count
```

`ic_mean`、`icir` 和 `gini` 已是方向选择后的绝对统计量。分位数使用
`numpy.quantile(..., method="linear")` 计算：

```text
p95   = 0.950
p99   = 0.990
p99.9 = 0.999
```

p99 三项值写入 `configs/default.yaml` 的正式准入配置；p95 和 p99.9 只用于报告和显著性
分级，不作为正式入库门槛。

## 6. 因子筛查与准入规则

正式库和两个补充库的真实标签指标均在同一 649 日训练窗口上重新计算，不读取现有报告中
的样本外字段。筛查结果保存到：

```text
data/artifacts/placebo_factor_screening.parquet
```

每个因子包含：

```text
scope
library_path
factor_name
ic_mean_signed
ic_mean
icir_signed
icir
gini_signed
gini
ic_level
icir_level
gini_level
overall_level
candidate_eligible
suggest_evict
```

每项指标按严格大于关系分级：

1. `value > p99.9`：`超 p99.9`
2. `value > p99`：`超 p99`
3. `value > p95`：`超 p95`
4. 其余：`低于随机水平`

综合等级取三项中的最低等级。只有三项都严格超过 p99 时，
`candidate_eligible = true`。

正式库中不满足该条件的因子标记 `suggest_evict = true`，但脚本不修改正式库。补充库的
`candidate_eligible` 始终为 `false`，`suggest_evict` 为空；报告只展示其按正式门槛得到的
反事实等级，并明确其 G3 否决状态不改变。

## 7. 配置设计

新增独立的因子准入配置，不把阈值混入 GP 搜索超参数。配置结构如下；三个浮点值在首次
真实数据校准完成后直接固化为本次正式分布的 p99，设计阶段不预设数值：

| YAML 路径 | 类型 | 值来源 |
|---|---|---|
| `factor_admission.placebo_threshold.ic_mean` | 非负有限浮点数 | 正式安慰剂分布 IC mean 的 p99 |
| `factor_admission.placebo_threshold.icir` | 非负有限浮点数 | 正式安慰剂分布 ICIR 的 p99 |
| `factor_admission.placebo_threshold.gini` | 非负有限浮点数 | 正式安慰剂分布 gini 的 p99 |
| `factor_admission.placebo_threshold.quantile` | 浮点数 | 固定为 `0.99` |
| `factor_admission.placebo_threshold.train_start` | 日期字符串 | 固定为 `2022-01-04` |
| `factor_admission.placebo_threshold.train_end` | 日期字符串 | 固定为 `2024-09-04` |

对应 Pydantic 模型要求三个门槛为有限非负数，`quantile` 严格位于 `(0, 1)`，训练起止日期
非空且有序。新增纯函数读取该配置并判断三个指标是否全部严格超过门槛。校准/筛查命令
必须调用该函数生成 `candidate_eligible`，后续入库工具可以复用同一入口。

该配置不接入 `run_search()` 或 `_select_factors()`，从而遵守“不修改现有因子挖掘主流程”
的红线。它是入库校验基准，不是新的搜索目标。

## 8. 报告与治理文档

生成 `docs/risk/placebo_ic_calibration.md`，包含：

1. 数据文件、训练窗口、标签、种子、置换数、正式因子数和补充因子数。
2. 安慰剂三指标的均值、标准差、最小值、中位数、最大值。
3. p95、p99、p99.9 核心阈值表。
4. 正式因子逐项指标、等级、候选资格和淘汰建议。
5. `argus_n40`、`argus_multi` 的独立补充对照表与等级计数。
6. 明确声明阈值仅由训练集正式库生成，补充库和样本外数据均未参与。

更新 `docs/factor-governance.md`：

- 将 G1 表中的 IC mean、ICIR 和 gini 更新为本次 p99 实测值并链接校准报告。
- 明确三项必须同时超过 p99。
- 将 D12 标为“已量化闭环”，记录训练窗口、1000 次截面置换和报告路径。
- 保留 G3 的否决权；显著性通过不等于正式入库。

## 9. 错误处理与审计断言

校准程序遇到以下情况必须失败，不得静默降级：

- `--train-end` 缺失、不存在于输入数据或晚于正式库训练截止日。
- 过滤后的数据包含晚于 `train_end` 的日期。
- 标签不是 `{0, 1, NaN}`，或某日没有正负两类导致指标不可用。
- 正式因子库为空、因子计算失败、所需字段缺失。
- 置换后的任一交易日、任一组正负样本数与原标签不一致。
- 1000 组中任一最终指标非有限。
- 配置中的 p99 与本次经验分布重算结果不一致。

报告记录固定种子、训练窗口、样本数和库路径，使结果能够复算和审计。

## 10. 测试策略

测试遵循红—绿—重构流程，至少覆盖：

1. **截面置换守恒**：用两个正样本率不同的日期，验证 1000 组逐日正负数完全不变。
2. **不跨日**：给两个日期使用互不相交的标签结构，验证标签只在各自日期内部重排。
3. **可复现性**：相同种子得到完全相同的随机标签，不同种子至少一组不同。
4. **向量化指标一致性**：小样本上将批量结果与现有 `daily_ic`、`daily_gini` 的单组结果逐项比较。
5. **分位数准确性**：对已知数组验证 `linear` 方法的 p95、p99、p99.9 精确结果。
6. **联合准入**：验证任一指标未严格超过 p99 时均不得成为候选。
7. **训练截止日防泄漏**：加入样本外极端信号后，阈值与仅含训练数据时完全一致。
8. **正式/补充隔离**：改变补充库指标不影响正式阈值和正式配置。
9. **配置校验**：拒绝负值、非有限门槛、无效分位数和倒置日期。

实现后运行：

```text
.venv/bin/pytest -q
.venv/bin/ruff check .
```

并用真实数据运行一次校准命令，核对 Parquet 行数、训练截止日、阈值表、正式筛查数量和
补充对照数量。

## 11. 预计文件变更

新增：

```text
helix/eval/placebo.py
scripts/calibrate_placebo.py
tests/test_placebo_calibration.py
data/artifacts/placebo_ic_distribution.parquet
data/artifacts/placebo_factor_screening.parquet
docs/risk/placebo_ic_calibration.md
```

修改：

```text
helix/config.py
configs/default.yaml
docs/factor-governance.md
```

现有 `scripts/make_cs_columns.py` 只作为置换契约来源，不修改；三个 `event_factors.json`
均保持原样。
