# Helix 架构设计文档

> 本文按**代码现状**编写（非规划稿），用于架构确认。每一节末尾的「待确认」标出需要你拍板的设计决策。
> 研究结论与负面结果见 `README.md`，本文只讲结构、契约和不变量。

---

## 1. 预测目标与时点契约

这是整个系统唯一的对齐基准，所有模块的正确性都相对它定义。

```
    D0 收盘              D+1 开盘             D+2 盘中
      │                    │                    │
      │  特征计算截止        │  建仓（entry）      │  high ≥ entry×1.08 ?
      │  （只用 ≤D0 信息）   │                    │
      ▼                    ▼                    ▼
   决策时点              成交时点              结算时点
```

| 契约 | 值 | 代码位置 |
|---|---|---|
| 决策时点 | D0 收盘 | 所有特征只用 `≤D0` 数据 |
| 建仓 | D+1 开盘 | `label.entry_offset = 1` |
| 结算 | D+2 最高价 | `label.touch_offset = 2` |
| 触及阈值 | `entry × 1.08` | `label.target_ratio = 1.08` |
| 标签取值 | 1 触及 / 0 未触及 / **NaN 未定义** | `LabelSet.y` |

**行对齐规则（贯穿全系统）**：`(T, N)` 数组的第 `t` 行永远表示「在 `dates[t]` 收盘做出的决策」，而不是「`dates[t]` 发生的事」。标签、特征、预测、掩码全部按这一条对齐，`lead()` 是唯一允许向前看的算子，且只在 `helix/labels/` 内使用。

**待确认（§13.2 决策点 1）**：目标定义是否仍是 `+8% 触及`。按此目标**字面执行**（触及即止盈）是负期望，回测的默认退出已改为持到 D+2 收盘（§9）——但标签本身没动，所以排序目标与执行仍然不对口：分类器对 D+2 收盘收益的 IC 是 **−0.064**。

---

## 2. 总体结构

系统有**两条通路**，共用算子集、指标、切分纪律和 GP 引擎，但输入形态和产物完全不同。

```
                 ┌─────────────────── A. 面板通路（Tushare 全市场）────────────────────┐
                 │                                                                    │
   Tushare ─► ParquetStore ─► Panel(T×N) ─► base fields(22~28) ─┐                     │
                                  │                             │                     │
                             universe mask ──► touch label ─────┤                     │
                                                                ▼                     │
                                                        GP 搜索（DEAP，只看最老训练块）  │
                                                                │                     │
                                                        factors.json（可读表达式）      │
                                                                ▼                     │
                                              截面标准化 → GRU 合成 → walk-forward     │
                                                                ▼                     │
                                                       OOS 预测 → 回测 / 实盘打分       │
                 └────────────────────────────────────────────────────────────────────┘

                 ┌─────────────────── B. 事件表通路（argus_quant 长表）────────────────┐
                 │                                                                    │
   parquet 长表 ─► SlotIndex ─► EventPanel(T×N_max 槽位) ─► 特征预筛(IC/ICIR + 去相关)  │
                                                                │                     │
                                                        GP 搜索（残差 IC 适应度）        │
                                                                ▼                     │
                                          event_factors.json + apply_factors.py（独立脚本）│
                 └────────────────────────────────────────────────────────────────────┘
```

**为什么是两条**：A 从零构建全市场面板，时序算子有效，产物是**预测**；B 输入是每天几百只入选股 + 数百个已算好的特征，槽位 `j` 在不同日子是不同公司，时序算子无意义，产物是**因子列**（追加到下游训练表）。

---

## 3. 数据层（`helix/data/`）

| 模块 | 职责 | 关键不变量 |
|---|---|---|
| `tushare_source.py` | 增量下载日线/复权因子/涨跌停/每日指标 | 限速 + 重试，token 只从 `HELIX_TUSHARE_TOKEN` 读 |
| `store.py` | parquet 落盘 | 增量 append，不重下 |
| `panel.py` | 长表 → `(T, N)` 面板 | 行=交易日，列=股票；`panel.f64(name)` 统一取数 |
| `universe.py` | 时点股票池掩码 | **ST 用 `namechange` 历史还原当日股票名**，不用当前名 |
| `event_table.py` | 长表 → 槽位面板 | 见下 |

### 3.1 面板布局（通路 A）

`(T, N)` 二维：时序算子沿 `axis=0`，截面算子沿 `axis=1`。这个布局是 GP 能跑动的前提——每个候选表达式一次 numpy 运算算完全历史全市场。

股票池过滤（`configs/default.yaml: universe`）：排除 ST、次新（<120 日）、北交所、成交额 <2000 万、价格区间外。

### 3.2 槽位面板（通路 B）

每天的行压进槽位 `0..n_t-1`，得到 `(T, N_max)`。

- **槽位不是股票**。`occupied` 掩码标记真实行。截面算子有效，时序算子**被硬断言拒绝**（`build_event_pset` → `assert_no_time_series`），不是靠注释提醒。
- **列式流式读取**：459 列 × 1083 日 × 2656 槽 ≈ 5GB，所以 `SlotIndex` 只持有索引，`stream_feature_grids` 分批读列，用完即弃。
- **标签列按前缀排除**：`LABEL_PREFIXES = ("label","target","y_","fwd_","future_")`。不用枚举列表——源表同时带 `label_d2_hit_3pct/5pct/8pct`，漏一个进特征集就能互相预测出 IC>0.6 的垃圾因子。`assert_no_label_columns` 是硬断言。

---

## 4. 标签层（`helix/labels/touch_label.py`）

```python
target   = open_hfq[D+1] * 1.08
touched  = high_hfq[D+2] >= target
```

四条判定，任何一条不满足则标签 **NaN（未定义）而非 0**：

| # | 规则 | 理由 |
|---|---|---|
| 1 | 用**后复权价** `*_hfq` 算比值 | 原始价在除权日会凭空跌一笔分红，把命中算没 |
| 2 | D+1 开盘价 ≥ 当日涨停价 → **剔除** | 一字涨停买不进；留着会恰好在策略看起来最好的日子虚高命中率。涨停价是**原始价报价**，所以这一条用原始价比 |
| 3 | D+1 或 D+2 停牌 → 未定义 | 结果不可观测。填 0 会低估命中率 |
| 4 | 不在 D0 股票池 → 未定义 | — |

产物 `LabelSet`：`y` / `valid` / `touch_tradable` / `entry_price` / `target_price` /
`exit_price`（D+2 收盘，供未触及退出用），全部 `(T, N)` 且按 D0 行对齐。
`touch_tradable` 对下游只暴露给回测成交校验，不参与因子 population 或候选池构建。

`tests/test_labels.py` 逐条钉死这四条。

---

## 5. 特征与算子层（`helix/features/`）

### 5.1 基础字段（22 个固定 + 最多 6 个条件）

刻意保持**原始**——收益、振幅、量、估值这些人能认出的观测量。组合交给 GP 发现；在这里预烘焙聪明的复合指标只会把搜索偏向我们已经相信的东西。

| 组 | 字段 |
|---|---|
| 多周期收益 | `ret1` `ret5` `ret20` |
| D0 日内形态 | `gap` `intraday` `hl_range` `close_pos` `upper_shadow` `lower_shadow` |
| 趋势/回归 | `ma_dev5` `ma_dev20` `ma_dev60` `rsv20` `vola20` `max_ret20` |
| 流动性 | `log_amount` `amount_z20` `amihud20` |
| 涨停距离 | `to_up_limit` `limitup_cnt20` |
| 开盘行为 | `open_gap_mean5` `oc_corr20` |
| 条件字段 | `turnover` `turnover_z20` `volume_ratio` `log_circ_mv` `bp` `ep`（源列存在才建） |

`to_up_limit` / `limitup_cnt20` 是针对本目标专门加的：涨停板限制了 D+2 能走多远，直接约束 +8% 可达性。

### 5.2 算子集（`operators.py`）

| 类别 | 个数 | 成员 |
|---|---|---|
| 一元 | 8 | `neg` `abs` `sign` `log` `sqrt` `cs_rank` `cs_zscore` `cs_demean` |
| 二元 | 4 | `add` `sub` `mul` `div` |
| 时序（带窗口） | 13 | `ts_mean/std/max/min/sum/rank/delta/delay/zscore/pct/argmax/argmin/decay` |
| 时序二元 | 2 | `ts_corr` `ts_cov` |

全部严格后视。`lead` 存在但**故意不进算子集**，只有标签用。

---

## 6. GP 搜索层（`helix/gp/`）

### 6.1 类型化算子集

`Window` 是独立类型（`int` 子类）。无类型的算子集会让 GP 写出 `ts_mean(x, close)` 这种「能算但无意义」的树，把大半个种群浪费掉。窗口终端只能取 `gp.windows = [3,5,10,20,60]`。

### 6.2 三重防自欺

这是本层最重要的设计，三条互相独立：

**① 搜索只看最老的训练块。** `search_window()` 返回 `slice(0, train_days)`，之后所有数据留给 walk-forward。在全历史上挖因子再「验证」是自欺欺人。

**② fit / sel 双窗口。** 搜索块内部再按 `fit_fraction=0.8` 切开，中间隔 `embargo_days`：

```
[──────── fit 段（驱动进化，最大化 Top4 净收益）────────][embargo][── sel 段（决定去留）──]
```

进化只看 fit 段；一个因子只有在 sel 段上生产 Top4 D+2 收盘净收益严格为正（`sel_net_return > 0`）才会被保留。两个窗口都在 D+2 结果完整的训练范围内，sel 段不足 20 行直接报错。

**③ 方向必须显式。** 适应度不取绝对值、不隐式翻 sign；反向由表达式中的 `neg(...)` 表达，新因子统一保存 `sign=+1`。

适应度：`fitness = 10_000 × mean(fit 日 production Top4 D+2-close net portfolio return)`。节点数只在 P&L 完全相等时作为次级排序键。

### 6.3 残差 IC 中性化（`neutralize.py`）

**默认适应度不是原始 IC，而是对基列正交化后的残差 IC。**

来源是一个真实负面结果：首轮因子 `gp_000` 原始 IC +0.087，对自身三个输入正交化后 −0.006——它只是三个已有列的线性混合，而那三列本来就在训练集里。

实现要点：

- 每日基组（截距 + K 个基列的截面秩）用**批量 QR** 一次性正交化，秩亏方向置零；因子残差化退化成两次 BLAS matmul，70 列基下是秒级而非分钟级。
- **幅度守卫不可省略。** 下游指标基于秩、尺度无关。被完全解释的因子残差只剩 `1e-16` 浮点噪声，但那噪声**仍然保留原始排序**，会被当成信号打分。残差占比 `< 1e-6` 的交易日直接判为已解释，返回 NaN。
- **已知局限**：投影在秩空间是线性的，移除不了基列的单调非线性变换。存活因子是**待验证候选**，不是独立性证明。

### 6.4 去相关与保留

名人堂（60）→ 过 `sel_net_return > 0` → 按 `(-sel_net_return, n_nodes)` 排序 → 贪心按截面秩相关去重（`max_abs_corr=0.7`）→ 保留 `n_keep=24`。

喂 24 个同一个想法的副本给网络，效果不如喂 8 个真正不同的。

### 6.5 事件表专用算子集（`event_primitives.py`）

- `FORBIDDEN`：所有窗口算子，**永久禁止**（槽位面板上无意义），`assert_no_time_series` 硬断言。
- `SEARCH_EXCLUDED`：目前 `{"sign"}`，**仅从搜索中摘除**，不是禁止——已保存的表达式必须仍能回放。`FactorLibrary.build_pset` 用 `exclude=frozenset()` 重建完整算子集来解析历史因子。

  动机：上一轮 30 个因子里 27 个含 `sign(...)`，典型长相 `sub(sign(A), B)`——树在 A 和 B 上各劈一刀就复现了。它吃光了搜索预算，而真正算不出来的截面结构没人碰。（`744bc31`）

---

## 7. 深度学习合成层（`helix/dl/`）

### 7.1 输入构造（`dataset.py`）

```
factors (T,N,K) ──截面 z-score（按日，population = D0 股票池）──► winsorize ±4σ
                                                                      │
                          + traded 通道 (T,N,1) ────────────────────► (B, L=20, K+1)
```

两条设计约束：

- **标准化的 population 必须是 `universe`，不是 `labels.valid`。** 后者额外依赖 D+1 是否一字涨停、D+1/D+2 是否停牌——用它当 population，未来信息就会影响 D0 特征的均值和标准差。而且它会让最后两行全 NaN，正好是实盘打分需要数值的地方。
- **停牌日在回看窗内会变成 0**，和「真的是平均值」不可区分。所以额外拼一个 `traded` 通道让网络能分辨，而不是从伪造的平坦段里学东西。

### 7.2 模型（`models.py`）

`GRUCombiner`：`LayerNorm → GRU(hidden=96, layers=2, dropout=0.2) → 取末步 → MLP head → logit`

**为什么是 GRU 不是 Transformer**：~20 个时间步、几十个输入、正样本率几个百分点。注意力没有可注意的东西，只会更快过拟合。循环路径捕捉的是「这个因子已经积累了一周」，这正是两日触及需要的信息形状。

### 7.3 训练（`train.py`）

每折一个模型：训练用该折 train 行，早停看 valid 行，只给 test 行打分。拼接各折 test 预测 → 每个点都出自没见过该日期的模型。

训练/验证索引用 `labels.valid`，因为损失与早停指标只能消费可观测标签；test 打分索引则用
D0 `universe` 加当日因子覆盖率。两者必须分开：如果 test 也用 `labels.valid`，预测矩阵
里的 NaN 分布本身就会泄露 D+1/D+2 是否可交易，回测即使显式传入 `universe` 也会在
`isfinite(predictions)` 处再次把未来不可交易股票删掉。

| 决策 | 值 | 理由 |
|---|---|---|
| 损失 | `BCEWithLogitsLoss(pos_weight)` | `pos_weight = n_neg/n_pos`，**上限 20**。不封顶时 2% 正样本率给出权重 49，模型去追异常值 |
| 早停指标 | **日频 gini**，不是 pooled loss | loss 会因为「学会哪些周整体好做」而下降；日频 gini 只有当天内排序变好才涨。策略要做的是当天在几千只票里选 20 只 |
| 梯度裁剪 | 5.0 | — |
| DataLoader workers | 0 | 面板已在内存，spawn 平台上每个 worker 会整份复制 |

**所有指标一律按交易日算完再平均，从不跨日池化。**

### 7.4 断点（`checkpoint.py`）

每折存权重 + `factor_names` + `seq_len` + 训练区间。`require_matching_factors` 在打分时校验因子集一致——因子库变了却用旧权重打分，是静默错到底的那类 bug。

---

## 8. 切分纪律（`helix/splits.py`）

```
[──── train 750 ────][emb][── valid 120 ──][emb][── test 120 ──]
                                                                 step 120 →  下一折
```

- `embargo_days = 5`，**配置加载时强制校验 `embargo_days ≥ touch_offset + 1`**，不满足直接 `ValueError`（`config.py:125`）。D0 的样本 D+2 才结算，紧邻切分会让训练集尾部泄漏到验证集头部。
- 数据不够一折时报错并给出可操作提示，不静默产出空折。

---

## 9. 评估与回测层（`helix/eval/`）

| 模块 | 内容 |
|---|---|
| `metrics.py` | `daily_gini`（日频 AUC 的线性变换）、`summarize_daily`（mean/IR/coverage）、`lift_at_k`、`pairwise_max_abs_corr` |
| `ic.py` | `daily_ic` / `summarize_ic`（IC、ICIR、年化 ICIR、正 IC 占比） |
| `backtest.py` | 交易级回测 |

**退出规则**（`backtest.exit_rule`，默认 `close`）：

| 取值 | 含义 | 为什么 |
|---|---|---|
| `close` | 全部持到 D+2 收盘 | 默认。触及一个价格不等于在那个价格成交；止盈会把唯一赚钱的一侧削平 |
| `target` | 触及 `entry × 1.08` 即按目标价成交，否则收盘出 | 标签的字面口径。**负期望**（CAGR −46.7% vs +51.6%），保留仅为复现该负面结果 |

**成本**按 A 股法定口径逐笔计，乘法而非减法：`(1+g)(1−sell)/(1+buy) − 1`。佣金 2.5bp + 过户费 0.1bp 双边、印花税仅卖出、滑点单边默认 10bp。印花税 2023-08-28 由 10bp 减半为 5bp，面板数据从 2018 年起，所以**卖出费率按交易日取值**（`STAMP_CUT_DATE`），而不是全程一个数——盈亏平衡点在单边 19bp，5bp 的误差就是四分之一的余量。

`BacktestConfig` 设了 `extra="forbid"`：遗留的 `cost_bps` 键会直接报错，而不是被 pydantic 静默忽略后悄悄改用新默认值。

**持仓规模** `top_k = 4`（`configs/default.yaml`、`argus_neutral.yaml`、pydantic 默认值三处一致）。资金按 `overlap = touch_offset − entry_offset + 1 = 2` 拆分（新一批每天开仓，上一批还没了结）。

**候选与成交严格分层**：`run_backtest` 先用 D0 `universe` 和当时已有的预测固定
`top_k` 名单，再对名单内股票应用 `labels.valid` / `touch_tradable` 成交校验。D+1
涨停或 D+2 停牌可以让已选股票不计入成交，但不会从更深排名补位。`lift_at_k` 也按
同一顺序先选股；若入选股票的结果不可观测，该日指标为 NaN，而不是偷换成下一名。
标签的 `valid` / `y` 生成口径保持不变，因此停牌结果仍是 NaN、不会被填成 0。
未成交槽位的资金留在现金；组合收益按 `sum(已成交净收益) / top_k / overlap` 计算，
全未成交日保留为 0 收益，不删除交易日或把资金集中到剩余成交股票。

---

## 10. 配置与入口

**配置**（`helix/config.py`，pydantic）：YAML → 强类型模型，每节 1:1 对应。密钥不进 YAML，只从 `HELIX_TUSHARE_TOKEN` / `.env` 读。跨节校验（embargo vs touch_offset）在 `Config.load` 里做。

**CLI**（`helix/cli.py`）：

```
helix download → prepare → mine → evaluate → train → backtest → score
helix run       # 一条龙
```

**事件表脚本**：

```bash
python scripts/mine_argus.py --input train.parquet \
       --lineage train.lineage.json --calendar calendar.parquet --out artifacts \
       --n-features 70 --rounds 5 --neutralize-base 8
python artifacts/apply_factors.py --input train.parquet \
       --lineage train.lineage.json --calendar calendar.parquet \
       --output train_with_factors.parquet \
       --output-lineage train_with_factors.lineage.json
```

`--rounds K` 让每一轮把已找到的因子加进中性化基，逼出互不相关的因子。

**验证脚本**（都是为了不相信自己的结论而写的）：

| 脚本 | 回答什么 |
|---|---|
| `ablate_factors.py` | 加了因子列，下游模型真的变好了吗（多种子） |
| `check_fillability.py` / `fill_impact.py` | 源表有没有剔除买不进的样本，影响多大 |
| `check_suspension.py` | 停牌是被剔了还是被静默填 0（带**正对照**：正对照全零直接 `SystemExit`） |
| `backtest_argus.py` | 排序能力换算成钱是多少 |
| `window_stats.py` | 短窗口对照表的数字在长曲线上是什么分位 |

---

## 11. 产物目录

```
data/
├── raw/                     # Tushare parquet + 事件表
├── cache/                   # panel.npz, base_fields.npz（贵的部分只算一次）
└── artifacts/
    ├── factors.json         # 面板通路因子库
    ├── predictions.npz      # 拼接后的 OOS 预测
    ├── backtest_summary.json
    ├── scores_YYYYMMDD.csv  # 实盘打分
    ├── models/fold_NNN.pt   # 每折断点
    └── argus*/              # 事件表产物：event_factors.json / IC 报告 / apply_factors.py
```

---

## 12. 全局不变量清单

这些是任何改动都不能破坏的：

1. 第 `t` 行 = 在 `dates[t]` 收盘做的决策。
2. `lead()` 只出现在 `helix/labels/`。
3. 标签不可观测时是 NaN，永远不是 0。
4. 后复权价算收益比值；原始价只用于和涨跌停价比较。
5. GP 搜索只看 `search_window()`。
6. 所有指标按日算完再平均，不跨日池化。
7. 截面标准化的 population 是 `universe`，不是 `labels.valid`。
8. D0 候选排名只用 `universe`；未来可交易性只能在固定名单后的成交校验中使用。
9. 槽位面板上时序算子是硬断言拒绝，不是约定。
10. 特征集里不能出现 `label*` 前缀列，硬断言。
11. `embargo_days ≥ touch_offset + 1`，配置加载时校验。

---

## 13. 决策点

### 13.1 已落地

| # | 决策点 | 结论 | 提交 |
|---|---|---|---|
| 2 | **回测退出规则** | 回填 `exit_rule`，默认 `close`；成本一并对齐法定口径（乘法、印花税按交易日）。`target` 保留仅为复现负面结果 | `6e080c9` |
| 3 | **持仓规模** | TOP20 → TOP4，三处配置同步 | `36c070d` |
| 4 | **`sign` 排除** | 从事件表搜索中摘除，保留回放能力 | `744bc31` |

`scripts/backtest_argus.py` 的 `--hold-grid` 与 `--signal-k` 替补深度**未**回填（见 §9 两条已知差距）。

### 13.2 未决

| # | 决策点 | 现状 | 备选 |
|---|---|---|---|
| 1 | **预测目标** | `P(D+2 触及 +8%)` 二分类 | 与「持到收盘」执行不对口（分类器对 D+2 收益的 IC 是 **−0.064**）；但换成回归 `label_d2_return` 在 hold≤4 时反而更差。是否保持不变 |
| 5 | **截面算子缺口** | 459 列里只有 41 个 rank 型，400 列没有截面版本；上轮搜索也没走进这个缺口 | 是否作为下一步：直接批量生成 `cs_rank/cs_zscore` 列做消融，而不是让 GP 去碰运气 |
| 6 | **两条通路的关系** | 各自独立，共用底层 | 是否需要收敛（面板通路目前没有跑过完整实证，README 的所有结论都来自事件表通路） |
