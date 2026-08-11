# Helix

用遗传规划挖掘因子表达式 + 深度学习合成，预测 A 股短周期"触及"事件。

**预测目标**

> D0 收盘后用截至 D0 的信息计算因子 → D+1 开盘买入 → **D+2 最高价是否触及 D+1 开盘价 × 1.08**

```
D0 收盘          D+1 开盘          D+2 盘中
  │                │                 │
  └─ 特征计算       └─ 建仓           └─ high ≥ entry × 1.08 ?
```

---

## 三个决定成败的实现细节

这类项目最常见的失败不是模型不够强，而是标签或切分里藏了未来函数，跑出漂亮的回测然后实盘归零。Helix 在这三处做了明确处理：

| 问题 | 错误做法 | Helix 的做法 |
|------|---------|-------------|
| **除权除息** | 用原始价算 `high[D+2]/open[D+1]` | 用后复权价（`*_hfq`）算比值。原始价只用于和当日涨跌停价比较 |
| **一字涨停买不进** | 把这些样本当正样本留在训练集 | D+1 开盘价 ≥ 当日涨停价的样本**直接剔除**（标签置为未定义，不是 0） |
| **ST 判定** | 用当前股票名过滤 | 用 `namechange` 历史还原**每个交易日当时**的股票名 |

此外：

- **停牌** → 标签未定义（NaN），不是 0。D+1 或 D+2 停牌就无法观测结果。
- **切分隔离** → train/valid/test 之间强制 `embargo_days` 间隔。D0 的样本在 D+2 才结算，紧邻切分会让训练集尾部泄漏到验证集头部。配置加载时会校验 `embargo_days ≥ touch_offset + 1`，不满足直接报错。
- **挖掘只看最早的训练块** → GP 搜索限制在第一个 `train_days` 窗口内，之后的数据全部留给 walk-forward。在全历史上挖因子再"验证"是自欺欺人。

---

## 架构

```
Tushare ──► ParquetStore ──► Panel (T×N)  ──► base fields (~26)
                                 │                  │
                            universe mask           ▼
                                 │            GP 搜索 (DEAP)
                                 ▼                  │
                            touch label      factors.json (可读表达式)
                                 │                  │
                                 └────────┬─────────┘
                                          ▼
                              截面标准化 → GRU 合成 → walk-forward
                                          ▼
                                    OOS 预测 → 回测
```

面板是 `(T, N)` 的二维数组：行是交易日，列是股票。时序算子沿 axis 0，截面算子沿 axis 1 —— 这个布局是 GP 能跑得动的前提。

### 模块

| 路径 | 职责 |
|------|------|
| `helix/data/` | Tushare 增量下载、parquet 存储、面板构建、时点股票池 |
| `helix/labels/` | 触及标签，含可成交性过滤 |
| `helix/features/operators.py` | 面板算子（`ts_*` 严格后视，`lead` 仅供标签使用） |
| `helix/features/base_fields.py` | GP 的原料字段 |
| `helix/gp/` | 类型化算子集、适应度、进化循环、因子库持久化 |
| `helix/dl/` | 序列构造、GRU 合成模型、walk-forward 训练 |
| `helix/eval/` | 日频 AUC/gini、precision@k、交易级回测 |
| `helix/splits.py` | 带 embargo 的 walk-forward 切分 |

---

## 快速开始

```bash
uv venv --python 3.11
uv pip install -e ".[dev]"

cp .env.example .env        # 填入 Tushare Pro token
```

```bash
.venv/bin/helix download    # 全市场日线/复权因子/涨跌停/每日指标（首次较慢）
.venv/bin/helix prepare     # 建面板 + 基础字段 + 标签，打印正样本率
.venv/bin/helix mine        # GP 挖因子 → data/artifacts/factors.json
.venv/bin/helix evaluate    # 每个因子的样本内 vs 搜索窗口外表现
.venv/bin/helix train       # walk-forward 训练合成模型 → predictions.npz
.venv/bin/helix backtest    # top-k 回测 → backtest_summary.json
```

或者一条龙：`helix run`。

所有参数在 `configs/default.yaml`，改完用 `-c my.yaml` 指定。

---

## 关键设计说明

### GP 适应度：两个窗口，不是一个

搜索窗口内部再切成 **fit 段**（驱动进化）和**隔离后的 sel 段**（决定去留）。进化最大化 fit 段的 `|gini|`；一个因子只有在 sel 段上**符号仍然一致且为正**才会被保留。在同一批行上既优化又筛选，是 GP 流水线产出"样本内绝美、样本外归零"因子的标准姿势。

因子的符号是自由参数：gini 为 `-0.2` 的因子取负后和 `+0.2` 一样好用，所以适应度用绝对值，符号单独记录在 `factors.json` 里。

### 为什么是日频 AUC 而不是池化 AUC

所有指标都**按交易日算完再平均**，从不跨日池化。池化会让"知道哪些周整体好做"的因子拿高分，但策略要做的是**当天在几千只票里选 20 只**，跨日信息毫无用处。早停也盯日频 gini，不盯 loss。

### 去相关

GP 会收敛到一堆近乎同义的表达式。名人堂按 sel 段表现排序后，贪心地按截面秩相关（阈值 `max_abs_corr`）去重再交给网络。喂 24 个同一个想法的副本，效果不如喂 8 个真正不同的。

### 为什么用 GRU 不用 Transformer

回看窗口 20 天、输入二十来个因子、正样本率个位数百分比 —— 注意力机制没什么可注意的，只会更快过拟合。循环结构能抓"这个因子连涨了一周"这种形态，正好是两日触及关心的信息。

停牌日在序列里会变成 0，和"真的很平均"无法区分，所以额外拼了一个 `traded` 通道让网络能分辨。

### 回测的退出假设

和标签严格一致：触及则按目标价成交（+8%），未触及则按 D+2 收盘价离场。只报命中率会掩盖真正的问题 —— 没命中的那批往往是低开砸下去的。

净值曲线按 `overlap = touch_offset - entry_offset + 1 = 2` 拆分资金（每天开新仓时上一批还没平），所以不是逐日收益的简单累加。

---

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python -m ruff check helix tests
```

`tests/test_pipeline_smoke.py` 在合成行情上跑完整链路（埋了一个真实可发现的信号），不需要 Tushare。它验证 GP 确实能找到该信号，并且该信号在搜索窗口之外仍然有效。

标签语义由 `tests/test_labels.py` 逐条钉死，包括除权、涨停买不进、停牌三种情况。

---

## 尚未包含

- **实盘打分链路**：`train` 目前不保存模型权重，所以没有"用最新一天数据打分"的命令。要上实盘需要先在 `train_fold` 里落盘 `state_dict`，再加一个加载最后一折模型、对最新交易日推理的入口。
- **成本模型**：只有固定的双边 bps。没有冲击成本、没有按成交量约束仓位。
- **指数/行业中性化**：因子未做行业中性，选出来的票可能高度集中在少数行业。

---

## 一个必要的预期管理

这个框架解决的是**方法论正确性**问题 —— 它保证你测出来的数字是真的。它不保证数字好看。

D+2 单日触及 +8%，在剔除 ST（5% 涨停会让标签几乎不可达）后的全 A 股上，正样本率大概率是个位数百分比。先跑 `helix prepare` 看 base rate，再看 `helix backtest` 的 `lift` —— **lift 是唯一重要的数字**。lift 接近 1 说明模型没有信息，无论命中率看起来多高。
