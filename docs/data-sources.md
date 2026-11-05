# Helix 数据源目录

> **文档定位**：记录 Tushare Pro token 当前实际可访问的接口目录，作为评估"是否值得为某个因子方向新增数据接入"时的查表依据。`helix/data/schema.py` 是**已接入**表的唯一权威契约；本文覆盖的是**已验证可访问但尚未接入**的接口，帮助判断 `docs/factor-governance.md` §3.7 所说的"引入新的原始信息"具体有哪些选项。
>
> **核验方式**：对每个接口发一次最小参数的真实请求（`trade_date=20260731` 或 `ts_code=000001.SZ` 或最近一期报告期），记录返回行数与字段数；权限错误与参数错误分别标注，不混为一谈。
> **核验日期**：2026-08-15（token 更换后）

---

## 1. 已接入 helix 的数据源

这 7 张表由 `helix/data/schema.py` 定义、`tushare_source.py` 下载、`ParquetStore` 落盘，是当前面板构建（`panel.py`）唯一依赖的数据来源。字段契约、单位口径、复权规则见 `docs/architecture.md` §3 与本次基线校验报告。

| 接口 | 用途 | 状态 |
|---|---|---|
| `daily` | 日线行情（不复权） | 已接入 |
| `adj_factor` | 复权因子 | 已接入 |
| `daily_basic` | 每日指标（换手率、量比、PE/PB、流通市值） | 已接入 |
| `stk_limit` | 涨跌停价格 | 已接入 |
| `stock_basic` | 股票基本信息 | 已接入 |
| `namechange` | 名称变更历史（点时 ST 重建依赖） | 已接入 |
| `trade_cal` | 交易日历 | 已接入 |

---

## 2. 新 token 已验证可访问的接口

以下均为**尚未接入 helix 管道**、但用当前 token 实测可正常返回数据的接口，按用途分类。字段数 / 行数为单次探测样本（`trade_date=20260731` 或对应最小参数），仅供规模参考，非官方字段契约——真正要接入时仍需去 Tushare 文档核对完整字段表与单位口径。

### 2.1 行情类（在已用 7 表之外）

| 接口 | 说明 | 探测行数 | 字段数 | 备注 |
|---|---|---|---|---|
| `weekly` | 周线行情 | 5,610 | 11 | |
| `monthly` | 月线行情 | 5,617 | 11 | |
| `stk_factor` | 股票技术因子（简版） | 1（单股） | 35 | |
| `stk_factor_pro` | 股票技术因子（专业版） | 1（单股） | 261 | 官方 bfq/hfq/qfq 三态技术指标全集，见基线校验报告"Step 2 专项" |
| `bak_daily` | 备用行情 | 1（单股） | 31 | **`amount` 单位为万元，与 `daily.amount` 的千元不同，接入时需换算** |
| `suspend_d` | 每日停复牌信息 | 8 | 4 | |
| `limit_list_d` | 涨跌停 / 炸板列表（新版） | 206 | 18 | 含首次封板时间、炸板次数等，`limit_list_ths` 的可替代方案 |
| `kpl_list` | 开盘啦榜单 | 101 | 24 | 题材 / 游资风格的另类数据源 |
| `hm_list` | 游资名录 | 113 | 3 | |
| `ccass_hold` | 中央结算系统持股明细 | 5,000 | 6 | 港股通相关 |
| `hsgt_top10` | 沪深股通十大成交股 | 20 | 11 | |
| `ggt_top10` | 港股通十大成交股 | 20 | 17 | |
| `ggt_daily` | 港股通每日成交统计 | 1 | 5 | |

### 2.2 资金面

| 接口 | 说明 | 探测行数 | 字段数 | 备注 |
|---|---|---|---|---|
| `moneyflow` | 个股资金流向（新浪口径，大中小单买卖） | 5,197 | 20 | |
| `moneyflow_dc` | 个股资金流向（东财口径） | 5,990 | 15 | |
| `moneyflow_ind_dc` | 行业资金流向（东财） | 1,031 | 18 | |
| `moneyflow_mkt_dc` | 大盘资金流向（东财） | 1 | 15 | |
| `moneyflow_hsgt` | 沪深港通资金流向 | 1 | 7 | |
| `margin` | 融资融券交易汇总 | 3 | 9 | |
| `margin_detail` | 融资融券交易明细（个股） | 4,419 | 10 | |
| `margin_secs` | 融资融券标的名单 | 4,090 | 4 | |
| `top_list` | 龙虎榜每日明细 | 80 | 15 | |
| `top_inst` | 龙虎榜机构明细 | 828 | 10 | |
| `block_trade` | 大宗交易 | 153 | 7 | |
| `stk_holdertrade` | 股东增减持 | 26 | 11 | |
| `pledge_stat` | 股权质押统计 | 639 | 7 | |
| `share_float` | 限售股解禁 | 6,000 | 7 | |
| `repurchase` | 股票回购 | 35 | 9 | |

### 2.3 财务类

| 接口 | 说明 | 探测行数 | 字段数 | 备注 |
|---|---|---|---|---|
| `income` | 利润表 | 1（单股单期） | 85 | |
| `balancesheet` | 资产负债表 | 2（单股单期，含合并/母公司口径） | 152 | |
| `cashflow` | 现金流量表 | 1（单股单期） | 97 | |
| `fina_indicator` | 财务指标（ROE、毛利率等衍生指标） | 1（单股单期） | 108 | |
| `fina_mainbz` | 主营业务构成 | 37 | 8 | |
| `forecast` | 业绩预告 | 8 | 13 | |
| `express` | 业绩快报 | 2 | 15 | |
| `dividend` | 分红送股 | 96 | 14 | |
| `disclosure_date` | 财报披露计划 | 120 | 5 | |

### 2.4 基础参考

| 接口 | 说明 | 探测行数 | 字段数 | 备注 |
|---|---|---|---|---|
| `stock_company` | 上市公司基本信息 | 1 | 18 | |
| `stk_managers` | 上市公司管理层 | 264 | 11 | |
| `new_share` | IPO 新股列表 | 2,000 | 12 | |
| `concept` | 概念股分类（旧接口，需带 `trade_date`） | 1,345 | 2 | |
| `index_classify` | 申万行业分类 | 511 | 7 | |
| `index_member` | 申万行业成分股 | 184（单行业） | 5 | |
| `hs_const` | 沪深股通成分股 | 581 | 5 | |

### 2.5 指数

| 接口 | 说明 | 探测行数 | 字段数 | 备注 |
|---|---|---|---|---|
| `index_basic` | 指数基本信息 | 208（SSE） | 8 | |
| `index_daily` | 指数日线行情 | 1（单指数单日） | 11 | |
| `index_weight` | 指数成分和权重 | 300（沪深300单日） | 4 | |
| `index_dailybasic` | 指数每日指标 | 12 | 12 | |

### 2.6 舆情 / 另类

| 接口 | 说明 | 探测行数 | 字段数 | 备注 |
|---|---|---|---|---|
| `report_rc` | 分析师盈利预测 | 5,000 | 21 | |

### 2.7 宏观

| 接口 | 说明 | 探测行数 | 字段数 | 备注 |
|---|---|---|---|---|
| `cn_gdp` | 国内生产总值 | 177 | 9 | |
| `cn_cpi` | 居民消费价格指数 | 511 | 13 | |
| `cn_ppi` | 工业生产者出厂价格指数 | 418 | 31 | |
| `shibor` | 上海银行间同业拆放利率 | 1 | 9 | |

### 2.8 衍生品（与当前 A 股现货因子项目关联度低，仅确认可访问）

| 接口 | 说明 | 探测行数 | 字段数 |
|---|---|---|---|
| `fut_basic` | 期货合约信息 | 716 | 15 |
| `opt_basic` | 期权合约信息 | 12,000 | 20 |

---

## 3. 当前仍不可访问的接口

| 接口 | 说明 | 报错 |
|---|---|---|
| `limit_list_ths` | 同花顺涨跌停榜单 | 无接口访问权限 |
| `limit_step` | 连板天梯 | 无接口访问权限 |
| `hm_detail` | 游资每日交易明细 | 无接口访问权限 |
| `anns_d` | 上市公司公告 | 无接口访问权限 |
| `news` | 新闻快讯 | 无接口访问权限 |
| `cctv_news` | 新闻联播文字稿 | 无接口访问权限 |

`limit_list_d`（新版涨跌停/炸板列表）已可访问，字段覆盖首次封板时间、炸板次数等，可作为 `limit_list_ths` 的功能替代评估对象。

---

## 4. 接入前必须核对的口径陷阱

1. **`bak_daily.amount` 是万元，`daily.amount` 是千元**——相差 10 倍，混用会产生量级错误（详见基线校验报告 Step 2）。**已加装守护工具** `helix/data/unit_registry.py::bak_daily_amount_to_kcny()`（含真实数值对回归测试 `tests/test_unit_registry.py`），但 `bak_daily` 本身仍未接入下载管道——真正接入时必须走这个转换，不能绕开。
2. **`daily`/`adj_factor`/`daily_basic`/`stk_limit` 等已接入表全部用 `trade_date=YYYYMMDD`**；新接口逐个确认日期字段格式后再对齐。**已加装统一工具** `helix/data/dates.py::normalize_trade_date()`，兼容 `YYYYMMDD` 与 `YYYY-MM-DD`。这解决的是"以后要 join 不同格式的表"这个通用问题；`argus_quant_working.parquet` 本身用 `YYYY-MM-DD`，`docs/factor-governance.md` 已把它冻结为 legacy 基线，**不会**被这个工具改写——它是给未来 join 代码用的桥接函数，不是文件迁移脚本。
3. 财务类接口（`income`/`balancesheet`/`cashflow`/`fina_indicator`）返回**多个口径版本**（如 `balancesheet` 同一 `period` 返回合并 / 母公司两行），接入前需明确按 `report_type` / `comp_type` 过滤，避免重复行造成面板错位。
4. 任何新接入的数据源，落盘前都必须过 `helix/data/schema.py` 同款的显式列契约校验，不能绕开 `docs/factor-governance.md` §5.1 的标签前缀盲扫等既有防泄漏机制。

---

## 5. 与因子挖掘方向的对应关系

`docs/factor-governance.md` §3.7 指出，继续挖因子必须引入"表外新信息"（盘口、资讯、其他频率），而不是对已有列重新编码。上表里对应的候选方向：

| 方向 | 候选接口 |
|---|---|
| 资金流 / 主力行为 | `moneyflow`、`moneyflow_dc`、`moneyflow_ind_dc`、`margin_detail`、`top_list`、`top_inst`、`block_trade` |
| 游资 / 题材情绪 | `kpl_list`、`hm_list`、`limit_list_d` |
| 基本面 | `income`、`balancesheet`、`cashflow`、`fina_indicator`、`forecast`、`express` |
| 股东行为 | `stk_holdertrade`、`pledge_stat`、`share_float`、`repurchase` |
| 分析师预期 | `report_rc` |
| 行业 / 概念结构 | `index_classify`、`index_member`、`concept` |

任何一个方向真正接入前仍需走完整的 G0~G3 准入流程，本文只解决"数据拿不拿得到"，不构成任何因子有效性的背书。

---

## 6. 本轮修复状态（2026-08-15，`fix/data-baseline-remediation`）

针对《Helix 数据基线校验》审计发现的 P1/P2 缺陷与口径陷阱，本轮已完成：

| 项目 | 状态 |
|---|---|
| `namechange` 分页截断（P1） | **已修复**：`TushareSource._call_paginated()`，本地行数已核实变为 14,178，与官方全量一致 |
| 配置声明区间与实际覆盖不符 | **已修复**：默认 `start_date` 纠正为如实的 `20211201`；`build_panel` 新增 `_validate_panel_coverage()` 强制校验，区间缺口直接 `PanelCoverageError` |
| `stk_limit` 兜底规则缺 ST 5% 分支（P2） | **已修复**：`helix/data/st_status.py` + `panel.py::_limit_pct()`，仅对主板 ST/*ST 生效（科创/创业/北交所维持原板块比例，"退" 类不适用），真实数据核验 95.6% 落在 0.01 元误差内 |
| 每日数据完整性巡检 | **已交付**：`scripts/check_data_freshness.py`（默认模式纯本地、`--deep` 模式含实时分页复核） |
| 历史回填（2018-01~2021-11） | **工具已交付，未实际执行**：`configs/backfill_2018_2021.yaml` |
| `bak_daily` 单位陷阱 | **已加装守护**，见 §4 第 1 条 |
| `trade_date` 格式桥接 | **已加装工具**，见 §4 第 2 条 |

完整的修复前后对比、校验数据与剩余风险见修复验证报告（Artifact，随本轮提交发布）。
