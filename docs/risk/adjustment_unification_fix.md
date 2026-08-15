# 后复权基线统一修复说明

**生成日期：** 2026-08-14

**生成入口：** `scripts/adjustment_unification_baseline.py`

**证据性质：** 固定 `gp_000` 分数与选股的只读 before/after 对照；不重训、不重新挖掘、不覆盖历史产物。

## 执行摘要

**复权口径问题存在，但不是核心或主导亏损原因。** 同一组冻结的 `gp_000` 分数和 Top4 选择从 legacy raw outcome 切换为同日点时 HFQ outcome 后，单笔净收益仅从 -0.545565% 改善至 -0.523340%，变化 0.022226%，收益仍为负。

**目标错配是主导亏损原因。** `gp_000` 的历史准入目标与 D+2 收盘净收益目标错配；本次工作只建立可审计的合规复权基线，不修复老因子的盈利能力。后续新一代 GP 因子必须从带四元血缘的 HFQ 新链路生成。

现有 `gp_000` 成品特征与 event 表仅作为“修复前 legacy 基线”保留。其上游价格口径和 `source_date/as_of_time/price_basis/adj_factor_version` 不可追溯，本报告不将其补写为已验证 raw 或 HFQ，也不改写任何既有实验结论。详见[治理台账 D10](../factor-governance.md)与[既有专项审计](gp000_loss_attribution.md)。

## 修复合同

- 新链路中，跨日价格因子、标签、成交计价与收益核算只接受带四元血缘的点时 HFQ 价格，校验失败即终止。
- 原始 OHLC 只用于同日涨跌停状态特征（涨跌停距离与计数）以及涨跌停/可成交性校验，不参与跨日价格因子、标签值或持仓收益核算。
- 本对照是 legacy baseline adapter：只读取既有 event 特征、正式因子库和行情缓存，并复用专项审计的纯计算函数；不调用历史报告写入器。
- 两个对照臂共享同一分数、同一 Top4 选择和同一成本/滑点配置，仅 outcome 价格口径不同。

## 固定窗口与 D+2 边界

| 项目 | 值 |
| --- | ---: |
| 名义训练窗 | 2022-01-04 至 2024-09-04 |
| D+2 完整 D0 数 | 647 |
| 最后 D0 | 2024-09-02 |
| 最后 D+2 退出日 | 2024-09-04 |
| Top4 冻结选择摘要 | `2fd376727e7cd83ee664f58b16da871bb0a96ce555d9d75cc156c6415804c740` |

最后两个没有完整 D+2 outcome 的 D0 被严格排除；任何 D0 数量、最后 D0 或退出日变化都会触发 fail-closed，不会自动更新基线。

## Legacy outcome 重建闸门

| 校验 | 原始统计值 | 要求 |
| --- | ---: | --- |
| event return 与 raw 重建一致 | `true` | 必须为 `true` |
| 最大舍入误差 | 5.00000000046e-07 | 不超过 `1e-6` |

该闸门只证明本次 legacy event outcome 能由指定 raw 行情缓存重建；不证明 legacy 因子特征的上游复权口径或四元血缘。

## gp_000 修复前后核心指标

| 指标 | 修复前：legacy raw outcome | 修复后：点时 HFQ outcome | 变化 |
| --- | ---: | ---: | ---: |
| D+2 close IC | -0.0627748064 | -0.0628999742 | -0.0001251678 |
| Top4 单笔净收益 | -0.545565% | -0.523340% | 0.022226% |
| 年化 Sharpe | -1.442030 | -1.388278 | +0.053752 |
| CAGR（补充） | -55.173493% | -53.857140% | 1.316353% |
| 期末净值（补充） | 0.127447 | 0.137278 | +0.009831 |

该变化与既有专项审计预估一致：Top4 单笔净收益改善约 `0.022226` 个百分点，修复后约 `-0.5233%`，未逆转亏损结论。

## 输入追溯

| 输入 | 绝对路径 | SHA-256/集合摘要 |
| --- | --- | --- |
| legacy event 表 | `/Users/aochong/code/helix/data/raw/argus_quant_working.parquet` | `300d9542435735c16701d3360ed7f76bedfa98d647903ce4bf384e4d19e38903` |
| 正式 gp_000 因子库 | `/Users/aochong/code/helix/data/artifacts/argus/event_factors.json` | `6823a9e7d76caa4adcd21cd82e781d85e70407191aab9ef1835138934fb05391` |
| D+2 行情缓存 | `/Users/aochong/code/helix/data/raw/d2_exit_cache` | `c0601a8626de4446703c210e8f5d27debc611dd1bd53291fdef7bba859bfb2c6` |
| 成本与 Top4 配置 | `/Users/aochong/code/helix/.worktrees/adjustment-unification/configs/default.yaml` | `b209a5a2302089edf2a1dd0c3201e063c6ead2ff581f1bce63243ff1ee41f137` |
| 本报告生成脚本 | `/Users/aochong/code/helix/.worktrees/adjustment-unification/scripts/adjustment_unification_baseline.py` | `2187f6cf2464c5f8aa182efb1cd90d60858acc6d457e3ab9be9059aaa945d50d` |

所有输入通过从可信 `/` 开始、逐组件 `O_NOFOLLOW` 的源 fd 复制到私有只读快照；SHA-256 与解析器消费的是同一份复制字节。行情只从该 manifest 的快照文件集合加载。计算结束后脚本重新枚举源缓存，并对全部源文件重新 stat 与 SHA-256 校验；任何内容、身份或成员变化都会终止发布。

## 运行来源

| 项目 | 值 |
| --- | --- |
| Git HEAD | `8c5a7fd7c180b87fa9ea931ee03174e25cca6114` |
| Git HEAD tree | `64c21fb7e0ef412889e86027a3ca316725248911` |
| 排除生成报告后的工作树是否脏 | `true` |
| 工作树 diff SHA-256 | `6ae0bacf37495800fb326de05dc1ed6a2a22c55af415ad20f27bc97c68b79702` |
| 工作树 status SHA-256 | `92e3f4e3d121b29afd8a2d8800e6d298396bcfc59c0399e13c5e4c27df003f9f` |
| Python | `3.11.15 (main, Apr 14 2026, 14:45:51) [Clang 22.1.3 ]` |
| NumPy | `2.4.6` |
| pandas | `3.0.5` |

严格 JSON 还记录了比较脚本、专项审计、配置、回测、IC 与因子库模块的逐文件 SHA-256。

CLI 标准输出同时提供严格 JSON，其中 `legacy_unverified_lineage=true`、`historical_reports_rewritten=false`、`loss_conclusion_unchanged=true`。这些标志防止把 outcome 修正误解为对 legacy 特征血缘或盈利能力的认证。

## 限制与后续基线

- “修复前 raw”只描述 event outcome 与历史行情缓存的重建关系，不代表 legacy 特征的上游价格口径已获认证。
- 本次没有重新训练、调参或改变正式因子方向；分数由冻结的正式表达式和既有成品特征确定，两个对照臂不重新选股。
- 本报告不替代、不覆盖 `docs/risk/gp000_loss_attribution.md`，也不修改历史 artifacts。
- 新一代 GP 因子不得通过 legacy adapter 进入正式训练；必须使用新的 HFQ 血缘强契约链路。
- 报告发布与输入完整性实现依赖 POSIX `dir_fd`、`O_DIRECTORY`、`O_NOFOLLOW` 和 descriptor-relative rename；不具备这些能力的平台会在读取或写入前明确终止。
