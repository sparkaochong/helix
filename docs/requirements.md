# 需求与上线状态总表

存量盘点未完成：本次仅核对代码同步范围；历史计划、Issue/PR 和生产发布尚未全量逐条复核。未知业务状态不由 WIP、提交或合并推导。

## 2026-11-04 代码协作同步

| ID | 名称与可验证验收标准 | 来源/Issue | 优先级 | 当前持有者 | 研发状态 | 上线状态 | 计划/PR/提交与证据 | 阻塞与下一步 | 最近更新 |
|---|---|---|---|---|---|---|---|---|---|
| REQ-GH-7 | mac-dev 现有源码/文档归并 GitHub 并同步 Mac mini；适用验证通过后经 PR 合并；Mini 实际 SHA 与 origin/main 一致且工作区干净；运行数据/凭据/其他会话保留 | [Issue #7](https://github.com/sparkaochong/helix/issues/7)；用户 2026-11-04 请求 | P1 | Codex /root；GitHub sparkaochong | 已验收 | 无需上线 | [同步验收记录](macmini-sync-20261104.md)；分支 `feat/github-mini-sync-20261104` | [源码 PR](https://github.com/sparkaochong/helix/pull/8)；[Mini SHA 与验证](macmini-sync-20261104.md) | 已核对源码合并版本与 Mini；本需求仅源码协作归档，无生产发布 |
