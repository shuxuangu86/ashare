# AQuant L2 因子扩展报告

生成日期：2026-07-31（Asia/Shanghai）

## 来源复现

| 来源 | 结果 | 状态 |
|---|---:|---|
| Alpha101 | 0/101 EXACT | PARTIAL / FORMULA_AMBIGUOUS |
| Alpha191 | 0/191 EXACT | PARTIAL / SOURCE_UNAVAILABLE |
| 中银技术指标 | 16 个基础指标，810 个受控表示 | IMPLEMENTED / DERIVED_VARIANT |
| 东海技术指标 | 7 个基础指标，375 个受控表示 | IMPLEMENTED / DERIVED_VARIANT |
| 学术技术规则 | 1 个基础规则族，90 个受控表示 | PARTIAL |
| 中信建投日频原型 | 3 个基础指标，195 个受控表示 | IMPLEMENTED / DERIVED_VARIANT |
| 基本面学术因子 | 既有 BM、动量、Amihud；未新增伪造字段 | PARTIAL |
| 分钟级延期 | 2 个来源 | DEFERRED_INTRADAY |
| 公式歧义 | Alpha101、Alpha191 两个来源级条目 | FORMULA_AMBIGUOUS |

Alpha101/191 没有在缺少逐公式审计和授权本地来源时伪报 EXACT。VWAP 不会被
静默近似；既有 `vwap_deviation` 已记录标准化数据契约中的 CNY/股单位。

## 因子数量

| 指标 | 数量 |
|---|---:|
| 开始时基线因子 | 73 |
| 既有 second-wave | 28 |
| 新增注册 | 1,470 |
| 总注册候选 | 1,571 |
| 成功目录化 | 1,571 |
| 表达式重复合并 | 0 |
| RESEARCH_VALIDATED | 0（待真实评价） |
| FEATURE_ELIGIBLE | 0（待真实评价） |
| STANDALONE_PRODUCTION_ALPHA | 0（未降低严格门槛） |
| REJECTED | 0（尚未执行真实评价） |

新增量位于用户要求的 1,000–3,000 区间。它来自 98 个有界基础参数变体与 15
种统一表示，不是无限笛卡尔积。目录生成时强制检查 factor id 与表达式 hash 唯一性。

## 评价结果

| 项目 | 当前状态 |
|---|---|
| 评价区间 | 2021-07-18 至 2026-07-17 |
| 数据发布 | `cn_equity_20260717_001` |
| horizons | 1、5、10、20、40 |
| 首批评价 | `technical_classic_level_v1`，98 个新因子 |
| OOS | 最终 20% 时间段；固定时序切分 |
| 批处理 | batch 8、逐批重载、resume、单进程 |
| 股票日数量 | PENDING |
| 最强家族 | PENDING |
| 新增独立信息最多家族 | PENDING |
| 与原 73 因子重叠率 | PENDING |
| L3 smoke 最佳方法 | PENDING；FEATURE_ELIGIBLE 为空时不运行 |

目录发布不等于研究验证。当前 pool 文件明确为 `NOT_EVALUATED`，不存在把
DRAFT 候选提前写成 FEATURE_ELIGIBLE 或 Production 的情况。

## 质量

| 门禁 | 结果 |
|---|---|
| pytest | 688 passed |
| coverage | 87.30%（硬门槛 85%） |
| L2 v2 ruff | PASS |
| L2 v2 format | PASS |
| L2 v2 mypy | PASS（9 个相关源文件） |
| 全仓 ruff | BLOCKED：非本任务 `regime`/微盘未提交修改 |
| 全仓 mypy | BLOCKED：非本任务 `regime` 两文件 6 个错误 |
| leakage tests | PASS：未来末行扰动不改变历史结果 |
| deterministic hash | PASS：1,470/1,470 唯一且重复运行稳定 |

全仓测试包含新增 golden、NaN/无穷处理、常数序列、重复运行和末端扰动防泄漏
smoke。未修改并行工作区文件来掩盖 ruff/mypy 失败。

## 性能

| 项目 | 结果 |
|---|---|
| 目录构建耗时 | 0.301 秒 |
| 目录行数 | 1,571 |
| Parquet 大小 | 约 100 KiB（Zstandard） |
| 断点恢复 | PASS；input hash 相同且状态 PASS 时跳过 |
| 默认并发 | 2 |
| 评价峰值内存 | PENDING；`/usr/bin/time -v` 写入评价日志 |
| 失败批次 | 0（目录构建）；真实评价 PENDING |

## 版本

| 项目 | 值 |
|---|---|
| branch | `v0.3.0-microcap-baseline` |
| start commit | `a20374f` |
| library build commit | `81df1f8` |
| data_release_id | `cn_equity_20260717_001` |
| factor_library_hash | `2acf60fe4eb0127255241128a7302d3df9c2cfd220e70147e9a55622de8e176d` |
| feature_eligible_hash | SHA-256 of empty JSON array（待评价） |
| PR | 现有 PR #1；不创建重复 PR |

## 未完成事项

- `PARTIAL`：Alpha101 逐公式转录、VWAP/行业语义和原始/中性版尚未完成。
- `BLOCKED`：Alpha191 缺少授权本地研报；公开转录不标记 EXACT。
- `DEFERRED`：UTD/UTR 与高频技术聚合缺少分钟历史数据。
- `DEFERRED`：Gross Profitability、Asset Growth 缺少完整财务 PIT 字段。
- `PARTIAL`：输出相关、RankIC/多空收益相关去重等待真实物化评价。
- `PARTIAL`：FEATURE_ELIGIBLE、家族 Pareto 与 L3 smoke 等待首批评价完成。
- `BLOCKED`：全仓 ruff/mypy 被本轮范围外的并行未提交文件阻断。

这些状态不会阻塞已完成的日频技术候选目录发布，也不会被含糊表述为完成。
