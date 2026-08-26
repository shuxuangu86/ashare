# AQuant L2 因子扩展报告

生成日期：2026-07-31（Asia/Shanghai）

## 来源复现

| 来源 | 完成情况 | 可信度说明 |
|---|---:|---|
| Alpha101 | 101/101 可执行 | 53 EXACT；30 NORMALIZED_EQUIVALENT；18 A_SHARE_ADAPTED |
| GTJA Alpha191 | 190/191 可执行，191/191 已登记 | 116 EXACT；46 CORRECTED_AMBIGUITY；29 A_SHARE_ADAPTED |
| GTJA Alpha143 | 0/1 | `SELF` 无可验证定义，FORMULA_AMBIGUOUS |
| 经典技术指标 | 1,470 个受控候选 | DERIVED_VARIANT |
| Han–Yang–Zhou 2013 | 13 个 | 7 个基础成分；6 个可解释二阶交互 |
| China 7000 Rules | 687 个 | 五类规则的受控参数空间；原始 7,846 次试验已登记 |
| Technical Sentiment 2023 | 7 个 | 基于 10 个预声明日频信号；不使用测试期择优 |
| Fama–French 2015 | 2/4 个核心维度 | size、book-to-market 已实现；profitability、investment 缺 PIT 字段 |
| 分钟级来源 | 2 个来源延期 | DEFERRED_INTRADAY |

两份来源均保存 PDF SHA-256、原始公式、页码和采用公式。PDF 不提交 Git。
GTJA 的 OCR 修复、缺省窗口解释和市场基准代理均显式标记，不冒充 EXACT。
收盘后计算统一设置 `availability_lag=1`。

## 因子数量

| 指标 | 数量 |
|---|---:|
| 开始时基线 | 73 |
| second-wave | 28 |
| 技术候选 | 1,470 |
| Alpha101 | 101 |
| GTJA Alpha191 | 191 |
| 学术扩展候选 | 707 |
| 总注册候选 | 2,570 |
| 相对 73 基线新增注册 | 2,497 |
| 公开公式运行审计通过 | 291/292 |
| IMPLEMENTED | 2,569 |
| FORMULA_AMBIGUOUS | 1 |
| RESEARCH_VALIDATED / FEATURE_ELIGIBLE | 待真实 OOS 评价 |
| STANDALONE_PRODUCTION_ALPHA | 0（未降低门槛） |

目录构建 PASS，`factor_library_v2` 输入哈希为
`f922c94a4bc1b9489e1030b5ff17de61592d6f23139a1a23fafc222999c18b35`。

## 评价任务

- 数据：`cn_equity_20260717_001`
- 区间：2021-07-18 至 2026-07-17
- horizons：1、5、10、20、40
- Alpha101：101/101 完成，耗时 1:34:59，峰值内存 3.58 GiB
- GTJA Alpha191：68/191 后因历史批次缺 `open` 字段失败；checkpoint 可恢复
- 学术扩展：已启动，顺序为 Han、Technical Sentiment、Fama–French、China Rules
- 设置：最终 20% OOS、batch=4、单进程、逐批重载、支持 resume
- 学术扩展任务 PID：79766；绑定代码 `2a0f80580f53abb15ab97dc693c259a8f3dd0aa7`

评价完成前不生成虚假的 FEATURE_ELIGIBLE 或 Production 结论。

## 质量与未完成项

- PASS：291 条可执行公式确定性运行；代表公式防未来扰动测试。
- PASS：相关文件 ruff、format、mypy。
- PASS：全量 pytest 711 passed；coverage 87.05%（硬门槛 85%）。
- PASS：学术扩展模块 coverage 93%；代表因子确定性和防未来扰动测试。
- PASS：全仓 ruff、format、mypy。
- PARTIAL：真实覆盖率、RankIC、去重、家族 Pareto、L3 smoke 待评价完成。
- FAILED：GTJA 5 年评价在第 69 个因子的数据字段加载处退出；已保留 68 个 checkpoint。
- DEFERRED：分钟数据因子和缺失财务 PIT 字段。
- BLOCKED：无。
