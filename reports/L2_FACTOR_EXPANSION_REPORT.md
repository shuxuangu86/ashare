# AQuant L2 因子扩展报告

生成日期：2026-07-31（Asia/Shanghai）

## 来源复现

| 来源 | 完成情况 | 可信度说明 |
|---|---:|---|
| Alpha101 | 101/101 可执行 | 53 EXACT；30 NORMALIZED_EQUIVALENT；18 A_SHARE_ADAPTED |
| GTJA Alpha191 | 190/191 可执行，191/191 已登记 | 116 EXACT；46 CORRECTED_AMBIGUITY；29 A_SHARE_ADAPTED |
| GTJA Alpha143 | 0/1 | `SELF` 无可验证定义，FORMULA_AMBIGUOUS |
| 经典技术指标 | 1,470 个受控候选 | DERIVED_VARIANT |
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
| 总注册候选 | 1,863 |
| 公开公式运行审计通过 | 291/292 |
| FORMULA_AMBIGUOUS | 1 |
| RESEARCH_VALIDATED / FEATURE_ELIGIBLE | 待真实 OOS 评价 |
| STANDALONE_PRODUCTION_ALPHA | 0（未降低门槛） |

目录构建 PASS，`factor_library_v2` 输入哈希为
`e6cee6e00201cafa482dc8b468d4ffcb73ef8a229ceb475ec54cf885aad2ce2b`。

## 评价任务

- 数据：`cn_equity_20260717_001`
- 区间：2021-07-18 至 2026-07-17
- horizons：1、5、10、20、40
- 顺序：Alpha101 后 GTJA Alpha191
- 设置：最终 20% OOS、batch=4、单进程、逐批重载、支持 resume
- 状态：已启动；结果写入 `reports/l2-v2-alpha101-5y-v1` 和
  `reports/l2-v2-alpha191-5y-v1`

评价完成前不生成虚假的 FEATURE_ELIGIBLE 或 Production 结论。

## 质量与未完成项

- PASS：291 条可执行公式确定性运行；代表公式防未来扰动测试。
- PASS：相关文件 ruff、format、mypy；新增测试 5 passed。
- PASS：全量 pytest 703 passed；coverage 87.06%（硬门槛 85%）。
- PASS：全仓 ruff、format、mypy。
- PARTIAL：真实覆盖率、RankIC、去重、家族 Pareto、L3 smoke 待评价完成。
- DEFERRED：分钟数据因子和缺失财务 PIT 字段。
- BLOCKED：无。
