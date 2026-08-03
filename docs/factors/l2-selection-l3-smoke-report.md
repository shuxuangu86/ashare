# L2 去重、FEATURE_ELIGIBLE 与 L3 家族 Smoke 报告

## 结论

市值中性五年 OOS 的 2,569 个可执行因子已完成本阶段筛选：2,517 个通过
`RESEARCH_VALIDATED`，140 个进入 `FEATURE_ELIGIBLE`，52 个硬拒绝。新增因子在
入池数量、显著候选和家族覆盖上均提供了明确增量，足以进入正式 L3 建设；但本次
L3 仅是接口与收敛能力 smoke，不是独立的最终测试或 L4 策略证据。

## 去重与筛选

- 表达式完全重复：0。
- 家族内 RankIC 行为近重复：762，阈值为绝对相关 0.995。
- 硬拒绝：退化输出 38，覆盖不足 14。
- BH-FDR 5%：入池 140 个中 35 个显著。
- 每家族最多保留 9 个，覆盖最强、最稳定、最低换手、最独立、状态互补及 Pareto
  代表。

原始 73 因子有 34 个入池（46.6%），新增因子有 106 个入池。入池因子的平均绝对
OOS RankIC：原始因子 0.0237，新增因子 0.0264；FDR 显著数分别为 7 和 28。
因此新增库不是单纯扩数量，它贡献了 75.7% 的最终候选，并扩展到公式 Alpha、技术
规则、通道、震荡、量价、波动、流动性和状态家族。

## L3 Smoke

19 个可组合家族完成 equal-weight、滚动 IC、滚动 ICIR、Ridge、Elastic Net 五类
三折 expanding walk-forward。按家族平均 RankIC，滚动 IC 为 0.0530，滚动 ICIR
为 0.0527，Ridge 为 0.0503，等权为 0.0486。最佳单项是 microcap_risk Ridge，
三折 RankIC 为 0.0945、0.1144、0.0502，均值 0.0863。

该 smoke 的所有拟合和标准化只使用折内训练窗；但 L2 池本身使用了同一历史的 OOS
证据，因此结果明确标记 `RESEARCH_ONLY`。正式 L3 应采用嵌套 walk-forward 或后续
新数据发布作为最终隔离测试。

## 待完成

- `RUNNING`：industry/style proxy 稳健性评价，最近为 943/2,569；它不是真实行业
  中性化。
- `NOT_RUN`：严格 `STANDALONE_PRODUCTION_ALPHA` 门禁，本阶段未降低阈值。
- `DEFERRED`：获得完整 PIT 行业历史后，补做真实行业中性化稳健性验证。
