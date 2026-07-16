# 实施状态

更新时间：2026-07-16

## Day 1：基础设施与领域骨架

| 项目 | 状态 | 验收方式 |
| --- | --- | --- |
| Monorepo 与 `src` 布局 | 完成 | 包可编辑安装 |
| uv 项目与锁文件 | 完成 | `uv sync --group dev` |
| Ruff / mypy / pytest / pre-commit | 完成 | `make quality` |
| PostgreSQL / Redis / Prefect / MLflow 配置 | 完成 | `docker compose config --quiet` |
| 本机容器启动 | 待机器验收 | 当前执行环境没有 Docker |
| 分层 YAML 配置与环境覆盖 | 完成 | 单元测试和 `config-check` |
| 实盘安全默认值 | 完成 | 配置验证测试 |
| 核心领域对象 | 完成 | Symbol、TargetPortfolio、OrderIntent、AuditEvent |
| 订单状态机与幂等键 | 完成 | 单元测试 |

自动验收：Ruff、格式检查、mypy strict 通过；133 项测试通过，覆盖率 99.13%。

## Day 2：可信数据协议

| 项目 | 状态 | 验收方式 |
| --- | --- | --- |
| 证券主表领域模型与 SQL | 完成 | 不变量测试 + migration |
| 交易日历领域模型与 SQL | 完成 | 开闭市、前后交易日测试 |
| Raw 原始响应协议 | 完成 | 原子写入、SHA-256、不可覆盖测试 |
| 数据源 Adapter 协议 | 完成 | Runtime Protocol + 离线 Provider |
| Tushare `stock_basic` Adapter | 完成 | HTTP 信封与错误归档测试 |
| Tushare `trade_cal` Adapter | 完成 | 字段白名单与 Fixture 测试 |
| Standard 标准化 | 完成 | 沪深代码、板块、日期和日历转换测试 |
| Point-in-time 内存参考实现 | 完成 | 未来修订隔离和数据集白名单测试 |
| PostgreSQL migration 实际执行 | 待机器验收 | 当前执行环境没有 Docker |
| Tushare 真实拉取 | 待用户配置 | 需要本机设置 `TUSHARE_TOKEN` |

## 当前已知限制

- 已定义首批 PostgreSQL Schema，尚未在当前环境实际执行 migration。
- 暂按架构中的示例选择 Tushare 为首个主源；若用户使用商业数据源，应替换 Adapter，
  不改变领域和 PIT 接口。
- Compose 镜像版本是开发阶段固定值，Day 20 才做最终依赖冻结与升级审计。
- QMT / PTrade 不在当前 Linux 环境内联调。

## 下一验收点

Day 3：

1. 日线未复权行情与独立复权因子 Schema；
2. 停牌、ST 和涨跌停状态 Schema；
3. Tushare `daily` / `adj_factor` / `daily_basic` 增量 Adapter；
4. Standard Parquet 分区写入和批次血缘；
5. 首批行情结构与金融逻辑校验。
