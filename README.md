# AQuant

AQuant 是一套面向沪深 A 股日频/低频研究的本地量化系统。核心设计目标是：

- 所有研究都绑定不可变的数据发布版本；
- 所有可修订数据都以 `available_at` 做 Point-in-time 查询；
- 量化雷达与证据驱动的主观基本面策略构成两条并列的核心研究路径；
- 策略只生成目标仓位，研究进程不直接连接券商；
- 订单意图具有稳定幂等键；
- 实盘默认关闭，数据或状态异常时停止交易；
- 相同代码、配置、数据和随机种子产生相同结果。

当前状态：Day 1 至 Day 20 的本地 MVP、v0.2 主观基本面核心策略、v0.2.1 日终增量更新
与 v0.3.0a3 历史 Standard/PIT 物化代码、离线自动验收均已完成。系统包含可信数据、
PIT 查询、因子研究、主观证据研究、
双路径回测、机器学习、组合约束、模拟券商、QMT 适配边界、执行门控、核对、Kill Switch、
API、界面原型和运维手册。
开发阶段默认使用 AKShare 主源和 BaoStock 校验源；实盘始终默认关闭。真实数据、Docker
服务和 Windows QMT 联调仍需在目标电脑验收。v0.1 基线详见
`docs/acceptance/final-report.md`，v0.2 验收见
`docs/acceptance/v0.2-discretionary-report.md`，v0.2.1 日更验收见
`docs/acceptance/v0.2.1-daily-update-report.md`。

## 环境要求

- Windows 11 + WSL2；
- Docker Desktop（启用 WSL2 后端）；
- Python 3.11；
- `uv`；
- Git。

QMT / XtQuant 或 PTrade 只在后续 Windows 原生执行代理中安装。不要把券商 SDK
装入研究服务的 Python 环境。

## 快速开始

```bash
cd aquant
cp .env.example .env
uv sync --group dev \
  --extra data --extra research --extra free-data \
  --extra optimization --extra services --extra ui
uv run aquant config-check \
  --config config/base.yaml \
  --config config/data.yaml \
  --config config/backtest.yaml \
  --config config/risk.yaml \
  --config config/strategies/discretionary.yaml \
  --config config/brokers/paper.yaml
docker compose config --quiet
docker compose up -d --build
uv run pytest
```

本地服务仅绑定 `127.0.0.1`：

| 服务 | 地址 | 角色 |
| --- | --- | --- |
| PostgreSQL + pgvector | `127.0.0.1:5432` | 元数据、交易、审计、知识库 |
| Redis | `127.0.0.1:6379` | 缓存与短期事件；不是账务真相源 |
| Prefect | `http://127.0.0.1:4200` | 工作流调度 |
| MLflow | `http://127.0.0.1:5000` | 实验、指标和模型登记 |

首次启动前必须修改 `.env` 中的 PostgreSQL 密码。`.env` 已被 Git 忽略。

## 常用命令

```bash
make bootstrap      # 安装开发环境并安装 pre-commit hooks
make config-check   # 校验基础配置并输出脱敏摘要
make infra-config   # 校验 Compose 配置
make infra-up       # 构建并启动基础服务
make quality        # Ruff + mypy + pytest + 覆盖率门槛
make tushare-backfill # 断点续传归档 Tushare 全历史日频研究数据
make tushare-status   # 查看 Tushare 归档检查点
make tushare-daily-close   # 19:30 收盘后完整增量更新
make tushare-daily-morning # 08:30 最近三个交易日补漏
make tushare-daily-status  # 查看两档日更的最近状态
make tushare-history-materialize # 离线生成不可变历史 Standard/PIT 发布
make microcap-history-smoke      # 短区间数据/PIT/执行链路冒烟
```

Windows 自动任务安装与失败恢复见
`docs/runbooks/tushare-daily-update.md`。默认只创建 19:30 与次日 08:30 两个任务；
不再保留 17:20 的独立任务。
历史物化的单位、PIT和公司行为边界见
`docs/data/tushare-history-materialization.md`。

其他入口：

```bash
uv run uvicorn aquant.api:create_app --factory --host 127.0.0.1 --port 8000
uv run streamlit run apps/research_ui/app.py
uv run streamlit run apps/operations_ui/app.py
uv run python apps/execution_agent_windows/main.py
```

Windows 执行代理入口只验证安全配置，不会自行解锁 LIVE，也不会在未注入实际 QMT Gateway
时提交订单。

## 主观基本面核心策略

`discretionary-fundamental-v1` 将“价格发现—竞争假设—产业因果链—PIT 证据—市场预期差—
分级仓位—人工审批—后验校准”纳入与量化策略相同的版本、审计和执行边界。

- 价格只触发研究，单独不能形成仓位；
- 每个正式研究案维护基本面、资金/技术和噪声三类解释；
- 非标数据按依赖簇去重，并用后来经营结果校准未来权重；
- 实时判断与事后平滑版本隔离，回顾性结果不能进入仓位；
- 默认观察仓、证据仓、核心仓上限为 0.5%、2%、5%；
- 加仓必须来自经营证据升级，所有仓位建议必须人工批准。

完整方法论见 `docs/strategies/discretionary-fundamental.md`，例行操作见
`docs/runbooks/discretionary-research-sop.md`，研究案模板位于
`research/templates/discretionary-case.yaml`。

## 安全边界

- `config/base.yaml` 固定为 `BACKTEST`，且 `live_trading.enabled=false`。
- MVP 代码拒绝 `LIVE_FULL`。
- `LIVE` 必须同时显式启用并保留人工确认。
- QMT 和 PTrade 配置默认 `enabled=false`、`live_submission_enabled=false`。
- 配置检查输出不会显示数据库密码。

这些配置保护只是第一层。实盘前还必须通过数据库状态、当日数据发布、账户核对、
风控服务、Kill Switch、券商连接和人工审批的联合门控。

## 开发规则

- 生产逻辑只能存在于 `src/aquant`，Notebook 仅调用已测试代码。
- 新领域逻辑先声明不变量，再实现测试。
- 任何财务、成分股或公司行为查询都必须经过 PIT 接口。
- 不允许绕过 `OrderIntent` 直接提交订单。
- 数据目录中的运行数据不进入 Git；数据批次通过校验和与发布 ID 追踪。

## 验收基线

- 离线测试、覆盖率、Ruff 与 mypy 验收结果见对应版本验收报告；
- Ruff、mypy strict 和 pre-commit 必须全部通过；
- 合成市场覆盖停牌、涨跌停、费用、部分成交、幂等、乱序回调与故障闭锁；
- 真实收益、真实数据供应商稳定性和真实券商连接不属于离线测试结论。

## 许可证

私有项目，未经授权不得分发。
