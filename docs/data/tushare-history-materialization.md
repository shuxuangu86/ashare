# Tushare 历史 Standard/PIT 物化

历史归档完成后，Raw JSON 仍不能直接进入回测。物化器从只读 SQLite 检查点解析已经完成的
Raw 页面，选择最新且分页闭合的请求族，校验每页 SHA-256，再生成不可变 Parquet 发布。
该过程完全离线，不读取 `TUSHARE_TOKEN`，也不调用网络。

## 固定窗口

第一轮可信微盘实验使用 `2007-01-04` 至 `2026-07-17`。开始日期由 `stk_limit` 的首个
非空交易日决定；更早区间缺少历史涨跌停价格，不能与后续区间使用同一交易规则口径。

运行：

```bash
make tushare-history-materialize
```

等价的完整命令：

```bash
uv run --extra data python -u scripts/materialize_tushare_history.py \
  --state artifacts/tushare-backfill/state.sqlite3 \
  --standard-root data/standard \
  --release-id cn_equity_history_20260717_001 \
  --start-date 20070104 \
  --end-date 20260717
```

默认至少要求 20 GiB 可用空间。每250个市场页或500个股票页输出一次 JSON 进度。已完成的
数据集写入隐藏的 building 目录并可在重跑时复用；只有全部数据集完成、清单写入并同步
到磁盘后，整个目录才原子改名为正式发布目录。

## 单位和时点

- `daily.vol` 从“手”乘100转换为“股”；
- `daily.amount` 从“千元”乘1000转换为“元”；
- `daily_basic.total_mv/circ_mv` 从“万元”乘10000转换为“元”；
- 成交容量使用仅由 `T-20..T-1` 构成的平均成交量；
- 换手率波动使用收盘后已知的 `T-19..T` 总体标准差；
- `stock_basic.exchange` 从代码后缀恢复；
- `stock_basic.list_status` 从三次 `L/D/P` 请求参数恢复；
- 只有D状态且在物化窗口内存在日线的股票，才用最后交易日保守推断缺失退市日期；
- 财务字段保留原始公告日，PIT读取强制 `ann_date < signal_trade_date`。

历史名称不存在的股票被视为风险状态，不会默认成非ST。截面不足400只时选择器直接失败，
不会静默缩小股票池。

## 当前发布状态

发布状态固定为 `MATERIALIZED_NOT_BACKTEST_APPROVED`。原因是未复权价格、复权因子与
分红送转事件虽已分别保存，但公司行为尚未进入回测账本。可先运行短区间链路冒烟：

```bash
make microcap-history-smoke
```

其收益字段明确标记为 `SMOKE_ONLY_UNADJUSTED_RETURNS_NOT_RESEARCH_RESULT`，只能验证：

- Parquet可读；
- 历史名称、停牌和涨跌停状态可连接；
- 财务字段没有使用同日公告；
- T日信号与T+1成交链路能够运行。

在现金分红、送转和除权数量调整进入账本前，不得发布72组正式历史收益。
