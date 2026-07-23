# Tushare 日终增量更新

## 运行时刻

v0.2.1 只保留两个 Windows 计划任务，时间均按 Windows 当前本地时区解释：

| 任务 | 时间 | 数据窗口 |
| --- | --- | --- |
| `AQuant-Daily-Close` | 19:30 | 当日收盘行情、指数、财务与公告增量 |
| `AQuant-Daily-Morning-Recheck` | 08:30 | 最近三个已收盘交易日重新请求 |

原计划的 17:20 快照已合并到 19:30。资金流等接口晚于基础行情到达，因此 19:30 是唯一
正式收盘后版本。次晨任务使用独立检查点，会重新请求最近三日而不是复用前一晚结果。

## 数据范围与门禁

每次运行先读取上交所交易日历。非交易日的 `close` 模式跳过行情，但仍检查当天公告；
`morning` 模式始终选择当前日期之前最近三个开市日。

核心接口失败、预期非空却返回空集、返回了错误交易日、主键重复、OHLCV 不合法或
Parquet 标准化失败时，运行状态为 `QUARANTINED`，不会发布可供策略使用的数据版本。
核心范围包括：

- `daily`、`adj_factor`、`daily_basic`、`suspend_d`、`stk_limit`；
- 七个基准的 `index_daily`；
- `income`、`balancesheet`、`cashflow`、`fina_indicator`。

`limit_list_d`、`moneyflow`、指数每日指标、预告、快报、披露日、分红和解禁属于扩展
接口。权限不足或单次失败会形成质量警告并写入报告，不会被静默丢弃。依赖这些扩展字段
的下游策略仍应检查对应警告后再运行。

原始响应继续追加到 `data/raw/provider=tushare/`，每个批次保存响应完成时间、请求参数、
行数与 SHA-256。日线写入 `data/standard/` 的不可变 Parquet，标准批次绑定全部 Raw
batch ID。成功后发布 `data/releases/cn_equity_YYYYMMDD_NNN/`；失败版本只进入
`data/quarantine/`。

财务与公司行为在 v0.2.1 中完成 Raw 归档和可用时间留痕，尚未写入专用财务 PIT
Parquet。研究读取这些字段前仍须经过既有 PIT Repository，不能直接扫描 Raw 文件。

## 手工烟雾测试

确认 `.env` 已有 `TUSHARE_TOKEN`，并已安装 data extra：

```bash
uv sync --group dev --extra data
make tushare-daily-close
make tushare-daily-status
```

测试次晨窗口时可以显式指定日期：

```bash
uv run --extra data python -u scripts/tushare_daily_update.py \
  --mode morning \
  --as-of-date 20260724 \
  --workers 4 \
  --interval 0.25
```

相同日期、相同模式重复启动会从独立 SQLite 检查点续传。成功运行再次触发时直接返回原
发布版本；只有人工核实确需再次抓取时才使用 `--force-refresh`。

## 安装 Windows 任务

先在 Windows PowerShell 检查本地时区和 WSL 发行版：

```powershell
Get-TimeZone
wsl.exe --list --quiet
```

北京时间机器应显示 China Standard Time。然后在仓库根目录执行：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\install_daily_update_tasks.ps1 `
  -Distro Ubuntu `
  -RepoPath /home/gsx1339/aquant
```

安装脚本会：

- 创建 19:30 与 08:30 两个任务；
- 允许唤醒电脑并在错过计划后尽快运行；
- 失败后每 15 分钟重试，最多三次；
- 已有实例运行时拒绝启动第二个实例；
- 固定使用 4 个线程、全局 0.25 秒最短请求间隔。

任务使用当前 Windows 登录用户运行，不要求打开 WSL 终端，但用户需要保持已登录状态。
安装后立即做一次人工触发：

```powershell
Start-ScheduledTask -TaskName AQuant-Daily-Close
Get-ScheduledTaskInfo -TaskName AQuant-Daily-Close
```

WSL 中查看日志和状态：

```bash
tail -f artifacts/daily-update/logs/scheduler-close.log
make tushare-daily-status
```

卸载两个任务：

```powershell
powershell.exe -ExecutionPolicy Bypass -File .\scripts\install_daily_update_tasks.ps1 -Remove
```

## 状态与恢复

每次运行目录为：

```text
artifacts/daily-update/runs/YYYYMMDD-close/
artifacts/daily-update/runs/YYYYMMDD-morning/
```

其中包含 `state.sqlite3`、`batch-index.json`、`quality-report.json` 和 `report.json`。
`latest-close.json` 与 `latest-morning.json` 指向最近一次状态摘要。成功日志必须出现：

```text
"event": "daily_update_complete"
"status": "COMPLETED"
"release_id": "cn_equity_..."
```

`FAILED_RESUMABLE` 可以原命令重跑；`QUARANTINED` 应先查看质量报告，修复根因后再用
`--force-refresh`。不要删除 Raw 批次或直接篡改检查点。
