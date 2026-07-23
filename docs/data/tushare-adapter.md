# Tushare 数据 Adapter

状态：首个主数据源实现，等待用户 Token 和本机网络联调。

## 协议

Adapter 按 Tushare 官方 HTTP 协议发送 JSON POST：`api_name`、`token`、`params` 和
逗号分隔的 `fields`。响应中的 `code`、`msg`、`data.fields`、`data.items` 在 Raw
归档后解析。参考：

- [Tushare HTTP 调用协议](https://tushare.pro/document/1?doc_id=130)
- [股票基础列表 stock_basic](https://tushare.pro/document/2?doc_id=25)
- [Pro 数据与交易日历示例](https://tushare.pro/document/1?doc_id=40)

## 当前白名单

| 内部数据集 | Tushare 接口 | 标准化结果 |
| --- | --- | --- |
| `instruments` | `stock_basic` | `Instrument` |
| `trading_calendar` | `trade_cal` | `TradingSession` |

任意其他 `api_name` 会被 Adapter 拒绝，新增接口必须先定义字段白名单、Raw Fixture、
标准化规则和异常测试。

## 安全与血缘

- Token 只从运行环境读取，不写入 YAML、日志、Manifest 或测试 Fixture。
- 原始响应无论业务 `code` 是否成功，都应先归档再校验。
- Raw Manifest 保存 HTTP 状态、请求 ID、请求参数、响应时刻、字节数和 SHA-256。
- 标准化代码支持上交所 `XSHG`、深交所 `XSHE` 与北交所 `XBSE`。
- `stock_basic` 需要分别拉取上市、退市和暂停状态，不能只依赖默认 `L`。

## 待本机联调

在 `.env` 设置 `TUSHARE_TOKEN` 后执行后续 Day 2 拉取命令。不要把 Token 粘贴到聊天、
提交到 Git，或写进 `config/data.yaml`。
