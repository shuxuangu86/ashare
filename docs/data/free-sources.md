# 免费数据源策略

开发阶段使用 AKShare 作为主源、BaoStock 作为校验源。二者都只拉取未复权日线，复权因子
必须独立存储，避免把会随时间变化的复权序列当作不可变原始价格。

## 字段和单位约定

| 内部字段 | AKShare | BaoStock | Standard |
| --- | --- | --- | --- |
| 股票代码 | 六位代码 | `sh./sz.` 前缀 | `CODE.XSHG/XSHE` |
| OHLC | 元 | 元 | Decimal 元 |
| 成交量 | 手 | 股 | 股 |
| 成交额 | 元 | 元 | Decimal 元 |
| 复权参数 | 空字符串 | `adjustflag=3` | 未复权 |

AKShare 日线文档注明成交量单位为手，因此标准化时乘以 100；BaoStock 日线成交量按股进入
Standard。跨源比较只在完成单位统一后执行。

## 免费源限制

- 上游网页字段或访问策略可能变化；
- 历史财报修订和精确公告可用时间可能不完整；
- 历史指数成分、行业分类和证券状态覆盖可能不齐；
- 免费接口可能限流、短时不可用或返回空表；
- 数据许可必须在后续分发或商业使用前单独核对。

因此免费数据可用于工程开发、粗筛和发现明显无效策略，但任何“有收益”结论都必须经过
PIT 检查、跨源核对、交易成本和严格样本外测试；在投入资金前再用更可靠的数据复验。

## 官方接口参考

- [AKShare A 股历史行情](https://akshare.akfamily.xyz/data/stock/stock.html)
- [BaoStock Python API](https://www.baostock.com/mainContent?file=pythonAPI.md)
