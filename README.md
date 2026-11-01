# factor-research

Binance USDⓈ-M Futures 的订单簿与成交流高频因子研究流水线，以及基于此的多horizon LightGBM 预测与定价模型。

- **Rust**：回放历史bbo, depth, trade数据，逐事件驱动计算生成因子，并将样本写入 Parquet。
- **Python**：将Rust生成的样本按时间切分成train/validate/test三个集合，为每个horizon训练一个 LightGBM 回归模型，并与单因子OBI baseline进行对比。
- **Rust**：加载训练好的模型，复用生成训练样本的因子计算代码，在每个行情事件上进行实时推理预测。


## 核心结果

BTCUSDT Rank IC

| Horizon | Multi-factor LightGBM | OBI linear | OBI LightGBM |
|---|---:|---:|---:|
| 1ms | **.5906** | .4980 | .4951 |
| 2ms | **.6023** | .5111 | .5086 |
| 5ms | **.6275** | .5347 | .5325 |
| 25ms | **.6751** | .5893 | .5878 |
| 50ms | **.6900** | .6122 | .6111 |
| 100ms | **.7048** | .6385 | .6378 |
| 200ms | **.6977** | .6433 | .6426 |
| 300ms | **.6842** | .6377 | .6371 |
| 500ms | **.6547** | .6202 | .6196 |
| 750ms | **.6167** | .5930 | .5925 |
| 1s | **.5878** | .5678 | .5673 |
| 2s | **.4956** | .4832 | .4827 |
| 5s | **.3634** | .3563 | .3558 |

APTUSDT Rank IC

| Horizon | Multi-factor LightGBM | OBI linear | OBI LightGBM |
|---|---:|---:|---:|
| 1ms | **.5548** | .2321 | .2039 |
| 2ms | **.5610** | .2311 | .2025 |
| 5ms | **.5717** | .2333 | .2047 |
| 25ms | **.5560** | .2449 | .2127 |
| 50ms | **.5327** | .2528 | .2219 |
| 100ms | **.5187** | .2666 | .2399 |
| 200ms | **.4814** | .2708 | .2511 |
| 300ms | **.4548** | .2710 | .2532 |
| 500ms | **.4162** | .2655 | .2494 |
| 750ms | **.3848** | .2565 | .2416 |
| 1s | **.3604** | .2449 | .2301 |
| 2s | **.2982** | .2076 | .1988 |
| 5s | **.1954** | .1464 | .1397 |

*OBI: 单因子 L1 订单簿失衡  
*数据时间范围：2026-10-06 - 2026-10-16


## 因子体系 (152 个因子)

所有因子均仅使用截至采样时刻已收到的信息，时间窗口覆盖 30 µs 到 3 s。

| 因子族 (Family) | 数量 | 定义 |
|---|---:|---|
| `obi_*` | 7 | 订单簿失衡 `(bid − ask) / (bid + ask)`。`obi_0bps` 仅使用 BBO；其余因子按 `0.5^(distance / h)` 对各档位加权。 |
| `ofi_*` | 18 | 最优买卖档（BBO）的订单流失衡（Order Flow Imbalance），在时间窗口内累加后，除以过去 60 s 的时间加权平均 L1 深度。 |
| `mlofi_*` | 72 | 多档位 OFI（Multi-level OFI）。在整个订单簿上逐价格档位对比订单流变化，按距 mid 的距离进行指数衰减加权，并除以过去 60 s 的时间加权衰减深度。 |
| `tfi_*` | 18 | 窗口内主动买单减去主动卖单的成交额（Trade Flow Imbalance），除以过去 1 h 成交额。 |
| `trade_notional_ratio_*` | 18 | 窗口内总成交额相对于过去 1 h 成交额的比值。 |
| `mid_ret_*` | 18 | 窗口内的 mid price 收益率。 |
| `l1_depth_ratio` | 1 | 当前 L1 深度相对于过去 60 s 时间加权平均深度的比值。 |


## 标签与采样 (Labels & Sampling)

- **Labels：** `fmid_h = mid(exchange transaction time now + h) / mid(local now) − 1`，覆盖 0 到 60 s 共 19 个预测周期（horizons）。
- **Sampling：** 经过 1 h 的warmup后，每 100 个行情事件（BBO、trade 和 depth）采样一次。


## 目录结构 (Layout)

```
src/strategy/
  factors_generator.rs            样本生成策略
  factors_generator/
    bbo_factors.rs                OFI, mid return, L1 depth ratio
    depth_factors.rs              OBI, multi-level OFI
    trade_factors.rs              TFI, trade notional ratio
    sample_writer.rs              样本writer
  pricing_model_factors.rs        基于训练好的模型的多horizon定价策略
  pricing_model_factors/
    pricing_model.rs              加载 LightGBM 模型，运行推理
    factors_generator.rs          生成与训练阶段相同的因子
train/
  train.py                        按时间切分数据，训练 LightGBM
  baseline.py                     与 OBI baseline进行对比评估
```
