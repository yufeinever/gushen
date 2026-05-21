# 开源底座选型

日期：2026-05-21

## 结论

第一阶段建议复刻 Microsoft Qlib 的研究流水线，而不是直接复刻完整交易平台。

我们的项目定位是 A 股高流动性股票池、AI 辅助研究、回测验证、人工确认执行。Qlib 的核心价值在于把数据、特征、模型、回测、实验记录串成一条 AI 量化投研流水线，和我们已有的 AI 资源最匹配。

执行层暂不复刻。等模拟盘稳定后，再借鉴 vn.py 的交易接口、仿真账户、风控模块和事件引擎思想。

## 推荐路线

### 主底座：Qlib

适合复刻的部分：

- 数据层：日历、股票池、特征存储。
- 特征表达式：用统一表达式生成 Alpha/因子。
- 模型训练：LightGBM、线性模型、神经网络等统一接口。
- 回测和组合管理：从预测分数到持仓再到绩效分析。
- 实验管理：每次研究保留配置、参数、指标和产物。

不直接照搬的部分：

- 默认数据格式需要适配 A 股成交额前 100 的动态股票池。
- 默认策略示例更偏模型研究，需要补充 A 股涨跌停、T+1、停牌、ST 等交易约束。
- 第一阶段不启用在线交易和自动执行。

### 执行参考：vn.py

适合后续借鉴的部分：

- 事件驱动架构。
- gateway 接口设计。
- paper account 仿真账户。
- risk manager 风控模块。
- data recorder 数据记录模块。

暂不作为第一阶段主底座的原因：

- vn.py 是成熟交易平台，范围很大。
- 它更适合“已经准备好接行情和交易接口”的阶段。
- 我们当前最缺的是研究数据、策略验证和 AI 研发闭环。

### 数据入口：AKShare

AKShare 适合作为第一版公开数据入口之一，用于快速获取行情、基础信息和部分财经数据。它是数据接口库，不是完整策略框架，所以不能单独作为项目底座。

### A 股研究参考：Hikyuu

Hikyuu 是 A 股友好的量化研究框架，性能强，策略部件化思想值得参考。它包含 C++/Python 技术栈，第一阶段直接复刻会增加工程复杂度。建议先阅读并吸收设计，不作为主工程模板。

## 候选项目对比

| 项目 | 更适合做什么 | 优点 | 风险 |
| --- | --- | --- | --- |
| Qlib | AI 投研、因子、模型、回测 | AI 量化流水线完整，MIT，活跃，适合我们已有 AI 资源 | A 股交易细节需要自行补强 |
| vn.py | 实盘交易平台、接口、风控 | 生态成熟，A 股/国内接口丰富，MIT，活跃 | 第一阶段过重，容易先陷入交易接口 |
| Hikyuu | A 股研究和高速回测 | A 股味道强，性能好，Apache-2.0，活跃 | C++/Python 混合，复刻成本更高 |
| RQAlpha | 事件驱动回测 | 思路清晰，适合学习回测框架 | 数据和生态依赖需要额外评估 |
| AKShare | 数据获取 | 简洁、活跃、MIT，适合作为数据入口 | 不是交易/回测框架 |
| qka / OSkhQuant | 个人 A 股/QMT 方向参考 | 更贴近个人投资者实盘场景 | 生态和可维护性弱于头部项目 |
| backtrader | 通用回测 | 经典成熟 | GPL-3.0 许可证不适合直接融合进本项目主代码 |

## 我们要复刻的最小闭环

先做一个 “Qlib-like Mini Research Loop”：

1. `data`：用 AKShare 或其他数据源拉取 A 股日线行情。
2. `universe`：每日生成成交额前 100 股票池。
3. `features`：计算趋势、量价、波动、流动性、涨跌停距离。
4. `labels`：生成未来 1/3/5 日收益和风险标签。
5. `model`：训练第一版 LightGBM 或线性模型。
6. `backtest`：按每日 Top-K 候选做回测，显式处理费用、滑点、T+1、涨跌停。
7. `reports`：输出收益、回撤、换手、命中率、最大连续亏损、候选清单解释。
8. `ai_review`：AI 只读结构化报告，生成观察和复盘，不直接下单。

## 第一阶段目录建议

```text
gushen/
  data/
    raw/
    processed/
  configs/
  scripts/
  src/gushen/
    data/
    universe/
    features/
    labels/
    models/
    backtest/
    reports/
    ai/
    risk/
  notebooks/
  reports/
  docs/
```

## 决策

- 不 fork Qlib。
- 不把 Qlib 源码复制进本项目。
- 先复刻 Qlib 的投研流水线结构和关键概念。
- 第一版代码保持轻量，等最小闭环跑通后，再决定是否把 Qlib 作为依赖深度集成。
- vn.py 只进入执行层设计文档，不进入第一阶段代码。

## 参考项目

- Qlib：https://github.com/microsoft/qlib
- Qlib 文档：https://qlib.readthedocs.io/en/latest/
- vn.py：https://github.com/vnpy/vnpy
- Hikyuu：https://github.com/fasiondog/hikyuu
- RQAlpha：https://github.com/ricequant/rqalpha
- AKShare：https://github.com/akfamily/akshare
