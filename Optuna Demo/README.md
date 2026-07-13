# Optuna TPE 多目标优化 Demo

这个目录是一个独立的 Optuna（tpe based multi-objective Bayesian optimizer） demo，用来测试未来 DSE 外层硬件搜索流程中可能采用的 `TPESampler` 多目标优化能力。它不会调用真实的硬件建模、mapping search 或性能模拟器，只用 mock 公式模拟：

```text
采样硬件参数 -> mock mapping search -> mock PPAC 评估 -> 反馈给 Optuna
```

## 环境

需要安装

```bash
python -m pip install optuna
```


## Demo 内容

脚本使用 `optuna.samplers.TPESampler` 创建多目标 study，目标包括：

- 最大化 `tokens_per_dollar`
- 最小化 `tpot_ms`
- 最小化 `energy_per_token`

同时使用 constraints 模拟 DSE 中常见的限制：

- 功耗上限
- 面积上限
- IO SRAM 是否足够
- TPOT 是否满足目标

## 输出文件

运行后会自动创建 `results/` 目录，并输出：

- `results/trials.csv`：所有 completed trials 的参数和 mock PPAC 结果
- `results/pareto_front.csv`：Optuna 找到的 Pareto front trials

这些结果文件只是 demo 运行产物，可以删除后重新生成。

## 注意

这只是个 demo，用于初步验证 optuna 的可行性
