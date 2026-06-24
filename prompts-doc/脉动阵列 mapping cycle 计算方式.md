# 脉动阵列 Mapping Cycle 计算方式

本文档描述在理想条件下，Output Stationary、Weight Stationary、Input Stationary 三种脉动阵列 mapping 策略的 compute cycle 计算方式。

这里的理想条件是：

- 不考虑 systolic array 的启动延迟和退出延迟；
- buffer 带宽资源充足；
- DRAM 带宽资源充足；
- 不考虑 D2D / C2C 通信；
- 不考虑 bank conflict、page miss、partial sum reduction 的额外 latency / bandwidth 开销、vector op 和 softmax；
- 只计算完美流水下的 compute cycle。

这里的“不考虑 partial sum reduction”不是指不进行矩阵乘内部的累加。矩阵乘在 `K` 维上的 MAC 累加仍然包含在 compute cycle 中。这里忽略的是当 partial sum 需要跨 PE、跨 systolic array、跨 chiplet 或回到 IO chiplet 时，额外产生的传输、同步和 reduction 开销。

## 基础符号

一个 PIM chiplet 内部有：

- `PE_M * PE_N` 个 PE，构成二维 PE layout；
- 每个 PE 内部有 `num_SA` 个 systolic array；
- 每个 systolic array 的 MAC 阵列大小是 `SA_M * SA_N`。

要计算的矩阵 tile 定义为：

```text
A[MatSA_M, MatSA_K] * W[MatSA_K, MatSA_N] = O[MatSA_M, MatSA_N]
```

其中：

- `MatSA_M` 是输入矩阵和输出矩阵的 M 维；
- `MatSA_K` 是 reduction 维；
- `MatSA_N` 是权重矩阵和输出矩阵的 N 维；
- `ceil_div(x, y)` 表示向上取整除法。

## Output Stationary

Output Stationary 的核心是让输出 `O[M, N]` stationary。每个 MAC 阵列在本地累加 partial sum，直到一个 output tile 计算完成。

### 空间并行方式

在本文档的抽象下：

- `M` 维映射到单个 systolic array 的 `SA_M` 维；
- `N` 维映射到 `PE_N * SA_N`；
- `K` 维映射到 `PE_M * num_SA`。

因此，一个完美流水 step 中可以覆盖的计算并行度是：

```text
M_parallel = SA_M
K_parallel = PE_M * num_SA
N_parallel = PE_N * SA_N
```

### Cycle 公式

对于矩阵 tile：

```text
A[MatSA_M, MatSA_K] * W[MatSA_K, MatSA_N]
```

Output Stationary 的理想 compute cycle 是：

```text
cycle_OS =
    ceil_div(MatSA_M, SA_M)
  * ceil_div(MatSA_K, PE_M * num_SA)
  * ceil_div(MatSA_N, PE_N * SA_N)
```

这里不额外增加 systolic fill / drain cycle，因为假设没有启动和退出延迟。

### Weight 最小映射粒度

在 Output Stationary 下，weight 被按照 `K` 和 `N` 维切分到 PE local banks 中。

如果要求每个 PE 在 `K` 方向至少能够形成一次有效的 DRAM page 访问，则 chiplet 上 weight 的最小映射粒度可以写为：

```text
K_min = PE_M * (Bank_Page_Size * num_mc / SA_N)
N_min = PE_N * SA_N
```

即：

```text
W_min_OS = [PE_M * (Bank_Page_Size * num_mc / SA_N), PE_N * SA_N]
```

## Weight Stationary

Weight Stationary 的核心是让权重 `W[K, N]` stationary。权重 tile 固定在 PE / systolic array 附近，输入 `A[M, K]` 按照 `M` 维流过阵列，并生成输出 `O[M, N]`。

### 空间并行方式

在 Weight Stationary 下，权重本身占据空间并行资源：

- `K` 维映射到 `PE_M * num_SA * SA_M`；
- `N` 维映射到 `PE_N * SA_N`；
- `M` 维不作为 stationary 维度，而是按时间流过 stationary weight。

因此，一个完美流水 step 中可以覆盖的计算并行度是：

```text
K_parallel = PE_M * num_SA * SA_M
N_parallel = PE_N * SA_N
```

`M` 维需要逐个或逐块流过。由于这里已经假设完美流水，且不考虑启动和退出延迟，可以把 `M` 维的时间开销记为 `MatSA_M`。

### Cycle 公式

对于矩阵 tile：

```text
A[MatSA_M, MatSA_K] * W[MatSA_K, MatSA_N]
```

Weight Stationary 的理想 compute cycle 是：

```text
cycle_WS =
    MatSA_M
  * ceil_div(MatSA_K, PE_M * num_SA * SA_M)
  * ceil_div(MatSA_N, PE_N * SA_N)
```

这个公式表示：每一轮 stationary weight 覆盖一个 `K_parallel * N_parallel` 的权重块，输入在 `M` 维上流过该权重块。

### Weight 最小映射粒度

Weight Stationary 下，weight 是 stationary tensor，因此最小 weight 映射粒度由空间展开的 `K` 和 `N` 维共同决定：

```text
K_min = PE_M * num_SA * SA_M
N_min = PE_N * SA_N
```

即：

```text
W_min_WS = [PE_M * num_SA * SA_M, PE_N * SA_N]
```

## Input Stationary

Input Stationary 的核心是让输入 `A[M, K]` stationary。输入 tile 固定在 PE / systolic array 附近，权重 `W[K, N]` 按照 `N` 维流过阵列，并生成输出 `O[M, N]`。

### 空间并行方式

在 Input Stationary 下，输入本身占据空间并行资源：

- `M` 维映射到 `PE_N * SA_N`；
- `K` 维映射到 `PE_M * num_SA * SA_M`；
- `N` 维不作为 stationary 维度，而是按时间流过 stationary input。

因此，一个完美流水 step 中可以覆盖的计算并行度是：

```text
M_parallel = PE_N * SA_N
K_parallel = PE_M * num_SA * SA_M
```

`N` 维需要逐个或逐块流过。由于这里假设完美流水，且不考虑启动和退出延迟，可以把 `N` 维的时间开销记为 `MatSA_N`。

### 数据流方向

按照上面的映射方式，单个 stationary input tile 的局部形状可以理解为：

```text
stationary input tile = A[M, K]
K 维沿阵列纵向展开
M 维沿阵列横向展开
```

对于某一个输出列 `n`，weight 向量 `W[:, n]` 流过 stationary input tile。每个 MAC 单元计算：

```text
A[m, k] * W[k, n]
```

同一个 `m` 上的所有 `k` 需要在纵向上完成累加：

```text
O[m, n] = sum_k A[m, k] * W[k, n]
```

因此，在这个方向定义下，partial sum 沿 `K` 维纵向传播，最终的 `O[m, n]` 从阵列下方输出。

从局部物理放置看，这等价于把 stationary input 以 `A^T[K, M]` 的方向放进阵列；但数学上仍然是在计算 `A[M, K] * W[K, N] = O[M, N]`。

### Cycle 公式

对于矩阵 tile：

```text
A[MatSA_M, MatSA_K] * W[MatSA_K, MatSA_N]
```

Input Stationary 的理想 compute cycle 是：

```text
cycle_IS =
    MatSA_N
  * ceil_div(MatSA_M, PE_N * SA_N)
  * ceil_div(MatSA_K, PE_M * num_SA * SA_M)
```

这个公式表示：每一轮 stationary input 覆盖一个 `M_parallel * K_parallel` 的输入块，权重在 `N` 维上流过该输入块。

### Weight 最小映射粒度

Input Stationary 下，weight 不是 stationary tensor，而是沿 `N` 维流过 stationary input。为了匹配 stationary input 的 `K` 维并行度，weight 的 `K` 维至少需要覆盖：

```text
K_min = PE_M * num_SA * SA_M
```

同时，由于 `N` 是流入维度，如果要求 weight 读取满足 DRAM page 访问粒度，则 `N` 维最小规模受到 page size 约束：

```text
N_min = Bank_Page_Size * num_mc / SA_N
```

因此，Input Stationary 下 chiplet 上 weight 的最小映射粒度可以写为：

```text
W_min_IS = [PE_M * num_SA * SA_M, Bank_Page_Size * num_mc / SA_N]
```

## 三种策略的对比

在只考虑理想 compute cycle 的情况下，三种策略的公式可以汇总为：

```text
cycle_OS =
    ceil_div(MatSA_M, SA_M)
  * ceil_div(MatSA_K, PE_M * num_SA)
  * ceil_div(MatSA_N, PE_N * SA_N)

cycle_WS =
    MatSA_M
  * ceil_div(MatSA_K, PE_M * num_SA * SA_M)
  * ceil_div(MatSA_N, PE_N * SA_N)

cycle_IS =
    MatSA_N
  * ceil_div(MatSA_M, PE_N * SA_N)
  * ceil_div(MatSA_K, PE_M * num_SA * SA_M)
```

直观上：

- Output Stationary 同时利用 `M`、`K`、`N` 三个方向的空间并行，但 output partial sum 需要保持在本地；
- Weight Stationary 把 `K`、`N` 维 weight 固定住，`M` 维输入流过；
- Input Stationary 把 `M`、`K` 维 input 固定住，`N` 维 weight 流过；`K` 维沿纵向累加，结果从阵列下方输出。

## 适用范围

本文档中的 cycle 只表示理想 compute cycle，不表示真实端到端 latency。

如果后续性能模型加入 memory bandwidth、interconnect bandwidth 和 IO chiplet 上的后处理开销，则完整 latency 应扩展为：

```text
latency_cycle = max(compute_cycle, memory_cycle, communication_cycle)
```

其中：

- `compute_cycle` 来自本文档中的 OS / WS / IS 公式；
- `memory_cycle` 来自 local DRAM bank / memory channel / page policy；
- `communication_cycle` 来自 D2D、C2C、IO SRAM 读写、gather、跨计算单元 partial sum reduction 等数据移动和同步开销。
