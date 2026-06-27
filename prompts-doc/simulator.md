# Tree Mapping Encoding 与事件驱动 Matrix Performance Simulator

## 目标和范围

第一版实现 Mapping 表达和给定 MappingObj 的性能仿真，不实现搜索。

范围固定为：

- 单 chip；
- 单 MatrixOp：`A[M, K] * W[K, N] = O[M, N]`；
- chip 内多个 PIM chiplet；
- chiplet 间只切 `K/N`；
- 跨 `K` 切分产生的 partial sum 在 IO chiplet 上 reduction；
- 输出单个矩阵任务完成时间，单位为 `cycle`。

不包含：

- C2C；
- 完整 LLM layer / decode step；
- tokens/s 推导；
- power / area / cost；
- mapping search。

## 模块边界

```text
dsemachine/encoding/hardware.py
  硬件参数表达

dsemachine/encoding/matrix_mapping.py
  tree mapping 表达和展开

dsemachine/perfsim/resources.py
  硬件资源抽象

dsemachine/perfsim/instructions.py
  指令抽象

dsemachine/perfsim/engine.py
  事件/拓扑执行引擎

dsemachine/perfsim/perf_core.py
  simulate_matrix 入口
```

## Mapping Encoding

核心对象：

```text
Dataflow:
  OS / WS / IS

MatrixShape:
  m, k, n
  input_bytes
  weight_bytes
  accumulator_bytes

Rect:
  rect_id
  k_size
  n_size

Tile:
  rect_id
  tile_id
  coordinate
  k_offset
  n_offset
  k_size
  n_size

TileOrdering:
  num_k_tiles
  num_n_tiles
  group_k
  group_n
  coord_to_id()
  id_to_coord()

MappingSplitNode:
  parent
  rect
  tile_k
  tile_n
  ordering
  children
  tile_ids_from_parent

MappingLeafNode:
  parent
  chiplet_id
  tile_ids_from_parent
```

`TileOrdering` 通过函数计算 `(k_tile_id, n_tile_id) <-> linear_tile_id`，不保存 dict。这样 encoding 只描述规则，不枚举完整映射表。

`Rect` 是 `MappingSplitNode` 持有的待切分 K/N shape，不记录 start。`rect_id` 在创建 `Rect` 时自动自增生成。`Tile` 是该 split node 从自己的 `Rect` 切出来的局部基本单位，不包含 `Rect`。

`Tile.rect_id` 表示该 tile 来自哪个 `Rect`。`Tile.tile_id` 是所属 rect 的局部 linear id。`Tile.coordinate` 是所属 rect tile grid 中的 `(k_id, n_id)`。

`MappingSplitNode` 表示一个 `Rect` 继续被切分。`MappingLeafNode` 表示 parent split node 下的一组 tile 被分配给某个 PIM chiplet。

`tile_ids_from_parent` 是 child node 对 parent split node 生成的 tile id 的引用：

- 对 non-root split node：这些 parent tiles 会合并成当前 split node 的 `Rect`；
- 对 leaf node：这些 parent tiles 会展开成 simulator 可执行的 assigned tiles。

Root split node 的 `parent=None`，`tile_ids_from_parent=None`。

展开 mapping 后得到：

```text
AssignedTile:
  chiplet_id
  tile
```

`AssignedTile.tile` 是展开后给 simulator 执行的 tile。它保留原始 `rect_id/tile_id/coordinate/k_size/n_size`，其中 `k_offset/n_offset` 表示全局 K/N offset。

非法情况直接报错，包括：

- tile id 越界；
- chiplet id 越界；
- child.parent 与当前 split node 不一致；
- children 没有完整且唯一覆盖 parent 的全部 tile id；
- non-root split 的 parent tiles 不能合并成 rectangle；
- non-root split 的 `rect` 与 parent tiles 合并结果不一致；
- 非正 tile size 或硬件参数。

## Simulator Architecture

模拟器采用统一资源抽象和指令执行模型。

所有硬件资源继承同一个父类：

```text
HardwareResource:
  name
  free_time
  estimate_cycles(instruction)
  reserve(ready_time, instruction)
```

具体资源：

```text
InputD2DResource
OutputD2DResource
DRAMResource
MatrixComputeResource
IOReductionResource
```

每个 PIM chiplet 拥有：

```text
InputD2DResource
OutputD2DResource
DRAMResource
MatrixComputeResource
```

IO chiplet 拥有：

```text
IOReductionResource
```

每条指令绑定一种资源，并声明依赖：

```text
Instruction:
  id
  kind
  resource_name
  deps
  payload
```

执行引擎接收 instruction DAG，检查无环后按拓扑序执行。每条指令的开始时间为：

```text
start_time = max(resource.free_time, max(dep.end_time))
```

每条指令占用自己的资源一段时间，时长由对应 resource 的 `estimate_cycles()` 计算。

## Tile Instruction DAG

每个 assigned tile 生成四条指令：

```text
InputD2D
DRAMLoad
Compute
OutputD2D
```

依赖关系采用 tile 级 streaming overlap：

```text
InputD2D  -> DRAMLoad
InputD2D  -> Compute
DRAMLoad  -> OutputD2D
Compute   -> OutputD2D
```

含义：

- input slice 到达后，DRAM 权重读取和 compute 可以并行发生；
- `OutputD2D` 必须等待 DRAMLoad 和 Compute 都完成；
- 第一版不拆 micro-tile。

每个 tile 的数据量：

```text
input_bytes  = M * tile_K * input_bytes
weight_bytes = tile_K * tile_N * weight_bytes
output_bytes = M * tile_N * accumulator_bytes
```

DRAM latency：

```text
aligned_weight_bytes =
  ceil_div(weight_bytes, dram_page_bytes) * dram_page_bytes

dram_cycles =
  ceil_div(aligned_weight_bytes, dram_bytes_per_cycle)
```

D2D latency：

```text
input_d2d_cycles =
  ceil_div(input_bytes, d2d_input_bytes_per_cycle)

output_d2d_cycles =
  ceil_div(output_bytes, d2d_output_bytes_per_cycle)
```

Compute latency：

```text
OS =
ceil(M / SA_M)
* ceil(tile_K / (PE_M * num_SA))
* ceil(tile_N / (PE_N * SA_N))

WS =
M
* ceil(tile_K / (PE_M * num_SA * SA_M))
* ceil(tile_N / (PE_N * SA_N))

IS =
tile_N
* ceil(M / (PE_N * SA_N))
* ceil(tile_K / (PE_M * num_SA * SA_M))
```

## Partial Sum Reduction

如果同一个 output `N` 区间由多个 `K` tile 产生，则需要 IO reduction。

模拟器根据 `AssignedTile.tile` 的 `(n_offset, n_size)` 分组：

- 同组只有一个 tile 时，不生成 reduction；
- 同组有多个 tile 时，生成一条 `IOReductionInst`；
- `IOReductionInst` 依赖该组所有 partial tile 的 `OutputD2DInst`；
- IO reduction resource 用自己的 `free_time` 串行化多个 reduction。

reduction latency：

```text
reduction_bytes =
num_partials * M * n_size * accumulator_bytes

reduction_cycles =
ceil_div(reduction_bytes, io_reduction_bytes_per_cycle)
```

最终：

```text
total_cycles = max(all_instruction_end_times, all_resource_free_times)
```

## Public API

主入口：

```python
simulate_matrix(
    shape: MatrixShape,
    hw: HardwareConfig,
    mapping: MappingSplitNode,
    dataflow: Dataflow,
) -> MatrixSimResult
```

结果对象：

```text
MatrixSimResult:
  total_cycles
  input_d2d_cycles
  output_d2d_cycles
  dram_cycles
  compute_cycles
  reduction_cycles
  instruction_trace
```

`instruction_trace` 用于 debug：

```text
inst_id, inst_type, resource_name, start, end, deps
```

## Assumptions

- 本阶段不做 mapping search。
- 本阶段不模拟 C2C。
- 本阶段不做 bank-level free_time，只做 page-size 取整。
- 本阶段不做 input cache reuse，每个 tile 独立产生 InputD2D。
- 本阶段采用 tile 级 streaming overlap，不拆 micro-tile。
- 非法输入直接 `raise ValueError`，不做自动修复或隐式兜底。
