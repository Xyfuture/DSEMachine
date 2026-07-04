# DSE Cost Parameter List

这份文档只回答一个问题：DSE 的 cost 模块需要从上游拿到哪些参数，以及最后如何算 cost。

## 1. 需要从 DSE 获取的参数

### 1.1 结构与数量参数

- `system_chip_count`
  - 单位：`chip`
  - 整个系统里有多少个 chip。

- `pim_chiplet_per_chip`
  - 单位：`PIM chiplet / chip`
  - 单个 chip 里有多少个 PIM chiplet。

- `io_chiplet_count_per_chip`
  - 单位：`IO chiplet / chip`
  - 单个 chip 里有多少个 IO chiplet。默认是 1。

- `logic_die_count_per_pim_chiplet`
  - 单位：`logic die / PIM chiplet`
  - 单个 PIM chiplet 里包含多少颗 logic die。

- `dram_die_count_per_pim_chiplet`
  - 单位：`DRAM die / PIM chiplet`
  - 单个 PIM chiplet 对应多少颗 DRAM die。

### 1.2 工艺与封装参数

- `logic_process_node`
  - 单位：`process option`
  - logic die 使用的工艺节点，用来查 logic 的 `cost_per_mm2`，也用来决定 logic penalty 的参考系数。
  - CATCH 已提供的典型选项：
    - `40nm` -> `combined_40nm`
    - `45nm` -> `combined_45nm`
    - `12nm` -> `combined_12nm`
    - `10nm` -> `combined_10nm`
    - `7nm` -> `combined_7nm`
    - `5nm` -> `combined_5nm`
    - `3nm` -> `combined_3nm`
    - `3nm_0.001` -> `combined_3nm_0.001`
    - `3nm_0.005` -> `combined_3nm_0.005`
    - `3nm_0.01` -> `combined_3nm_0.01`
  - 如果要使用 CATCH 里没有的 logic 工艺，只能通过外部数据或者你们自己的拟合结果来补 `cost_per_mm2` 和相应 penalty 参考。

- `io_process_node`
  - 单位：`process option`
  - IO die 使用的工艺节点，用来查 IO 的 `cost_per_mm2`。
  - CATCH 没有单独的 IO 专用工艺表，通常直接复用 logic 的 `combined_*` 选项作为近似参考：
    - `40nm` -> `combined_40nm`
    - `45nm` -> `combined_45nm`
    - `12nm` -> `combined_12nm`
    - `10nm` -> `combined_10nm`
    - `7nm` -> `combined_7nm`
    - `5nm` -> `combined_5nm`
    - `3nm` -> `combined_3nm`
  - 如果要使用 CATCH 里没有的 IO 工艺，只能通过外部数据或者你们自己的拟合结果来补 `cost_per_mm2`。

- `dram_process_node`
  - 单位：`memory-process option`
  - DRAM die 使用的参考工艺 / memory layer 选项，用来查 DRAM 的 `cost_per_mm2`。在当前 CATCH 中，它不是完整的 DRAM 工艺表，而更像少量 memory-layer 参考，例如 `combined_hbm2_12nm`、`combined_hbm_7nm`。
  - CATCH 已提供的典型选项：
    - `hbm2_12nm` -> `combined_hbm2_12nm`
    - `hbm_7nm` -> `combined_hbm_7nm`
  - CATCH 没有完整的 DRAM 工艺节点表；如果要使用其他 DRAM / HBM 选项，只能通过外部数据或者你们自己的拟合结果来补 `cost_per_mm2`。

- `package_integration_type`
  - 单位：`package option`
  - 封装形式，例如 organic substrate、silicon interposer、glass interposer，用来查 package 的 `cost_per_mm2`。
  - CATCH 已提供的典型选项：
    - `organic_substrate` -> `organic_substrate`
    - `organic_interposer` -> `combined_interposer_organic`
    - `silicon_interposer` -> `combined_interposer_silicon`
    - `glass_interposer` -> `combined_interposer_glass`
  - 如果要使用 CATCH 里没有的封装类型，只能通过外部数据或者你们自己的拟合结果来补 `cost_per_mm2`。

### 1.3 面积参数

- `estimated_pim_logic_area_per_die`
  - 单位：`mm^2 / logic die`
  - 单颗 logic die 的面积。

- `estimated_io_chiplet_area_per_die`
  - 单位：`mm^2 / IO die`
  - 单颗 IO die 的面积。

- `estimated_dram_area_per_die`
  - 单位：`mm^2 / DRAM die`
  - 单颗 DRAM die 的面积。

- `estimated_package_area_per_chip`
  - 单位：`mm^2 / chip`
  - 单个 chip 对应的 package / interposer / substrate 有效面积。这里可以直接理解为一个 chip 中所有 PIM chiplet 和 IO die 需要承载的总面积，再加上必要的间距与布线余量。

## 2. cost 计算方法

下面这些公式里除了 DSE 直接提供的参数外，还会用到 cost 模块内部维护的映射或系数：

- `logic_process_cost_per_mm2`：由 `logic_process_node` 映射得到
- `io_process_cost_per_mm2`：由 `io_process_node` 映射得到
- `dram_cost_per_mm2`：由 `dram_process_node` 映射得到
- `package_cost_per_mm2`：由 `package_integration_type` 映射得到
- `logic_yield_coefficient_by_process`：由 `logic_process_node` 对应的 `defect_density` 派生得到
- `logic_penalty_coefficient`：cost 模块内部配置系数
- `bonding_cost_factor`：cost 模块内部配置系数
- `stacking_penalty_coefficient`：cost 模块内部配置系数
- `total_bonded_die_pairs = system_chip_count * pim_chiplet_per_chip * logic_die_count_per_pim_chiplet`
- `total_tsv_related_stacks = system_chip_count * pim_chiplet_per_chip`

### 2.1 制造成本

```text
PIM Logic Manufacturing Cost
= system_chip_count
* pim_chiplet_per_chip
* logic_die_count_per_pim_chiplet
* estimated_pim_logic_area_per_die
* logic_process_cost_per_mm2

IO Die Manufacturing Cost
= system_chip_count
* io_chiplet_count_per_chip
* estimated_io_chiplet_area_per_die
* io_process_cost_per_mm2

DRAM Manufacturing Cost
= system_chip_count
* pim_chiplet_per_chip
* dram_die_count_per_pim_chiplet
* estimated_dram_area_per_die
* dram_cost_per_mm2

Manufacturing Cost
= PIM Logic Manufacturing Cost
+ IO Die Manufacturing Cost
+ DRAM Manufacturing Cost
```

### 2.2 封装与 bonding 成本

```text
package_area_cost
= system_chip_count
* estimated_package_area_per_chip
* package_cost_per_mm2

bonding_overhead_cost
= bonding_cost_factor
* total_bonded_die_pairs
```

### 2.3 penalty

```text
logic_yield_penalty
= 1
+ logic_penalty_coefficient
* system_chip_count
* pim_chiplet_per_chip
* logic_die_count_per_pim_chiplet
* estimated_pim_logic_area_per_die
* logic_yield_coefficient_by_process

stacking_penalty
= 1
+ stacking_penalty_coefficient
* total_tsv_related_stacks
```

说明：

- 当 penalty = 1 时，表示这一项不额外抬高成本。
- 当 penalty > 1 时，表示把前面的基础成本按比例放大。

### 2.4 最终 cost

```text
Effective Cost
= (Manufacturing Cost
 + package_area_cost
 + bonding_overhead_cost)
* logic_yield_penalty
* stacking_penalty
```

## 3. 备注

- D2D / C2C 不单独建立成本项。
  - 它们对成本的影响主要通过增大 `estimated_io_chiplet_area_per_die` 体现。

- TSV 不单独建立直接成本项。
  - 它主要通过 `estimated_dram_area_per_die`、`estimated_package_area_per_chip` 和 `stacking_penalty` 间接影响成本。

- `logic_yield_coefficient_by_process` 建议参考 CATCH 的 `defect_density` 做相对映射，但不要把它理解为真实良率。

- `total_bonded_die_pairs` 不需要 DSE 单独输入。
  - 当前可直接取
    `system_chip_count * pim_chiplet_per_chip * logic_die_count_per_pim_chiplet`
  - 在你们当前口径下，它也等于系统中的 DRAM die 总数。

- `total_tsv_related_stacks` 不需要 DSE 单独输入。
  - 当前可直接取
    `system_chip_count * pim_chiplet_per_chip`
