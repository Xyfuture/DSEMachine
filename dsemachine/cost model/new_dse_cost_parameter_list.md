# DSE Cost Parameter List 2

这份文档是面向 DSE cost 模块的实现版参数清单。

和 [dse_cost_parameter_list.md](/Users/waynex/terminal_files/from_github/CATCH/dse_cost_parameter_list.md) 的区别是：

- 原版默认 DSE 可以直接给出 die 面积
- 这一版默认 DSE 只能给出结构、数量、容量和工艺参数
- die 面积由 cost / area 模块根据 component 面积系数计算得到
- 当前先不考虑布线面积
- `estimated_package_area_per_chip` 不作为 DSE 原始输入，而由各 die 面积继续推导

## 1. 需要从 DSE 或 cost 模块获取的参数

### 1.1 结构与数量参数

- `system_chip_count`
  - 单位：`chip`
  - 整个系统中的 chip 数量。

- `pim_chiplet_per_chip`
  - 单位：`PIM chiplet / chip`
  - 单个 chip 中的 PIM chiplet 数量。

- `io_chiplet_count_per_chip`
  - 单位：`IO chiplet / chip`
  - 单个 chip 中的 IO chiplet 数量。

- `logic_die_count_per_pim_chiplet`
  - 单位：`logic die / PIM chiplet`
  - 单个 PIM chiplet 中的 logic die 数量。

- `dram_die_count_per_pim_chiplet`
  - 单位：`DRAM die / PIM chiplet`
  - 单个 PIM chiplet 中的 DRAM die 数量。

- `pe_layout_m`
  - 单位：`PE / dim`
  - 单个 PIM chiplet 中 PE 阵列的 M 方向规模。

- `pe_layout_n`
  - 单位：`PE / dim`
  - 单个 PIM chiplet 中 PE 阵列的 N 方向规模。

- `pe_bank_capacity`
  - 单位：`capacity / bank`
  - 单个 bank 的容量。

- `pe_bank_number`
  - 单位：`bank / PE`
  - 单个 PE 对应的 bank 数量。

- `pe_memory_controller_number`
  - 单位：`MC / PE`
  - 单个 PE 内的 memory controller 数量。

- `pe_systolic_array_m`
  - 单位：`MAC / dim`
  - 单个 systolic array 的 M 方向规模。

- `pe_systolic_array_n`
  - 单位：`MAC / dim`
  - 单个 systolic array 的 N 方向规模。

- `pe_systolic_array_number`
  - 单位：`SA / PE`
  - 单个 PE 内的 systolic array 数量。

- `d2d_serdes_count_per_pim_chiplet`
  - 单位：`serdes / PIM chiplet`
  - 单个 PIM chiplet 面向 IO chiplet 的 D2D serdes 数量。

- `c2c_serdes_count_per_chip`
  - 单位：`serdes / chip`
  - 单个 chip 的 C2C serdes 数量。

- `io_chiplet_buffer_capacity`
  - 单位：`capacity / IO chiplet`
  - 单个 IO chiplet 的 buffer 容量。

- `pe_local_buffer_capacity`
  - 单位：`capacity / PE`
  - 单个 PE 的 local buffer 容量。
  - 如果当前 DSE 没有显式暴露，可先作为固定配置。

- `global_buffer_capacity_per_chiplet`
  - 单位：`capacity / PIM chiplet`
  - 单个 PIM chiplet 的 global buffer 容量。
  - 如果当前 DSE 没有显式暴露，可先作为固定配置。

### 1.2 工艺与封装参数

- `logic_process_node`
  - 单位：`process option`
  - logic die 工艺节点。
  - CATCH 现成参考：
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

- `io_process_node`
  - 单位：`process option`
  - IO die 工艺节点。
  - CATCH 没有单独的 IO 工艺表，通常复用：
    - `40nm` -> `combined_40nm`
    - `45nm` -> `combined_45nm`
    - `12nm` -> `combined_12nm`
    - `10nm` -> `combined_10nm`
    - `7nm` -> `combined_7nm`
    - `5nm` -> `combined_5nm`
    - `3nm` -> `combined_3nm`

- `dram_process_node`
  - 单位：`memory-process option`
  - DRAM die 的参考 memory layer。
  - CATCH 现成参考：
    - `hbm2_12nm` -> `combined_hbm2_12nm`
    - `hbm_7nm` -> `combined_hbm_7nm`
  - CATCH 没有的 DRAM 工艺，需要通过外部数据或拟合补齐。

- `package_integration_type`
  - 单位：`package option`
  - 封装类型。
  - CATCH 现成参考：
    - `organic_substrate` -> `organic_substrate`
    - `organic_interposer` -> `combined_interposer_organic`
    - `silicon_interposer` -> `combined_interposer_silicon`
    - `glass_interposer` -> `combined_interposer_glass`
  - CATCH 没有的封装类型，需要通过外部数据或拟合补齐。

### 1.3 面积建模系数参数

这一节只保留面积建模绕不开、但不是 DSE 原始硬件参数的面积库参数。

#### Logic die 面积系数

- `area_per_mac`
  - 单位：`mm^2 / MAC`
  - 单个 MAC 的面积系数。

- `area_per_mc`
  - 单位：`mm^2 / MC`
  - 单个 memory controller 的面积系数。

- `area_per_d2d_serdes_on_logic_die`
  - 单位：`mm^2 / serdes`
  - logic die 上单个 D2D serdes 接口的面积系数。

- `area_per_pe_vector_unit`
  - 单位：`mm^2 / PE`
  - 单个 PE 内 vector unit 的固定面积。

- `area_per_pe_local_buffer_capacity`
  - 单位：`mm^2 / capacity`
  - PE 内 local buffer 的单位容量面积系数。

- `area_per_global_buffer_capacity`
  - 单位：`mm^2 / capacity`
  - global buffer 的单位容量面积系数。

#### DRAM die 面积系数

- `area_per_bank`
  - 单位：`mm^2 / bank`
  - 单个 DRAM bank 的面积系数。

- `area_mini_tsv_per_mc`
  - 单位：`mm^2 / MC`
  - 每个 memory controller 对应的 mini-TSV 等效面积开销。

#### IO die 面积系数

- `area_per_io_buffer_capacity`
  - 单位：`mm^2 / capacity`
  - IO buffer 的单位容量面积系数。

- `area_per_d2d_serdes`
  - 单位：`mm^2 / serdes`
  - IO die 上单个 D2D serdes 的面积系数。

- `area_per_c2c_serdes`
  - 单位：`mm^2 / serdes`
  - IO die 上单个 C2C serdes 的面积系数。

- `area_io_switch_base`
  - 单位：`mm^2`
  - IO die 上 switch 基础面积。
  - 可理解为 IO chiplet 中与数据交换、仲裁、流控、地址分发等相关的固定基础逻辑面积；这部分不随 serdes 数量线性增长。

- `area_io_control_logic_fixed`
  - 单位：`mm^2`
  - IO die 上固定 control logic 面积。

- `area_io_vector_unit_fixed`
  - 单位：`mm^2`
  - IO die 上固定 vector / control resource 面积。

## 2. 面积与 cost 计算方法

### 2.1 die 面积计算

#### Logic die

```text
pe_count
= pe_layout_m * pe_layout_n

logic_compute_area
= pe_count
 * pe_systolic_array_number
 * pe_systolic_array_m
 * pe_systolic_array_n
 * area_per_mac(logic_process_node)

logic_mc_area
= pe_count
 * pe_memory_controller_number
 * area_per_mc(logic_process_node)

logic_d2d_area
= d2d_serdes_count_per_pim_chiplet
 * area_per_d2d_serdes_on_logic_die

logic_buffer_area
= pe_count
 * pe_local_buffer_capacity
 * area_per_pe_local_buffer_capacity(logic_process_node)
 + global_buffer_capacity_per_chiplet
 * area_per_global_buffer_capacity(logic_process_node)

logic_vector_area
= pe_count * area_per_pe_vector_unit

estimated_pim_logic_area_per_die
= logic_compute_area
 + logic_mc_area
 + logic_d2d_area
 + logic_buffer_area
 + logic_vector_area
```

说明：

- `logic_d2d_area` 只表示 logic die 上面向 IO chiplet 的 D2D 接口面积，不包含 C2C。
- 当前不额外加入片上网络附加面积项。
- 当前不把 hybrid bonding 或 mini-TSV 显式并入 logic die 面积。

#### DRAM die

使用 `area_per_bank` 时：

```text
pe_count
= pe_layout_m * pe_layout_n

total_bank_count_per_pim_chiplet
= pe_count * pe_bank_number

bank_count_per_dram_die
= total_bank_count_per_pim_chiplet
 / dram_die_count_per_pim_chiplet

total_mc_per_pim_chiplet
= pe_count * pe_memory_controller_number

area_mini_tsv_overhead_per_die
= total_mc_per_pim_chiplet
 * area_mini_tsv_per_mc
 / dram_die_count_per_pim_chiplet

estimated_dram_area_per_die
= bank_count_per_dram_die * area_per_bank(pe_bank_capacity, dram_process_node)
 + area_mini_tsv_overhead_per_die
```

说明：

- mini-TSV 不在 logic die 中单列，而是通过 `area_mini_tsv_overhead_per_die` 并入 DRAM die 面积。

#### IO die

```text
d2d_serdes_per_io_die
= pim_chiplet_per_chip
 * d2d_serdes_count_per_pim_chiplet
 / io_chiplet_count_per_chip

c2c_serdes_per_io_die
= c2c_serdes_count_per_chip
 / io_chiplet_count_per_chip

io_buffer_area
= io_chiplet_buffer_capacity
 * area_per_io_buffer_capacity(io_process_node)

io_d2d_area
= d2d_serdes_per_io_die
 * area_per_d2d_serdes(io_process_node)

io_c2c_area
= c2c_serdes_per_io_die
 * area_per_c2c_serdes(io_process_node)

io_fixed_overhead_area
= area_io_switch_base
 + area_io_control_logic_fixed
 + area_io_vector_unit_fixed

estimated_io_chiplet_area_per_die
= io_buffer_area
 + io_d2d_area
 + io_c2c_area
 + io_fixed_overhead_area
```

说明：

- C2C 面积只在 IO die 侧体现。
- 当前不再加入额外的 IO 端口线性面积项。

#### Package area

```text
estimated_package_area_per_chip
```

由单个 chip 内所有 PIM chiplet 与 IO die 的 footprint 求和后，再加必要的 package margin 得到；它不是 DSE 原始输入。

### 2.2 cost 计算

下面这些公式除了 DSE 直接提供的参数外，还会用到 cost 模块内部维护的映射或系数：

- `logic_process_cost_per_mm2`
- `io_process_cost_per_mm2`
- `dram_cost_per_mm2`
- `package_cost_per_mm2`
- `logic_yield_coefficient_by_process`
- `logic_penalty_coefficient`
- `bonding_cost_factor`
- `stacking_penalty_coefficient`
- `total_bonded_die_pairs = system_chip_count * pim_chiplet_per_chip * logic_die_count_per_pim_chiplet`
- `total_tsv_related_stacks = system_chip_count * pim_chiplet_per_chip`

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

package_area_cost
= system_chip_count
 * estimated_package_area_per_chip
 * package_cost_per_mm2

bonding_overhead_cost
= bonding_cost_factor * total_bonded_die_pairs

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

Effective Cost
= (Manufacturing Cost
 + package_area_cost
 + bonding_overhead_cost)
 * logic_yield_penalty
 * stacking_penalty
```

## 3. 备注

- D2D / C2C 不单独建成本项，而是主要通过 logic die 和 IO die 面积进入成本。
- TSV 不单独建直接成本项，而是通过 DRAM 面积和 `stacking_penalty` 间接进入成本。
- hybrid bonding 的直接成本走 `bonding_overhead_cost`，风险走 `stacking_penalty`，面积当前不单列。
- `logic_yield_coefficient_by_process` 建议参考 CATCH 的 `defect_density` 做相对映射，不把它理解成真实良率。
