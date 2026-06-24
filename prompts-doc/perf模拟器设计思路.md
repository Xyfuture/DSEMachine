# 性能模拟部分设计思路

性能模拟的核心目标是给出一个硬件在一套映射下的延迟数据情况。

输入:
- 硬件参数
- Mapping 表示

输出:



工作逻辑：

这里采用离散时间时序仿真完成对一个流水线时间的仿真。 
我们拥有多种硬件资源
- chiplet InputD2DLink
- chiplet OutputD2DLink
- chiplet DRAM 
    - 使用其带宽资源
- chiplet MatrixCompute
- chiplet VectorCompute 


