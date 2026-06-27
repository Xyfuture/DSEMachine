# 性能模拟部分设计思路

性能模拟的核心目标是给出一个硬件在一套映射下的延迟数据情况。

输入:
- 硬件参数
- Mapping 表示

输出:
- 延迟的情况 



## Mapping 表示


你帮我想一种对于 Mapping 的表示，我要用在模拟器中，根据这个编码，我可以知道 Mapping 是什么样子的，能够开展对这个 Mapping 的 performance simulation 
我们的任务是将一个 Matrix Mapping 到多个 PIM Chiplet 上，以及 PIM Chiplet 内部的脉动阵列上
我现在有一个大致的策略
1. 选择脉动阵列的数据流 IS，OS，WS，每个数据流都对应一个映射在该 chiplet 上的最小 A-level-Tile
2. 使用 A-level-Tile 的尺寸对 weight 划分，得到一个新的 weight （例如 1024*2048 使用 128 *64 的 A-level—tile 划分之后是 8*32 ）
3. 针对新的 weight(8*32) 进行下一轮的划分，这次选择新的 tile size -- B-level-Tile，新的 B-level-Tile 是用来将一个矩阵划分到多个 PIM chiplet 中的，一个 PIM chiplet 会分到多个 B-level-Tile
4. 划分完毕的 B-level-Tile 会按照 Group K 和 Group N 的方式进行一个 linear 的编码， 得到一个 id-> (x,y) 的映射
5. 下面进行一个含有递归操作的分配策略 
6. 根据 PIM chiplet 的数量和 tile 的数量计算一下每个 PIM chiplet 可以分到几个B-level-tile ， 以及有几个 residual tile 不能分配到 PIM chiplet 上
7. 确定 residual tile 的编号，让所有的 residual tile 尽可能组合成一个或这个两个 rectangle 新的矩阵。
8. 按照编号，一次性为每个 PIM chiplet 分配好其需要执行的 B-level-Tile，直到所有的 PIM chiplet 都分配完，这个过程中要跳过分配到 residual tile 中的那些
9. 将剩下的由 residual tile 构成的 1-2 个 rectangle 作为一个子问题，回归到步骤 3 进行一个递归操作



### Mapping Encoding 代码参考 


详细看一下 @mapping_tree.py 中的实现。 



## 工作逻辑

这里采用离散时间时序仿真完成对一个流水线时间的仿真。 
我们拥有多种硬件资源
- chiplet InputD2DLink
- chiplet OutputD2DLink
- chiplet DRAM 
    - 使用其带宽资源
- chiplet MatrixCompute
- chiplet VectorCompute 

每种硬件资源都是一个独立可并行的流水级，我们为每个硬件资源都维护一个 free_time 表示最早在这个时间是空闲的，每次这个硬件资源要执行一个操作的时候，都会在 free time 开始执行，然后依据操作的时间更新这个 free time。






