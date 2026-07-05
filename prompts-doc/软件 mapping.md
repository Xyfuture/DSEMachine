首先看一下 @prompts-doc/背景.md 中的大背景
然后看一下 @dsemachine/encoding/matrix_mapping.py 中的代码， 这个其中包含了 Tree-based Representation



现在要求你编写 mapping 的 dse 代码。

我大致需要搜索这几个参数
- Dataflow : OS IS WS
- MappingTree
    - 每一层的 SplitNode 用多大的粒度划分
    - 每一层 SplitNode 的 TileOrdering 
    - child SplitNode 是什么样子的， 其使用了哪些 residual tile，构建成为了什么新的 rect


Mapping DSE 算法
输入：
    - 硬件encoding 
    - 待mapping的 weight，给出 B，M，K，N 四个参数
输出：
    - Best MappingTree


Mapping 部分的流程可以参考
- 首先枚举一个 Dataflow 的形式
    - 根据 @prompts-doc/脉动阵列 mapping cycle 计算方式.md 中的guide， 计算 base block的尺寸
    - 依据这个 base block 的尺寸对输入的 BMKN 的 rect 进行初步划分
- 对于 Root Rect 开始构建 MappingTree 
    - 递归式的构建方式
        - 首先对 Rect 进行划分操作，构建一个 SplitNode
            - 枚举一个 tile 的尺寸
            - 设定一个 ordering 的取值
        - 构建这个 SplitNode 的 children
            - 根据 num—tiles/num-chiplets 下取整来计算一个 chiplet 分到多少个 tile， 以及剩余多少个 residual tile
            - 根据 residual tile 的个数开始构建 child split node ， 具体构建方式参考 "residual tile 构建方法" 中给出的算法
            - 有多少个 chiplet 就有多少个 leaf node
                - 按照 ordering 中给出的编号逻辑， tile id按照从小到大(排除掉分配给 residual 的)开始连续的将每次 num_tile_per_chiplet 个 tile 分配给一个 chiplet， chiplet id 也从 0 开始计数。
            - 需要检测递归的层数，如果达到了 max 的递归层数，强制采用能均分的tile 尺寸，如果 tile 尺寸不符合要求，则跳过。
    - residual tile 构建方法
        -  residual tile 从右下角开始选取。对应 coordinate 的 (x_max,y_max) 开始选择，对应的 tile id 需要按照 ordering 给出的方式来找出
        - residual tiles 最好只构建一个 child rect
            - 根据 residual tile 的数量执行质因数分解计算(使用 function 中的 LRU 来记录一下)，找到所有可行的 child rect shape， 检测一下这个 child rect shape 和 parent rect shape 是否兼容，如果不兼容则抛弃。对于每一个可行的 child rect shape，都为其构建出一个单独的tree。
        - 如果因为 parent rect 的 shape 限制， 无法构建为一个 child rect，则构建为两个  child rect 。
            - 首先确定每个 child rect 有多大, 让 A+B = num_residual_rect , A 和 B 是对称的，求取所有的 pair
            - 然后对 A 和 B 的数量分别进行质因数分解，得到所有的候选集合
            - 然后用parent rect 的 shape 来判断能不能同时放下 A 和 B 的 child rect
                - 一个放到左上角，一个放到右下角
    - 注意事项
        - 在 ordering  之后， 要先给 residual tile 分配 tile id， 然后从剩下的 tile id 中从小到大开始分配给 leaf node
        - 需要设定递归的层级
- 每次得到一个 Mapping Tree 之后，评估一下性能
- 可能存在剪枝算法，还在考虑中。






---

我需要你在 @dsemachine/encoding/matrix_mapping.py 添加一个 class BaseBlock 的 class，然后将其加入到 class Rect 中。 BaseBlock 中记录一个 k 和 n 的大小，然后记录采取什么 Dataflow

每个 Rect 有 k—size * n_size 个 element， 每个 element 都是一个 base block，baseblock 里面的 k 和n 才是最小的单位--即每个数。一个 PIM chiplet 一次执行一个 base block。
BaseBlock 的尺寸是和硬件相关的，具体的算法在 @prompts-doc/脉动阵列 mapping cycle 计算方式.md 中有给出

你要看看这个变化会影响什么，同步这个改动







