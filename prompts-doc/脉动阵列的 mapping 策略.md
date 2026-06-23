# 关于脉动阵列的映射

先定义一下基础的粒度
一个 chiplet 有 PE_M * PE_N 个 PE
一个 PE 内部有 Num_SA * SA_M * SA_N 个 MAC 单元
一个 PE 的 SA 要运算的矩阵是 MatSA_M * MatSA_K * MatSA_N -- 这个是最小的单元。

在 3D-stacked PIM 中，对于 input tensor * weight = output tensor 的形式， weight 需要拆分直接映射到 PE 的 local banks 中


num_mc (num_memory_channel)


- Output Stationary
    - 对于 weight 来说，映射到一个 chiplet 上，最小的规模是
        - (PE_M * (Bank_Page_Size * num_mc/SA_N) , PE_N * SA_N)
    - 对于 MatSA_M * MatSA_K * MatSA_N 的矩阵运算来说，SA 需要的 cycle 数是
        - (MatSA_M/SA_M) * (MatSA_K/(PE_M*num_SA)) * (MatSA_N/PE_N)
        - 需要上取整
    - Memory的访存时间是
        - 

- Weight Stationary
    - 对于 weight 来说，映射到一个 chiplet 上，最小的规模是
        - (PE_M*SA_M * num_SA, PE_N * SA_N)
    - 对于 MatSA_M * MatSA_K * MatSA_N 的矩阵运算来说，SA 需要的 cycle 数是
        - 你帮我推导一下

- Input Stationary
    - 对于 weight 来说，映射到一个 chiplet 上，最小的规模是：
        - 你帮我推导一下
    - 对于 MatSA_M * MatSA_K * MatSA_N 的矩阵运算来说，SA 需要的 cycle 数是
        - 你帮我推导一下


