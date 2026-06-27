class MappingTreeNode:
    """
    基础的 base class

    """
    pass 


class MappingLeafNode(MappingTreeNode):
    """
    这个 node 意味着来到了分配的终点。
    这里记录一个 PIM chiplet 要执行那些 其 parent SpiltNode 分配的Tile
    
    split node 中存在一个编码的规则，leaf node 中记录其使用了哪些 tile 的 tile id。
    
    """
    pass 


class MappingSplitNode(MappingTreeNode):
    """
    这个 node 表示对于这些 tile 还要进一步进行拆分操作
    不是最终的 leaf, 没有办法进行最后的仿真运算。

    
    这个 node 需要知道自己持有的 tile 的有哪些，持有的 tile 是什么形状 (必须是一个 rectangle, 不然没法进行下一步的分配)

    然后这个 node 还需要知道下一步要对自己的 rectangle 进行下一步的细分策略

    """
    
    pass 


class Ordering:
    """
    用于给不同的 tile 进行编号。
    可以参考 triton 中 group M 和 group N 的操作，相当于是一个线性化的过程
    给出 (x,y) 能返回 tile 的 id, 同时给出 tile id 能反推出 (x,y) 的坐标
    

    可以看看怎么设计比较好，可以直接融入到 MappingSplitNode 中
    
    """
    pass 
