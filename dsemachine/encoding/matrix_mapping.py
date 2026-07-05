from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import count

from dsemachine.encoding.hardware import HardwareConfig


_rect_id_counter = count()


def ceil_div(x: int, y: int) -> int:
    if y <= 0:
        raise ValueError("divisor must be positive")
    return -(-x // y)


class Dataflow(Enum):
    OS = "os"
    WS = "ws"
    IS = "is"


@dataclass(frozen=True)
class BaseBlock:
    k_size_per_block: int
    n_size_per_block: int
    dataflow: Dataflow

    def __post_init__(self) -> None:
        if self.k_size_per_block <= 0 or self.n_size_per_block <= 0:
            raise ValueError("base block size must be positive")


@dataclass(frozen=True)
class MatrixShape:
    m: int
    k: int
    n: int
    input_bytes: int = 2
    weight_bytes: int = 2
    accumulator_bytes: int = 4

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int")


@dataclass(frozen=True)
class Rect:
    """
    一个等待分割的 rectangle, 是 SplitNode 操作的对象
    """
    num_k_blocks: int
    num_n_blocks: int
    base_block: BaseBlock
    rect_id: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rect_id", next(_rect_id_counter))
        if self.num_k_blocks <= 0 or self.num_n_blocks <= 0:
            raise ValueError("rect size must be positive")


@dataclass(frozen=True)
class Tile:
    rect_id: int
    tile_id: int
    coordinate: tuple[int, int]
    k_block_offset: int
    n_block_offset: int
    num_k_blocks: int
    num_n_blocks: int
    base_block: BaseBlock

    def __post_init__(self) -> None:
        if self.rect_id < 0:
            raise ValueError("rect id must be non-negative")
        if self.tile_id < 0:
            raise ValueError("tile id must be non-negative")
        if self.k_block_offset < 0 or self.n_block_offset < 0:
            raise ValueError("tile offset must be non-negative")
        if self.num_k_blocks <= 0 or self.num_n_blocks <= 0:
            raise ValueError("tile size must be positive")


@dataclass(frozen=True)
class TileOrdering:
    num_k_tiles: int
    num_n_tiles: int
    group_k: int = 1
    group_n: int = 1

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int")

    @property
    def num_tiles(self) -> int:
        return self.num_k_tiles * self.num_n_tiles

    def coord_to_id(self, k_id: int, n_id: int) -> int:
        if not (0 <= k_id < self.num_k_tiles and 0 <= n_id < self.num_n_tiles):
            raise ValueError("tile coordinate out of range")

        target_gk = k_id // self.group_k
        target_gn = n_id // self.group_n
        tile_id = 0
        for gk in range(ceil_div(self.num_k_tiles, self.group_k)):
            for gn in range(ceil_div(self.num_n_tiles, self.group_n)):
                k0, n0 = gk * self.group_k, gn * self.group_n
                k_count = min(self.group_k, self.num_k_tiles - k0)
                n_count = min(self.group_n, self.num_n_tiles - n0)
                if gk == target_gk and gn == target_gn:
                    return tile_id + (k_id - k0) * n_count + (n_id - n0)
                tile_id += k_count * n_count
        raise ValueError("tile coordinate out of range")

    def id_to_coord(self, tile_id: int) -> tuple[int, int]:
        if not (0 <= tile_id < self.num_tiles):
            raise ValueError("tile id out of range")

        remaining = tile_id
        for gk in range(ceil_div(self.num_k_tiles, self.group_k)):
            for gn in range(ceil_div(self.num_n_tiles, self.group_n)):
                k0, n0 = gk * self.group_k, gn * self.group_n
                k_count = min(self.group_k, self.num_k_tiles - k0)
                n_count = min(self.group_n, self.num_n_tiles - n0)
                group_tiles = k_count * n_count
                if remaining < group_tiles:
                    return k0 + remaining // n_count, n0 + remaining % n_count
                remaining -= group_tiles
        raise ValueError("tile id out of range")


class MappingTreeNode:
    parent: "MappingSplitNode | None"


@dataclass(eq=False)
class MappingLeafNode(MappingTreeNode):
    parent: "MappingSplitNode" = field(repr=False)
    chiplet_id: int
    tile_ids_from_parent: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.parent is None:
            raise ValueError("leaf node must have a parent")
        if not self.tile_ids_from_parent:
            raise ValueError("leaf node must have tile_ids_from_parent")


@dataclass(eq=False)
class MappingSplitNode(MappingTreeNode):
    parent: "MappingSplitNode | None" = field(repr=False)
    rect: Rect
    num_k_blocks_per_tile: int
    num_n_blocks_per_tile: int
    ordering: TileOrdering
    children: tuple[MappingTreeNode, ...] = ()
    tile_ids_from_parent: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.num_k_blocks_per_tile <= 0 or self.num_n_blocks_per_tile <= 0:
            raise ValueError("tile size must be positive")
        expected_k = ceil_div(self.rect.num_k_blocks, self.num_k_blocks_per_tile)
        expected_n = ceil_div(self.rect.num_n_blocks, self.num_n_blocks_per_tile)
        if self.ordering.num_k_tiles != expected_k:
            raise ValueError("ordering num_k_tiles does not match split rect")
        if self.ordering.num_n_tiles != expected_n:
            raise ValueError("ordering num_n_tiles does not match split rect")


@dataclass(frozen=True)
class AssignedTile:
    chiplet_id: int
    tile: Tile


def expand_mapping(root: MappingSplitNode, hw: HardwareConfig) -> list[AssignedTile]:
    if root.parent is not None:
        raise ValueError("root split node must not have a parent")
    if root.tile_ids_from_parent is not None:
        raise ValueError("root split node must not have tile_ids_from_parent")
    assigned: list[AssignedTile] = []
    _expand_split(root, hw, 0, 0, assigned)
    return assigned


def _expand_split(
    node: MappingSplitNode,
    hw: HardwareConfig,
    global_k_block_start: int,
    global_n_block_start: int,
    assigned: list[AssignedTile],
) -> None:
    if not node.children:
        raise ValueError("split node must have children")

    consumed_tile_ids: list[int] = []

    for child in node.children:
        if child.parent is not node:
            raise ValueError("child parent does not match current split node")

        if isinstance(child, MappingLeafNode):
            if not (0 <= child.chiplet_id < hw.num_pim_chiplets):
                raise ValueError("chiplet id out of range")
            tiles = tiles_from_parent(node, child.tile_ids_from_parent)
            consumed_tile_ids.extend(child.tile_ids_from_parent)
            for tile in tiles:
                assigned.append(
                    AssignedTile(
                        chiplet_id=child.chiplet_id,
                        tile=Tile(
                            rect_id=tile.rect_id,
                            tile_id=tile.tile_id,
                            coordinate=tile.coordinate,
                            k_block_offset=global_k_block_start + tile.k_block_offset,
                            n_block_offset=global_n_block_start + tile.n_block_offset,
                            num_k_blocks=tile.num_k_blocks,
                            num_n_blocks=tile.num_n_blocks,
                            base_block=tile.base_block,
                        ),
                    )
                )
        elif isinstance(child, MappingSplitNode):
            if child.tile_ids_from_parent is None:
                raise ValueError("non-root split node must have tile_ids_from_parent")
            tiles = tiles_from_parent(node, child.tile_ids_from_parent)
            consumed_tile_ids.extend(child.tile_ids_from_parent)
            rect, k_offset, n_offset = merge_tiles_to_rect(tiles)
            if not _same_rect_shape(rect, child.rect):
                raise ValueError("child split rect does not match parent tiles")
            _expand_split(
                child,
                hw,
                global_k_block_start + k_offset,
                global_n_block_start + n_offset,
                assigned,
            )
        else:
            raise ValueError("unknown mapping node type")

    _validate_child_tile_coverage(node, consumed_tile_ids)


def split_rect_to_tiles(node: MappingSplitNode) -> list[Tile]:
    tiles: list[Tile] = []
    for tile_id in range(node.ordering.num_tiles):
        k_id, n_id = node.ordering.id_to_coord(tile_id)
        k_block_offset = k_id * node.num_k_blocks_per_tile
        n_block_offset = n_id * node.num_n_blocks_per_tile
        tiles.append(
            Tile(
                rect_id=node.rect.rect_id,
                tile_id=tile_id,
                coordinate=(k_id, n_id),
                k_block_offset=k_block_offset,
                n_block_offset=n_block_offset,
                num_k_blocks=min(
                    node.num_k_blocks_per_tile,
                    node.rect.num_k_blocks - k_block_offset,
                ),
                num_n_blocks=min(
                    node.num_n_blocks_per_tile,
                    node.rect.num_n_blocks - n_block_offset,
                ),
                base_block=node.rect.base_block,
            )
        )
    return tiles


def tiles_from_parent(
    parent: MappingSplitNode,
    tile_ids_from_parent: tuple[int, ...],
) -> list[Tile]:
    if not tile_ids_from_parent:
        raise ValueError("tile_ids_from_parent must not be empty")
    if len(set(tile_ids_from_parent)) != len(tile_ids_from_parent):
        raise ValueError("tile_ids_from_parent must not contain duplicates")

    parent_tiles = split_rect_to_tiles(parent)
    if min(tile_ids_from_parent) < 0 or max(tile_ids_from_parent) >= len(parent_tiles):
        raise ValueError("tile id out of range")
    return [parent_tiles[tile_id] for tile_id in tile_ids_from_parent]


def merge_tiles_to_rect(tiles: list[Tile]) -> tuple[Rect, int, int]:
    if not tiles:
        raise ValueError("tiles must not be empty")

    k_start = min(tile.k_block_offset for tile in tiles)
    n_start = min(tile.n_block_offset for tile in tiles)
    k_end = max(tile.k_block_offset + tile.num_k_blocks for tile in tiles)
    n_end = max(tile.n_block_offset + tile.num_n_blocks for tile in tiles)
    area = sum(tile.num_k_blocks * tile.num_n_blocks for tile in tiles)
    merged = Rect(
        num_k_blocks=k_end - k_start,
        num_n_blocks=n_end - n_start,
        base_block=tiles[0].base_block,
    )
    if area != merged.num_k_blocks * merged.num_n_blocks:
        raise ValueError("tiles must form a rectangle")
    return merged, k_start, n_start


def _validate_child_tile_coverage(
    node: MappingSplitNode,
    consumed_tile_ids: list[int],
) -> None:
    expected = list(range(node.ordering.num_tiles))
    if sorted(consumed_tile_ids) != expected:
        raise ValueError("children must cover every parent tile exactly once")


def _same_rect_shape(a: Rect, b: Rect) -> bool:
    return (
        a.num_k_blocks == b.num_k_blocks
        and a.num_n_blocks == b.num_n_blocks
        and a.base_block == b.base_block
    )
