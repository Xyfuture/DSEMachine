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
    k_size: int
    n_size: int
    rect_id: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rect_id", next(_rect_id_counter))
        if self.k_size <= 0 or self.n_size <= 0:
            raise ValueError("rect size must be positive")


@dataclass(frozen=True)
class Tile:
    rect_id: int
    tile_id: int
    coordinate: tuple[int, int]
    k_offset: int
    n_offset: int
    k_size: int
    n_size: int

    def __post_init__(self) -> None:
        if self.rect_id < 0:
            raise ValueError("rect id must be non-negative")
        if self.tile_id < 0:
            raise ValueError("tile id must be non-negative")
        if (
            not isinstance(self.coordinate, tuple)
            or len(self.coordinate) != 2
            or not isinstance(self.coordinate[0], int)
            or not isinstance(self.coordinate[1], int)
            or self.coordinate[0] < 0
            or self.coordinate[1] < 0
        ):
            raise ValueError("tile coordinate must be a non-negative (k_id, n_id) tuple")
        if self.k_offset < 0 or self.n_offset < 0:
            raise ValueError("tile offset must be non-negative")
        if self.k_size <= 0 or self.n_size <= 0:
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
    tile_k: int
    tile_n: int
    ordering: TileOrdering
    children: tuple[MappingTreeNode, ...] = ()
    tile_ids_from_parent: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.tile_k <= 0 or self.tile_n <= 0:
            raise ValueError("tile size must be positive")
        expected_k = ceil_div(self.rect.k_size, self.tile_k)
        expected_n = ceil_div(self.rect.n_size, self.tile_n)
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
    global_k_start: int,
    global_n_start: int,
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
                            k_offset=global_k_start + tile.k_offset,
                            n_offset=global_n_start + tile.n_offset,
                            k_size=tile.k_size,
                            n_size=tile.n_size,
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
                global_k_start + k_offset,
                global_n_start + n_offset,
                assigned,
            )
        else:
            raise ValueError("unknown mapping node type")

    _validate_child_tile_coverage(node, consumed_tile_ids)


def split_rect_to_tiles(node: MappingSplitNode) -> list[Tile]:
    tiles: list[Tile] = []
    for tile_id in range(node.ordering.num_tiles):
        k_id, n_id = node.ordering.id_to_coord(tile_id)
        k_offset = k_id * node.tile_k
        n_offset = n_id * node.tile_n
        tiles.append(
            Tile(
                rect_id=node.rect.rect_id,
                tile_id=tile_id,
                coordinate=(k_id, n_id),
                k_offset=k_offset,
                n_offset=n_offset,
                k_size=min(node.tile_k, node.rect.k_size - k_offset),
                n_size=min(node.tile_n, node.rect.n_size - n_offset),
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

    k_start = min(tile.k_offset for tile in tiles)
    n_start = min(tile.n_offset for tile in tiles)
    k_end = max(tile.k_offset + tile.k_size for tile in tiles)
    n_end = max(tile.n_offset + tile.n_size for tile in tiles)
    area = sum(tile.k_size * tile.n_size for tile in tiles)
    merged = Rect(k_size=k_end - k_start, n_size=n_end - n_start)
    if area != merged.k_size * merged.n_size:
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
    return a.k_size == b.k_size and a.n_size == b.n_size
