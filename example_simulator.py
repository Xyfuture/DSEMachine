from dsemachine.encoding.hardware import HardwareConfig
from dsemachine.encoding.matrix_mapping import (
    BaseBlock,
    Dataflow,
    MappingLeafNode,
    MappingSplitNode,
    MatrixShape,
    Rect,
    TileOrdering,
    expand_mapping,
)
from dsemachine.perfsim.perf_core import simulate_matrix


def build_example_mapping() -> MappingSplitNode:
    base_block = BaseBlock(
        k_size_per_block=4,
        n_size_per_block=4,
        dataflow=Dataflow.OS,
    )
    root = MappingSplitNode(
        parent=None,
        rect=Rect(num_k_blocks=8, num_n_blocks=8, base_block=base_block),
        num_k_blocks_per_tile=4,
        num_n_blocks_per_tile=4,
        ordering=TileOrdering(num_k_tiles=2, num_n_tiles=2),
    )

    root.children = (
        MappingLeafNode(parent=root, chiplet_id=0, tile_ids_from_parent=(0,)),
        MappingLeafNode(parent=root, chiplet_id=1, tile_ids_from_parent=(1,)),
        MappingLeafNode(parent=root, chiplet_id=2, tile_ids_from_parent=(2,)),
        MappingLeafNode(parent=root, chiplet_id=3, tile_ids_from_parent=(3,)),
    )
    return root


def main() -> None:
    hw = HardwareConfig(
        num_pim_chiplets=4,
        pe_m=2,
        pe_n=2,
        num_sa=2,
        sa_m=4,
        sa_n=4,
        sa_frequency_mhz=1000,
        dram_bytes_per_cycle=64,
        d2d_input_bytes_per_cycle=32,
        d2d_output_bytes_per_cycle=32,
        io_reduction_bytes_per_cycle=64,
        dram_page_bytes=128,
    )
    shape = MatrixShape(m=16, k=32, n=32)
    mapping = build_example_mapping()

    print("Assigned tiles:")
    for assigned in expand_mapping(mapping, hw):
        tile = assigned.tile
        print(
            f"  chiplet={assigned.chiplet_id} "
            f"tile_id={tile.tile_id} "
            f"k_blocks=[{tile.k_block_offset}, "
            f"{tile.k_block_offset + tile.num_k_blocks}) "
            f"n_blocks=[{tile.n_block_offset}, "
            f"{tile.n_block_offset + tile.num_n_blocks})"
        )

    result = simulate_matrix(shape, hw, mapping)
    trace_path = "example_perf_trace.json"
    result.save_perf_trace(trace_path, ns_per_cycle=1.0)

    print("\nSimulation result:")
    print(f"  total_cycles={result.total_cycles}")
    print(f"  input_d2d_cycles={result.input_d2d_cycles}")
    print(f"  dram_cycles={result.dram_cycles}")
    print(f"  compute_cycles={result.compute_cycles}")
    print(f"  output_d2d_cycles={result.output_d2d_cycles}")
    print(f"  reduction_cycles={result.reduction_cycles}")
    print(f"  perf_trace={trace_path}")

    print("\nInstruction trace:")
    for item in result.instruction_trace:
        print(
            f"  {item.inst_id:22s} "
            f"{item.resource_name:18s} "
            f"{item.start:4d} -> {item.end:4d} "
            f"deps={item.deps}"
        )


if __name__ == "__main__":
    main()
