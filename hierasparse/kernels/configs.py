import itertools

import tilelang


def acc_t_C_mapping(atom_idx: int, atom_col: int):
    group_idx = atom_idx // 2
    in_group_idx = atom_idx % 2
    return (group_idx // atom_col) * 2 + in_group_idx, group_idx % atom_col


def acc_t_B_mapping_inv(row, col, atom_col):
    row_group = row // 4
    in_group_idx = row % 4
    return 4 * (row_group * atom_col + col) + in_group_idx


def flashattn_sp_tune_configs():
    iter_params = dict(
        block_M=[32 * i for i in range(1, 5)],
        block_N=[32 * i for i in range(1, 5)],
        threads=[128, 256],
        use_movmatrix=[True, False],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


def flashattn_tune_configs():
    iter_params = dict(
        block_M=[32 * i for i in range(1, 5)],
        block_N=[32 * i for i in range(1, 5)],
        threads=[128, 256],
    )
    return [dict(zip(iter_params, values)) for values in itertools.product(*iter_params.values())]


def pass_configs():
    return {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # tilelang.PassConfigKey.TL_DEBUG_MERGE_SHARED_MEMORY_ALLOCATIONS: True,
        # tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_ENABLE: True,
        # tilelang.PassConfigKey.TL_LAYOUT_VISUALIZATION_FORMATS: "png",
    }


def flashdecode_sp_tune_configs():
    block_N = [32, 64, 96, 128]
    block_H = [8]
    num_split = [1, 2, 4, 8, 16, 32]
    num_stages = [0, 1, 2, 3]
    threads = [32]
    use_movmatrix = [True, False]
    _configs = list(itertools.product(block_N, block_H, num_split, num_stages, threads, use_movmatrix))
    configs = [
        {
            "block_N": c[0],
            "block_H": c[1],
            "num_split": c[2],
            "num_stages": c[3],
            "threads": c[4],
            "use_movmatrix": c[5],
        }
        for c in _configs
    ]
    return configs


if __name__ == "__main__":
    atom_row = 2
    atom_col = 4
    for idx in range(atom_row * atom_col):
        i, j = acc_t_C_mapping(idx, atom_col)
