import tilelang
import tilelang.language as T
import torch


def convert_to_uint16(x):
    hval = T.Cast(T.float16, x)
    bits_uint = T.reinterpret(T.uint16, hval)
    bits_uint = T.if_then_else(x < 0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8


def convert_to_uint32(x):
    bits_uint = T.reinterpret(T.uint32, x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast(T.uint32, (0xFFFFFFFF)),
        bits_uint | T.Cast(T.uint32, (0x80000000)),
    )
    return bits_uint


def convert_to_uint16_full(x):
    bits_uint = T.reinterpret(T.uint16, x)
    bits_uint = T.if_then_else(
        x < 0,
        ~bits_uint & T.Cast(T.uint16, (0xFFFF)),
        bits_uint | T.Cast(T.uint16, (0x8000)),
    )
    return bits_uint


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def tl_topk_radix(topk, in_dtype=T.float16, out_dtype=T.int32):
    assert in_dtype in (T.float16, T.float32)
    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    RADIX = 1 << 8
    BLOCK_SIZE = 1024
    SMEM_INPUT_SIZE = 4096  # assume the threshold bucket size after first pass is less than 4K

    @T.prim_func
    def tl_topk_radix_kernel(
        input: T.Tensor[(batch, seq_len), in_dtype],
        index: T.Tensor[(batch, topk), out_dtype],
    ):
        with T.Kernel(batch, threads=BLOCK_SIZE) as (bx):
            tx = T.get_thread_binding()

            s_threshold_bin_id = T.alloc_shared([1], T.int32)
            s_histogram = T.alloc_shared([RADIX + 1], T.int32)
            s_num_input = T.alloc_shared([2], T.int32)
            s_input_idx = T.alloc_shared([2, SMEM_INPUT_SIZE], T.int32)

            l_threshold_bin_id = T.alloc_var(T.int32)
            l_new_topk = T.alloc_var(T.int32)
            l_num_input = T.alloc_var(T.int32)
            l_bin_id32 = T.alloc_var(T.int32)
            l_val = T.alloc_var(T.int32)
            l_start_pos = T.alloc_var(T.int32)
            l_start_idx = T.alloc_var(T.int32)
            l_end_idx = T.alloc_var(T.int32)
            l_out_pos = T.alloc_var(T.int32)
            pos = T.alloc_var(T.int32)

            l_new_topk = topk
            l_start_idx = 0
            l_end_idx = seq_len

            # stage 1: use 8bit to do quick topk
            T.fill(s_histogram, 0)
            T.fill(s_num_input[0], 0)

            T.sync_threads()
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    inval_int16 = convert_to_uint16(input[bx, input_idx])
                    T.atomic_add(s_histogram[inval_int16], 1)
            T.sync_threads()

            # cumsum
            if tx < RADIX:
                for i in T.serial(8):
                    offset = 1 << i
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        l_val = s_histogram[tx] + s_histogram[tx + offset]
                    T.sync_threads(3, RADIX)
                    if tx < RADIX - offset:
                        s_histogram[tx] = l_val

                # find threshold bin id
                T.sync_threads(3, RADIX)
                if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                    s_threshold_bin_id[0] = tx
            T.sync_threads()
            l_threshold_bin_id = s_threshold_bin_id[0]
            l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
            T.sync_threads()

            # collect all elements with exponent ≥ threshold
            for s in T.serial(T.ceildiv(seq_len, BLOCK_SIZE)):
                T.sync_threads()
                input_idx = s * BLOCK_SIZE + tx
                if input_idx < l_end_idx and input_idx >= l_start_idx and input_idx < seq_len:
                    bin_id = convert_to_uint16(input[bx, input_idx])
                    l_bin_id32 = T.Cast(T.int32, bin_id)
                    if l_bin_id32 > l_threshold_bin_id:
                        # need a pos = T.atomic_add(s_histogram[bin_id32+1], 1)
                        pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True)
                        index[bx, pos] = input_idx

                    elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                        # pos = s_num_input[0]
                        pos = T.atomic_add(s_num_input[0], 1, return_prev=True)
                        T.device_assert(pos < SMEM_INPUT_SIZE)
                        s_input_idx[0, pos] = input_idx

            # stage 2: tail pass
            for round in T.serial(4 if in_dtype == T.float32 else 2):
                if l_new_topk <= 0:
                    T.loop_break()

                r_idx = round % 2
                l_start_pos = topk - l_new_topk

                T.sync_threads()
                T.fill(s_histogram, 0)
                if tx == 0:
                    s_num_input[r_idx ^ 1] = 0
                T.sync_threads()

                l_num_input = s_num_input[r_idx]
                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    if s * BLOCK_SIZE + tx < l_num_input:
                        if in_dtype == T.float32:
                            l_bin_id32 = T.Cast(
                                T.int32,
                                (
                                    (
                                        convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]])
                                        >> (24 - round * 8)
                                    )
                                    & 0xFF
                                ),
                            )
                        else:
                            l_bin_id32 = T.Cast(
                                T.int32,
                                (
                                    (
                                        convert_to_uint16_full(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]])
                                        >> (8 - round * 8)
                                    )
                                    & 0xFF
                                ),
                            )
                        T.atomic_add(s_histogram[l_bin_id32], 1)
                T.sync_threads()
                # cumsum
                if tx < RADIX:
                    for i in T.serial(8):
                        offset = 1 << i
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            l_val = s_histogram[tx] + s_histogram[tx + offset]
                        T.sync_threads(3, RADIX)
                        if tx < RADIX - offset:
                            s_histogram[tx] = l_val

                    # find threshold bin id
                    T.sync_threads(3, RADIX)
                    if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
                        s_threshold_bin_id[0] = tx
                T.sync_threads()

                l_threshold_bin_id = s_threshold_bin_id[0]
                l_new_topk = l_new_topk - s_histogram[l_threshold_bin_id + 1]
                T.sync_threads()

                for s in T.serial(T.ceildiv(l_num_input, BLOCK_SIZE)):
                    T.sync_threads()
                    if s * BLOCK_SIZE + tx < l_num_input:
                        if in_dtype == T.float32:
                            l_bin_id32 = T.Cast(
                                T.int32,
                                (
                                    (
                                        convert_to_uint32(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]])
                                        >> (24 - round * 8)
                                    )
                                    & 0xFF
                                ),
                            )
                        else:
                            l_bin_id32 = T.Cast(
                                T.int32,
                                (
                                    (
                                        convert_to_uint16_full(input[bx, s_input_idx[r_idx, s * BLOCK_SIZE + tx]])
                                        >> (8 - round * 8)
                                    )
                                    & 0xFF
                                ),
                            )
                        if l_bin_id32 > l_threshold_bin_id:
                            pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                            index[bx, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                        elif l_bin_id32 == l_threshold_bin_id and l_new_topk > 0:
                            if round == (3 if in_dtype == T.float32 else 1):
                                l_out_pos = T.atomic_add(s_histogram[l_bin_id32 + 1], 1, return_prev=True) + l_start_pos
                                if l_out_pos < topk:
                                    index[bx, l_out_pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]
                            else:
                                pos = T.atomic_add(s_num_input[r_idx ^ 1], 1, return_prev=True)
                                s_input_idx[r_idx ^ 1, pos] = s_input_idx[r_idx, s * BLOCK_SIZE + tx]

    return tl_topk_radix_kernel


def topk_radix(input, topk):
    batch, seq_len = input.shape
    indexes = torch.zeros(batch, topk, dtype=torch.int16, device=input.device)
    kernel = tl_topk_radix(topk)
    kernel(input, indexes)
    return indexes


@tilelang.jit(out_idx=[-1])
def tl_topk_2of4(
    elem_per_threads=16,
    threads=128,
):
    M = T.dynamic("M")
    N = 4
    topk = 2
    group_per_threads = elem_per_threads // N
    group_per_block = group_per_threads * threads
    dtype = T.float16
    index_dtype = T.int32

    @T.prim_func
    def tl_topk_2of4_kernel(
        logits: T.Tensor([M, N], dtype),
        topk_indices: T.Tensor([M, topk], index_dtype),
    ):
        with T.Kernel(T.ceildiv(M, group_per_block), threads=threads) as bx:
            tid = T.get_thread_binding(0)
            logits_local = T.alloc_local([group_per_threads, N], dtype=dtype)

            T.copy(
                logits[
                    bx * group_per_block
                    + tid * group_per_threads : bx * group_per_block
                    + (tid + 1) * group_per_threads,
                    :,
                ],
                logits_local,
            )

            for i in T.serial(group_per_threads):
                max1_idx = T.alloc_var(dtype=index_dtype)
                max1_val = T.alloc_var(dtype=dtype)
                max2_idx = T.alloc_var(dtype=index_dtype)
                max2_val = T.alloc_var(dtype=dtype)

                max1_idx = 0
                max1_val = logits_local[i, 0]
                max2_idx = 1
                max2_val = logits_local[i, 1]

                if max2_val > max1_val:
                    tmp_idx = max1_idx
                    tmp_val = max1_val
                    max1_idx = max2_idx
                    max1_val = max2_val
                    max2_idx = tmp_idx
                    max2_val = tmp_val

                for j in range(2):
                    curr_idx = j + 2
                    val = logits_local[i, curr_idx]
                    if val > max1_val:
                        max2_idx = max1_idx
                        max2_val = max1_val
                        max1_idx = curr_idx
                        max1_val = val
                    elif val > max2_val:
                        max2_idx = curr_idx
                        max2_val = val

                topk_indices[bx * group_per_block + tid * group_per_threads + i, 0] = max1_idx
                topk_indices[bx * group_per_block + tid * group_per_threads + i, 1] = max2_idx

    return tl_topk_2of4_kernel


@tilelang.jit(out_idx=[-1])
def tl_topk_multipass(
    N,
    topk,
    blk_m=64,
    threads=128,
):
    M = T.dynamic("M")
    dtype = T.float16
    index_dtype = T.int32
    index_inf = 2_147_483_647

    @T.prim_func
    def tl_topk_multipass_kernel(
        logits: T.Tensor([M, N], dtype),
        topk_indices: T.Tensor([M, topk], index_dtype),
    ):
        with T.Kernel(T.ceildiv(M, blk_m), threads=threads) as bx:
            logits_frag = T.alloc_fragment([blk_m, N], dtype=dtype)
            max_val = T.alloc_fragment([blk_m], dtype=dtype)
            expand_max_idx = T.alloc_fragment([blk_m, N], index_dtype)
            max_idx = T.alloc_fragment([blk_m], index_dtype)

            T.copy(logits[bx * blk_m, 0], logits_frag)

            for k in T.serial(topk):
                T.fill(expand_max_idx, index_inf)

                T.reduce_max(logits_frag, max_val, dim=1, clear=True)

                for i, j in T.Parallel(blk_m, N):
                    expand_max_idx[i, j] = T.if_then_else(max_val[i] == logits_frag[i, j], j, expand_max_idx[i, j])

                T.reduce_min(expand_max_idx, max_idx, dim=1, clear=True)

                for i, j in T.Parallel(blk_m, N):

                    logits_frag[i, j] = T.if_then_else(
                        max_idx[i] == j,
                        -T.infinity(dtype),
                        logits_frag[i, j],
                    )

                for i in T.Parallel(blk_m):
                    if 0 <= bx * blk_m + i < M:
                        topk_indices[bx * blk_m + i, k] = max_idx[i]

    return tl_topk_multipass_kernel


@torch.library.custom_op("sparse_attn::topk_multipass", mutates_args=())
def topk_multipass(X: torch.Tensor, topk: int = 2) -> torch.Tensor:
    assert X.dim() == 2
    assert X.size(1) == 4
    topk_kernel = tl_topk_multipass(N=X.size(1), topk=topk, blk_m=64, threads=128)
    indices = topk_kernel(X)
    return indices


@topk_multipass.register_fake
def _(X: torch.Tensor) -> torch.Tensor:
    m, n = X.shape
    assert n == 4
    return torch.empty(m, 2, dtype=torch.int16, device=X.device)
