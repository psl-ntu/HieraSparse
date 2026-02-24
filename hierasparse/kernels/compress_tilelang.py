import tilelang
import tilelang.language as T

from hierasparse.utils import E_FACTOR


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    }
)
def prune_and_compress_key_kernel(B, H, D, block_M=E_FACTOR, elem_per_thread=32):
    dtype = T.float16
    meta_dtype = T.int16
    elem, group = 2, 4
    e_factor = 16

    assert D >= elem_per_thread, "D must be greater than or equal to elem_per_thread"
    assert D % elem_per_thread == 0, "D must be divisible by elem_per_thread"
    S = T.dynamic("S")

    @T.prim_func
    def prune_and_compress_key_kernel(
        Dense: T.Tensor([B, H, S, D], dtype),
        Sparse: T.Tensor([B, H, S, D // 2], dtype),
        Meta: T.Tensor([B, H, S, D // e_factor], meta_dtype),
    ):
        with T.Kernel(S // block_M, H, B, threads=(block_M, D // elem_per_thread)) as (bz, by, bx):
            tm, tn = T.get_thread_binding(0), T.get_thread_binding(1)
            dense_local = T.alloc_local([elem_per_thread], dtype)
            sparse_local = T.alloc_local([elem_per_thread // 2], dtype)
            meta_local = T.alloc_local([elem_per_thread // e_factor], meta_dtype)

            non_zero_elt_log_idx = T.alloc_local([elem], T.uint8)

            T.clear(sparse_local)
            T.clear(meta_local)

            T.copy(Dense[bx, by, bz * block_M + tm, tn * elem_per_thread : (tn + 1) * elem_per_thread], dense_local)

            for g_i in range(elem_per_thread // group):
                T.clear(non_zero_elt_log_idx)

                local_idx = g_i * group

                max1_idx = T.alloc_var(dtype=T.uint8)
                max1_val = T.alloc_var(dtype=dtype)
                max2_idx = T.alloc_var(dtype=T.uint8)
                max2_val = T.alloc_var(dtype=dtype)

                max1_idx = 0
                max1_val = T.abs(dense_local[local_idx])
                max2_idx = 1
                max2_val = T.abs(dense_local[local_idx + 1])

                if max2_val > max1_val:
                    tmp_idx = max1_idx
                    tmp_val = max1_val
                    max1_idx = max2_idx
                    max1_val = max2_val
                    max2_idx = tmp_idx
                    max2_val = tmp_val

                for i in range(2):
                    j = i + 2
                    val = T.abs(dense_local[local_idx + j])
                    if val > max1_val:
                        max2_idx = max1_idx
                        max2_val = max1_val
                        max1_idx = j
                        max1_val = val
                    elif val > max2_val:
                        max2_idx = j
                        max2_val = val

                non_zero_elt_log_idx[0] = T.min(max1_idx, max2_idx)
                non_zero_elt_log_idx[1] = T.max(max1_idx, max2_idx)
                sparse_local[local_idx // 2] = dense_local[local_idx + non_zero_elt_log_idx[0]]
                sparse_local[local_idx // 2 + 1] = dense_local[local_idx + non_zero_elt_log_idx[1]]

                for i in T.serial(elem):
                    val = non_zero_elt_log_idx[i]
                    meta_local[local_idx // e_factor] |= T.shift_left(
                        val.astype(meta_dtype), (4 * (g_i % (e_factor // group)) + 2 * i).astype(meta_dtype)
                    )
            T.copy(
                sparse_local,
                Sparse[bx, by, bz * block_M + tm, tn * (elem_per_thread // 2) : (tn + 1) * (elem_per_thread // 2)],
            )
            T.copy(
                meta_local,
                Meta[
                    bx,
                    by,
                    bz * block_M + tm,
                    tn * (elem_per_thread // e_factor) : (tn + 1) * (elem_per_thread // e_factor),
                ],
            )

    return prune_and_compress_key_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    }
)
def prune_and_compress_value_kernel(B, H, D, block_M=E_FACTOR, elem_per_thread=16):
    dtype = T.float16
    meta_dtype = T.int16
    elem, group = 2, 4
    e_factor = 16

    assert block_M >= elem_per_thread, "block_M must be greater than or equal to elem_per_thread"
    assert block_M % elem_per_thread == 0, "block_M must be divisible by elem_per_thread"
    S = T.dynamic("S")

    @T.prim_func
    def prune_and_compress_value_kernel(
        Dense: T.Tensor([B, H, S, D], dtype),
        Sparse: T.Tensor([B, H, S // 2, D], dtype),
        Meta: T.Tensor([B, H, S // e_factor, D], meta_dtype),
    ):
        with T.Kernel(B, H, S // block_M, threads=(block_M // elem_per_thread, D)) as (bx, by, bz):
            tm, tn = T.get_thread_binding(0), T.get_thread_binding(1)
            dense_local = T.alloc_local([elem_per_thread], dtype)
            sparse_local = T.alloc_local([elem_per_thread // 2], dtype)
            meta_local = T.alloc_local([elem_per_thread // e_factor], meta_dtype)

            non_zero_elt_log_idx = T.alloc_local([elem], T.uint8)

            T.copy(
                Dense[bx, by, bz * block_M + tm * elem_per_thread : bz * block_M + (tm + 1) * elem_per_thread, tn],
                dense_local,
            )

            T.clear(sparse_local)
            T.clear(meta_local)

            for g_i in range(elem_per_thread // group):
                T.clear(non_zero_elt_log_idx)

                local_idx = g_i * group

                max1_idx = T.alloc_var(dtype=T.uint8)
                max1_val = T.alloc_var(dtype=dtype)
                max2_idx = T.alloc_var(dtype=T.uint8)
                max2_val = T.alloc_var(dtype=dtype)

                max1_idx = 0
                max1_val = T.abs(dense_local[local_idx])
                max2_idx = 1
                max2_val = T.abs(dense_local[local_idx + 1])

                if max2_val > max1_val:
                    tmp_idx = max1_idx
                    tmp_val = max1_val
                    max1_idx = max2_idx
                    max1_val = max2_val
                    max2_idx = tmp_idx
                    max2_val = tmp_val

                for i in range(2):
                    j = i + 2
                    val = T.abs(dense_local[local_idx + j])
                    if val > max1_val:
                        max2_idx = max1_idx
                        max2_val = max1_val
                        max1_idx = j
                        max1_val = val
                    elif val > max2_val:
                        max2_idx = j
                        max2_val = val

                non_zero_elt_log_idx[0] = T.min(max1_idx, max2_idx)
                non_zero_elt_log_idx[1] = T.max(max1_idx, max2_idx)
                sparse_local[local_idx // 2] = dense_local[local_idx + non_zero_elt_log_idx[0]]
                sparse_local[local_idx // 2 + 1] = dense_local[local_idx + non_zero_elt_log_idx[1]]

                for i in T.serial(elem):
                    val = non_zero_elt_log_idx[i]
                    meta_local[local_idx // e_factor] |= T.shift_left(
                        val.astype(meta_dtype), (4 * (g_i % (e_factor // group)) + 2 * i).astype(meta_dtype)
                    )
            T.copy(
                sparse_local,
                Sparse[
                    bx,
                    by,
                    (bz * block_M + tm * elem_per_thread) // 2 : (bz * block_M + (tm + 1) * elem_per_thread) // 2,
                    tn,
                ],
            )
            T.copy(
                meta_local,
                Meta[
                    bx,
                    by,
                    (bz * block_M + tm * elem_per_thread)
                    // e_factor : (bz * block_M + (tm + 1) * elem_per_thread)
                    // e_factor,
                    tn,
                ],
            )

    return prune_and_compress_value_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    }
)
def block_compress_sparse_key(b, hn, block_M, hd, elem_per_thread=32):
    dtype = T.float16
    idx_dtype = T.int16
    meta_dtype = T.int16
    elem, group = 2, 4
    e_factor = 16

    s = T.dynamic("s")
    blocks_per_head = T.dynamic("blocks_per_head")
    sparse_blocks_per_head = T.dynamic("sparse_blocks_per_head")

    assert hd >= elem_per_thread, "hd must be greater than or equal to elem_per_thread"
    assert hd % elem_per_thread == 0, "hd must be divisible by elem_per_thread"

    @T.prim_func
    def block_compress_sparse_key_kernel(
        K: T.Tensor([b, hn, s, hd], dtype),
        block_mask: T.Tensor([b, hn, blocks_per_head], T.bool),
        idx_map: T.Tensor([b, hn, blocks_per_head], idx_dtype),
        sparse_nonzero: T.Tensor([b, hn, sparse_blocks_per_head, block_M, hd // 2], dtype),
        sparse_meta: T.Tensor([b, hn, sparse_blocks_per_head, block_M, hd // e_factor], meta_dtype),
    ):
        with T.Kernel(hn, b, threads=(block_M, hd // elem_per_thread)) as (hx, bx):
            tm, tn = T.get_thread_binding(0), T.get_thread_binding(1)
            sparse_count = T.alloc_local([1], idx_dtype)
            sparse_count[0] = 0

            for k in range(blocks_per_head):
                mask_val = block_mask[bx, hx, k]

                if not mask_val:
                    sparse_idx = sparse_count[0]
                    sparse_count[0] += 1
                    idx_map[bx, hx, k] = -sparse_count[0]

                    dense_local = T.alloc_local([elem_per_thread], dtype)
                    sparse_local = T.alloc_local([elem_per_thread // 2], dtype)
                    meta_local = T.alloc_local([elem_per_thread // e_factor], meta_dtype)
                    non_zero_elt_log_idx = T.alloc_local([elem], T.uint8)

                    T.clear(sparse_local)
                    T.clear(meta_local)

                    T.copy(K[bx, hx, k * block_M + tm, tn * elem_per_thread : (tn + 1) * elem_per_thread], dense_local)

                    for g_i in range(elem_per_thread // group):
                        T.clear(non_zero_elt_log_idx)

                        local_idx = g_i * group

                        max1_idx = T.alloc_var(dtype=T.uint8)
                        max1_val = T.alloc_var(dtype=dtype)
                        max2_idx = T.alloc_var(dtype=T.uint8)
                        max2_val = T.alloc_var(dtype=dtype)

                        max1_idx = 0
                        max1_val = T.abs(dense_local[local_idx])
                        max2_idx = 1
                        max2_val = T.abs(dense_local[local_idx + 1])

                        if max2_val > max1_val:
                            tmp_idx = max1_idx
                            tmp_val = max1_val
                            max1_idx = max2_idx
                            max1_val = max2_val
                            max2_idx = tmp_idx
                            max2_val = tmp_val

                        for i in range(2):
                            j = i + 2
                            val = T.abs(dense_local[local_idx + j])
                            if val > max1_val:
                                max2_idx = max1_idx
                                max2_val = max1_val
                                max1_idx = j
                                max1_val = val
                            elif val > max2_val:
                                max2_idx = j
                                max2_val = val

                        non_zero_elt_log_idx[0] = T.min(max1_idx, max2_idx)
                        non_zero_elt_log_idx[1] = T.max(max1_idx, max2_idx)
                        sparse_local[local_idx // 2] = dense_local[local_idx + non_zero_elt_log_idx[0]]
                        sparse_local[local_idx // 2 + 1] = dense_local[local_idx + non_zero_elt_log_idx[1]]

                        for i in T.serial(elem):
                            val = non_zero_elt_log_idx[i]
                            meta_local[local_idx // e_factor] |= T.shift_left(
                                val.astype(meta_dtype), (4 * (g_i % (e_factor // group)) + 2 * i).astype(meta_dtype)
                            )
                    T.copy(
                        sparse_local,
                        sparse_nonzero[
                            bx, hx, sparse_idx, tm, tn * (elem_per_thread // 2) : (tn + 1) * (elem_per_thread // 2)
                        ],
                    )
                    T.copy(
                        meta_local,
                        sparse_meta[
                            bx,
                            hx,
                            sparse_idx,
                            tm,
                            tn * (elem_per_thread // e_factor) : (tn + 1) * (elem_per_thread // e_factor),
                        ],
                    )

    return block_compress_sparse_key_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    }
)
def block_compress_sparse_value(b, hn, block_M, hd, elem_per_thread=32):
    dtype = T.float16
    idx_dtype = T.int16
    meta_dtype = T.int16
    elem, group = 2, 4
    e_factor = 16

    s = T.dynamic("s")
    blocks_per_head = T.dynamic("blocks_per_head")
    sparse_blocks_per_head = T.dynamic("sparse_blocks_per_head")

    assert hd >= elem_per_thread, "hd must be greater than or equal to elem_per_thread"
    assert hd % elem_per_thread == 0, "hd must be divisible by elem_per_thread"

    @T.prim_func
    def block_compress_sparse_value_kernel(
        K: T.Tensor([b, hn, s, hd], dtype),
        block_mask: T.Tensor([b, hn, blocks_per_head], T.bool),
        idx_map: T.Tensor([b, hn, blocks_per_head], idx_dtype),
        sparse_nonzero: T.Tensor([b, hn, sparse_blocks_per_head, block_M // 2, hd], dtype),
        sparse_meta: T.Tensor([b, hn, sparse_blocks_per_head, block_M // e_factor, hd], meta_dtype),
    ):
        with T.Kernel(hn, b, threads=(block_M // elem_per_thread, hd)) as (hx, bx):
            tm, tn = T.get_thread_binding(0), T.get_thread_binding(1)
            sparse_count = T.alloc_local([1], idx_dtype)
            sparse_count[0] = 0

            for k in range(blocks_per_head):
                mask_val = block_mask[bx, hx, k]

                if not mask_val:
                    sparse_idx = sparse_count[0]
                    sparse_count[0] += 1
                    idx_map[bx, hx, k] = -sparse_count[0]

                    dense_local = T.alloc_local([elem_per_thread], dtype)
                    sparse_local = T.alloc_local([elem_per_thread // 2], dtype)
                    meta_local = T.alloc_local([elem_per_thread // e_factor], meta_dtype)
                    non_zero_elt_log_idx = T.alloc_local([elem], T.uint8)

                    T.clear(sparse_local)
                    T.clear(meta_local)

                    T.copy(
                        K[bx, hx, k * block_M + tm * elem_per_thread : k * block_M + (tm + 1) * elem_per_thread, tn],
                        dense_local,
                    )

                    for g_i in range(elem_per_thread // group):
                        T.clear(non_zero_elt_log_idx)

                        local_idx = g_i * group

                        max1_idx = T.alloc_var(dtype=T.uint8)
                        max1_val = T.alloc_var(dtype=dtype)
                        max2_idx = T.alloc_var(dtype=T.uint8)
                        max2_val = T.alloc_var(dtype=dtype)

                        max1_idx = 0
                        max1_val = T.abs(dense_local[local_idx])
                        max2_idx = 1
                        max2_val = T.abs(dense_local[local_idx + 1])

                        if max2_val > max1_val:
                            tmp_idx = max1_idx
                            tmp_val = max1_val
                            max1_idx = max2_idx
                            max1_val = max2_val
                            max2_idx = tmp_idx
                            max2_val = tmp_val

                        for i in range(2):
                            j = i + 2
                            val = T.abs(dense_local[local_idx + j])
                            if val > max1_val:
                                max2_idx = max1_idx
                                max2_val = max1_val
                                max1_idx = j
                                max1_val = val
                            elif val > max2_val:
                                max2_idx = j
                                max2_val = val

                        non_zero_elt_log_idx[0] = T.min(max1_idx, max2_idx)
                        non_zero_elt_log_idx[1] = T.max(max1_idx, max2_idx)
                        sparse_local[local_idx // 2] = dense_local[local_idx + non_zero_elt_log_idx[0]]
                        sparse_local[local_idx // 2 + 1] = dense_local[local_idx + non_zero_elt_log_idx[1]]

                        for i in T.serial(elem):
                            val = non_zero_elt_log_idx[i]
                            meta_local[local_idx // e_factor] |= T.shift_left(
                                val.astype(meta_dtype), (4 * (g_i % (e_factor // group)) + 2 * i).astype(meta_dtype)
                            )
                    T.copy(
                        sparse_local,
                        sparse_nonzero[
                            bx, hx, sparse_idx, tm * (elem_per_thread // 2) : (tn + 1) * (elem_per_thread // 2), tn
                        ],
                    )
                    T.copy(
                        meta_local,
                        sparse_meta[
                            bx,
                            hx,
                            sparse_idx,
                            tm * (elem_per_thread // e_factor) : (tm + 1) * (elem_per_thread // e_factor),
                            tn,
                        ],
                    )

    return block_compress_sparse_value_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
    }
)
def block_compress_dense(b, hn, block_s, hd, threads=128):
    dtype = T.float16
    idx_dtype = T.int16
    s = T.dynamic("s")
    blocks_per_head = T.dynamic("blocks_per_head")
    dense_blocks_per_head = T.dynamic("dense_blocks_per_head")

    @T.prim_func
    def block_compress_dense_kernel(
        K: T.Tensor([b, hn, s, hd], dtype),
        block_mask: T.Tensor([b, hn, blocks_per_head], T.bool),
        idx_map: T.Tensor([b, hn, blocks_per_head], idx_dtype),
        dense: T.Tensor([b, hn, dense_blocks_per_head, block_s, hd], dtype),
    ):
        with T.Kernel(b, hn, threads=threads) as (bx, hx):
            dense_count = T.alloc_var(idx_dtype)
            block_shared = T.alloc_shared([block_s, hd], dtype)
            dense_count = 0

            for k in T.Pipelined(blocks_per_head, num_stages=0):
                mask_val = block_mask[bx, hx, k]

                if mask_val:
                    T.copy(K[bx, hx, k * block_s : (k + 1) * block_s, :], block_shared)
                    dense_idx = dense_count
                    dense_count += 1
                    idx_map[bx, hx, k] = dense_count
                    T.copy(block_shared, dense[bx, hx, dense_idx, :, :])

    return block_compress_dense_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def prune_block_key_mask(b, hn, block_s, hd, elem_per_thread=32):
    dtype = T.float16
    accum_dtype = T.float32
    group = 4
    num_blocks = T.dynamic("num_blocks")

    @T.prim_func
    def prune_block_key_mask_kernel(
        K: T.Tensor([b, hn, num_blocks, block_s, hd], dtype),
        prune_loss: T.Tensor([b, hn, num_blocks], dtype),
    ):
        with T.Kernel(b, hn, num_blocks, threads=(block_s, hd // elem_per_thread)) as (bx, by, bz):
            tm, tn = T.get_thread_binding(0), T.get_thread_binding(1)
            dense_local = T.alloc_local([elem_per_thread], dtype)
            loss_local = T.alloc_var(dtype=accum_dtype)
            loss_shared = T.alloc_shared([1], accum_dtype)

            loss_local = 0.0

            if tn == 0 and tm == 0:
                loss_shared[0] = 0.0

            T.sync_threads()

            T.copy(K[bx, by, bz, tm, tn * elem_per_thread : (tn + 1) * elem_per_thread], dense_local)
            for g_i in range(elem_per_thread // group):

                local_idx = g_i * group

                min1_val = T.alloc_var(dtype=dtype)
                min2_val = T.alloc_var(dtype=dtype)

                min1_val = T.abs(dense_local[local_idx])
                min2_val = T.abs(dense_local[local_idx + 1])

                if min2_val < min1_val:
                    tmp_val = min1_val
                    min1_val = min2_val
                    min2_val = tmp_val

                for i in range(2):
                    j = i + 2
                    val = T.abs(dense_local[local_idx + j])
                    if val < min1_val:
                        min2_val = min1_val
                        min1_val = val
                    elif val < min2_val:
                        min2_val = val

                loss_local += min1_val + min2_val
            T.atomic_add(loss_shared[0], loss_local)
            T.sync_threads()
            if tm == 0 and tn == 0:
                prune_loss[bx, by, bz] = loss_shared[0]

    return prune_block_key_mask_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        # tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    }
)
def prune_block_value_mask(b, hn, block_s, hd, elem_per_thread=32):
    dtype = T.float16
    accum_dtype = T.float32
    group = 4
    num_blocks = T.dynamic("num_blocks")

    @T.prim_func
    def prune_block_value_mask_kernel(
        V: T.Tensor([b, hn, num_blocks, block_s, hd], dtype),
        prune_loss: T.Tensor([b, hn, num_blocks], dtype),
    ):
        with T.Kernel(b, hn, num_blocks, threads=(block_s // elem_per_thread, hd)) as (bx, by, bz):
            tm, tn = T.get_thread_binding(0), T.get_thread_binding(1)
            dense_local = T.alloc_local([elem_per_thread], dtype)
            loss_local = T.alloc_var(dtype=accum_dtype)
            loss_shared = T.alloc_shared([1], accum_dtype)

            loss_local = 0.0

            if tn == 0 and tm == 0:
                loss_shared[0] = 0.0

            T.sync_threads()

            T.copy(V[bx, by, bz, tm * elem_per_thread : (tm + 1) * elem_per_thread, tn], dense_local)
            for g_i in range(elem_per_thread // group):

                local_idx = g_i * group

                min1_val = T.alloc_var(dtype=dtype)
                min2_val = T.alloc_var(dtype=dtype)

                min1_val = T.abs(dense_local[local_idx])
                min2_val = T.abs(dense_local[local_idx + 1])

                if min2_val < min1_val:
                    tmp_val = min1_val
                    min1_val = min2_val
                    min2_val = tmp_val

                for i in range(2):
                    j = i + 2
                    val = T.abs(dense_local[local_idx + j])
                    if val < min1_val:
                        min2_val = min1_val
                        min1_val = val
                    elif val < min2_val:
                        min2_val = val

                loss_local += min1_val + min2_val
            T.atomic_add(loss_shared[0], loss_local)
            T.sync_threads()
            if tm == 0 and tn == 0:
                prune_loss[bx, by, bz] = loss_shared[0]

    return prune_block_value_mask_kernel
