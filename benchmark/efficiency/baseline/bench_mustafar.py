import math

import compression
import mustafar_package
import torch
import torch.nn.functional as F
import triton
from torch.nn.attention import SDPBackend, sdpa_kernel

shared_reduction_workspace = torch.zeros(1, dtype=torch.float16, device="cuda")


def assert_close(a, b, atol=1e-3):
    max_diff = (a - b).abs().max()
    assert max_diff <= atol, f"{max_diff=}, allow {atol}"


def mustafar_key_prune(K, k_sparsity):
    B, H, T, D = K.shape
    num_to_keep = max(1, int((k_sparsity) * D))

    key_states_flat = K.reshape(-1, D)

    threshold_values, _ = torch.kthvalue(torch.abs(key_states_flat), num_to_keep, dim=-1, keepdim=True)

    mask = torch.abs(key_states_flat) >= threshold_values

    pruned_key_states = key_states_flat * mask

    return pruned_key_states.view(B, H, T, D)


def mustafar_value_prune(V, v_sparsity):
    B, H, T, D = V.shape

    num_to_keep = max(1, int((v_sparsity) * D))

    value_states_flat = V.reshape(-1, D)

    threshold_values, _ = torch.kthvalue(torch.abs(value_states_flat), num_to_keep, dim=-1, keepdim=True)

    mask = torch.abs(value_states_flat) >= threshold_values

    pruned_value_states = value_states_flat * mask

    return pruned_value_states.view(B, H, T, D)


def mustafar_key_compress(K):
    b, groups, s, d = K.shape
    total_batch_kv = b * groups
    k_bmps, k_idxs, k_nzs = compression.convert_key_batched(K.reshape(total_batch_kv, s, d))
    k_nz_offset = torch.zeros(total_batch_kv, dtype=torch.int32, device=K.device)
    for i in range(1, total_batch_kv):
        k_nz_offset[i] = k_nz_offset[i - 1] + k_idxs[i - 1][-1] // 4
    return k_bmps, k_idxs, torch.cat(k_nzs), k_nz_offset


def mustafar_value_compress(V):
    b, groups, s, d = V.shape
    total_batch_kv = b * groups
    v_bmps, v_idxs, v_nzs = compression.convert_value_batched(V.reshape(total_batch_kv, s, d))
    v_nz_offset = torch.zeros(total_batch_kv, dtype=torch.int32, device=V.device)
    for i in range(1, total_batch_kv):
        v_nz_offset[i] = v_nz_offset[i - 1] + v_idxs[i - 1][-1] // 4
    return v_bmps, v_idxs, torch.cat(v_nzs), v_nz_offset


def mustafar_attn(Q, K_packed, V_packed, seq_kv, groups):
    global shared_reduction_workspace
    batch, heads, seq_q, dim = Q.shape
    assert seq_q == 1

    k_bmps, k_idxs, k_nzs, k_nz_offset = K_packed
    v_bmps, v_idxs, v_nzs, v_nz_offset = V_packed

    total_batch_size = batch * heads

    padded_query = F.pad(Q.view(total_batch_size, -1, dim), (0, 0, 0, 7), mode="constant", value=0)
    attn_weights = mustafar_package.mustafar_key_formulation(
        k_bmps, k_nzs, k_idxs, k_nz_offset, padded_query, seq_kv, dim, total_batch_size, heads // groups
    )
    attn_weights = attn_weights[:, 0:1, :].view(batch, heads, 1, seq_kv)

    attn_weights = attn_weights / math.sqrt(dim)
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(Q.dtype)

    padded_score = F.pad(attn_weights[:, :, :, :].view(total_batch_size, -1, seq_kv), (0, 0, 0, 7)).contiguous()
    O = mustafar_package.mustafar_value_formulation(
        v_bmps,
        v_nzs,
        v_idxs,
        v_nz_offset,
        padded_score,
        shared_reduction_workspace,
        dim,
        seq_kv,
        total_batch_size,
        heads // groups,
    )
    O = O[:, 0:1, :].view(batch, heads, 1, dim)
    return O


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def unfused(Q, K, V):
    K = repeat_kv(K, Q.size(1) // K.size(1))
    V = repeat_kv(V, Q.size(1) // V.size(1))
    attn_weights = torch.matmul(Q, K.transpose(-2, -1))
    attn_weights = attn_weights / math.sqrt(Q.size(-1))
    attn_weights = F.softmax(attn_weights, dim=-1)
    O = torch.matmul(attn_weights, V)
    return O


def main(args):
    batch = args.batch
    heads = args.heads
    groups = args.groups
    seq_kv = args.seq_kv
    dim = args.dim

    Q = torch.randn([batch, heads, 1, dim], dtype=torch.float16, device="cuda")
    K = torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda")
    V = torch.randn([batch, groups, seq_kv, dim], dtype=torch.float16, device="cuda")

    K_bytes = K.numel() * K.element_size()
    V_bytes = V.numel() * V.element_size()
    k_prune_latency = triton.testing.do_bench(
        lambda: mustafar_key_prune(K, args.key_sparsity),
        warmup=args.warmup,
        rep=args.rep,
    )
    v_prune_latency = triton.testing.do_bench(
        lambda: mustafar_value_prune(V, args.value_sparsity),
        warmup=args.warmup,
        rep=args.rep,
    )

    print(f"Key prune: {k_prune_latency:.2f} ms {(K_bytes / 1024 ** 3) / (k_prune_latency / 1e3):.2f} GiB/s")
    print(f"Value prune: {v_prune_latency:.2f} ms {(V_bytes / 1024 ** 3) / (k_prune_latency / 1e3):.2f} GiB/s")

    K = mustafar_key_prune(K, args.key_sparsity)
    V = mustafar_value_prune(V, args.value_sparsity)

    k_compress_latency = triton.testing.do_bench(
        lambda: mustafar_key_compress(K),
        warmup=args.warmup,
        rep=args.rep,
    )

    v_compress_latency = triton.testing.do_bench(
        lambda: mustafar_value_compress(V),
        warmup=args.warmup,
        rep=args.rep,
    )

    print(f"Key compress: {k_compress_latency:.2f} ms {(K_bytes / 1024 ** 3) / (k_compress_latency / 1e3):.2f} GiB/s")
    print(f"Value compress: {v_compress_latency:.2f} ms {(V_bytes / 1024 ** 3) / (v_compress_latency / 1e3):.2f} GiB/s")

    k_bmps, k_idxs, k_nzs, k_nz_offset = mustafar_key_compress(K)
    v_bmps, v_idxs, v_nzs, v_nz_offset = mustafar_value_compress(V)

    compressed_key_bytes = (
        k_bmps.numel() * k_bmps.element_size()
        + k_idxs.numel() * k_idxs.element_size()
        + k_nzs.numel() * k_nzs.element_size()
        + k_nz_offset.numel() * k_nz_offset.element_size()
    )
    compressed_value_bytes = (
        v_bmps.numel() * v_bmps.element_size()
        + v_idxs.numel() * v_idxs.element_size()
        + v_nzs.numel() * v_nzs.element_size()
        + v_nz_offset.numel() * v_nz_offset.element_size()
    )

    print(f"Key compression rate: {compressed_key_bytes / K_bytes*100:.2f}%")
    print(f"Value compression rate: {compressed_value_bytes / V_bytes*100:.2f}%")

    flops_per_matmul = 2.0 * batch * heads * 1 * seq_kv * dim
    total_flops = 2 * flops_per_matmul

    O = mustafar_attn(Q, (k_bmps, k_idxs, k_nzs, k_nz_offset), (v_bmps, v_idxs, v_nzs, v_nz_offset), seq_kv, groups)
    latency = triton.testing.do_bench(
        lambda: mustafar_attn(
            Q, (k_bmps, k_idxs, k_nzs, k_nz_offset), (v_bmps, v_idxs, v_nzs, v_nz_offset), seq_kv, groups
        ),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"Mustafar: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops")

    latency = triton.testing.do_bench(
        lambda: unfused(Q, K, V),
        warmup=args.warmup,
        rep=args.rep,
    )
    print(f"Unfused: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops")

    if args.check:
        O_ref = unfused(Q, K, V)
        assert_close(O, O_ref)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        latency = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=False, enable_gqa=True),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(f"CUDNN: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops")
        if args.check:
            O_spda = F.scaled_dot_product_attention(Q, K, V, is_causal=False, enable_gqa=True)
            assert_close(
                O_spda,
                O,
            )

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        latency = triton.testing.do_bench(
            lambda: F.scaled_dot_product_attention(Q, K, V, is_causal=False, enable_gqa=True),
            warmup=args.warmup,
            rep=args.rep,
        )
        print(f"FLASH_ATTENTION: {latency:.2f} ms {total_flops / latency * 1e-9:.2f} TFlops")
        if args.check:
            O_fa2 = F.scaled_dot_product_attention(Q, K, V, is_causal=False, enable_gqa=True)
            assert_close(
                O_fa2,
                O,
            )

    if args.check:
        print("Precision checking passed")

    print(f"Peak mem: {torch.cuda.max_memory_allocated() / 1024 ** 3} GiB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--seq_kv", type=int, default=32768)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rep", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--key_sparsity", type=float, default=0.5)
    parser.add_argument("--value_sparsity", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
