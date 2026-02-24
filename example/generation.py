import argparse
import os
from time import time

import torch
from transformers import AutoTokenizer

from hierasparse.caches.compressed_cache import (
    HieraSparseCache,
    HieraSparseDecodeCache,
    PrefillKVDecodeKVCache,
    PrefillVDecodeKVCache,
    PrefillVDecodeVCache,
)
from hierasparse.caches.simulator_cache import (
    DenseCache,
    FlashAttnSPSimulationCache,
    HieraSparseSimulationCache,
)
from hierasparse.models import get_model_cls

torch.manual_seed(42)
torch.set_printoptions(threshold=100000)


def visualize_all_heads_cache(
    past_key_values,
    layer_ids,
    visualization_path,
    head_id=None,
    draw_zero=True,
):
    import matplotlib.pyplot as plt
    import numpy as np

    # IEEE Style compliance
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2,
            "lines.markersize": 8,
        }
    )

    has_query = hasattr(past_key_values, "query_cache")
    os.makedirs(visualization_path, exist_ok=True)

    for layer_id in layer_ids:
        if has_query:
            query_cache = past_key_values.query_cache[layer_id].abs()
        key_cache = past_key_values.key_cache[layer_id].abs()  # [B, H, T, D]
        value_cache = past_key_values.value_cache[layer_id].abs()

        if has_query:
            q = query_cache.squeeze(0).detach().cpu()
        k = key_cache.squeeze(0).detach().cpu()
        v = value_cache.squeeze(0).detach().cpu()

        if head_id is not None:
            if has_query:
                q = q[head_id : head_id + 1, :, :]
            k = k[head_id : head_id + 1, :, :]
            v = v[head_id : head_id + 1, :, :]

        if has_query:
            q = q.permute(1, 0, 2)  # (seq_len, num_heads, head_dim)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)

        seq_len, num_groups, head_dim = k.shape

        if has_query:
            q = q.reshape(seq_len, q.shape[1] * head_dim).numpy()
        k = k.reshape(seq_len, num_groups * head_dim).numpy()
        v = v.reshape(seq_len, num_groups * head_dim).numpy()

        head_centers = [i * head_dim + head_dim / 2 for i in range(num_groups)]

        tensors_to_plot = [q, k, v] if has_query else [k, v]
        names = ["q", "k", "v"] if has_query else ["k", "v"]

        for idx, (tensor, kv_name) in enumerate(zip(tensors_to_plot, names)):
            tokens, channels = tensor.shape

            x = np.arange(channels)
            y = np.arange(tokens)
            X, Y = np.meshgrid(x, y)

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

            ax.view_init(elev=25, azim=-135)

            surf = ax.plot_surface(X, Y, tensor, cmap="coolwarm", antialiased=True, alpha=0.9)

            ax.set_xticks(head_centers)
            ax.set_xticklabels([f"H{i}" for i in range(num_groups)], fontsize=14)
            ax.tick_params(axis="y", labelsize=15)
            ax.tick_params(axis="z", labelsize=15)

            # Use standard z-label but with more padding and correct rotation for 3D view
            # If the view is elev=25, azim=-135, the z-axis is on the left.
            # ax.set_zlabel("Absolute Value", labelpad=20, fontsize=12, rotation=90)

            # Remove manual text2D
            # ax.text2D(0.05, 0.95, "Absolute Value", transform=ax.transAxes, fontsize=12, rotation=0)

            ax.set_xlabel("Head × Dim", labelpad=10, fontsize=18)
            ax.set_ylabel("Token", labelpad=10, fontsize=18)

            # Adjust subplot parameters to give more space on the left
            # Move the plot slightly to the right by adjusting left margin
            # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
            # Or use tight_layout with padding
            # plt.tight_layout(pad=3.0)
            # But specific margins give more control "increase canvas a bit left"

            if draw_zero:
                z_floor = -0.5 if kv_name != "q" else -5.0

                zero_mask = tensor == 0
                zy, zx = np.where(zero_mask)
                ax.scatter(zx, zy, np.full_like(zx, z_floor), marker=".", c="black", s=1, alpha=0.5)

                nz_mask = tensor != 0
                nzy, nzx = np.where(nz_mask)
                ax.scatter(nzx, nzy, np.full_like(nzx, z_floor), marker=".", c="red", s=1, alpha=0.3)

            save_filename = f"{visualization_path}/layer{layer_id}_{kv_name}.png"
            plt.savefig(save_filename, bbox_inches="tight", dpi=300)
            plt.close(fig)


def visualize_key_value_distribution(past_key_values, layer_ids, save_dir, bins=100):
    import matplotlib.pyplot as plt
    import numpy as np

    # IEEE Style compliance
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 12,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "lines.linewidth": 2,
            "lines.markersize": 8,
        }
    )

    os.makedirs(save_dir, exist_ok=True)

    for layer_id in layer_ids:
        fig, axs = plt.subplots(2, 1, figsize=(4, 6), gridspec_kw={"hspace": 0.15})

        for idx, (cache_name, cache) in enumerate(
            [
                ("key", past_key_values.key_cache[layer_id]),
                ("value", past_key_values.value_cache[layer_id]),
            ]
        ):
            ax = axs[idx]

            # cache shape: [B, H, T, D]
            # Convert to float32
            data = cache.float().abs().cpu().numpy().flatten()

            print(f"Layer {layer_id} {cache_name} abs mean: {cache.float().abs().mean().item():.6f}")

            # --- Set Max Value per user request ---
            clip_max = 10.0 if cache_name == "key" else 1.25

            # Filter NaNs/Infs and apply clip for x-axis range (hist/plot) visualization
            # But for cumulative stats, we should use ALL data to be correct.
            # However, histogram binning needs to respect the view range.

            if np.isnan(data).any() or np.isinf(data).any():
                data = data[np.isfinite(data)]

            # Keep original data for accurate stats
            stats_data = data

            # For visualization, we can limit the range
            # But the histogram density must be correct with respect to all data or just visible?
            # Usually density=True means integral is 1 over the whole range.
            # If we just cut off the plot, density is still correct.

            color_hist = "tab:blue"
            color_cum = "tab:orange"

            # --- Left Axis: Histogram ---
            # ax.set_xlabel("Absolute Value", fontsize=10, fontweight="bold")
            # ax.set_ylabel("Density", color=color_hist, fontsize=10, fontweight="bold")

            # Use range to limit histogram calculation/display
            # Use weights to sum to 1 (Probability) instead of integral to 1 (Density)
            weights = np.ones_like(data) / len(data)
            ax.hist(
                data, bins=bins, weights=weights, range=(0, clip_max), alpha=0.6, color=color_hist, label="Probability"
            )

            ax.set_xlim(0, clip_max)
            ax.tick_params(axis="y", labelcolor=color_hist, labelsize=11)
            ax.tick_params(axis="x", labelsize=11)

            # --- Right Axis: Average Prune Loss (Absolute) ---
            ax_cum_axis = ax.twinx()
            # ax_cum_axis.set_ylabel("Avg Pruned Value", color=color_cum, fontsize=10, fontweight="bold")

            # Calculate Average Prune Loss (Accumulate low to high)
            # Sort data ascending (Low -> High)
            sorted_data = np.sort(stats_data)
            cum_sum = np.cumsum(sorted_data)
            counts = np.arange(1, len(sorted_data) + 1)
            cum_mean = cum_sum / counts

            # Plot (Value, Average Absolute Value).
            # We only plot up to clip_max on X-axis
            ax_cum_axis.plot(sorted_data, cum_mean, color=color_cum, linewidth=2, label="Avg Pruned Value")
            ax_cum_axis.tick_params(axis="y", labelcolor=color_cum, labelsize=11)

            ax.set_title(f"Layer {layer_id} {cache_name.capitalize()}", fontsize=11, fontweight="bold")

            # Mark 50% element count cutoff (Pruning bottom 50%)
            idx_50 = int(len(stats_data) * 0.5)
            val_50 = sorted_data[idx_50]
            mean_50 = cum_mean[idx_50]

            # Draw vertical line at the Median value (only if within range)
            ax.axvline(val_50, color="red", linestyle="--", alpha=0.8, label=f"50% Cutoff")

            # Annotate the Average Value at that point
            ax_cum_axis.plot(val_50, mean_50, "o", color="red")
            ax_cum_axis.annotate(
                f"Prune 50%\nAvg Loss: {mean_50:.2f}",
                xy=(val_50, mean_50),
                xytext=(50, 0),  # Shift text further right to avoid overlap
                textcoords="offset points",
                arrowprops=dict(facecolor="red", shrink=0.05, width=1, headwidth=5),
                color="red",
                verticalalignment="center",
                horizontalalignment="left",
                fontsize=14,  # Smaller font
                fontweight="bold",
            )

        # Common labels
        fig.text(0.5, 0.01, "Absolute Value", ha="center", fontsize=13, fontweight="bold")
        fig.text(
            0.0, 0.5, "Density", va="center", rotation="vertical", color=color_hist, fontsize=12, fontweight="bold"
        )
        fig.text(
            0.94,
            0.5,
            "Avg Pruned Loss",
            va="center",
            rotation="vertical",
            color=color_cum,
            fontsize=13,
            fontweight="bold",
        )

        plt.subplots_adjust(hspace=0.35, left=0.15, right=0.82, top=0.95, bottom=0.08)
        save_path = os.path.join(save_dir, f"layer_{layer_id}_kv_distribution.png")
        plt.savefig(save_path)  # bbox_inches="tight" might override subplots_adjust, so careful
        # plt.close(fig) -- Close explicit figure if needed, here 'fig' is available
        plt.close()
        print(f"Saved Layer {layer_id} KV distribution to {save_path}")


def calculate_cache_sparsity(past_key_values):
    key_sparsity = []
    for key_cache in past_key_values.key_cache:
        zero_elements = (key_cache == 0).sum().item()
        total_elements = key_cache.numel()
        sparsity = zero_elements / total_elements
        key_sparsity.append(sparsity)
    value_sparsity = []
    for value_cache in past_key_values.value_cache:
        zero_elements = (value_cache == 0).sum().item()
        total_elements = value_cache.numel()
        sparsity = zero_elements / total_elements
        value_sparsity.append(sparsity)
    key_sparsity = sum(key_sparsity) / len(key_sparsity)
    value_sparsity = sum(value_sparsity) / len(value_sparsity)
    return key_sparsity, value_sparsity


def main(args):
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    MODEL_CLS = get_model_cls(args.model_name)
    attn_implementation = "flash_attention_2"

    if "llama-2" in args.model_name.lower():
        prompt = f"[INST]{args.prompt}[/INST]"
    else:
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": args.prompt,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = input_ids["input_ids"].shape[-1]

    if args.cache == "dense":
        past_key_values = DenseCache(cache_query=True)
    elif args.cache == "simu_flashattn_sp":
        past_key_values = FlashAttnSPSimulationCache(
            prune_key_prefill=args.prune_key_prefill,
            prune_key_decode=args.prune_key_decode,
            prune_value_prefill=args.prune_value_prefill,
            prune_value_decode=args.prune_value_decode,
            sink=args.sink,
            local_window=args.local_window,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
        )
    elif args.cache == "simu_blockattn_sp":
        past_key_values = HieraSparseSimulationCache(
            prune_key_prefill=args.prune_key_prefill,
            prune_key_decode=args.prune_key_decode,
            prune_value_prefill=args.prune_value_prefill,
            prune_value_decode=args.prune_value_decode,
            prune_key_prefill_ratio=args.prune_key_prefill_ratio,
            prune_value_prefill_ratio=args.prune_value_prefill_ratio,
            block_seq_size=args.block_seq_size,
            sink=args.sink,
            local_window=args.local_window,
            start_layer=args.start_layer,
        )
    elif args.cache == "best_perf":
        attn_implementation = PrefillKVDecodeKVCache.ATTN_IMPLEMENTATION
        past_key_values = PrefillKVDecodeKVCache(
            sink=args.sink,
            local_window=args.local_window,
        )
    elif args.cache == "balanced":
        attn_implementation = PrefillVDecodeKVCache.ATTN_IMPLEMENTATION
        past_key_values = PrefillVDecodeKVCache(
            sink=args.sink,
            local_window=args.local_window,
        )
    elif args.cache == "best_acc":
        attn_implementation = PrefillVDecodeVCache.ATTN_IMPLEMENTATION
        past_key_values = PrefillVDecodeVCache(
            sink=args.sink,
            local_window=args.local_window,
        )
    elif args.cache == "hierasparse":
        attn_implementation = HieraSparseCache.ATTN_IMPLEMENTATION
        past_key_values = HieraSparseCache(
            sink=args.sink,
            local_window=args.local_window,
            block_size=args.block_seq_size,
            key_prune_ratio=args.prune_key_prefill_ratio,
            value_prune_ratio=args.prune_value_prefill_ratio,
        )

    elif args.cache == "hierasparse_decode":
        attn_implementation = HieraSparseDecodeCache.ATTN_IMPLEMENTATION
        past_key_values = HieraSparseDecodeCache(
            sink=args.sink,
            local_window=args.local_window,
            block_size=args.block_seq_size,
            key_prune_ratio=args.prune_key_prefill_ratio,
            value_prune_ratio=args.prune_value_prefill_ratio,
        )
    else:
        raise ValueError(f"Unknown cache type: {args.cache}")

    model = MODEL_CLS.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        attn_implementation=attn_implementation,
        device_map="cuda",
    )

    print(f"\n=== {input_len} input tokens ===\n{prompt}\n")
    print("====================================\n")
    print("Generating...")
    start = time()
    output = model.generate(
        **input_ids,
        do_sample=False,
        past_key_values=past_key_values,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    print(f"Generation took {time() - start:.2f} seconds")  # no need to sync here
    output = output[0][input_len:]

    output_text = tokenizer.decode(output, skip_special_tokens=False)
    print(f"\n=== {len(output)} output tokens ===\n{output_text}\n")
    print("====================================\n")

    if "simu" in args.cache:
        key_sparsity, value_sparsity = calculate_cache_sparsity(past_key_values)
        print(f"Key cache sparsity: {key_sparsity*100:.2f}%")
        print(f"Value cache sparsity: {value_sparsity*100:.2f}%")
    else:
        key_bytes, value_bytes = past_key_values.memory_usage_bytes()
        print(f"Key cache size: {key_bytes / 1024 ** 2} MiB")
        print(f"value_bytes cache size: {value_bytes / 1024 ** 2} MiB")

    if len(args.visualize_layers) > 0:
        visualize_all_heads_cache(
            past_key_values,
            layer_ids=args.visualize_layers,
            visualization_path=args.visualization_path,
            draw_zero=False,
        )

        visualize_key_value_distribution(
            past_key_values, layer_ids=args.visualize_layers, save_dir=args.visualization_path
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLaMA-style model text generation with optional sparse cache.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name or path")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to generate text from",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default="dense",
        choices=[
            "dense",
            "simu_flashattn_sp",
            "simu_blockattn_sp",
            "best_perf",
            "balanced",
            "best_acc",
            "hierasparse",
            "hierasparse_decode",
        ],
        help="Cache type to use",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature")
    parser.add_argument("--prune_key_prefill", action="store_true", help="Enable sparse key cache")
    parser.add_argument("--prune_key_decode", action="store_true", help="Enable sparse key cache during decoding")
    parser.add_argument("--prune_value_prefill", action="store_true", help="Enable sparse value cache")
    parser.add_argument("--prune_value_decode", action="store_true", help="Enable sparse value cache during decoding")
    parser.add_argument("--start_layer", type=int, default=0, help="Starting layer for pruning")
    parser.add_argument("--end_layer", type=int, default=32, help="Ending layer for pruning")
    parser.add_argument("--sink", type=int, default=0, help="Sink parameter for pruning")
    parser.add_argument("--local_window", type=int, default=0, help="Local window size for pruning")
    parser.add_argument("--block_seq_size", type=int, default=64, help="Block size for pruning")
    parser.add_argument("--prune_key_prefill_ratio", type=float, default=0, help="Block pruning ratio")
    parser.add_argument("--prune_value_prefill_ratio", type=float, default=0, help="Block pruning ratio")
    parser.add_argument(
        "--visualize_layers", type=int, nargs="+", default=[], help="Layers to visualize (default: only layer 0)"
    )
    parser.add_argument("--visualization_path", type=str, default="./figures", help="Directory to save visualizations")
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = r"""CBC had originally decided that none of its rebroadcasters will transition to digital. Also, the CBC had originally planned to not convert any non-originating stations in mandatory markets to digital, which would have forced CBRFT in Calgary and CBXFT-3 in Lethbridge to sign off on the transition date. (Lloydminster, another mandatory market, had no local Radio-Canada transmitter.) On August 16, 2011, the Canadian Radio-television and Telecommunications Commission (CRTC) granted the CBC permission to continue operating 22 repeaters in mandatory markets, including CBRFT and CBXFT-3, in analog until August 31, 2012, by which time the transmitters had to convert to digital or shut down. The remaining transmitters were shut down in 2012."""
    with torch.no_grad():
        main(args)
