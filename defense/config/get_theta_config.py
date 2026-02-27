import argparse
import json
from pathlib import Path
import torch
from transformers.cache_utils import DynamicCache


def analyze_kv_cache(kvcache: DynamicCache):
    """
    Analyze DynamicCache to find the maximum absolute value for each layer and attention head.

    Args:
        kvcache (DynamicCache): A list where each element is a tuple containing
                               a Key Cache and a Value Cache tensor.
                               Tensor dimensions should be (batch_size, num_heads, sequence_length, head_dim).
    """
    analysis_results = {}

    # Iterate through each layer
    for i, (key_cache, value_cache) in enumerate(kvcache):
        layer_key = f"layer_{i}"
        analysis_results[layer_key] = {}

        # --- Analyze Key Cache ---
        abs_key_cache = torch.abs(key_cache)
        max_values_per_head_key = torch.amax(abs_key_cache, dim=(0, 2, 3))
        analysis_results[layer_key]["key_max_values"] = max_values_per_head_key.tolist()

        # --- Analyze Value Cache ---
        abs_value_cache = torch.abs(value_cache)
        max_values_per_head_value = torch.amax(abs_value_cache, dim=(0, 2, 3))
        # Convert to list and store
        analysis_results[layer_key][
            "value_max_values"
        ] = max_values_per_head_value.tolist()

    return analysis_results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze KV-cache and generate theta config."
    )
    parser.add_argument("--model-name", default="Llama-3.2-1B")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Direct path to past_key_values.pt. If not set, constructed from other arguments.",
    )
    parser.add_argument("--cache-root", default="cache")
    parser.add_argument(
        "--cache-layout",
        default="config",
        choices=["config", "dataset"],
        help="Use cache/<dtype>/config/<model>/... or cache/<dtype>/<dataset>/<model>/...",
    )
    parser.add_argument("--dataset-name", default="lmsys-chat-1m_1k")
    parser.add_argument(
        "--input-hash",
        default="301f7f48573352226c8b86de2a7eb654e9fef28b",
    )
    parser.add_argument("--protect-type", default="origin")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output theta json path. Defaults to defense/config/kvcloak/theta/<model>.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.serialization.add_safe_globals([DynamicCache, set])

    dtype_name = args.dtype
    if args.cache_path is not None:
        kvcache_path = Path(args.cache_path).expanduser()
    elif args.cache_layout == "config":
        kvcache_path = Path(
            f"{args.cache_root}/{dtype_name}/config/{args.model_name}/{args.input_hash}/{args.protect_type}/past_key_values.pt"
        ).expanduser()
    else:
        kvcache_path = Path(
            f"{args.cache_root}/{dtype_name}/{args.dataset_name}/{args.model_name}/{args.input_hash}/{args.protect_type}/past_key_values.pt"
        ).expanduser()

    if not kvcache_path.exists():
        raise FileNotFoundError(f"KV-Cache not found at {kvcache_path}")

    try:
        kvcache = torch.load(kvcache_path, weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"Error loading KV-Cache for {args.model_name} from {kvcache_path}: {e}"
        ) from e

    analysis_results = analyze_kv_cache(kvcache)
    output_filename = (
        Path(args.output_path).expanduser()
        if args.output_path is not None
        else Path(f"defense/config/kvcloak/theta/{args.model_name}.json")
    )
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(output_filename, "w") as f:
        json.dump(analysis_results, f, indent=4)

    print(f"KV Cache analysis completed, results saved to '{output_filename}'")


if __name__ == "__main__":
    main()
