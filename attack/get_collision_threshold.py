"""
Generate collision+ (Chosen-Plaintext Attack) threshold configuration.

This script combines statistic.py and analysis.py to generate the distance
distribution configuration for collision+ attack using the specific "bitter lesson"
input text.

Usage:
    python attack/get_collision_threshold.py \
        --model_path ~/model/Llama-3.2-1B \
        --target_data_path cache/float32/config/Llama-3.2-1B/<hash>/origin/past_key_values.pt \
        --device cuda:0 \
        --dtype float32
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Tuple


def get_bos_token_ids(config: AutoConfig) -> List[int]:
    """Extract BOS token IDs from config."""
    bos_token_id = getattr(config, "bos_token_id", None)
    if bos_token_id is None:
        return []
    if isinstance(bos_token_id, int):
        return [bos_token_id]
    return bos_token_id


def needs_bos_padding(inputs: Dict[str, Tensor], bos_token_ids: List[int]) -> bool:
    """Check if BOS padding is needed."""
    existing_bos = inputs["input_ids"][0, : len(bos_token_ids)].tolist()
    return existing_bos != bos_token_ids


def pad_bos_token(inputs: Dict[str, Tensor], bos_token_ids: List[int]):
    """Prepend BOS token to inputs."""
    bos_tensor = torch.tensor([bos_token_ids], dtype=torch.long)
    inputs["input_ids"] = torch.cat([bos_tensor, inputs["input_ids"]], dim=1)
    inputs["attention_mask"] = torch.cat(
        [torch.ones(1, len(bos_token_ids), dtype=torch.long), inputs["attention_mask"]],
        dim=1,
    )


def statistic_distance(
    model: AutoModelForCausalLM,
    target_datas: Any,
    target: str,
    inputs: Dict[str, Tensor],
    batch_size: int,
    target_dist_dir: Path,
    gap: int = 100,
):
    """Calculate distance statistics for each position."""
    device = model.device
    target_dist_dir.mkdir(parents=True, exist_ok=True)

    all_ids = torch.arange(model.config.vocab_size, device=device)
    seq_length = inputs.input_ids.shape[1]

    current_kvcache = None
    target_data_dists = []

    for seq_id in tqdm(range(seq_length), desc="Calculating Distance"):
        sorted_ids = all_ids.cpu().tolist()

        if target == "past_key_values":
            target_data_dist = [
                [torch.tensor([], device=device) for _ in range(len(target_datas))],
                [torch.tensor([], device=device) for _ in range(len(target_datas))],
            ]
        elif target == "hidden_states":
            target_data_dist = [
                [torch.tensor([], device=device) for _ in range(len(target_datas))]
            ]

        for batch_start in range(0, len(sorted_ids), batch_size):
            batch_ids = sorted_ids[batch_start : batch_start + batch_size]
            if not batch_ids:
                continue

            input_batch = torch.tensor(batch_ids, device=device).unsqueeze(1)

            # Build attention mask
            if current_kvcache is not None and len(current_kvcache.key_cache) > 0:
                past_length = current_kvcache.key_cache[0].shape[2]
            else:
                past_length = 0
            attention_mask = torch.ones(
                (input_batch.size(0), past_length + 1),
                dtype=torch.long,
                device=device,
            )

            # Expand cache
            if current_kvcache is not None:
                expanded_cache = DynamicCache()
                for layer in range(len(current_kvcache.key_cache)):
                    k = current_kvcache.key_cache[layer]
                    v = current_kvcache.value_cache[layer]
                    expanded_cache.update(
                        k.expand(len(batch_ids), -1, -1, -1),
                        v.expand(len(batch_ids), -1, -1, -1),
                        layer,
                    )
                past_key_values = expanded_cache
            else:
                past_key_values = None

            with torch.no_grad():
                outputs = model(
                    input_ids=input_batch,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

            if target == "past_key_values":
                current_pkv = outputs.past_key_values
                for layer_idx in range(len(target_datas)):
                    target_k = target_datas[layer_idx][0][:, :, seq_id, :].unsqueeze(2)
                    target_v = target_datas[layer_idx][1][:, :, seq_id, :].unsqueeze(2)
                    k_dist = torch.norm(
                        current_pkv.key_cache[layer_idx] - target_k, dim=-1
                    )
                    v_dist = torch.norm(
                        current_pkv.value_cache[layer_idx] - target_v, dim=-1
                    )
                    target_data_dist[0][layer_idx] = torch.cat(
                        [target_data_dist[0][layer_idx], k_dist.squeeze(-1).squeeze(-1)]
                    )
                    target_data_dist[1][layer_idx] = torch.cat(
                        [target_data_dist[1][layer_idx], v_dist.squeeze(-1).squeeze(-1)]
                    )

            del outputs
            torch.cuda.empty_cache()

        target_data_dists.append(target_data_dist)

        if (seq_id + 1) % gap == 0 or seq_id == seq_length - 1:
            save_path = target_dist_dir / f"seq={seq_id}.pt"
            torch.save(target_data_dists, save_path)
            target_data_dists = []

        # Update current_kvcache for next iteration
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids[:, seq_id : seq_id + 1],
                past_key_values=current_kvcache,
                use_cache=True,
            )
            current_kvcache = outputs.past_key_values

    return target_dist_dir


def load_target_dists(dir_path: Path) -> List[Tensor]:
    """Load key-value distances from .pt files."""
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory {dir_path} does not exist!")

    data = []
    for file_path in sorted(dir_path.glob("*.pt")):
        try:
            loaded_data = torch.load(file_path, weights_only=True)
            if isinstance(loaded_data, list):
                data.extend(loaded_data)
            else:
                print(f"Warning: Skipped non-list content in {file_path.name}")
        except (RuntimeError, IOError) as e:
            print(f"Error loading {file_path}: {e!r}")

    return data


def analyze_distances(
    model_path: Path,
    target_dists: List[Tensor],
    target: str,
    input_text: str,
) -> List[Tuple[Any, Any]]:
    """Analyze target distances against model outputs."""
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), trust_remote_code=True, attn_implementation="eager"
    )
    model.eval()

    config = model.config
    bos_token_ids = get_bos_token_ids(config)

    inputs = tokenizer(input_text, return_tensors="pt")

    if bos_token_ids and needs_bos_padding(inputs, bos_token_ids):
        pad_bos_token(inputs, bos_token_ids)

    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)

    result = []
    for layer_idx in range(len(target_dists)):
        layer_dist = target_dists[layer_idx]

        if target == "past_key_values":
            num_tokens = layer_dist[0][0].shape[0]
            target_dists_layer = []
            others_dists_layer = []

            for token_idx in range(num_tokens):
                token_id = input_ids[0, token_idx].item()
                k_dist = layer_dist[0][token_idx]
                v_dist = layer_dist[1][token_idx]
                dist = (k_dist + v_dist) / 2

                target_dist = dist[token_id].item()
                others_dist = torch.cat([dist[:token_id], dist[token_id + 1 :]])

                target_dists_layer.append(target_dist)
                others_dists_layer.append(others_dist)

            result.append((target_dists_layer, others_dists_layer))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate collision+ (CPA) threshold configuration."
    )
    parser.add_argument(
        "--model_path",
        default="~/model/Llama-3.2-1B",
        help="Path to the model.",
    )
    parser.add_argument(
        "--target_data_path",
        required=True,
        help="Path to the target KV-cache (past_key_values.pt).",
    )
    parser.add_argument(
        "--input_text",
        default='"One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning. The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries."',
        help="Input text used to generate the KV-cache.",
    )
    parser.add_argument(
        "--target",
        default="past_key_values",
        choices=["past_key_values", "hidden_states"],
        help="Target data type.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for distance calculation.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Data type.",
    )
    parser.add_argument(
        "--protect_type",
        default="origin",
        help="Protection type.",
    )
    parser.add_argument(
        "--target_model_name",
        default=None,
        help="Target model name (defaults to model_path name).",
    )
    args = parser.parse_args()

    torch.manual_seed(42)
    torch.serialization.add_safe_globals([DynamicCache, set])

    # Parse paths
    model_path = Path(args.model_path).expanduser()
    target_data_path = Path(args.target_data_path).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not target_data_path.exists():
        raise FileNotFoundError(f"Target data path not found: {target_data_path}")

    target_model_name = args.target_model_name or model_path.name
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    dtype_name = str(dtype).split(".")[-1]

    # Calculate input hash
    input_hash = hashlib.sha1(args.input_text.encode("utf-8")).hexdigest()

    print(f"Model: {model_path.name}")
    print(f"Target model: {target_model_name}")
    print(f"Target data: {target_data_path}")
    print(f"Input hash: {input_hash}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype_name}")
    print()

    # Load model and data
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation="eager"
    ).to(device, dtype)
    model.eval()

    print("Loading target data...")
    target_datas = torch.load(target_data_path, weights_only=True)

    if args.target == "past_key_values":
        target_datas = tuple(
            (k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype))
            for (k, v) in target_datas
        )

    # Prepare inputs
    inputs = tokenizer(args.input_text, return_tensors="pt").to(device)
    bos_token_id = model.config.bos_token_id

    if bos_token_id is not None:
        if isinstance(bos_token_id, int):
            bos_token_id = [bos_token_id]

        bos_len = len(bos_token_id)
        if inputs["input_ids"][-1, :bos_len].tolist() != bos_token_id:
            inputs["input_ids"] = torch.cat(
                [torch.tensor([bos_token_id]).to(device), inputs["input_ids"]], dim=1
            )
            inputs["attention_mask"] = torch.cat(
                [torch.ones(1, bos_len).to(device), inputs["attention_mask"]], dim=1
            )

    # Step 1: Generate distance statistics
    print("\n" + "=" * 60)
    print("Step 1: Generating distance statistics...")
    print("=" * 60)
    target_dist_dir = target_data_path.parent / f"{args.target}_dist"
    start_time = time.time()

    statistic_distance(
        model, target_datas, args.target, inputs, args.batch_size, target_dist_dir
    )

    print(f"\nStatistics saved to: {target_dist_dir}")
    print(f"Time: {time.time() - start_time:.2f}s")

    # Step 2: Analyze distances
    print("\n" + "=" * 60)
    print("Step 2: Analyzing distances...")
    print("=" * 60)

    target_dists = load_target_dists(target_dist_dir)
    dists = analyze_distances(model_path, target_dists, args.target, args.input_text)

    # Compile statistics
    statistics = [
        {
            "L0_model_name": model_path.name,
            "L1_model_name": target_model_name,
            "input_hash": input_hash,
            "seq_len": len(dists[0][0]),
            "target": args.target,
            "layer_idx": layer_idx,
            "target_mean": [
                target_dists[i].mean().item() for i in range(len(target_dists))
            ],
            "target_std": [
                target_dists[i].std().item() for i in range(len(target_dists))
            ],
            "target_max": [
                target_dists[i].max().item() for i in range(len(target_dists))
            ],
            "others_mean": [
                others_dists[i].mean().item() for i in range(len(target_dists))
            ],
            "others_std": [
                others_dists[i].std().item() for i in range(len(target_dists))
            ],
            "others_min": [
                others_dists[i].min().item() for i in range(len(target_dists))
            ],
        }
        for layer_idx, (target_dists, others_dists) in enumerate(dists)
    ]

    # Save configuration
    output_path = Path(
        f"attack/config/{args.protect_type}/{dtype_name}/{target_model_name}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(statistics, indent=4))

    print(f"\nCollision+ configuration saved to: {output_path}")
    print("\nYou can now run collision+ attack with: --enhance")


if __name__ == "__main__":
    main()
