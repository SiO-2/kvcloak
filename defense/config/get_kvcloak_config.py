import argparse
import json
from pathlib import Path
import torch
from transformers import AutoConfig
from typing import Any, Dict, List, Tuple, Optional


def random_orthogonal_matrix(dim: int) -> torch.Tensor:
    """Generate a random orthogonal matrix using QR decomposition."""
    random_matrix = torch.randn(dim, dim)
    S, _ = torch.linalg.qr(random_matrix)
    return S


def get_kvcloak_config(
    num_hidden_layers: int,
    num_key_value_heads: int,
    head_dim: int,
    block_size: int,
    theta_config: Dict,
    S_ratio: float,
    M_ratio: float,
    theta_ratio: float,
) -> List[List[List[Dict[str, torch.Tensor]]]]:
    """Generate KV-Cloak configuration for all layers and heads.
    
    The configuration structure is:
    kvcloak_config[num_hidden_layers][num_key_value_heads][k/v][config_dict]
    
    Args:
        num_hidden_layers: Number of transformer layers
        num_key_value_heads: Number of key/value heads
        head_dim: Dimension of each head
        block_size: Block size for permutation (b)
        theta_config: Configuration containing max values for each layer/head
        S_ratio: Scaling ratio for S matrix
        M_ratio: Scaling ratio for M matrix
        theta_ratio: Multiplier for theta values
        
    Returns:
        Nested list of configuration dictionaries
    """
    kvcloak_config = []
    for layer_idx in range(num_hidden_layers):
        layer_data = []
        for head_idx in range(num_key_value_heads):
            kv_data = []
            for kv_type in ["key", "value"]:
                S_ratios = torch.rand(block_size) * (S_ratio - (1 / S_ratio)) + (
                    1 / S_ratio
                )
                M_ratios = torch.rand(head_dim // 2) * (M_ratio - (1 / M_ratio)) + (
                    1 / M_ratio
                )
                M_angles = torch.rand(head_dim // 2) * 2 * torch.pi
                S = random_orthogonal_matrix(block_size)
                theta = (
                    theta_config[f"layer_{layer_idx}"][f"{kv_type}_max_values"][head_idx]
                    * theta_ratio
                )
                threshold = theta * M_ratio * 1.42
                a = (torch.rand(head_dim) + 3) * threshold
                # Constraints:
                # -threshold < kM < threshold
                # 3*threshold <= a < 4*threshold
                # 2*threshold < kM+a < 5*threshold
                # padding = 1.5*threshold (kM < padding < kM+a)

                data = {
                    "theta": theta,
                    "M_ratio": M_ratio,
                    "S_ratios": S_ratios,
                    "S": S,
                    "M_ratios": M_ratios,
                    "M_angles": M_angles,
                    "a": a,
                }
                kv_data.append(data)
            layer_data.append(kv_data)

        kvcloak_config.append(layer_data)

    return kvcloak_config


def get_kvcloak_config_mla(
    num_hidden_layers: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_size: int,
    theta: float,
    S_ratio: float,
    Mr_ratio: float,
) -> List[Dict[str, torch.Tensor]]:
    """
    kvcloak_config[num_hidden_layers][
        {
            "theta": theta,
            "Mr_ratio": Mr_ratio,
            "S_ratios": S_ratios,
            "S": S,
            "Mn_indices": Mn_indices,
            "Mr_ratios": Mr_ratios,
            "Mr_angles": Mr_angles,
            "a": a,
        }
    ]
    """
    threshold = theta * Mr_ratio * 1.42

    kvcloak_config = []
    for _ in range(num_hidden_layers):
        S_ratios = torch.rand(block_size) * (S_ratio - (1 / S_ratio)) + (1 / S_ratio)
        S = random_orthogonal_matrix(block_size)
        Mn_indices = torch.randperm(kv_lora_rank)
        Mr_ratios = torch.rand(qk_rope_head_dim // 2) * (
            Mr_ratio - (1 / Mr_ratio)
        ) + (
            1 / Mr_ratio
        )
        Mr_angles = torch.rand(qk_rope_head_dim // 2) * 2 * torch.pi
        a = (torch.rand(kv_lora_rank + qk_rope_head_dim) + 3) * threshold
        # -threshold < cM < threshold
        # 3*threshold <= a < 4*threshold
        # 2*threshold < cM+a < 5*threshold
        # padding = 1.5*threshold (cM < padding < cM+a)

        data = {
            "theta": theta,
            "Mr_ratio": Mr_ratio,
            "S_ratios": S_ratios,
            "S": S,
            "Mn_indices": Mn_indices,
            "Mr_ratios": Mr_ratios,
            "Mr_angles": Mr_angles,
            "a": a,
        }

        kvcloak_config.append(data)

    return kvcloak_config


def load_json_file(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"File path does not exist: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"File format is incorrect: {file_path}") from e


def main():
    parser = argparse.ArgumentParser(description="Generate KV-Cloak config file.")
    parser.add_argument("--model-name", default="Llama-3.2-1B")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local model path. Defaults to ~/model/<model-name>",
    )
    parser.add_argument(
        "--theta-config-path",
        default=None,
        help="Theta config JSON path. Defaults to defense/config/kvcloak/theta/<model-name>.json",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--S-ratio", type=float, default=1.0)
    parser.add_argument("--M-ratio", type=float, default=1.0)
    parser.add_argument("--theta-ratio", type=float, default=2.0)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output .pt path. Defaults to defense/config/kvcloak/b{b}_S{S}_M{M}_t{t}/<model>.pt",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = args.model_name
    model_path = (
        Path(args.model_path).expanduser()
        if args.model_path is not None
        else Path(f"~/model/{model_name}").expanduser()
    )
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    theta_config_path = (
        Path(args.theta_config_path).expanduser()
        if args.theta_config_path is not None
        else Path(f"defense/config/kvcloak/theta/{model_name}.json").expanduser()
    )
    theta_config = load_json_file(theta_config_path)

    if model_config.model_type == "gpt2":
        num_hidden_layers = model_config.n_layer
        num_key_value_heads = model_config.n_head
        head_dim = model_config.n_embd // model_config.n_head
    elif model_config.model_type in ["qwen2", "llama", "llama3"]:
        num_hidden_layers = model_config.num_hidden_layers
        num_key_value_heads = model_config.num_key_value_heads
        head_dim = model_config.hidden_size // model_config.num_attention_heads
    else:
        raise ValueError(
            f"Model type {model_config.model_type} is not supported for now."
        )

    kvcloak_config = get_kvcloak_config(
        num_hidden_layers=num_hidden_layers,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        block_size=args.block_size,
        theta_config=theta_config,
        S_ratio=args.S_ratio,
        M_ratio=args.M_ratio,
        theta_ratio=args.theta_ratio,
    )

    if args.output_path is not None:
        config_path = Path(args.output_path).expanduser()
    else:
        config_path = (
            Path("./defense/config/kvcloak").expanduser()
            / f"b{args.block_size}_S{args.S_ratio:g}_M{args.M_ratio:g}_t{args.theta_ratio:g}"
            / f"{model_name}.pt"
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(kvcloak_config, config_path)
    print(f"KV-Cloak config saved in {config_path}")


if __name__ == "__main__":
    main()
