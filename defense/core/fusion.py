import argparse
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Dict, List, Tuple, Optional


def get_rotation_matrix(angles: torch.Tensor) -> torch.Tensor:
    """Generate rotation matrix from angle vector."""
    cos = angles.cos()
    sin = angles.sin()
    # Generate diagonal matrices A and B
    A = torch.diag(cos)
    B = torch.diag(sin)

    # Construct block matrix
    row1 = torch.cat((A, B), dim=1)  # First row [A, B]
    row2 = torch.cat((-B, A), dim=1)  # Second row [-B, A]
    result = torch.cat((row1, row2), dim=0)
    return result


def fusion_llama(
    model: AutoModelForCausalLM,
    kvcloak_config: List[List[List[Dict[str, torch.Tensor]]]],
) -> AutoModelForCausalLM:
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    num_key_value_heads = (
        model.config.num_key_value_heads
        if hasattr(model.config, "num_key_value_heads")
        else model.config.num_attention_heads
    )
    head_dim = hidden_size // num_attention_heads

    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            device = layer.self_attn.q_proj.weight.device
            dtype = layer.self_attn.q_proj.weight.dtype
            q_proj_weight = layer.self_attn.q_proj.weight.view(
                num_attention_heads, head_dim, hidden_size
            ).to(device, dtype)
            if layer.self_attn.q_proj.bias is not None:
                q_proj_bias = layer.self_attn.q_proj.bias.view(
                    num_attention_heads, head_dim, 1
                ).to(device, dtype)

            k_proj_weight = layer.self_attn.k_proj.weight.view(
                num_key_value_heads, head_dim, hidden_size
            ).to(device, dtype)
            if layer.self_attn.k_proj.bias is not None:
                k_proj_bias = layer.self_attn.k_proj.bias.view(
                    num_key_value_heads, head_dim, 1
                ).to(device, dtype)

            v_proj_weight = layer.self_attn.v_proj.weight.view(
                num_key_value_heads, head_dim, hidden_size
            ).to(device, dtype)
            if layer.self_attn.v_proj.bias is not None:
                v_proj_bias = layer.self_attn.v_proj.bias.view(
                    num_key_value_heads, head_dim, 1
                ).to(device, dtype)

            o_proj_weight = (
                layer.self_attn.o_proj.weight.view(
                    hidden_size, num_attention_heads, head_dim
                )
                .transpose(0, 1)
                .to(device, dtype)
            )

            for head_idx in range(num_key_value_heads):
                M1_ratios = kvcloak_config[i][head_idx][0]["M_ratios"].to(device, dtype)
                M1_ratios = torch.cat((M1_ratios, M1_ratios)).to(device, dtype)
                M1_angles = kvcloak_config[i][head_idx][0]["M_angles"].to(device, dtype)
                M1_rotation = get_rotation_matrix(M1_angles).to(device, dtype)

                M2_ratios = kvcloak_config[i][head_idx][1]["M_ratios"].to(device, dtype)
                M2_ratios = torch.cat((M2_ratios, M2_ratios)).to(device, dtype)
                M2_angles = kvcloak_config[i][head_idx][1]["M_angles"].to(device, dtype)
                M2_rotation = get_rotation_matrix(M2_angles).to(device, dtype)

                # R = rotation_matrix
                # I = torch.eye(R.size(-1), device=R.device)
                # error = torch.norm(R @ R.T - I)  # Should be close to 0
                # print(f"rotation_matrix.error: {error.item():.2e}")

                # error = torch.norm(matrix @ inverse_matrix - I)
                # print(f"matrix.inverse_matrix.error: {error.item():.2e}")

                k_proj_weight[head_idx] = (
                    M1_rotation @ k_proj_weight[head_idx] * M1_ratios.unsqueeze(1)
                )
                if layer.self_attn.k_proj.bias is not None:
                    k_proj_bias[head_idx] = (
                        M1_rotation @ k_proj_bias[head_idx] * M1_ratios.unsqueeze(1)
                    )

                v_proj_weight[head_idx] = (
                    M2_rotation @ v_proj_weight[head_idx] * M2_ratios.unsqueeze(1)
                )
                if layer.self_attn.v_proj.bias is not None:
                    v_proj_bias[head_idx] = (
                        M2_rotation @ v_proj_bias[head_idx] * M2_ratios.unsqueeze(1)
                    )

                for j in range(num_attention_heads // num_key_value_heads):
                    attn_idx = (
                        head_idx * (num_attention_heads // num_key_value_heads) + j
                    )
                    q_proj_weight[attn_idx] = (
                        M1_rotation @ q_proj_weight[attn_idx] / M1_ratios.unsqueeze(1)
                    )
                    if layer.self_attn.q_proj.bias is not None:
                        q_proj_bias[attn_idx] = (
                            M1_rotation @ q_proj_bias[attn_idx] / M1_ratios.unsqueeze(1)
                        )

                    o_proj_weight[attn_idx] = o_proj_weight[attn_idx] @ (
                        M2_rotation.T / M2_ratios.unsqueeze(1)
                    )

            layer.self_attn.q_proj.weight.copy_(q_proj_weight.view(-1, hidden_size))
            if layer.self_attn.q_proj.bias is not None:
                layer.self_attn.q_proj.bias.copy_(q_proj_bias.view(-1))

            layer.self_attn.k_proj.weight.copy_(k_proj_weight.view(-1, hidden_size))
            if layer.self_attn.k_proj.bias is not None:
                layer.self_attn.k_proj.bias.copy_(k_proj_bias.view(-1))

            layer.self_attn.v_proj.weight.copy_(v_proj_weight.view(-1, hidden_size))
            if layer.self_attn.v_proj.bias is not None:
                layer.self_attn.v_proj.bias.copy_(v_proj_bias.view(-1))

            layer.self_attn.o_proj.weight.copy_(
                o_proj_weight.transpose(0, 1).view(-1, hidden_size)
            )

    return model


def fusion_gpt2(
    model: AutoModelForCausalLM,
    kvcloak_config: List[List[List[Dict[str, torch.Tensor]]]],
) -> AutoModelForCausalLM:
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    with torch.no_grad():
        for i, layer in enumerate(model.transformer.h):
            device = layer.attn.c_attn.weight.device
            dtype = layer.attn.c_attn.weight.dtype
            q_proj_weight = torch.stack(
                torch.split(layer.attn.c_attn.weight[:, :n_embd], head_dim, dim=-1),
                dim=0,
            ).to(device, dtype)

            q_proj_bias = torch.stack(
                torch.split(layer.attn.c_attn.bias[:n_embd], head_dim, dim=-1),
                dim=0,
            ).to(device, dtype)

            k_proj_weight = torch.stack(
                torch.split(
                    layer.attn.c_attn.weight[:, n_embd : 2 * n_embd], head_dim, dim=-1
                ),
                dim=0,
            ).to(device, dtype)
            k_proj_bias = torch.stack(
                torch.split(
                    layer.attn.c_attn.bias[n_embd : 2 * n_embd], head_dim, dim=-1
                ),
                dim=0,
            ).to(device, dtype)

            v_proj_weight = torch.stack(
                torch.split(
                    layer.attn.c_attn.weight[:, 2 * n_embd :], head_dim, dim=-1
                ),
                dim=0,
            ).to(device, dtype)
            v_proj_bias = torch.stack(
                torch.split(layer.attn.c_attn.bias[2 * n_embd :], head_dim, dim=-1),
                dim=0,
            ).to(device, dtype)

            o_proj_weight = torch.stack(
                torch.split(layer.attn.c_proj.weight[:, :], head_dim, dim=0), dim=0
            ).to(device, dtype)

            for head_idx in range(n_head):
                M1_ratios = kvcloak_config[i][head_idx][0]["M_ratios"].to(device, dtype)
                M1_ratios = torch.cat((M1_ratios, M1_ratios)).to(device, dtype)
                M1_angles = kvcloak_config[i][head_idx][0]["M_angles"].to(device, dtype)
                M1_rotation = get_rotation_matrix(M1_angles).to(device, dtype)

                M2_ratios = kvcloak_config[i][head_idx][1]["M_ratios"].to(device, dtype)
                M2_ratios = torch.cat((M2_ratios, M2_ratios)).to(device, dtype)
                M2_angles = kvcloak_config[i][head_idx][1]["M_angles"].to(device, dtype)
                M2_rotation = get_rotation_matrix(M2_angles).to(device, dtype)

                q_proj_weight[head_idx] = (
                    q_proj_weight[head_idx] @ M1_rotation * M1_ratios.unsqueeze(0)
                )
                q_proj_bias[head_idx] = (
                    q_proj_bias[head_idx] @ M1_rotation * M1_ratios.unsqueeze(0)
                )

                k_proj_weight[head_idx] = (
                    k_proj_weight[head_idx] @ M1_rotation / M1_ratios.unsqueeze(0)
                )
                k_proj_bias[head_idx] = (
                    k_proj_bias[head_idx] @ M1_rotation / M1_ratios.unsqueeze(0)
                )

                v_proj_weight[head_idx] = (
                    v_proj_weight[head_idx] @ M2_rotation * M2_ratios.unsqueeze(0)
                )
                v_proj_bias[head_idx] = (
                    v_proj_bias[head_idx] @ M2_rotation * M2_ratios.unsqueeze(0)
                )

                o_proj_weight[head_idx] = (
                    M2_rotation.T @ o_proj_weight[head_idx] / M2_ratios.unsqueeze(1)
                )

            layer.attn.c_attn.weight[:, :n_embd].copy_(
                q_proj_weight.permute(1, 0, 2).reshape(n_embd, n_embd)
            )
            layer.attn.c_attn.bias[:n_embd].copy_(q_proj_bias.view(-1))

            layer.attn.c_attn.weight[:, n_embd : 2 * n_embd].copy_(
                k_proj_weight.permute(1, 0, 2).reshape(n_embd, n_embd)
            )
            layer.attn.c_attn.bias[n_embd : 2 * n_embd].copy_(k_proj_bias.view(-1))

            layer.attn.c_attn.weight[:, 2 * n_embd :].copy_(
                v_proj_weight.permute(1, 0, 2).reshape(n_embd, n_embd)
            )
            layer.attn.c_attn.bias[2 * n_embd :].copy_(v_proj_bias.view(-1))

            layer.attn.c_proj.weight.copy_(o_proj_weight.reshape(-1, n_embd))

    return model


def get_rotation_matrix_interleave(angles: torch.Tensor) -> torch.Tensor:
    """
    Generates a rotation matrix that matches the standard RoPE geometry,
    acting on adjacent pairs of dimensions. This creates a block-diagonal
    matrix with 2x2 rotation blocks.

    Args:
        angles (torch.Tensor): A tensor of rotation angles of shape (head_dim / 2,).

    Returns:
        torch.Tensor: A (head_dim, head_dim) rotation matrix.
    """
    head_dim = angles.shape[0] * 2
    cos = angles.cos()
    sin = angles.sin()

    # Start with a zero matrix
    M = torch.zeros((head_dim, head_dim), device=angles.device, dtype=angles.dtype)

    # Populate the block-diagonal with 2x2 rotation matrices
    for i in range(head_dim // 2):
        c, s = cos[i], sin[i]
        block_start_idx = 2 * i
        block_end_idx = 2 * i + 2

        # Place the 2x2 rotation block on the diagonal
        M[block_start_idx:block_end_idx, block_start_idx:block_end_idx] = torch.tensor(
            [[c, -s], [s, c]],
            device=M.device,
            dtype=M.dtype,
        )

    return M


def fusion_deepseek(
    model: AutoModelForCausalLM,
    kvcloak_config: List[Dict[str, torch.Tensor]],
) -> AutoModelForCausalLM:
    hidden_size = model.config.hidden_size
    num_attention_heads = model.config.num_attention_heads
    qk_nope_head_dim = model.config.qk_nope_head_dim
    qk_rope_head_dim = model.config.qk_rope_head_dim
    kv_lora_rank = model.config.kv_lora_rank
    v_head_dim = model.config.v_head_dim

    with torch.no_grad():
        for i, layer in enumerate(model.model.layers):
            device = layer.self_attn.q_proj.weight.device
            dtype = layer.self_attn.q_proj.weight.dtype

            q_nope_proj_weight, q_rope_proj_weight = torch.split(
                layer.self_attn.q_proj.weight.view(
                    num_attention_heads,
                    qk_nope_head_dim + qk_rope_head_dim,
                    hidden_size,
                ).to(device, dtype),
                [qk_nope_head_dim, qk_rope_head_dim],
                dim=1,
            )
            if layer.self_attn.q_proj.bias is not None:
                q_nope_proj_bias, q_rope_proj_bias = torch.split(
                    layer.self_attn.q_proj.bias.view(
                        num_attention_heads, qk_nope_head_dim + qk_rope_head_dim, 1
                    ).to(device, dtype),
                    [qk_nope_head_dim, qk_rope_head_dim],
                    dim=1,
                )

            a_nope_proj_weight, a_rope_proj_weight = torch.split(
                layer.self_attn.kv_a_proj_with_mqa.weight.view(
                    1, kv_lora_rank + qk_rope_head_dim, hidden_size
                ).to(device, dtype),
                [kv_lora_rank, qk_rope_head_dim],
                dim=1,
            )
            if layer.self_attn.kv_a_proj_with_mqa.bias is not None:
                a_nope_proj_bias, a_rope_proj_bias = torch.split(
                    layer.self_attn.kv_a_proj_with_mqa.bias.view(
                        1, kv_lora_rank + qk_rope_head_dim, 1
                    ).to(device, dtype),
                    [kv_lora_rank, qk_rope_head_dim],
                    dim=1,
                )

            k_b_proj_weight, v_b_proj_weight = torch.split(
                layer.self_attn.kv_b_proj.weight.view(
                    num_attention_heads, qk_nope_head_dim + v_head_dim, kv_lora_rank
                ).to(device, dtype),
                [qk_nope_head_dim, v_head_dim],
                dim=1,
            )

            # nope
            Mn_indices = kvcloak_config[i]["Mn_indices"].to(device)
            # Mn_indices = torch.randperm(kv_lora_rank).to(device)
            Mn = torch.eye(kv_lora_rank).to(device, dtype)[Mn_indices]
            a_nope_proj_weight = Mn @ a_nope_proj_weight
            if layer.self_attn.kv_a_proj_with_mqa.bias is not None:
                a_nope_proj_bias = Mn @ a_nope_proj_bias

            k_b_proj_weight = k_b_proj_weight @ Mn.T
            v_b_proj_weight = v_b_proj_weight @ Mn.T

            layer.self_attn.kv_a_layernorm.weight.copy_(
                layer.self_attn.kv_a_layernorm.weight[Mn_indices]
            )

            # rope
            Mr_ratios = kvcloak_config[i]["Mr_ratios"].to(device, dtype)
            Mr_ratios = (
                torch.cat([Mr_ratios, Mr_ratios], dim=0)
                .view(2, -1)
                .transpose(0, 1)
                .reshape(-1)
            )
            # Mr_ratios = torch.cat((Mr_ratios, Mr_ratios)).to(device, dtype)
            Mr_angles = kvcloak_config[i]["Mr_angles"].to(device, dtype)
            Mr = get_rotation_matrix_interleave(Mr_angles).to(device, dtype)
            # Mr = get_rotation_matrix(Mr_angles).to(device, dtype)

            a_rope_proj_weight = Mr @ a_rope_proj_weight * Mr_ratios.unsqueeze(1)
            if layer.self_attn.kv_a_proj_with_mqa.bias is not None:
                a_rope_proj_bias = Mr @ a_rope_proj_bias * Mr_ratios.unsqueeze(1)

            q_rope_proj_weight = Mr @ q_rope_proj_weight / Mr_ratios.unsqueeze(1)
            if layer.self_attn.q_proj.bias is not None:
                q_rope_proj_bias = Mr @ q_rope_proj_bias / Mr_ratios.unsqueeze(1)

            # restore weights
            layer.self_attn.q_proj.weight.copy_(
                torch.cat((q_nope_proj_weight, q_rope_proj_weight), dim=1).view(
                    -1, hidden_size
                )
            )
            if layer.self_attn.q_proj.bias is not None:
                layer.self_attn.q_proj.bias.copy_(
                    torch.cat((q_nope_proj_bias, q_rope_proj_bias), dim=1).view(-1)
                )

            layer.self_attn.kv_a_proj_with_mqa.weight.copy_(
                torch.cat((a_nope_proj_weight, a_rope_proj_weight), dim=1).view(
                    -1, hidden_size
                )
            )
            if layer.self_attn.kv_a_proj_with_mqa.bias is not None:
                layer.self_attn.kv_a_proj_with_mqa.bias.copy_(
                    torch.cat((a_nope_proj_bias, a_rope_proj_bias), dim=1).view(-1)
                )

            layer.self_attn.kv_b_proj.weight.copy_(
                torch.cat((k_b_proj_weight, v_b_proj_weight), dim=1).view(
                    -1, kv_lora_rank
                )
            )

    return model


def fusion(
    model: AutoModelForCausalLM,
    kvcloak_config: List[List[List[Dict[str, torch.Tensor]]]],
):

    if model.config.model_type == "gpt2":
        return fusion_gpt2(model, kvcloak_config)
    elif model.config.model_type in ["llama", "llama3", "qwen2"]:
        return fusion_llama(model, kvcloak_config)
    elif model.config.model_type == "deepseek_v2":
        return fusion_deepseek(model, kvcloak_config)
    else:
        raise ValueError("Models are not supported for now.")


def main():
    parser = argparse.ArgumentParser(description="Process model parameters.")
    parser.add_argument(
        "--model_path",
        help="Input model path",
        # default="~/model/gpt2/",
        default="~/model/Llama-3.2-1B/",
        # default="~/model/Llama-3.2-3B-Instruct/",
        # default="~/model/Meta-Llama-3.1-8B/",
        # default="~/model/Qwen2.5-Math-7B/",
        # default="~/model/DeepSeek-R1-Distill-Llama-8B/",
    )
    parser.add_argument(
        "--device", help="Device to use (e.g., cuda:0, cpu)", default="cuda:0"
    )
    parser.add_argument(
        "--dtype", help="Data type (e.g., bfloat16, float32)", default="float32"
    )

    args = parser.parse_args()
    torch.manual_seed(42)

    # Expand ~ in path
    model_path = Path(args.model_path).expanduser()

    # Check if path exists
    if not model_path.exists():
        raise FileNotFoundError(f"Input model path does not exist: {model_path}")

    # torch.cuda.manual_seed_all(42)

    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.float32
    device = torch.device(args.device)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation="eager"
    ).to(device, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()

    torch_dtype = (
        model.config.torch_dtype
        if model.config.torch_dtype is not None
        else torch.bfloat16
    )

    model_name = model_path.name
    kvcloak_config_path = Path(f"defense/config/kvcloak/b16_S1_M1_t2/{model_name}.pt")

    if not kvcloak_config_path.exists():
        print(f"KV-Cloak config path does not exist in {kvcloak_config_path}")
    else:
        kvcloak_config = torch.load(kvcloak_config_path, weights_only=True)
        model_kvcloak = fusion(model, kvcloak_config).to(dtype=torch_dtype)

        # fusion_model_path = model_path.parent / f"{model_name}_kvcloak/"
        # fusion_model_path.mkdir(parents=True, exist_ok=True)
        # print("Waiting for the model to save...")
        # model_kvcloak.save_pretrained(fusion_model_path)
        # tokenizer.save_pretrained(fusion_model_path)
        # print(f"KV-Cloak model saved in {fusion_model_path}")


if __name__ == "__main__":
    main()
