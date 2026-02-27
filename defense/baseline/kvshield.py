import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def random_permutation_matrix(d):
    identity_matrix = torch.eye(d)
    permutation = torch.randperm(d)
    permuted_matrix = identity_matrix[permutation]
    return permuted_matrix


def kvshield_vo(model: AutoModelForCausalLM):
    device = model.device
    dtype = model.dtype
    hidden_size = model.config.hidden_size
    num_key_value_heads = model.config.num_key_value_heads
    num_attention_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim

    with torch.no_grad():
        for i in range(len(model.model.layers)):
            v_proj = (
                model.model.layers[i]
                .self_attn.v_proj.weight.view(
                    num_key_value_heads, head_dim, hidden_size
                )
                .to(device, dtype)
            )
            o_proj = (
                model.model.layers[i]
                .self_attn.o_proj.weight.view(
                    hidden_size, num_attention_heads, head_dim
                )
                .transpose(0, 1)
                .to(device, dtype)
            )
            for kv_idx in range(num_key_value_heads):
                permuted_matrix = random_permutation_matrix(head_dim).to(device, dtype)
                v_proj[kv_idx] = permuted_matrix @ v_proj[kv_idx]
                for j in range(num_attention_heads // num_key_value_heads):
                    attn_idx = kv_idx * (num_attention_heads // num_key_value_heads) + j
                    o_proj[attn_idx] = o_proj[attn_idx] @ permuted_matrix.T

            model.model.layers[i].self_attn.v_proj.weight.copy_(
                v_proj.view(-1, hidden_size)
            )
            model.model.layers[i].self_attn.o_proj.weight.copy_(
                o_proj.transpose(0, 1).view(-1, hidden_size)
            )

    return model


def kvshield(model: AutoModelForCausalLM):
    device = model.device
    dtype = model.dtype
    hidden_size = model.config.hidden_size
    num_key_value_heads = model.config.num_key_value_heads
    num_attention_heads = model.config.num_attention_heads
    head_dim = model.config.head_dim

    with torch.no_grad():
        for i in range(len(model.model.layers)):
            q_proj = (
                model.model.layers[i]
                .self_attn.q_proj.weight.view(
                    num_attention_heads, head_dim, hidden_size
                )
                .to(device, dtype)
            )
            k_proj = (
                model.model.layers[i]
                .self_attn.k_proj.weight.view(
                    num_key_value_heads, head_dim, hidden_size
                )
                .to(device, dtype)
            )
            v_proj = (
                model.model.layers[i]
                .self_attn.v_proj.weight.view(
                    num_key_value_heads, head_dim, hidden_size
                )
                .to(device, dtype)
            )
            o_proj = (
                model.model.layers[i]
                .self_attn.o_proj.weight.view(
                    hidden_size, num_attention_heads, head_dim
                )
                .transpose(0, 1)
                .to(device, dtype)
            )
            for kv_idx in range(num_key_value_heads):
                permuted_matrix = random_permutation_matrix(head_dim).to(device, dtype)
                k_proj[kv_idx] = permuted_matrix @ k_proj[kv_idx]
                v_proj[kv_idx] = permuted_matrix @ v_proj[kv_idx]
                for j in range(num_attention_heads // num_key_value_heads):
                    attn_idx = kv_idx * (num_attention_heads // num_key_value_heads) + j
                    q_proj[attn_idx] = permuted_matrix @ q_proj[attn_idx]
                    o_proj[attn_idx] = o_proj[attn_idx] @ permuted_matrix.T

            model.model.layers[i].self_attn.q_proj.weight.copy_(
                q_proj.view(-1, hidden_size)
            )
            model.model.layers[i].self_attn.k_proj.weight.copy_(
                k_proj.view(-1, hidden_size)
            )
            model.model.layers[i].self_attn.v_proj.weight.copy_(
                v_proj.view(-1, hidden_size)
            )
            model.model.layers[i].self_attn.o_proj.weight.copy_(
                o_proj.transpose(0, 1).view(-1, hidden_size)
            )

    return model


def kvshield_gpt2(model: AutoModelForCausalLM):
    device = model.device
    dtype = model.dtype
    n_embd = model.config.n_embd
    n_head = model.config.n_head
    head_dim = n_embd // n_head

    with torch.no_grad():
        for i, layer in enumerate(model.transformer.h):
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

            for kv_idx in range(n_head):
                permuted_matrix = random_permutation_matrix(head_dim).to(device, dtype)
                q_proj_weight[kv_idx] = q_proj_weight[kv_idx] @ permuted_matrix
                q_proj_bias[kv_idx] = q_proj_bias[kv_idx] @ permuted_matrix

                k_proj_weight[kv_idx] = k_proj_weight[kv_idx] @ permuted_matrix
                k_proj_bias[kv_idx] = k_proj_bias[kv_idx] @ permuted_matrix

                v_proj_weight[kv_idx] = v_proj_weight[kv_idx] @ permuted_matrix
                v_proj_bias[kv_idx] = v_proj_bias[kv_idx] @ permuted_matrix

                o_proj_weight[kv_idx] = permuted_matrix.T @ o_proj_weight[kv_idx]

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


def main():
    parser = argparse.ArgumentParser(description="Apply KVShield transform to a model.")
    parser.add_argument("--model-name", default="Llama-3.2-1B")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local model path. Defaults to ~/model/<model-name>/",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "full", "vo", "gpt2"],
        help="KVShield mode: full=QKVO, vo=VO-only, gpt2=GPT2-specific, auto=model-type based.",
    )
    parser.add_argument(
        "--save-dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Dtype used when saving transformed model.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output model directory. Defaults to ~/model/<model-name>_kvshield/",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    save_dtype = getattr(torch, args.save_dtype)
    device = torch.device(args.device)
    model_name = args.model_name
    model_path = (
        Path(args.model_path).expanduser()
        if args.model_path is not None
        else Path(f"~/model/{model_name}/").expanduser()
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation="eager"
    ).to(device, dtype)
    model.eval()

    mode = args.mode
    if mode == "auto":
        mode = "gpt2" if model.config.model_type == "gpt2" else "full"

    if mode == "gpt2":
        model = kvshield_gpt2(model)
    elif mode == "vo":
        model = kvshield_vo(model)
    else:
        model = kvshield(model)

    model = model.to(dtype=save_dtype)
    kvshield_model_path = (
        Path(args.output_path).expanduser()
        if args.output_path is not None
        else model_path.parent / f"{model_name}_kvshield/"
    )
    kvshield_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(kvshield_model_path)
    tokenizer.save_pretrained(kvshield_model_path)
    print(f"kvshield model saved in {kvshield_model_path}")


if __name__ == "__main__":
    main()
