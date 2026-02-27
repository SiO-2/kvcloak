import argparse
from datetime import datetime
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def remove_rotary_pos_emb(k, cos, sin, unsqueeze_dim=1):
    """Remove Rotary Position Embedding (RoPE) from keys.
    
    This is the inverse operation of applying RoPE, used to recover
    the original key vectors before position encoding.
    
    Args:
        k: Key tensor with shape [batch, num_heads, seq_len, head_dim]
        cos: Cosine component of RoPE
        sin: Sine component of RoPE
        unsqueeze_dim: Dimension to unsqueeze cos/sin tensors
        
    Returns:
        Unrotated key tensor with RoPE removed
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = -1 * sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed


def get_token_ids_from_hidden(hidden_states, model):
    """Convert hidden states back to token IDs using embedding matrix.
    
    This function performs the inverse of the embedding lookup by:
    1. Removing layer normalization
    2. Computing cosine similarity with embedding matrix
    3. Selecting the most similar token for each position
    
    Args:
        hidden_states: Hidden state tensor [batch, seq_len, hidden_dim]
        model: The language model containing embedding matrix
        
    Returns:
        Predicted token IDs [batch, seq_len]
    """
    layernorm = model.model.layers[0].input_layernorm
    layernorm_weights = layernorm.weight.data.float()

    unnormed_hidden = hidden_states / layernorm_weights
    embedding_matrix = model.model.embed_tokens.weight.data.float()
    embedding_matrix_norm = F.normalize(embedding_matrix, p=2, dim=1)
    unnormed_hidden_norm = F.normalize(unnormed_hidden, p=2, dim=2)
    scores = torch.matmul(unnormed_hidden_norm, embedding_matrix_norm.T)

    predicted_token_ids = torch.argmax(scores, dim=-1)

    return predicted_token_ids


def inversion(model, key_values):
    """Perform KV-cache inversion attack to recover input tokens.
    
    This attack reverses the attention computation by:
    1. Extracting K/V projections and biases from the model
    2. For Llama/Qwen: removing RoPE from keys
    3. Solving linear equations to recover hidden states
    4. Mapping hidden states back to token IDs
    
    Args:
        model: The language model (GPT-2, Llama, or Qwen)
        key_values: Tuple of (keys, values) from KV-cache
        
    Returns:
        Predicted token IDs as 1D tensor
        
    Raises:
        ValueError: If model type is not supported
    """
    if model.config.model_type == "gpt2":
        hidden_size = model.config.n_embd

        c_attn = model.transformer.h[0].attn.c_attn
        slice_k = slice(hidden_size, 2 * hidden_size)
        slice_v = slice(2 * hidden_size, None)
        W_k = c_attn.weight[:, slice_k].float()
        W_v = c_attn.weight[:, slice_v].float()
        B_k = c_attn.bias[slice_k].flatten().float()
        B_v = c_attn.bias[slice_v].flatten().float()
        keys = key_values[0].transpose(1, 2).reshape(-1, hidden_size).float() - B_k
        values = key_values[1].transpose(1, 2).reshape(-1, hidden_size).float() - B_v
        input_keys = torch.linalg.solve(W_k.T, keys.T)
        input_values = torch.linalg.solve(W_v.T, values.T)
        attn_inputs = (input_keys + input_values) / 2

        attn_inputs = attn_inputs.T.reshape(1, -1, hidden_size).to(dtype=model.dtype)
        lm_head_outputs = model.lm_head(attn_inputs)
        predicted_token_ids = torch.argmax(lm_head_outputs, dim=-1)

    elif model.config.model_type in ["llama", "llama3", "qwen2"]:
        hidden_size = model.config.hidden_size
        num_attention_heads = model.config.num_attention_heads
        num_key_value_heads = model.config.num_key_value_heads
        head_dim = hidden_size // num_attention_heads

        self_attn = model.model.layers[0].self_attn
        W_k = self_attn.k_proj.weight.data.float()
        W_v = self_attn.v_proj.weight.data.float()
        B_k = self_attn.k_proj.bias.float() if self_attn.k_proj.bias is not None else 0
        B_v = self_attn.v_proj.bias.float() if self_attn.v_proj.bias is not None else 0
        device = W_k.device

        # --- Inverse RoPE (Remove Rotary Position Embedding) ---
        # 1. Get RoPE module and dynamically generate sin/cos tables
        rotary_emb = model.model.rotary_emb
        seq_len = key_values[0].shape[2]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = rotary_emb(key_values[0], position_ids)

        # 2. Apply inverse rotation to key vectors
        keys_with_rope = key_values[0].to(device)
        unrotated_keys = remove_rotary_pos_emb(keys_with_rope, cos, sin)

        # 3. Use unrotated keys for subsequent computation
        keys = (
            unrotated_keys.transpose(1, 2)
            .reshape(-1, num_key_value_heads * head_dim)
            .float()
            .to(device)
        ) - B_k

        values = (
            key_values[1]
            .transpose(1, 2)
            .reshape(-1, num_key_value_heads * head_dim)
            .float()
            .to(device)
        ) - B_v

        # Solve linear least squares to recover input from K/V
        input_keys = torch.linalg.lstsq(W_k, keys.T).solution
        input_values = torch.linalg.lstsq(W_v, values.T).solution
        attn_inputs = (input_keys + input_values) / 2

        attn_inputs = attn_inputs.T.reshape(1, -1, hidden_size).to(dtype=model.dtype)
        predicted_token_ids = get_token_ids_from_hidden(attn_inputs, model)

    else:
        raise ValueError("Models are not supported for now.")

    return predicted_token_ids.squeeze(0)


def inversion_log(
    log_path,
    L1_model_name,
    L0_model_name,
    input_hash,
    target,
    result,
    result_ids,
    time_cost,
):
    log = {
        "L1 model": L1_model_name,
        "L0 model": L0_model_name,
        "input hash": input_hash,
        "attack type": "inversion",
        "target": target,
        "layer": 0,
        "threshold": None,
        "result": result,
        "result token ids": result_ids.tolist(),
        "time": time_cost,
        "date": datetime.now().isoformat(),
    }

    with open(log_path, "a", encoding="utf-8") as outfile:
        json.dump(log, outfile, ensure_ascii=False)
        outfile.write("\n")


def main():
    parser = argparse.ArgumentParser(description="KV-Cache Inversion Attack")
    parser.add_argument(
        "--model_path",
        help="Model path.",
        default="~/model/Llama-3.2-1B/",
    )
    parser.add_argument(
        "--target_data_path",
        help="Target data path.",
        default="cache/torch.float32/config/Llama-3.2-1B/6d0aba55c643e35809cae53f263941168b37b344/origin/past_key_values.pt",
    )
    parser.add_argument(
        "--device", help="Device to use (e.g., cuda:0, cpu).", default="cuda:2"
    )
    parser.add_argument(
        "--dtype", help="Data type (e.g., bfloat16, float32).", default="float32"
    )

    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser()
    target_data_path = Path(args.target_data_path).expanduser()
    target = target_data_path.stem

    # Check if paths exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not target_data_path.exists():
        raise FileNotFoundError(f"Path does not exist: {target_data_path}")

    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.bfloat16
    device = torch.device(args.device)

    L0_model_path_parts = list(model_path.parts[model_path.is_absolute() :])
    try:
        model_index = L0_model_path_parts.index("model")
        L0_model_name = L0_model_path_parts[model_index + 1]
    except (ValueError, IndexError):
        L0_model_name = None

    L1_kvcache_path_parts = list(
        target_data_path.parts[target_data_path.is_absolute() :]
    )
    try:
        cache_index = L1_kvcache_path_parts.index("cache")
        L1_model_name = L1_kvcache_path_parts[cache_index + 1]
        input_hash = L1_kvcache_path_parts[cache_index + 2]
    except (ValueError, IndexError):
        L1_model_name = None
        input_hash = None

    L0_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    L0_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation="eager"
    ).to(device, dtype)
    L0_model.eval()

    torch.serialization.add_safe_globals([DynamicCache, set])
    target_datas = torch.load(target_data_path, weights_only=True)

    target_datas = tuple(
        (k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype))
        for (k, v) in target_datas
    )

    log_path = target_data_path.parent / "attack_log.jsonl"

    target_datas = target_datas[0]
    start_time = time.time()
    result_ids = inversion(L0_model, target_datas)
    time_cost = time.time() - start_time

    result = L0_tokenizer.decode(result_ids, skip_special_tokens=True)

    # Uncomment to save results to log file
    # inversion_log(
    #     log_path=log_path,
    #     L1_model_name=L1_model_name,
    #     L0_model_name=L0_model_name,
    #     input_hash=input_hash,
    #     target=target,
    #     result=result,
    #     result_ids=result_ids,
    #     time_cost=time_cost,
    # )
    # print(f"Results saved to {log_path}")

    print(f"Inversion result:\n{result}")


if __name__ == "__main__":
    main()
