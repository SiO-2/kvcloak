import argparse
from datetime import datetime
import json
import numpy as np
from pathlib import Path
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
from typing import List, Optional, Tuple
from scipy.stats import norm
from scipy.optimize import minimize_scalar


def attack_log(
    log_path,
    target_model_name,
    base_model_name,
    input_hash,
    attack_type,
    layer_idx,
    threshold,
    result,
    result_ids,
    time_cost,
):
    log = {
        "target model": target_model_name,
        "base model": base_model_name,
        "input hash": input_hash,
        "attack type": attack_type,
        "layer": layer_idx,
        "threshold": threshold,
        "result": result,
        "result token ids": result_ids.tolist(),
        "time": time_cost,
        "date": datetime.now().isoformat(),
    }

    with open(log_path, "a", encoding="utf-8") as outfile:
        json.dump(log, outfile, ensure_ascii=False)
        outfile.write("\n")


class ThresholdClassifier:
    """
    Classifier for computing optimal threshold based on two normal distributions
    and evaluating the probability of successful classification.
    """

    def __init__(self, mean_t, std_t, mean_f, std_f):
        """
        Initialize the classifier.

        Args:
            mean_t (float): Mean of distribution T (target distances).
            std_t (float): Standard deviation of distribution T.
            mean_f (float): Mean of distribution F (other distances).
            std_f (float): Standard deviation of distribution F.
        """
        # Note: We expect mean_t < mean_f for proper separation

        self.mean_t = mean_t
        self.std_t = std_t
        self.mean_f = mean_f
        self.std_f = std_f

    def calculate_success_probability(self, threshold, n):
        """
        Calculate the probability of successfully classifying all elements
        given a threshold.

        Args:
            threshold (float): The threshold for classification.
            n (int): Number of tokens in the sequence.

        Returns:
            float: Probability of successful classification.
        """
        if n < 2:
            raise ValueError("n must be >= 2")

        # Calculate correct classification probability for distribution T
        if self.std_t == 0:
            prob_t = 1.0 if self.mean_t < threshold else 0.0
        else:
            prob_t = norm.cdf(threshold, loc=self.mean_t, scale=self.std_t)

        # Calculate correct classification probability for distribution F
        if self.std_f == 0:
            prob_f = 1.0 if self.mean_f > threshold else 0.0
        else:
            prob_f = 1.0 - norm.cdf(threshold, loc=self.mean_f, scale=self.std_f)

        # Probability that all n-1 F tokens are correctly classified
        prob_all_f = np.power(prob_f, n - 1)

        return prob_t * prob_all_f

    def find_optimal_threshold(self, n: int):
        """Find the optimal threshold that maximizes success probability.
        
        Args:
            n (int): Number of tokens in the sequence.
            
        Returns:
            float: Optimal threshold value.
        """
        # If T distribution has zero variance, use mean directly
        if self.std_t == 0:
            threshold0 = self.mean_t
            return threshold0

        # Define negative success probability function for minimization
        def neg_success(thresh):
            p_t = norm.cdf(thresh, loc=self.mean_t, scale=self.std_t)
            p_f = 1.0 - norm.cdf(thresh, loc=self.mean_f, scale=self.std_f)
            return -(p_t * (p_f ** (n - 1)))

        # Bounded search between the two means
        if self.mean_t < self.mean_f:
            bounds = (self.mean_t, self.mean_f)
        else:
            bounds = (self.mean_f, self.mean_t)

        res = minimize_scalar(neg_success, bounds=bounds, method="bounded")

        opt_thresh = res.x

        return opt_thresh


def get_collision_threshold(
    config_path: str, layer_idx: int, n: int = 5
) -> Tuple[List[float], List[float]]:
    config_path = Path(config_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file path does not exist: {config_path}")
    else:
        with open(config_path, "r") as f:
            config = json.load(f)

    target_max = config[layer_idx]["target_max"]
    target_mean = config[layer_idx]["target_mean"]
    target_std = config[layer_idx]["target_std"]
    others_min = config[layer_idx]["others_min"]
    others_mean = config[layer_idx]["others_mean"]
    others_std = config[layer_idx]["others_std"]

    gap = [
        (others_mean[i] - target_mean[i]) / (target_std[i] + others_std[i])
        for i in range(len(target_max))
    ]

    if all(target_max[i] < others_min[i] for i in range(len(target_max))):
        threshold = [
            (target_max[i] + others_min[i]) / 2 for i in range(len(target_max))
        ]
    else:
        clf = [
            ThresholdClassifier(
                target_mean[i], target_std[i], others_mean[i], others_std[i]
            )
            for i in range(len(target_max))
        ]
        threshold = [clf[i].find_optimal_threshold(n) for i in range(len(target_max))]
    return threshold, gap


def collision(
    model: AutoModelForCausalLM,
    target_datas: DynamicCache,
    layer_idx: int,
    logical_batch_size: int,
    micro_batch_size: int,
    stop_partition: int,
    target_gap: int,
    threshold: Optional[List[float]] = None,
) -> torch.Tensor:
    """Perform collision attack to recover input from KV-cache.
    
    This attack searches for tokens that produce matching KV-cache entries
    by comparing distances between generated and target KV-cache.
    
    Args:
        model: The language model to use for attack
        target_datas: Target KV-cache to reconstruct
        layer_idx: Which layer to attack
        logical_batch_size: Number of tokens to consider per logical batch
        micro_batch_size: Batch size for inference
        stop_partition: Fraction of vocabulary to search (e.g., 3 means top 1/3)
        target_gap: Number of standard deviations below mean for auto-threshold
        threshold: Optional pre-computed distance threshold
        
    Returns:
        Reconstructed token IDs
    """
    device = model.device

    initial_sorted_ids = torch.arange(model.config.vocab_size, device=device)
    bos_ids = model.config.bos_token_id if model.config.bos_token_id else None

    result_ids = torch.empty((1, 0), dtype=torch.long, device=device)
    current_kvcache = None
    last_logits = None

    if bos_ids is not None:
        if isinstance(bos_ids, int):
            bos_ids = [bos_ids]
        bos_len = len(bos_ids)
        result_ids = torch.cat(
            [result_ids, torch.tensor([bos_ids], device=device)], dim=1
        )
        attention_mask = torch.ones((1, bos_len), dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = model(
                input_ids=result_ids, attention_mask=attention_mask, use_cache=True
            )
        last_logits = outputs.logits[-1, -1, :].squeeze(0)
        current_kvcache = DynamicCache.from_legacy_cache(outputs.past_key_values)
    else:
        bos_len = 0

    seq_len = target_datas[layer_idx][0].shape[2]

    for seq_id in range(bos_len, seq_len):
        if last_logits is None:
            sorted_ids = initial_sorted_ids
        else:
            probs = torch.nn.functional.softmax(last_logits, dim=-1)
            _, sorted_ids = torch.sort(probs, descending=True)

        sorted_ids = sorted_ids.cpu().tolist()
        found = False

        for logical_batch_start in range(
            0, len(sorted_ids) // stop_partition, logical_batch_size
        ):
            logical_batch_ids_list = sorted_ids[
                logical_batch_start : logical_batch_start + logical_batch_size
            ]
            if not logical_batch_ids_list:
                continue

            # Store results for all micro-batches within this logical batch
            all_micro_batch_data_dist = []
            all_micro_batch_kv_caches = []
            all_micro_batch_logits = []

            # Inner loop: process in micro-batches for inference
            for micro_batch_start in range(
                0, len(logical_batch_ids_list), micro_batch_size
            ):
                micro_batch_ids = logical_batch_ids_list[
                    micro_batch_start : micro_batch_start + micro_batch_size
                ]
                if not micro_batch_ids:
                    continue

                input_batch = torch.tensor(micro_batch_ids, device=device).unsqueeze(1)
                current_micro_batch_size = input_batch.size(0)

                if current_kvcache is not None and len(current_kvcache.key_cache) > 0:
                    past_length = current_kvcache.key_cache[0].shape[2]
                else:
                    past_length = 0

                attention_mask = torch.ones(
                    (current_micro_batch_size, past_length + 1),
                    dtype=torch.long,
                    device=device,
                )

                if current_kvcache is not None:
                    expanded_cache = DynamicCache()
                    for tmp_layer_idx in range(len(current_kvcache.key_cache)):
                        k = current_kvcache.key_cache[tmp_layer_idx]
                        v = current_kvcache.value_cache[tmp_layer_idx]
                        expanded_cache.update(
                            k.expand(current_micro_batch_size, -1, -1, -1),
                            v.expand(current_micro_batch_size, -1, -1, -1),
                            tmp_layer_idx,
                        )
                    past_key_values = expanded_cache
                else:
                    past_key_values = None

                with torch.no_grad():
                    outputs = model(
                        input_ids=input_batch,
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_hidden_states=True,
                    )

                batch_kvcache = DynamicCache.from_legacy_cache(outputs.past_key_values)
                current_datas = [
                    batch_kvcache[layer_idx][i][:, :, seq_id, :]
                    for i in range(len(batch_kvcache[layer_idx]))
                ]
                target_data = [
                    target_datas[layer_idx][i][0, :, seq_id, :]
                    for i in range(len(target_datas[layer_idx]))
                ]
                data_dist = [
                    torch.norm(current_datas[i] - target_data[i], dim=(1, 2))
                    for i in range(len(current_datas))
                ]

                # Store results for current micro-batch
                all_micro_batch_data_dist.append(data_dist)
                all_micro_batch_kv_caches.append(batch_kvcache.to_legacy_cache())
                all_micro_batch_logits.append(outputs.logits)

                torch.cuda.empty_cache()

            # Concatenate all micro-batch results
            # data_dist contains K and V distances, concatenate them separately
            num_kv_components = len(all_micro_batch_data_dist[0])
            full_batch_data_dist = [
                torch.cat(
                    [batch_dist[i] for batch_dist in all_micro_batch_data_dist], dim=0
                )
                for i in range(num_kv_components)
            ]

            # Concatenate logits and kv_cache for later selection
            full_batch_logits = torch.cat(all_micro_batch_logits, dim=0)
            num_layers = len(all_micro_batch_kv_caches[0])
            full_batch_kv_cache = tuple(
                (
                    torch.cat(
                        [cache[layer][0] for cache in all_micro_batch_kv_caches], dim=0
                    ),  # key
                    torch.cat(
                        [cache[layer][1] for cache in all_micro_batch_kv_caches], dim=0
                    ),  # value
                )
                for layer in range(num_layers)
            )

            if threshold is None:
                data_dist_mean = [d.mean() for d in full_batch_data_dist]
                data_dist_std = [d.std() for d in full_batch_data_dist]
                valid_mask = [
                    full_batch_data_dist[i]
                    < data_dist_mean[i] - target_gap * data_dist_std[i]
                    for i in range(len(full_batch_data_dist))
                ]
            else:
                valid_mask = [
                    full_batch_data_dist[i] < threshold[i] + 0.01 * threshold[i]
                    for i in range(len(full_batch_data_dist))
                ]

            stacked = torch.stack(valid_mask)
            all_true = torch.all(stacked, dim=0)
            valid_indices = torch.where(all_true)[0]

            if valid_indices.size(0) > 0:
                best_idx_in_logical_batch = valid_indices[0].item()
                selected_token = logical_batch_ids_list[best_idx_in_logical_batch]

                result_ids = torch.cat(
                    [result_ids, torch.tensor([[selected_token]], device=device)], dim=1
                )

                # Extract cache and logits for the best sample
                current_kvcache = DynamicCache.from_legacy_cache(
                    [
                        (
                            k[
                                best_idx_in_logical_batch : best_idx_in_logical_batch
                                + 1
                            ],
                            v[
                                best_idx_in_logical_batch : best_idx_in_logical_batch
                                + 1
                            ],
                        )
                        for k, v in full_batch_kv_cache
                    ]
                )
                last_logits = full_batch_logits[
                    best_idx_in_logical_batch, -1, :
                ].squeeze(0)

                found = True
                break  # Found match, exit logical batch loop

        if not found:
            break

    return result_ids.squeeze(0)


def main():
    parser = argparse.ArgumentParser(description="KV-Cache Collision Attack")
    parser.add_argument(
        "--model_path",
        help="Model path.",
        default="~/model/Llama-3.2-1B/",
        # default="~/model/DeepSeek-V2-Lite/",
    )
    parser.add_argument(
        "--target_data_path",
        help="Target data path.",
        default="cache/torch.float32/config/Llama-3.2-1B/6d0aba55c643e35809cae53f263941168b37b344/origin/past_key_values.pt",
    )
    parser.add_argument(
        "--config_path",
        help="Config path.",
        default=None,
    )
    parser.add_argument(
        "--layer_idx", help="The layer of KV-Cache (e.g., 0, 1, -1).", default=0
    )
    parser.add_argument(
        "--threshold",
        help="Distance matching threshold list. Accept JSON (e.g., [3.1,2.9]) or comma-separated values.",
        default=None,
    )
    parser.add_argument(
        "--logical_batch_size",
        type=int,
        default=256,
        help="Logical batch size for vocabulary candidates.",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=256,
        help="Micro batch size for each model forward.",
    )
    parser.add_argument(
        "--stop_partition",
        type=int,
        default=3,
        help="Only search top vocab_size/stop_partition candidates.",
    )
    parser.add_argument(
        "--target_gap",
        type=int,
        default=3,
        help="Auto-threshold gap when --threshold is not provided.",
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

    # Check if paths exist
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    if not target_data_path.exists():
        raise FileNotFoundError(f"Path does not exist: {target_data_path}")
    if args.config_path is not None:
        config_path = Path(args.config_path).expanduser()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file path does not exist: {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config_path = None
        config = None

    layer_idx = args.layer_idx
    if layer_idx is not None:
        layer_idx = int(layer_idx)
    threshold = args.threshold
    if threshold is not None:
        threshold = threshold.strip()
        if threshold.startswith("["):
            threshold = [float(x) for x in json.loads(threshold)]
        else:
            threshold = [float(x) for x in threshold.split(",") if x.strip()]
    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.bfloat16
    device = torch.device(args.device)

    base_model_path_parts = list(model_path.parts[model_path.is_absolute() :])
    try:
        model_index = base_model_path_parts.index("model")
        base_model_name = base_model_path_parts[model_index + 1]
    except (ValueError, IndexError):
        base_model_name = None

    target_data_path_parts = list(
        target_data_path.parts[target_data_path.is_absolute() :]
    )
    try:
        cache_index = target_data_path_parts.index("cache")
        target_model_name = target_data_path_parts[cache_index + 1]
        input_hash = target_data_path_parts[cache_index + 2]
    except (ValueError, IndexError):
        target_model_name = None
        input_hash = None

    base_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation="eager"
    ).to(device, dtype)
    base_model.eval()

    torch.serialization.add_safe_globals([DynamicCache, set])
    target_datas = torch.load(target_data_path, weights_only=True)

    target_datas = tuple(
        (k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype))
        for (k, v) in target_datas
    )

    log_path = target_data_path.parent / "attack_log.jsonl"

    if layer_idx is not None:
        layers = [layer_idx]
    else:
        # layers = list(range(len(target_datas)))
        layers = [0, len(target_datas) // 2, len(target_datas) - 1]

    # layers = [0, 1]
    test_times = 5
    for layer_idx in layers:
        if threshold is None and config_path is not None:
            target_mean = config[layer_idx]["target_mean"]
            target_std = config[layer_idx]["target_std"]
            others_mean = config[layer_idx]["others_mean"]
            others_std = config[layer_idx]["others_std"]
            # starts = target_mean + 2.5 * target_std
            starts = config[layer_idx]["target_max"]
            ends = [
                others_mean[i] - 2.4 * others_std[i] for i in range(len(others_mean))
            ]
            steps = [(ends[i] - starts[i]) / test_times for i in range(len(ends))]

            if any(step < 0 for step in steps):
                continue

            for idx in range(test_times):
                cur_threshold = [starts[i] + idx * steps[i] for i in range(len(starts))]
                start_time = time.time()
                result_ids = collision(
                    model=base_model,
                    target_datas=target_datas,
                    layer_idx=layer_idx,
                    logical_batch_size=args.logical_batch_size,
                    micro_batch_size=args.micro_batch_size,
                    stop_partition=args.stop_partition,
                    target_gap=args.target_gap,
                    threshold=cur_threshold,
                )
                time_cost = time.time() - start_time

                result = base_tokenizer.decode(result_ids, skip_special_tokens=True)

                attack_log(
                    log_path=log_path,
                    target_model_name=target_model_name,
                    base_model_name=base_model_name,
                    input_hash=input_hash,
                    attack_type="collision",
                    layer_idx=layer_idx,
                    threshold=cur_threshold,
                    result=result,
                    result_ids=result_ids,
                    time_cost=time_cost,
                )
        else:
            start_time = time.time()
            result_ids = collision(
                model=base_model,
                target_datas=target_datas,
                layer_idx=layer_idx,
                logical_batch_size=args.logical_batch_size,
                micro_batch_size=args.micro_batch_size,
                stop_partition=args.stop_partition,
                target_gap=args.target_gap,
                threshold=threshold,
            )
            time_cost = time.time() - start_time

            result = base_tokenizer.decode(result_ids, skip_special_tokens=True)

            attack_log(
                log_path=log_path,
                target_model_name=target_model_name,
                base_model_name=base_model_name,
                input_hash=input_hash,
                attack_type="collision",
                layer_idx=layer_idx,
                threshold=threshold,
                result=result,
                result_ids=result_ids,
                time_cost=time_cost,
            )

    print(f"Results saved to {log_path}")


if __name__ == "__main__":
    main()
