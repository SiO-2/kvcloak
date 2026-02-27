import argparse
import numpy as np
from pathlib import Path
import torch
from transformers.cache_utils import DynamicCache
from typing import Tuple, List
from tqdm import tqdm

from dp_accounting.rdp import RdpAccountant
from dp_accounting.dp_event import GaussianDpEvent

from defense.config.get_dp_norm import load_norms_from_file


def calibrate_noise_multiplier(
    target_epsilon: float,
    target_delta: float,
    composition_steps: int,
    tolerance: float = 1e-10,
) -> float:
    """
    Calibrate noise multiplier via binary search to satisfy given (epsilon, delta).

    Args:
        target_epsilon: Target privacy budget epsilon.
        target_delta: Target privacy budget delta.
        composition_steps: Number of DP event compositions (2 here: one for K, one for V).
        tolerance: Search precision.

    Returns:
        Minimum noise multiplier required to satisfy privacy budget.
    """
    if target_epsilon <= 0:
        raise ValueError("Target epsilon must be positive.")

    # Define a helper function to calculate epsilon for a given noise multiplier
    def get_epsilon(noise_multiplier: float) -> float:
        if noise_multiplier <= 0:
            return float("inf")
        accountant = RdpAccountant()
        dp_event = GaussianDpEvent(noise_multiplier)
        accountant.compose(dp_event, count=composition_steps)
        return accountant.get_epsilon(target_delta)

    # Binary search range
    low = 0.0
    high = 100.0  # A sufficiently large initial upper bound

    # Pre-check if upper bound can satisfy requirements, expand if not
    while get_epsilon(high) > target_epsilon:
        low = high
        high *= 2
        if high > 1e10:  # Prevent infinite loop
            raise ValueError(
                "Could not find a noise multiplier to satisfy the target epsilon. Epsilon might be too small."
            )

    # Start binary search
    while high - low > tolerance:
        mid = (low + high) / 2
        epsilon = get_epsilon(mid)
        if epsilon > target_epsilon:
            low = mid
        else:
            high = mid

    return high


class KVCacheDPApplier:
    def __init__(
        self, key_clip_norm: float, value_clip_norm: float, noise_multiplier: float
    ):
        if key_clip_norm <= 0 or value_clip_norm <= 0:
            raise ValueError("Clipping norms must be positive.")
        if noise_multiplier < 0:
            raise ValueError("Noise multiplier cannot be negative.")

        self.key_clip_norm = key_clip_norm
        self.value_clip_norm = value_clip_norm
        self.noise_multiplier = noise_multiplier
        self.key_noise_std = self.key_clip_norm * self.noise_multiplier
        self.value_noise_std = self.value_clip_norm * self.noise_multiplier

    def __call__(self, past_key_values: DynamicCache) -> DynamicCache:
        if not past_key_values:
            return past_key_values

        # Get device and batch_size info from the first layer
        first_key_tensor = past_key_values[0][0]
        device = first_key_tensor.device
        batch_size = first_key_tensor.shape[0]

        # --- First pass: Calculate sum of squared norms layer by layer to avoid creating huge tensors ---
        key_norms_sq = torch.zeros(batch_size, device=device)
        value_norms_sq = torch.zeros(batch_size, device=device)

        for key_layer, value_layer in past_key_values:
            # key_layer shape: (B, H, S, D)
            # Flatten each sample to calculate norm
            key_layer_flat = key_layer.reshape(batch_size, -1)
            value_layer_flat = value_layer.reshape(batch_size, -1)

            # Calculate and accumulate squared norms for current layer
            key_norms_sq.add_(torch.linalg.norm(key_layer_flat, ord=2, dim=-1).pow(2))
            value_norms_sq.add_(
                torch.linalg.norm(value_layer_flat, ord=2, dim=-1).pow(2)
            )

        # Take square root to get total L2 norm per sample across all layers
        key_norms = torch.sqrt(key_norms_sq)
        value_norms = torch.sqrt(value_norms_sq)

        # --- Calculate clipping scales ---
        key_clip_scales = torch.clamp(self.key_clip_norm / (key_norms + 1e-6), max=1.0)
        value_clip_scales = torch.clamp(
            self.value_clip_norm / (value_norms + 1e-6), max=1.0
        )

        # Reshape clipping scales to broadcastable shape (B,) -> (B, 1, 1, 1)
        # to multiply with layer tensors of shape (B, H, S, D)
        key_clip_scales_b = key_clip_scales.view(batch_size, 1, 1, 1)
        value_clip_scales_b = value_clip_scales.view(batch_size, 1, 1, 1)

        # --- Second pass: Apply clipping and noise ---
        private_kv_cache_list: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for key_layer, value_layer in past_key_values:
            # Apply clipping (this creates a new tensor, but only single-layer size)
            private_key_layer = key_layer * key_clip_scales_b
            private_value_layer = value_layer * value_clip_scales_b

            # Add noise (can be done in-place to save memory)
            if self.key_noise_std > 0:
                noise_k = torch.randn_like(private_key_layer)
                private_key_layer.add_(noise_k, alpha=self.key_noise_std)

            if self.value_noise_std > 0:
                noise_v = torch.randn_like(private_value_layer)
                private_value_layer.add_(noise_v, alpha=self.value_noise_std)

            private_kv_cache_list.append((private_key_layer, private_value_layer))

        # Convert list of tuples back to DynamicCache
        private_kv_cache = tuple(private_kv_cache_list)
        return DynamicCache.from_legacy_cache(private_kv_cache)

    def _apply_dp_to_tensor_per_sample(
        self, tensor: torch.Tensor, clip_norm: float, noise_std: float
    ) -> torch.Tensor:
        num_layers, batch_size, head_num, seq_len, head_dim = tensor.shape

        # Aggregate all vectors of one sample to calculate overall norm
        tensor_reshaped_for_norm = (
            tensor.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1)
        )

        # Calculate overall L2 norm for each sample
        norms = torch.linalg.norm(tensor_reshaped_for_norm, ord=2, dim=-1)

        # Calculate clipping scale for each sample (shape: B)
        clip_scales = torch.clamp(clip_norm / (norms + 1e-6), max=1.0)

        # To multiply with original tensor (L, B, H, S, D), need to reshape clipping scales to broadcastable shape
        # (B,) -> (1, B, 1, 1, 1)
        clip_scales_broadcastable = clip_scales.view(1, batch_size, 1, 1, 1)

        # Apply clipping (need to create new tensor here as in-place ops may not directly support broadcasting)
        private_tensor = tensor * clip_scales_broadcastable

        # Add noise
        if noise_std > 0:
            noise = torch.randn_like(private_tensor)
            private_tensor.add_(noise, alpha=noise_std)

        return private_tensor


class KVCacheDPProtecter:
    def __init__(
        self,
        key_clip_norm: float,
        value_clip_norm: float,
        epsilon: float,
        delta: float,
    ):
        self.epsilon = epsilon
        self.delta = delta

        # K and V each once, total 2 compositions
        self.noise_multiplier = calibrate_noise_multiplier(
            epsilon, delta, composition_steps=2
        )
        print(
            f"For epsilon={epsilon:.2e}, calculated noise_multiplier is: {self.noise_multiplier}"
        )
        # Use calculated noise multiplier to initialize data applier
        self.applier = KVCacheDPApplier(
            key_clip_norm, value_clip_norm, self.noise_multiplier
        )

    def protect(self, past_key_values: DynamicCache) -> DynamicCache:
        if not past_key_values:
            return past_key_values
        return self.applier(past_key_values)


def empirical_norm(
    past_key_values: DynamicCache,
) -> Tuple[np.ndarray, np.ndarray]:
    if not past_key_values:
        return np.array([]), np.array([])

    first_key_tensor = past_key_values[0][0]
    device = first_key_tensor.device
    batch_size = first_key_tensor.shape[0]

    # Also use layer-by-layer accumulation to calculate norms
    key_norms_sq = torch.zeros(batch_size, device=device)
    value_norms_sq = torch.zeros(batch_size, device=device)

    for key_layer, value_layer in past_key_values:
        key_layer_flat = key_layer.view(batch_size, -1)
        value_layer_flat = value_layer.view(batch_size, -1)

        key_norms_sq.add_(torch.linalg.norm(key_layer_flat, ord=2, dim=-1).pow(2))
        value_norms_sq.add_(torch.linalg.norm(value_layer_flat, ord=2, dim=-1).pow(2))

    k_norms_tensor = torch.sqrt(key_norms_sq)
    v_norms_tensor = torch.sqrt(value_norms_sq)

    k_norms = k_norms_tensor.to(device="cpu", dtype=torch.float32).numpy()
    v_norms = v_norms_tensor.to(device="cpu", dtype=torch.float32).numpy()

    return k_norms, v_norms


def test_per_sample_dp():
    torch.manual_seed(42)
    # Parameter settings
    model_name = "Llama-3.2-1B"
    norm_percentile = 50
    epsilon = 1e8
    delta = 1e-5
    dtype = torch.float32

    dtype_name = str(dtype).split(".")[-1]
    item_dir = Path(
        f"cache/{dtype_name}/config/{model_name}/301f7f48573352226c8b86de2a7eb654e9fef28b/"
    ).expanduser()
    try:
        kvcache_path = item_dir / "origin/past_key_values.pt"
        past_key_values_legacy = torch.load(kvcache_path, weights_only=True)
        past_key_values = DynamicCache.from_legacy_cache(past_key_values_legacy)

    except FileNotFoundError:
        print(f"Warning: KVCache file not found. Using dummy data.")

    dp_norm_path = Path(f"defense/config/dp_norm/{model_name}.json").expanduser()
    k_sample_norms, v_sample_norms = load_norms_from_file(dp_norm_path)

    clip_norm_k = float(np.percentile(k_sample_norms, norm_percentile))
    clip_norm_v = float(np.percentile(v_sample_norms, norm_percentile))

    print(f"Analysis complete:")
    print(f"  - Per-Sample Clip Norm (Key) at {norm_percentile}%: {clip_norm_k:.4f}")
    print(f"  - Per-Sample Clip Norm (Value) at {norm_percentile}%: {clip_norm_v:.4f}")

    dp_protecter = KVCacheDPProtecter(
        key_clip_norm=clip_norm_k,
        value_clip_norm=clip_norm_v,
        epsilon=epsilon,
        delta=delta,
    )

    dp_kvcache = dp_protecter.protect(past_key_values)

    dp_output_dir = item_dir / f"dp_cn{norm_percentile}_e{epsilon:.2e}"
    dp_output_dir.mkdir(parents=True, exist_ok=True)
    dp_kvcache_path = dp_output_dir / "past_key_values.pt"
    torch.save(dp_kvcache, dp_kvcache_path)
    print(dp_output_dir)


def process_model_cache(
    model_name: str,
    cache_path: str,
    protect_type: str,
    dp_protecter: KVCacheDPProtecter,
    device: torch.device,
):
    if not cache_path.exists():
        print(f"Warning: KV cache directory not found for {model_name} at {cache_path}")
        return

    # Use glob to get subdirectories
    sub_directories = [item for item in cache_path.glob("*") if item.is_dir()]

    if not sub_directories:
        print(f"No subdirectories found in '{cache_path}' to process.")
        return

    print(f"Processing directories in '{cache_path}':")
    for item_dir in tqdm(sub_directories, desc=f"Processing {model_name}"):
        input_hash = item_dir.name
        orig_kvcache_path = item_dir / "origin" / "past_key_values.pt"

        if not orig_kvcache_path.exists():
            print(f"Warning: Original KV cache not found at {orig_kvcache_path}")
            continue

        try:
            orig_kvcache = torch.load(orig_kvcache_path, weights_only=True)
        except Exception as e:
            print(
                f"Error loading original KV cache for {input_hash} in {model_name}: {e}"
            )
            continue

        orig_kvcache = tuple(
            (
                k.to(device=device),
                v.to(device=device),
            )
            for (k, v) in orig_kvcache
        )

        # Process and save DP protected KV cache
        dp_kvcache = dp_protecter.protect(orig_kvcache)
        dp_kvcache = tuple(
            (k.to(device="cpu"), v.to(device="cpu")) for (k, v) in dp_kvcache
        )
        dp_output_dir = item_dir / protect_type
        dp_output_dir.mkdir(parents=True, exist_ok=True)
        dp_kvcache_path = dp_output_dir / "past_key_values.pt"
        torch.save(dp_kvcache, dp_kvcache_path)


def main():
    parser = argparse.ArgumentParser(description="Apply DP protection to KV-cache.")
    parser.add_argument("--model-name", default="Llama-3.2-1B")
    parser.add_argument(
        "--dataset-path",
        default="./dataset/lmsys-chat-1m_1k.jsonl",
        help="Dataset path used to infer cache directory name.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--norm-percentile", type=float, default=50)
    parser.add_argument("--epsilon", type=float, default=1e8)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--dp-norm-path",
        default=None,
        help="Path to precomputed DP norm JSON. Defaults to defense/config/dp_norm/<model>.json",
    )
    parser.add_argument(
        "--cache-path",
        default=None,
        help="Direct cache directory path. If not set, built from dataset/model/dtype.",
    )
    parser.add_argument(
        "--protect-type",
        default=None,
        help="Output subdirectory name. Defaults to dp_cn<percentile>_e<epsilon>.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.serialization.add_safe_globals([DynamicCache, set])

    dp_norm_path = (
        Path(args.dp_norm_path).expanduser()
        if args.dp_norm_path is not None
        else Path(f"defense/config/dp_norm/{args.model_name}.json").expanduser()
    )
    k_sample_norms, v_sample_norms = load_norms_from_file(dp_norm_path)

    clip_norm_k = float(np.percentile(k_sample_norms, args.norm_percentile))
    clip_norm_v = float(np.percentile(v_sample_norms, args.norm_percentile))

    dp_protecter = KVCacheDPProtecter(
        key_clip_norm=clip_norm_k,
        value_clip_norm=clip_norm_v,
        epsilon=args.epsilon,
        delta=args.delta,
    )

    dtype_name = args.dtype
    dataset_name = Path(args.dataset_path).stem
    cache_path = (
        Path(args.cache_path).expanduser()
        if args.cache_path is not None
        else Path(f"cache/{dtype_name}/{dataset_name}/{args.model_name}").expanduser()
    )
    protect_type = args.protect_type or (
        f"dp_cn{args.norm_percentile:g}_e{args.epsilon:.2e}"
    )

    process_model_cache(
        args.model_name,
        cache_path,
        protect_type,
        dp_protecter,
        args.device,
    )


if __name__ == "__main__":
    main()
