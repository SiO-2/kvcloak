import argparse
import torch
import time
import os
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Any
from transformers.cache_utils import DynamicCache
from transformers import AutoConfig

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from defense.core.kvcloak import KVCloak
try:
    from defense.baseline.aes_kvcache import KVCacheAESProtecter
except ModuleNotFoundError:
    KVCacheAESProtecter = None
from defense.config.get_kvcloak_config import get_kvcloak_config
from defense.config.get_theta_config import analyze_kv_cache
from defense.config.get_dp_norm import load_norms_from_file
from defense.baseline.dp_kvcache import KVCacheDPProtecter


def create_synthetic_kv_cache(
    num_hidden_layers: int,
    batch_size: int,
    num_key_value_heads: int,
    sequence_length: int,
    head_dim: int,
    dtype: torch.dtype,
    device: str,
) -> DynamicCache:
    """Generate a synthetic KV-Cache with specified dimensions filled with random data."""
    kv_cache_list = []
    with torch.no_grad():
        for _ in range(num_hidden_layers):
            key_states = torch.randn(
                (batch_size, num_key_value_heads, sequence_length, head_dim),
                dtype=dtype,
                device=device,
            )
            value_states = torch.randn(
                (batch_size, num_key_value_heads, sequence_length, head_dim),
                dtype=dtype,
                device=device,
            )
            kv_cache_list.append((key_states, value_states))
    return DynamicCache.from_legacy_cache(kv_cache_list)


def get_cache_size_in_gb(cache: DynamicCache) -> float:
    """Calculate the total size of the KV-Cache in GB."""
    total_size_bytes = 0
    for key_states, value_states in zip(cache.key_cache, cache.value_cache):
        total_size_bytes += key_states.nelement() * key_states.element_size()
        total_size_bytes += value_states.nelement() * value_states.element_size()
    return total_size_bytes / (1 << 30)


class BenchmarkRunner:
    """Encapsulates configuration, execution, and reporting logic for benchmark tests."""

    def __init__(
        self,
        model_name: str,
        model_config: dict,
        scenarios: dict,
        kvcloak_params: dict,
        dp_params: dict,
        num_trials: int,
        output_filepath: str,
        device: str,
        dtype: torch.dtype,
    ):
        self.model_name = model_name
        self.model_config = model_config
        self.scenarios = scenarios
        self.kvcloak_params = kvcloak_params
        self.dp_params = dp_params
        self.num_trials = num_trials
        self.output_filepath = output_filepath
        self.device = device
        self.dtype = dtype

        # _initialize_protectors creates multiple KVCloak instances
        self.protectors = self._initialize_protectors()

    def _initialize_protectors(self) -> Dict[str, Any]:
        """Initialize all protector objects centrally, including multiple KVCloak configurations."""
        print("--- Initializing Protectors ---")
        protectors = {}

        if KVCacheAESProtecter is None:
            print(
                "Warning: cryptography is not installed. Skipping AES protector in micro benchmark."
            )
        else:
            aes_key = os.urandom(16)
            protectors["AES"] = KVCacheAESProtecter(key=aes_key, device=self.device)

        dp_norm_path = Path(
            f"defense/config/dp_norm/{self.model_name}.json"
        ).expanduser()
        k_norms, v_norms = load_norms_from_file(dp_norm_path)
        clip_norm_k_empirical = float(
            np.percentile(k_norms, self.dp_params["norm_percentile"])
        )
        clip_norm_v_empirical = float(
            np.percentile(v_norms, self.dp_params["norm_percentile"])
        )
        protectors["DP"] = KVCacheDPProtecter(
            clip_norm_k_empirical,
            clip_norm_v_empirical,
            self.dp_params["epsilon"],
            self.dp_params["delta"],
        )

        print("Initializing KV-Cloak configs with a template cache...")
        template_cache = create_synthetic_kv_cache(
            num_hidden_layers=self.model_config["num_hidden_layers"],
            batch_size=1,
            num_key_value_heads=self.model_config["num_key_value_heads"],
            sequence_length=256,
            head_dim=self.model_config["head_dim"],
            dtype=self.dtype,
            device=self.device,
        )
        theta_config = analyze_kv_cache(template_cache)
        del template_cache

        for fused in self.kvcloak_params["fused_options"]:
            for block_size in self.kvcloak_params["block_sizes"]:
                # Generate a unique, descriptive name for each configuration
                name = f"KVCloak-B{block_size}-{'Fused' if fused else 'No_fuse'}"
                print(f"  - Initializing {name}...")

                mock_kvcloak_config = get_kvcloak_config(
                    num_hidden_layers=self.model_config["num_hidden_layers"],
                    num_key_value_heads=self.model_config["num_key_value_heads"],
                    head_dim=self.model_config["head_dim"],
                    block_size=block_size,
                    theta_config=theta_config,
                    S_ratio=1,
                    M_ratio=1,
                    theta_ratio=2.5,
                )
                protectors[name] = KVCloak(
                    kvcloak_config=mock_kvcloak_config,
                    dtype=self.dtype,
                    fused=fused,
                    need_ratio=False,
                    add_a=True,
                )

        return protectors

    def _time_operation(
        self, operation_func: Callable, kv_cache: DynamicCache
    ) -> tuple[float, float]:
        """Generic timing function to measure the average time of a single protection/recovery operation."""
        times = []
        with torch.no_grad():
            for _ in range(self.num_trials):
                torch.cuda.synchronize(self.device)
                start_time = time.time()
                intermediate_result = operation_func(kv_cache)
                torch.cuda.synchronize(self.device)
                end_time = time.time()
                times.append(end_time - start_time)
                del intermediate_result

        times_ms = np.array(times) * 1000
        return np.mean(times_ms), np.std(times_ms)

    def run(self):
        """Execute benchmarks for all scenarios."""
        print("\n--- Starting Micro-benchmark ---")
        print(
            f"Device: {self.device}, DType: {self.dtype}, Trials per scenario: {self.num_trials}"
        )

        print("Performing warm-up run...")
        warmup_cache = create_synthetic_kv_cache(
            num_hidden_layers=self.model_config["num_hidden_layers"],
            batch_size=1,
            num_key_value_heads=self.model_config["num_key_value_heads"],
            sequence_length=128,
            head_dim=self.model_config["head_dim"],
            dtype=self.dtype,
            device=self.device,
        )
        for name, protector in self.protectors.items():
            print(f"  - Warming up {name}...")
            if name == "AES":
                _ = protector.decrypt(protector.encrypt(warmup_cache))
            elif name == "DP":
                _ = protector.protect(warmup_cache)
            else:
                _ = protector.deobfuscate(protector.obfuscate(warmup_cache))
        del warmup_cache
        torch.cuda.empty_cache()
        print("Warm-up complete.")

        try:
            with open(self.output_filepath, "w", encoding="utf-8") as f:
                for scenario_name, params in self.scenarios.items():
                    print(f"\n--- Scenario: {self.model_name} {scenario_name} ---")
                    kv_cache = create_synthetic_kv_cache(
                        **self.model_config,
                        **params,
                        dtype=self.dtype,
                        device=self.device,
                    )
                    cache_size_gb = get_cache_size_in_gb(kv_cache)
                    print(
                        f"KV-Cache Spec: Batch={params['batch_size']}, SeqLen={params['sequence_length']}, Size={cache_size_gb:.3f} GB"
                    )

                    for protector_name, protector in self.protectors.items():
                        if protector_name == "AES":
                            op = lambda cache: protector.decrypt(
                                protector.encrypt(cache)
                            )
                        elif protector_name == "DP":
                            op = lambda cache: protector.protect(cache)
                        else:
                            op = lambda cache: protector.deobfuscate(
                                protector.obfuscate(cache)
                            )

                        mean_latency, std_latency = self._time_operation(op, kv_cache)
                        cv = (std_latency / mean_latency) if mean_latency > 0 else 0

                        # --- Build and write record in real-time ---
                        record = {
                            "scenario": scenario_name,
                            "cache_size_gb": cache_size_gb,
                            "protector": protector_name,
                            "latency_ms_mean": mean_latency,
                            "latency_ms_std": std_latency,
                            "latency_cv": cv,
                            "num_trials": self.num_trials,
                            "batch_size": params["batch_size"],
                            "sequence_length": params["sequence_length"],
                            "model_name": self.model_name,
                            "dtype": str(self.dtype),
                            "device": self.device,
                            "timestamp": datetime.now().isoformat(),
                        }
                        f.write(json.dumps(record) + "\n")
                        print(
                            f"  - Logged result for {protector_name}: {mean_latency:.2f} ± {std_latency:.2f} ms (CV: {cv:.2%})"
                        )

                    del kv_cache
                    torch.cuda.empty_cache()
            print(f"\nBenchmark finished. All results saved to {self.output_filepath}")
        except IOError as e:
            print(f"Error during benchmark and file writing: {e}")


def main():
    """Configure and start the benchmark."""

    def parse_int_list(raw: str) -> list[int]:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]

    def parse_bool_list(raw: str) -> list[bool]:
        mapping = {
            "1": True,
            "0": False,
            "true": True,
            "false": False,
            "yes": True,
            "no": False,
        }
        values = []
        for item in raw.split(","):
            token = item.strip().lower()
            if not token:
                continue
            if token not in mapping:
                raise ValueError(f"Invalid boolean value in list: {item}")
            values.append(mapping[token])
        return values

    parser = argparse.ArgumentParser(description="Run micro benchmark for KV protections.")
    parser.add_argument("--model-name", default="Llama-3.2-1B")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local model path. Defaults to ~/model/<model-name>.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-seq-len", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--seq-step", type=int, default=256)
    parser.add_argument("--kvcloak-block-sizes", default="16,32,64")
    parser.add_argument("--kvcloak-fused-options", default="false,true")
    parser.add_argument("--dp-norm-percentile", type=float, default=50)
    parser.add_argument("--dp-epsilon", type=float, default=1e8)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--output-dir", default="defense/result/micro_benchmark")
    args = parser.parse_args()

    if args.seq_step <= 0:
        raise ValueError("--seq-step must be > 0")
    if args.max_seq_len < args.min_seq_len:
        raise ValueError("--max-seq-len must be >= --min-seq-len")

    model_path = (
        Path(args.model_path).expanduser()
        if args.model_path is not None
        else Path(f"~/model/{args.model_name}").expanduser()
    )
    model_config = AutoConfig.from_pretrained(model_path)

    if model_config.model_type == "gpt2":
        num_hidden_layers = model_config.n_layer
        num_key_value_heads = model_config.n_head
        head_dim = model_config.n_embd // model_config.n_head
    elif model_config.model_type in ["qwen2", "llama", "llama3"]:
        num_hidden_layers = model_config.num_hidden_layers
        num_key_value_heads = model_config.num_key_value_heads
        head_dim = model_config.hidden_size // model_config.num_attention_heads
    else:
        raise ValueError(f"Model type {model_config.model_type} not supported.")

    scenarios = {}
    for s in range(args.min_seq_len, args.max_seq_len + 1, args.seq_step):
        scenarios[f"b{args.batch_size}_s{s}"] = {
            "batch_size": args.batch_size,
            "sequence_length": s,
        }

    dtype = getattr(torch, args.dtype)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dtype_str = str(dtype).split(".")[-1]
    output_filename = f"{args.model_name}_{dtype_str}_{timestamp}.jsonl"
    output_filepath = output_dir / output_filename

    config = {
        "device": args.device,
        "dtype": dtype,
        "model_name": args.model_name,
        "num_trials": args.num_trials,
        "output_filepath": output_filepath,
        "model_config": {
            "num_hidden_layers": num_hidden_layers,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
        },
        "scenarios": scenarios,
        "kvcloak_params": {
            "block_sizes": parse_int_list(args.kvcloak_block_sizes),
            "fused_options": parse_bool_list(args.kvcloak_fused_options),
        },
        "dp_params": {
            "norm_percentile": args.dp_norm_percentile,
            "epsilon": args.dp_epsilon,
            "delta": args.dp_delta,
        },
    }

    runner = BenchmarkRunner(**config)
    runner.run()


if __name__ == "__main__":
    main()
