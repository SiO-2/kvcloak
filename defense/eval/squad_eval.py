import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from datasets import load_dataset, disable_progress_bar
from datetime import datetime
import gc
import json
import numpy as np
import os
import random
import re
import string
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

try:
    from defense.baseline.aes_kvcache import KVCacheAESProtecter
except ModuleNotFoundError:
    KVCacheAESProtecter = None
from defense.core.fusion import fusion
from defense.core.kvcloak import KVCloak
from defense.baseline.dp_kvcache import KVCacheDPProtecter
from defense.config.get_dp_norm import load_norms_from_file

# --- Global Settings ---
disable_progress_bar()


class SQuADEvaluator:
    """
    Class for Few-Shot evaluation of a given causal language model on the SQuAD v1.1 dataset.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: dict,
        device: str,
        output_filepath: Path,
        shot_count: int,
        kvcloak: KVCloak = None,
        dp_protecter: KVCacheDPProtecter = None,
        aes_protecter=None,
    ):
        """
        Initialize SQuADEvaluator.

        Args:
            model (AutoModelForCausalLM): Language model to evaluate.
            tokenizer (AutoTokenizer): Tokenizer for the model.
            dataset (dict): Loaded SQuAD dataset.
            device (str): Device to run the model on (e.g., "cuda:0").
            output_filepath (Path): Full path to save JSONL results file.
            shot_count (int): Number of examples for Few-shot evaluation.
            kvcloak (Optional[KVCloak]): (Optional) KVCloak instance for KV cache protection.
            dp_protecter (Optional[KVCacheDPProtecter]): (Optional) KVCacheDPProtecter instance for KV cache protection.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.device = device
        self.output_filepath = output_filepath
        self.shot_count = shot_count
        self.kvcloak = kvcloak
        self.dp_protecter = dp_protecter
        self.aes_protecter = aes_protecter

        self.model.eval()
        newline_token_id = self.tokenizer.encode("\n\n", add_special_tokens=False)[-1]
        self.stop_token_ids = {self.tokenizer.eos_token_id, newline_token_id}

    @staticmethod
    def _normalize_text(s: str) -> str:
        """Text normalization: lowercase, remove punctuation, articles, and extra whitespace."""
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = "".join(ch for ch in s if ch not in set(string.punctuation))
        s = " ".join(s.split())
        return s

    def _calculate_em(self, prediction: str, ground_truths: list[str]) -> bool:
        """Calculate Exact Match (EM) score."""
        normalized_prediction = self._normalize_text(prediction)
        for truth in ground_truths:
            if self._normalize_text(truth) == normalized_prediction:
                return 1
        return 0

    def _create_prompt(self, test_instance: dict, example: dict) -> str:
        """Build SQuAD prompt with 1-shot example."""
        return (
            f"Context: {example['context']}\n"
            f"Question: {example['question']}\n"
            f"Answer: {example['answers']['text'][0]}\n\n"
            f"Context: {test_instance['context']}\n"
            f"Question: {test_instance['question']}\n"
            f"Answer"
        )

    def _get_model_prediction(
        self, prompt: str, trigger: str, max_new_tokens: int = 64
    ) -> tuple[str, float]:
        """Get model prediction for the given prompt."""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=1024, truncation=True
        ).to(self.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(
                **inputs, use_cache=True, pad_token_id=self.tokenizer.pad_token_id
            )
            past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)

            # Apply protection if protector instances are provided
            if self.kvcloak:
                obfuscated_cache = self.kvcloak.obfuscate(past_key_values)
                past_key_values = self.kvcloak.deobfuscate(obfuscated_cache)
            elif self.dp_protecter:
                past_key_values = self.dp_protecter.protect(past_key_values)
            elif self.aes_protecter:
                encrypted_cache = self.aes_protecter.encrypt(past_key_values)
                past_key_values = self.aes_protecter.decrypt(encrypted_cache)

            input_ids = self.tokenizer.encode(
                trigger, add_special_tokens=False, return_tensors="pt"
            ).to(self.model.device)

            generated_ids = []
            for _ in range(max_new_tokens):
                outputs = self.model(
                    input_ids=input_ids, past_key_values=past_key_values, use_cache=True
                )
                next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)

                if next_token_id.item() in self.stop_token_ids:
                    break

                generated_ids.append(next_token_id.item())
                past_key_values = outputs.past_key_values
                input_ids = next_token_id.unsqueeze(0)

        time_cost = time.time() - start_time
        return (
            self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
            time_cost,
        )

    def run_evaluation(self):
        """Run the evaluation loop."""
        validation_set = self.dataset["validation"]
        train_set = self.dataset["train"]
        em_scores = []
        for test_instance in tqdm(
            validation_set, desc=f"Processing {self.shot_count}-shot SQuAD"
        ):
            one_shot_example = random.choice(train_set)
            prompt = self._create_prompt(test_instance, one_shot_example)
            trigger = ":"
            try:
                model_prediction, infer_time = self._get_model_prediction(
                    prompt, trigger
                )
            except:
                continue

            ground_truths = test_instance["answers"]["text"]
            is_em = self._calculate_em(model_prediction, ground_truths)
            em_scores.append(is_em)

            # Write results in real-time
            result = {
                "id": test_instance["id"],
                "question": test_instance["question"],
                "prediction": model_prediction,
                "ground_truths": ground_truths,
                "is_em": is_em,
                "infer_time": infer_time,
                "date": datetime.now().isoformat(),
            }
            with open(self.output_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")

        # --- Calculate and report final results ---
        total_questions = len(validation_set)
        em_score = sum(em_scores) / total_questions if total_questions > 0 else 0

        print("\n\n----- SQuAD Evaluation Summary -----")
        print(
            f"exact_match_score: {em_score:.4f}\nAll evaluation results saved to:\n{str(self.output_filepath)}"
        )


def main():
    parser = argparse.ArgumentParser(description="Run SQuAD evaluation.")
    parser.add_argument("--model-name", default="Llama-3.2-1B")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local model path. Defaults to ~/model/<model-name>.",
    )
    parser.add_argument("--dataset-path", default="~/dataset/squad/plain_text")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument(
        "--protect-type",
        default=os.getenv("PROTECT_TYPE", "origin"),
        choices=["origin", "kvcloak", "dp", "aes"],
    )
    parser.add_argument("--shot-count", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--theta-ratio", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--S-ratio", type=float, default=1.0)
    parser.add_argument("--M-ratio", type=float, default=1.0)
    parser.add_argument(
        "--fuse",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--add-a",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--need-ratio",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--epsilon", type=float, default=1e8)
    parser.add_argument("--norm-percentile", type=float, default=50)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to defense/result/squad/<protect-type>.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.serialization.add_safe_globals([DynamicCache, set])

    model_name = args.model_name
    device = args.device
    dtype = getattr(torch, args.dtype)
    dtype_name = str(dtype).split(".")[-1]
    protect_type = args.protect_type
    theta_ratio = args.theta_ratio
    if theta_ratio is None:
        theta_ratio = 2.5 if model_name == "llama-7b" else 2

    ratio_s = f"{args.S_ratio:g}"
    ratio_m = f"{args.M_ratio:g}"
    ratio_t = f"{theta_ratio:g}"

    dataset_path = Path(args.dataset_path).expanduser()
    output_dir = Path(
        args.output_dir or f"defense/result/squad/{protect_type}"
    ).expanduser()
    model_path = (
        Path(args.model_path).expanduser()
        if args.model_path is not None
        else Path(f"~/model/{model_name}").expanduser()
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, attn_implementation="eager"
        ).to(device, dtype)
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        return

    try:
        data_files = {
            "train": str(dataset_path / "train-00000-of-00001.parquet"),
            "validation": str(dataset_path / "validation-00000-of-00001.parquet"),
        }
        squad_dataset = load_dataset("parquet", data_files=data_files)
    except Exception as e:
        print(f"Error loading SQuAD dataset: {e}")
        return

    kvcloak = None
    dp_protecter = None
    aes_protecter = None
    if protect_type == "kvcloak":
        kvcloak_config_path = Path(
            f"defense/config/kvcloak/b{args.block_size}_S{ratio_s}_M{ratio_m}_t{ratio_t}/{model_name}.pt"
        )
        output_filename = f"{model_name}_squad_{protect_type}_{dtype_name}_a{args.add_a}_f{args.fuse}_b{args.block_size}_S{ratio_s}_M{ratio_m}_t{ratio_t}_{datetime.now().strftime('%Y%m%d%H%M')}.jsonl"

        try:
            kvcloak_config = torch.load(kvcloak_config_path)
            if args.fuse:
                model = fusion(model, kvcloak_config)
            kvcloak = KVCloak(
                kvcloak_config,
                dtype,
                args.fuse,
                args.need_ratio,
                args.add_a,
            )
        except FileNotFoundError:
            print(
                f"Error: KV Cloak config not found for {model_name} at {kvcloak_config_path}"
            )
            return
        except Exception as e:
            print(f"Error loading KV Cloak config for {model_name}: {e}")
            return
    elif protect_type == "dp":
        dp_norm_path = Path(f"defense/config/dp_norm/{model_name}.json").expanduser()
        k_norms, v_norms = load_norms_from_file(dp_norm_path)
        clip_norm_k_empirical = float(np.percentile(k_norms, args.norm_percentile))
        clip_norm_v_empirical = float(np.percentile(v_norms, args.norm_percentile))
        dp_protecter = KVCacheDPProtecter(
            clip_norm_k_empirical,
            clip_norm_v_empirical,
            args.epsilon,
            args.delta,
        )
        output_filename = f"{model_name}_squad_{protect_type}_{dtype_name}_cn{args.norm_percentile:g}_e{args.epsilon:.2e}_m{dp_protecter.noise_multiplier}_{datetime.now().strftime('%Y%m%d%H%M')}.jsonl"
    elif protect_type == "aes":
        if KVCacheAESProtecter is None:
            raise ModuleNotFoundError(
                "cryptography is required for AES mode. Install dependencies with `pip install -r requirements.txt`."
            )
        aes_key = os.urandom(16)
        aes_protecter = KVCacheAESProtecter(key=aes_key, device=device)
        output_filename = f"{model_name}_squad_{protect_type}_{dtype_name}_{datetime.now().strftime('%Y%m%d%H%M')}.jsonl"
    else:
        output_filename = f"{model_name}_squad_{protect_type}_{dtype_name}_{datetime.now().strftime('%Y%m%d%H%M')}.jsonl"

    output_filepath = output_dir / output_filename
    print(output_filepath, device)

    evaluator = SQuADEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset=squad_dataset,
        device=device,
        output_filepath=output_filepath,
        shot_count=args.shot_count,
        kvcloak=kvcloak,
        dp_protecter=dp_protecter,
        aes_protecter=aes_protecter,
    )
    evaluator.run_evaluation()


# --- How to Call ---
if __name__ == "__main__":
    main()
