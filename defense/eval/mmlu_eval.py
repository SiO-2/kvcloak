import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from datasets import load_dataset, disable_progress_bar
from datetime import datetime
import json
import numpy as np
import os
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

# Disable Hugging Face dataset progress bar
disable_progress_bar()


class MMLUEvaluator:
    """
    A class for evaluating large language models on the MMLU dataset.
    """

    CHOICES = ["A", "B", "C", "D"]

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
        Initialize MMLUEvaluator.

        Args:
            model (AutoModelForCausalLM): Model to evaluate.
            tokenizer (AutoTokenizer): Tokenizer for the model.
            dataset (dict): MMLU dataset.
            device (str): Device to run the model on (e.g., 'cuda:0').
            output_filepath (Path): Output file path to save results.
            shot_count (int): Number of examples for few-shot evaluation.
            kvcloak (KVCloak, optional): KVCloak protector instance. Defaults to None.
            dp_protecter (KVCacheDPProtecter, optional): DP protector instance. Defaults to None.
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

    def _format_subject(self, subject: str) -> str:
        """Format subject name, e.g., 'abstract_algebra' -> 'Abstract Algebra'"""
        return " ".join(word.capitalize() for word in subject.split("_"))

    def _create_prompt(
        self, subject: str, question: str, choices: list, examples: list
    ) -> str:
        """Build prompt with few-shot examples."""
        formatted_subject = self._format_subject(subject)
        prompt = f"The following are multiple choice questions (with answers) about {formatted_subject}.\n\n"
        for ex in examples:
            prompt += f"Question: {ex['question']}\n"
            for i, choice in enumerate(ex["choices"]):
                prompt += f"{self.CHOICES[i]}. {choice}\n"
            prompt += f"Answer: {self.CHOICES[ex['answer']]}\n\n"
        prompt += f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{self.CHOICES[i]}. {choice}\n"
        prompt += "Answer"
        return prompt

    def _get_model_prediction(self, prompt: str, trigger: str) -> tuple[str, float]:
        """Get model prediction for the given prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model(
                **inputs, use_cache=True, pad_token_id=self.tokenizer.pad_token_id
            )
        past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)

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

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids, past_key_values=past_key_values, use_cache=True
            )
        time_cost = time.time() - start_time

        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
        prediction = self.tokenizer.decode(next_token_id, skip_special_tokens=True)
        clean_prediction = prediction.strip().upper()
        return (
            clean_prediction[0]
            if clean_prediction and clean_prediction[0] in self.CHOICES
            else "Z"
        ), time_cost

    def run_evaluation(self):
        """Run the MMLU evaluation loop."""
        all_subject_accuracies = {}
        subjects = sorted(self.dataset["test"].unique("subject"))

        for subject in tqdm(
            subjects, desc=f"Processing {self.shot_count}-shot Evaluation"
        ):
            infer_time = 0
            test_df = self.dataset["test"].filter(lambda x: x["subject"] == subject)
            dev_df = self.dataset["dev"].filter(lambda x: x["subject"] == subject)

            if len(dev_df) < self.shot_count:
                print(
                    f"Warning: Not enough dev examples for subject {subject}. "
                    f"Found {len(dev_df)}, need {self.shot_count}. Skipping."
                )
                continue

            dev_examples = dev_df.shuffle(seed=42).select(range(self.shot_count))
            correct_predictions = 0
            total_questions = len(test_df)

            for test_instance in test_df:
                prompt = self._create_prompt(
                    subject,
                    test_instance["question"],
                    test_instance["choices"],
                    dev_examples,
                )
                trigger = ":"

                model_pred_char, time_cost = self._get_model_prediction(prompt, trigger)
                infer_time += time_cost
                correct_answer_char = self.CHOICES[test_instance["answer"]]
                if model_pred_char == correct_answer_char:
                    correct_predictions += 1

            accuracy = (
                correct_predictions / total_questions if total_questions > 0 else 0
            )
            all_subject_accuracies[subject] = accuracy

            subject_result = {
                "subject": subject,
                "accuracy": accuracy,
                "correct_predictions": correct_predictions,
                "total_questions": total_questions,
                "infer_time": infer_time,
                "date": datetime.now().isoformat(),
            }
            with open(self.output_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(subject_result) + "\n")

        print("\n\n----- MMLU Evaluation Summary -----")
        if all_subject_accuracies:
            macro_avg_acc = sum(all_subject_accuracies.values()) / len(
                all_subject_accuracies
            )
            print(f"Final Macro Average Accuracy: {macro_avg_acc:.4f}")
        else:
            print("No subjects were evaluated.")

        print(f"All evaluation results saved to:\n{self.output_filepath}")


def main():
    parser = argparse.ArgumentParser(description="Run MMLU evaluation.")
    parser.add_argument("--model-name", default="Llama-3.2-1B")
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local model path. Defaults to ~/model/<model-name>.",
    )
    parser.add_argument("--dataset-path", default="~/dataset/mmlu/all")
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
    parser.add_argument("--shot-count", type=int, default=5)
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
        help="Output directory. Defaults to defense/result/mmlu/<protect-type>.",
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

    output_dir = Path(
        args.output_dir or f"defense/result/mmlu/{protect_type}"
    ).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path).expanduser()
    model_path = (
        Path(args.model_path).expanduser()
        if args.model_path is not None
        else Path(f"~/model/{model_name}").expanduser()
    )

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
            "test": str(dataset_path / "test-00000-of-00001.parquet"),
            "dev": str(dataset_path / "dev-00000-of-00001.parquet"),
            "validation": str(dataset_path / "validation-00000-of-00001.parquet"),
        }
        mmlu_dataset = load_dataset("parquet", data_files=data_files)
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")
        return

    kvcloak = None
    dp_protecter = None
    aes_protecter = None
    if protect_type == "kvcloak":
        kvcloak_config_path = Path(
            f"defense/config/kvcloak/b{args.block_size}_S{ratio_s}_M{ratio_m}_t{ratio_t}/{model_name}.pt"
        )
        output_filename = f"{model_name}_mmlu_{protect_type}_{dtype_name}_a{args.add_a}_f{args.fuse}_b{args.block_size}_S{ratio_s}_M{ratio_m}_t{ratio_t}_{datetime.now().strftime('%Y%m%d%H%M')}.jsonl"

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
        output_filename = f"{model_name}_mmlu_{protect_type}_{dtype_name}_cn{args.norm_percentile:g}_e{args.epsilon:.2e}_m{dp_protecter.noise_multiplier}_{datetime.now().strftime('%Y%m%d%H%M')}.jsonl"
    elif protect_type == "aes":
        if KVCacheAESProtecter is None:
            raise ModuleNotFoundError(
                "cryptography is required for AES mode. Install dependencies with `pip install -r requirements.txt`."
            )
        aes_key = os.urandom(16)
        aes_protecter = KVCacheAESProtecter(key=aes_key, device=device)
        output_filename = f"{model_name}_mmlu_{protect_type}_{dtype_name}_{datetime.now().strftime('%Y%m%d%H%M')}.jsonl"
    else:
        output_filename = f"{model_name}_mmlu_{protect_type}_{dtype_name}_{datetime.now().strftime('%Y%m%d%H%M')}.jsonl"

    output_filepath = output_dir / output_filename
    print(output_filepath, device)

    evaluator = MMLUEvaluator(
        model=model,
        tokenizer=tokenizer,
        dataset=mmlu_dataset,
        device=device,
        output_filepath=output_filepath,
        shot_count=args.shot_count,
        kvcloak=kvcloak,
        dp_protecter=dp_protecter,
        aes_protecter=aes_protecter,
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
