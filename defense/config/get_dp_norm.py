import argparse
from datasets import load_dataset, disable_progress_bar
from datetime import datetime
import gc
import json
import numpy as np
from pathlib import Path
import time
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.cache_utils import DynamicCache
import matplotlib.pyplot as plt


CHOICES = ["A", "B", "C", "D"]


def create_prompt(question: str, choices: list, examples: list) -> str:
    """Build a prompt containing few-shot examples."""
    prompt = f"The following are multiple choice questions (with answers).\n\n"
    for ex in examples:
        prompt += f"Question: {ex['question']}\n"
        for i, choice in enumerate(ex["choices"]):
            prompt += f"{CHOICES[i]}. {choice}\n"
        prompt += f"Answer: {CHOICES[ex['answer']]}\n\n"
    prompt += f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{CHOICES[i]}. {choice}\n"
    prompt += "Answer"
    return prompt


def calc_norms(model, tokenizer, dataset, num, shot_count, device):
    """
    Iterate through the dataset and calculate the L2 norm of Key and Value Cache for each sample during the Prefill phase.
    """
    test_df = dataset["test"]
    dev_df = dataset["dev"]

    if len(dev_df) < shot_count:
        print(
            f"Warning: Not enough dev examples for {shot_count}-shot. Using {len(dev_df)} examples."
        )
        shot_count = len(dev_df)

    dev_examples = dev_df.shuffle(seed=42).select(range(shot_count))
    test_examples = test_df.shuffle(seed=42).select(range(num))

    key_norms = []
    value_norms = []

    print(f"Calculating norms for {num} samples...")
    with torch.no_grad():
        for test_instance in tqdm(test_examples, desc="Processing Samples"):
            prompt = create_prompt(
                test_instance["question"],
                test_instance["choices"],
                dev_examples,
            )
            try:
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                outputs = model(**inputs)
            except:
                continue
            past_key_values = outputs.past_key_values

            all_keys = torch.stack([kv[0] for kv in past_key_values], dim=0)
            all_values = torch.stack([kv[1] for kv in past_key_values], dim=0)

            batch_size = all_keys.shape[1]
            assert batch_size == 1, "Please process texts one by one for this analysis."

            keys_reshaped = (
                all_keys.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1)
            )
            norm_k = torch.linalg.norm(keys_reshaped, ord=2, dim=-1).item()
            key_norms.append(norm_k)

            values_reshaped = (
                all_values.permute(1, 0, 2, 3, 4).contiguous().view(batch_size, -1)
            )
            norm_v = torch.linalg.norm(values_reshaped, ord=2, dim=-1).item()
            value_norms.append(norm_v)

    return key_norms, value_norms


def load_norms_from_file(filepath: Path):
    """
    Load norm data from file and calculate percentiles to select clipping norms.
    """
    if not filepath.exists():
        print(f"Error: File not found at {filepath}")
        return

    with open(filepath, "r") as f:
        data = json.load(f)

    key_norms = np.array(data["key_norms"])
    value_norms = np.array(data["value_norms"])

    return key_norms, value_norms


def main():
    parser = argparse.ArgumentParser(description="Compute DP clipping norm statistics.")
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
    parser.add_argument("--shot-count", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="defense/config/dp_norm")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        Path(args.model_path).expanduser()
        if args.model_path is not None
        else Path(f"~/model/{args.model_name}").expanduser()
    )
    dataset_path = Path(args.dataset_path).expanduser()
    output_filepath = output_dir / f"{args.model_name}.json"
    dtype = getattr(torch, args.dtype)

    try:
        print(f"Loading model '{args.model_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=dtype,
        ).to(args.device)
        model.eval()
    except Exception as e:
        print(f"Error loading model '{args.model_name}': {e}")
        return

    try:
        data_files = {
            "test": str(dataset_path / "test-00000-of-00001.parquet"),
            "dev": str(dataset_path / "dev-00000-of-00001.parquet"),
        }
        mmlu_dataset = load_dataset("parquet", data_files=data_files)
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")
        return

    key_norms, value_norms = calc_norms(
        model,
        tokenizer,
        mmlu_dataset,
        args.num_samples,
        args.shot_count,
        args.device,
    )

    norm_data = {"key_norms": key_norms, "value_norms": value_norms}
    with open(output_filepath, "w") as f:
        json.dump(norm_data, f, indent=4)
    print(f"Successfully saved norm data to {output_filepath}")

    del model, tokenizer, mmlu_dataset
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
