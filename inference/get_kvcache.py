import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from pdsplit import prefill


def main(model_name: str, device: torch.device, dtype: torch.dtype, dataset_name: str, max_samples: int = None):
    """
    加载模型和数据集，为指定数据集中的每个用户输入生成并保存 KV 缓存。
    """
    model_path = Path(f"~/model/{model_name}").expanduser()
    dataset_path = Path(dataset_name)

    if not model_path.is_dir():
        print(f"Warning: Model path does not exist, skipping: {model_path}")
        return
    if not dataset_path.is_file():
        print(f"Warning: Dataset path does not exist, skipping: {dataset_path}")
        return

    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation="eager"
    ).to(device, dtype)
    model.eval()
    print("Model loaded successfully.")

    print(f"Loading dataset from {dataset_path}...")
    dataset = []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        return

    if max_samples:
        dataset = dataset[:max_samples]
        print(f"Processing {len(dataset)} samples (limited to {max_samples}) from {dataset_path.name}...")
    else:
        print(f"Processing {len(dataset)} samples from {dataset_path.name}...")

    dtype_name = str(dtype).split(".")[-1]
    base_cache_dir_parent = Path(
        f"cache/{dtype_name}/{dataset_path.stem}/{model_path.name}/"
    )

    dataset_filename = dataset_path.name
    for i, item in enumerate(tqdm(dataset, desc=f"Processing {dataset_path.name}")):
        try:
            user_input = None

            if dataset_filename == "lmsys-chat-1m_1k.jsonl":
                conversation = item.get("conversation", [])
                for turn in conversation:
                    if turn.get("role") == "user":
                        user_input = turn.get("content")
                        break

            elif dataset_filename == "gsm8k_1k.jsonl":
                user_input = item.get("question")

            elif dataset_filename == "alpaca_1k.jsonl":
                instruction = item.get("instruction", "")
                input_text = item.get("input", "")

                # 构建 Alpaca 格式的 prompt
                if input_text:
                    # 包含 input 的格式
                    user_input = (
                        "Below is an instruction that describes a task, paired with an input that provides further context. "
                        "Write a response that appropriately completes the request.\n\n"
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Input:\n{input_text}"
                    )
                else:
                    # 不含 input 的格式
                    user_input = (
                        "Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        f"### Instruction:\n{instruction}"
                    )

            else:
                print(
                    f"Warning: Unknown dataset format for {dataset_filename}. Skipping sample {i}."
                )
                continue

            if user_input:
                input_hash = hashlib.sha1(user_input.encode("utf-8")).hexdigest()
                cache_dir = base_cache_dir_parent / input_hash

                prefill(model, tokenizer, user_input, cache_dir)
            else:
                print(
                    f"Warning: No user input extracted for sample {i} in {dataset_filename}. Skipping."
                )

        except Exception as e:
            print(
                f"An error occurred while processing sample {i} from {dataset_filename}: {e}"
            )
            continue

    print(
        f"\nFinished processing {dataset_path.name}. KV-Cache data saved in '{base_cache_dir_parent}'"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Generate KV-cache for a dataset.")
    parser.add_argument(
        "--model-name",
        default="Llama-3.2-1B",
        help="Model directory name under ~/model/.",
    )
    parser.add_argument(
        "--dataset",
        default="./dataset/lmsys-chat-1m_1k.jsonl",
        help="Path to dataset jsonl file.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Computation device (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing).",
    )
    args = parser.parse_args()

    print(
        f"\n--- Processing model: {args.model_name} on dataset: {args.dataset} ---"
    )
    main(
        model_name=args.model_name,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        dataset_name=args.dataset,
        max_samples=args.max_samples,
    )
