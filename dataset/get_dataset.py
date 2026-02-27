import pandas as pd
import json
import csv
import ast
from pathlib import Path
import random

# --- Fix for lmsys-chat-1m field size limit ---
# Increase field size limit to 10MB (10 * 1024 * 1024), default 131072 is too small
csv.field_size_limit(10 * 1024 * 1024)

# --- Configuration ---
BASE_DIR = Path.home()
# Output directory
OUTPUT_DIR = BASE_DIR / "kvcache" / "dataset"
# Max samples required per dataset
MAX_SAMPLES = 1000

# Dataset processing definitions
DATASETS_CONFIG = [
    {
        "name": "gsm8k",
        "input_file": BASE_DIR
        / "dataset"
        / "gsm8k"
        / "main"
        / "train-00000-of-00001.parquet",
        "output_file": OUTPUT_DIR / f"gsm8k_{MAX_SAMPLES//1000}k.jsonl",
        "max_len": 150,
        "processor": "process_gsm8k",
    },
    {
        "name": "alpaca",
        "input_file": BASE_DIR
        / "dataset"
        / "alpaca"
        / "data"
        / "train-00000-of-00001-a09b74b3ef9c3b56.parquet",
        "output_file": OUTPUT_DIR / f"alpaca_{MAX_SAMPLES//1000}k.jsonl",
        "max_len": 100,
        "processor": "process_alpaca",
    },
    {
        "name": "lmsys-chat-1m",
        "input_file": BASE_DIR
        / "dataset"
        / "lmsys-chat-1m"
        / "data"
        / "train-00000-of-00006-4feeb3f83346a0e9.csv",
        "output_file": OUTPUT_DIR / f"lmsys-chat-1m_{MAX_SAMPLES//1000}k.jsonl",
        "max_len": 100,
        "processor": "process_lmsys",
    },
]


def save_jsonl(data, filepath: Path):
    """Save data to a JSONL file"""
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Successfully saved {len(data)} items to {filepath}")
    except Exception as e:
        print(f"Error saving file {filepath}: {e}")


# --- Processor 1: gsm8k (Parquet) ---
def process_gsm8k(config):
    """Processing gsm8k dataset"""
    print(f"\nProcessing {config['name']}...")
    try:
        df = pd.read_parquet(config["input_file"])
        # Filter condition: 'question' field length < max_len
        condition = df["question"].str.len() < config["max_len"]
        filtered_df = df[condition]

        sample_size = min(len(filtered_df), MAX_SAMPLES)
        sampled_df = filtered_df.sample(n=sample_size)
        data = sampled_df.to_dict("records")

        if len(data) < MAX_SAMPLES:
            print(
                f"Warning: Found only {len(data)} items matching criteria (required {MAX_SAMPLES})"
            )

        save_jsonl(data, config["output_file"])

    except FileNotFoundError:
        print(f"Error: Input file not found {config['input_file']}")
    except Exception as e:
        print(f"Error processing {config['name']}: {e}")


# --- Processor 2: alpaca (Parquet) ---
def process_alpaca(config):
    """Processing alpaca dataset"""
    print(f"\nProcessing {config['name']}...")
    try:
        df = pd.read_parquet(config["input_file"])

        instruction = df["instruction"].fillna("")
        input_text = df["input"].fillna("")

        # Filter condition: 'instruction' + 'input' total length < max_len
        combined_len = (instruction + input_text).str.len()
        condition = combined_len < config["max_len"]
        filtered_df = df[condition]

        sample_size = min(len(filtered_df), MAX_SAMPLES)
        sampled_df = filtered_df.sample(n=sample_size)
        data = sampled_df.to_dict("records")

        if len(data) < MAX_SAMPLES:
            print(
                f"Warning: Found only {len(data)} items matching criteria (required {MAX_SAMPLES})"
            )

        save_jsonl(data, config["output_file"])

    except FileNotFoundError:
        print(f"Error: Input file not found {config['input_file']}")
    except Exception as e:
        print(f"Error processing {config['name']}: {e}")


# --- Processor 3: lmsys-chat-1m (CSV) ---
def process_lmsys(config):
    """Processing lmsys-chat-1m dataset (reading CSV row by row)"""
    print(f"\nProcessing {config['name']}...")
    filtered_data = []

    try:
        with open(config["input_file"], "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if row.get("language") != "English":
                    continue

                convo_str = row.get("conversation")
                if not convo_str:
                    continue

                # --- Fix: Manually correcting formatting errors in CSV ---
                convo_str = convo_str.replace("}\n {", "}, {")

                moderation_str = row.get("openai_moderation", "[]")
                if moderation_str:
                    moderation_str = moderation_str.replace("}\n {", "}, {")

                try:
                    convo_list = ast.literal_eval(convo_str)

                    if (
                        isinstance(convo_list, list)
                        and len(convo_list) > 0
                        and isinstance(convo_list[0], dict)
                        and convo_list[0].get("role") == "user"
                    ):
                        first_prompt = convo_list[0].get("content")

                        # Check length of the first 'user' 'content'
                        if (
                            isinstance(first_prompt, str)
                            and len(first_prompt) < config["max_len"]
                        ):
                            moderation_data = ast.literal_eval(moderation_str)

                            output_row = {
                                "conversation_id": row.get("conversation_id"),
                                "model": row.get("model"),
                                "conversation": convo_list,
                                "turn": int(row.get("turn", 0)),
                                "language": row.get("language"),
                                "openai_moderation": moderation_data,
                                "redacted": (
                                    row.get("redacted", "false").lower() == "true"
                                ),
                            }
                            filtered_data.append(output_row)

                except (SyntaxError, ValueError, TypeError):
                    continue

        # --- NEW: Random Sampling Logic ---
        if len(filtered_data) < MAX_SAMPLES:
            print(
                f"Warning: Found only {len(filtered_data)} items matching criteria (required {MAX_SAMPLES})"
            )
            sampled_data = filtered_data
        else:
            # Randomly sample MAX_SAMPLES if we have more
            print(
                f"Found {len(filtered_data)} matching items. Sampling {MAX_SAMPLES}..."
            )
            # random.seed(42)  # --- REMOVED: No fixed seed for reproducibility ---
            sampled_data = random.sample(filtered_data, MAX_SAMPLES)

        # --- MODIFIED: Save the sampled_data ---
        save_jsonl(sampled_data, config["output_file"])

    except FileNotFoundError:
        print(f"Error: Input file not found {config['input_file']}")
    except Exception as e:
        print(f"Error processing {config['name']}: {e}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}\n")

    processors = {
        "process_gsm8k": process_gsm8k,
        "process_alpaca": process_alpaca,
        "process_lmsys": process_lmsys,
    }

    for config in DATASETS_CONFIG:
        processor_func = processors.get(config["processor"])
        if processor_func:
            processor_func(config)
        else:
            print(f"Warning: Processor named {config['processor']} not found")

    print("\nAll datasets processed successfully.")


if __name__ == "__main__":
    main()
