import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def prefill(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_text: str,
    cache_dir: Path,
):
    device = model.device
    dtype = model.dtype
    cache_dir.mkdir(parents=True, exist_ok=True)

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # L0 模型推理时缺失 bos，手动补上
    bos_token_id = model.config.bos_token_id if model.config.bos_token_id else None
    if bos_token_id is not None:
        if isinstance(bos_token_id, int):
            bos_token_id = [bos_token_id]

        bos_len = len(bos_token_id)
        if inputs["input_ids"][-1, :bos_len].tolist() != bos_token_id:
            inputs["input_ids"] = torch.cat(
                [torch.tensor([bos_token_id]).to(device), inputs["input_ids"]], dim=1
            )
            inputs["attention_mask"] = torch.cat(
                [torch.ones(1, bos_len).to(device), inputs["attention_mask"]], dim=1
            )

    start_time = time.time()
    with torch.no_grad():
        outputs = model(
            **inputs, use_cache=True, output_hidden_states=True, output_attentions=True
        )

    past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)

    past_key_values = tuple(
        (k.to(device="cpu", dtype=dtype), v.to(device="cpu", dtype=dtype))
        for (k, v) in past_key_values
    )
    # 保存中间结果
    intermediate = {
        "attentions": outputs.attentions,
        "hidden_states": outputs.hidden_states,
        "past_key_values": past_key_values,
    }

    for name, data in intermediate.items():
        origin_dir = cache_dir / "origin"
        origin_dir.mkdir(parents=True, exist_ok=True)
        path = origin_dir / f"{name}.pt"
        torch.save(data, path)

    # 获取下一个 token 的 logits
    logits = outputs.logits[:, -1, :]  # 取最后一个 token 的 logits
    probs = torch.nn.functional.softmax(logits, dim=-1)  # 转换为概率分布
    next_token_id = torch.argmax(probs, dim=-1)  # 选择概率最高的 token

    # 将 token ID 转换为文本
    next_token = tokenizer.decode(next_token_id, skip_special_tokens=True)

    # 保存结果
    result = {
        "input": input_text,
        "input length": inputs.input_ids.shape[-1],
        "next token": next_token,
        "time": time.time() - start_time,
        "date": datetime.now().isoformat(),
        "input token ids": inputs.input_ids.squeeze().tolist(),
        "next token id": next_token_id.item(),
        "dtype": str(model.dtype),
    }

    result_path = cache_dir / "decode.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 不再在此处打印，由主循环控制
    # print(f"Results saved to {result_path}")
    return past_key_values, next_token_id


def decode(
    model: AutoModelForCausalLM,
    past_key_values: DynamicCache,
    next_token_id: torch.Tensor,
    max_length: int,
):
    generated_tokens = [next_token_id.item()]
    current_token = next_token_id.view(1, 1)  # 保持输入维度为(batch_size=1, seq_len=1)
    device = model.device  # 获取模型所在的设备
    dtype = model.dtype
    past_key_values = tuple(
        (k.to(device=device, dtype=dtype), v.to(device=device, dtype=dtype))
        for (k, v) in past_key_values
    )
    past_key_values = DynamicCache.from_legacy_cache(past_key_values)

    for _ in range(max_length - 1):  # 已生成1个token，继续生成max_length-1个
        # 计算当前的past序列长度
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        else:
            past_length = 0
        current_length = past_length + current_token.size(1)
        # 构造attention_mask，形状为 (1, current_length)
        attention_mask = torch.ones(
            (1, current_length),
            dtype=torch.long,  # 使用与prefill阶段相同的类型
            device=device,
        )

        with torch.no_grad():
            outputs = model(
                input_ids=current_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

        # 获取下一个token
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)

        # 更新状态
        generated_tokens.append(next_token_id.item())
        past_key_values = DynamicCache.from_legacy_cache(outputs.past_key_values)
        current_token = next_token_id.view(1, 1)  # 保持输入维度

    # 将token IDs转换为文本
    return generated_tokens


def main():
    parser = argparse.ArgumentParser(description="Prefill-Decode split inference.")
    parser.add_argument(
        "--model_path",
        help="Model path.",
        default="~/model/Llama-3.2-1B",
        # default="~/model/Llama-3.2-3B-Instruct/",
        # default="~/model/Meta-Llama-3.1-8B/",
        # default="~/model/DeepSeek-R1-Distill-Llama-8B/",
        # default="~/model/Qwen2.5-Math-7B",
        # default="~/model/gpt2",
        # default="~/model/llama-7b",
    )
    parser.add_argument(
        "--input",
        help="User input.",
        # default="A robot may not injure a human being, or, through inaction, allow a human being to come to harm.",
        default="One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning. The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries.",
        # default="The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation.  There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computerchess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that 'brute force' search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research.In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning.In speech recognition, there was an early competition, sponsored by DARPA, in the 1970s. Entrants included a host of special methods that took advantage of human knowledge---knowledge of words, of phonemes, of the human vocal tract, etc. On the other side were newer methods that were more statistical in nature and did much more computation, based on hidden Markov models (HMMs). Again, the statistical methods won out over the human-knowledge-based methods. This led to a major change in all of natural language processing, gradually over decades, where statistics and computation came to dominate the field. The recent rise of deep learning in speech recognition is the most recent step in this consistent direction. Deep learning methods rely even less on human knowledge, and use even more computation, together with learning on huge training sets, to produce dramatically better speech recognition systems. As in the games, researchers always tried to make systems that worked the way the researchers thought their own minds worked---they tried to put that knowledge in their systems---but it proved ultimately counterproductive, and a colossal waste of researcher's time, when, through Moore's law, massive computation became available and a means was found to put it to good use.In computer vision, there has been a similar pattern. Early methods conceived of vision as searching for edges, or generalized cylinders, or in terms of SIFT features. But today all this is discarded. Modern deep-learning neural networks use only the notions of convolution and certain kinds of invariances, and perform much better.This is a big lesson. As a field, we still have not thoroughly learned it, as we are continuing to make the same kind of mistakes. To see this, and to effectively resist it, we have to understand the appeal of these mistakes. We have to learn the bitter lesson that building in how we think we think does not work in the long run. The bitter lesson is based on the historical observations that 1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term, and is personally satisfying to the researcher, but 3) in the long run it plateaus and even inhibits further progress, and 4) breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning. The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning.The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries. All these are part of the arbitrary, intrinsically-complex, outside world. They are not what should be built in, as their complexity is endless; instead we should build in only the meta-methods that can find and capture this arbitrary complexity. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.",
    )
    parser.add_argument(
        "--max_length", help="Maximum inference length (e.g., 1024).", default=16
    )
    parser.add_argument(
        "--device", help="Device to use (e.g., cuda:0, cpu).", default="cuda:6"
    )
    parser.add_argument(
        "--dtype", help="Data type (e.g., bfloat16, float32).", default="float32"
    )

    args = parser.parse_args()
    torch.manual_seed(42)

    # 解析路径中的 ~
    model_path = Path(args.model_path).expanduser()
    # 检查路径是否存在
    if not Path(model_path).is_dir():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    input = args.input
    max_length = int(args.max_length)
    dtype = getattr(torch, args.dtype) if hasattr(torch, args.dtype) else torch.float32
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )  # 加载tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation="eager"
    ).to(
        device, dtype
    )  # 加载预训练模型
    model.eval()

    dtype_name = str(dtype).split(".")[-1]
    cache_dir = Path(
        f'./cache/{dtype_name}/config/{model_path.name}/{hashlib.sha1(bytes(input, encoding="utf8")).hexdigest()}/'
    )
    past_key_values, next_token_id = prefill(model, tokenizer, input, cache_dir)
    generated_tokens = decode(model, past_key_values, next_token_id, max_length)
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Generated text:\n{generated_text}")
    print(str(cache_dir))


if __name__ == "__main__":
    main()
