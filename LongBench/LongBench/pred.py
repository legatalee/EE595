import os
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, LlamaTokenizer, LlamaForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import numpy as np
import random
import argparse
# from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
import torch.distributed as dist
import torch.multiprocessing as mp
from backends.llada_local import LLaDALocal


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        choices=["llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k",
                                 "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k",
                                 "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
                                 "llada-local"])  # ← 추가
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")

    # LLaDA 전용 옵션 추가
    parser.add_argument('--llada_gen_length', type=int, default=128)
    parser.add_argument('--llada_steps', type=int, default=128)
    parser.add_argument('--llada_block_size', type=int, default=32)
    parser.add_argument('--llada_use_cache', action='store_true')
    parser.add_argument('--llada_if_cache_position', action='store_true')
    parser.add_argument('--llada_threshold', type=float, default=None)
    parser.add_argument('--llada_delay_commit', action='store_true')
    parser.add_argument('--llada_max_length', type=int, default=32768)

    return parser.parse_args(args)

# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(rank, world_size, data, max_length, max_gen, prompt_format,
             dataset, device, model_name, model2path, out_path,
             backend="hf", runner=None):
    device = torch.device(f'cuda:{rank}')

    if backend == "llada-local":
        pass
    else:
        model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        if backend == "llada-local":
            pred, _ = runner.generate_once(prompt_text=prompt, system_prompt=None)
        else:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in model_name:
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt", add_special_tokens=False).input_ids[0]
            if len(tokenized_prompt) > max_length:
                half = int(max_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                         tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt, model_name)
            if "chatglm3" in model_name:
                if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                else:
                    input = prompt.to(device)
            else:
                input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            context_length = input.input_ids.shape[-1]

            if dataset == "samsum":
                output = model.generate(
                    **input, max_new_tokens=max_gen, num_beams=1, do_sample=False,
                    temperature=1.0, min_length=context_length+1,
                    eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
                )[0]
            else:
                output = model.generate(
                    **input, max_new_tokens=max_gen, num_beams=1, do_sample=False, temperature=1.0
                )[0]
            pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
            pred = post_process(pred, model_name)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"]
            }, f, ensure_ascii=False)
            f.write('\n')

    dist.destroy_process_group()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name, device):
    if "chatglm" in model_name or "internlm" in model_name or "xgen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    elif "llama2" in model_name:
        replace_llama_attn_with_flash_attn()
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
        replace_llama_attn_with_flash_attn()
        model, _ = load_model(
            path,
            device='cpu',
            num_gpus=0,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        model = model.to(device)
        model = model.bfloat16()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    model = model.eval()
    return model, tokenizer

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()

    if args.model != "llada-local":
        try:
            from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
            replace_llama_attn_with_flash_attn()
        except Exception as e:
            print("[warn] flash-attn not available; continuing without it:", e)

    if args.model == "llada-local":
        world_size = 1
    else:
        world_size = torch.cuda.device_count()

    mp.set_start_method('spawn', force=True)

    if args.model == "llada-local":
        max_length = args.llada_max_length
        runner = LLaDALocal(
            gen_length=args.llada_gen_length,
            steps=args.llada_steps,
            block_size=args.llada_block_size,
            use_cache=args.llada_use_cache,
            if_cache_position=args.llada_if_cache_position,
            threshold=args.llada_threshold,
            delay_commit=args.llada_delay_commit
        )
        backend = "llada-local"
        model2path = {}
        model_name = "llada-local"
    else:
        model2path = json.load(open("config/model2path.json", "r"))
        model2maxlen = json.load(open("config/model2maxlen.json", "r"))
        max_length = model2maxlen[args.model]
        backend = "hf"
        runner = None
        model_name = args.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
                    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
                    "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
                    "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
                    "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
                    "lsht", "passage_count", "passage_retrieval_en",
                    "passage_retrieval_zh", "lcc", "repobench-p"]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")

    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            out_dir = f"pred_e/{model_name}"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            out_dir = f"pred/{model_name}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{dataset}.jsonl"

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [d for d in data]
        data_subsets = [data_all[i::world_size] for i in range(world_size)]

        processes = []
        for rank in range(world_size):
            p = mp.Process(target=get_pred, args=(
                rank, world_size, data_subsets[rank], max_length, max_gen,
                prompt_format, dataset, device, model_name, model2path, out_path,
                backend, runner
            ))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
