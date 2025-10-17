import os
import json
import argparse
from pathlib import Path

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np
import random
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)

from backends.llada_local import LLaDALocal

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "config"
PRED_DIR   = BASE_DIR / "pred"
PRED_E_DIR = BASE_DIR / "pred_e"


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None,
                        choices=[
                            "llama2-7b-chat-4k", "longchat-v1.5-7b-32k", "xgen-7b-8k",
                            "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k",
                            "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
                            "llada-local"
                        ])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")

    parser.add_argument('--llada_gen_length', type=int, default=128)
    parser.add_argument('--llada_steps', type=int, default=128)
    parser.add_argument('--llada_block_size', type=int, default=32)
    parser.add_argument('--llada_use_cache', action='store_true')
    parser.add_argument('--llada_if_cache_position', action='store_true')
    parser.add_argument('--llada_threshold', type=float, default=None)
    parser.add_argument('--llada_delay_commit', action='store_true')
    parser.add_argument('--llada_max_length', type=int, default=32768,
                        help='프롬프트 토큰을 중간 자르기(truncation)할 때 쓰는 한도')

    return parser.parse_args(args)


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


def seed_everything(seed: int):
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
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
    elif "llama2" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16
        ).to(device)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import load_model
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


def get_pred(rank, world_size, data, max_length, max_gen, prompt_format,
             dataset, device, model_name, model2path, out_path,
             backend="hf", runner=None):

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    if backend == "llada-local":
        pass
    else:
        model, tokenizer = load_model_and_tokenizer(model2path[model_name], model_name, device)

    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)

        if backend == "llada-local":
            pred, _ = runner.generate_once(prompt, None)

        else:
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            if "chatglm3" in model_name:
                tokenized_prompt = tokenizer(
                    prompt, truncation=False, return_tensors="pt", add_special_tokens=False
                ).input_ids[0]

            if len(tokenized_prompt) > max_length:
                half = int(max_length / 2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) + \
                         tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)

            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt, model_name)

            if "chatglm3" in model_name:
                if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                    _input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
                else:
                    _input = prompt.to(device)
            else:
                _input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)

            context_length = _input.input_ids.shape[-1]

            if dataset == "samsum":
                output = model.generate(
                    **_input, max_new_tokens=max_gen, num_beams=1, do_sample=False,
                    temperature=1.0, min_length=context_length + 1,
                    eos_token_id=[
                        tokenizer.eos_token_id,
                        tokenizer.encode("\n", add_special_tokens=False)[-1]
                    ],
                )[0]
            else:
                output = model.generate(
                    **_input, max_new_tokens=max_gen, num_beams=1, do_sample=False, temperature=1.0
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

    if dist.is_initialized():
        dist.destroy_process_group()

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
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1

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
        model2path   = json.load(open(CONFIG_DIR / "model2path.json", "r"))
        model2maxlen = json.load(open(CONFIG_DIR / "model2maxlen.json", "r"))
        max_length = model2maxlen[args.model]
        backend = "hf"
        runner = None
        model_name = args.model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.e:
        datasets = [
            "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
            "multi_news", "trec", "triviaqa", "samsum", "passage_count",
            "passage_retrieval_en", "lcc", "repobench-p"
        ]
    else:
        datasets = [
            "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
            "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
            "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
            "lsht", "passage_count", "passage_retrieval_en",
            "passage_retrieval_zh", "lcc", "repobench-p"
        ]

    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(PRED_E_DIR, exist_ok=True)

    dataset2prompt = json.load(open(CONFIG_DIR / "dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open(CONFIG_DIR / "dataset2maxlen.json", "r"))

    for dataset in datasets:
        if args.e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split='test')
            out_dir = PRED_E_DIR / model_name
        else:
            data = load_dataset('THUDM/LongBench', dataset, split='test')
            out_dir = PRED_DIR / model_name

        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / f"{dataset}.jsonl"

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
