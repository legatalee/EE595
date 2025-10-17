# LongBench/backends/llada_local.py
import torch
from pathlib import Path

import sys
BASE = Path(__file__).resolve().parents[2]  # LongBench/.. (repo 루트)
sys.path.append(str(BASE / "llada"))

from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache

class LLaDALocal:
    def __init__(self,
                 model_name="GSAI-ML/LLaDA-8B-Instruct",
                 device="cuda",
                 torch_dtype=torch.bfloat16,
                 gen_length=128,
                 steps=128,
                 block_size=32,
                 use_cache=True,
                 if_cache_position=True,
                 threshold=None,
                 delay_commit=False):
        self.device = device
        self.gen_length = gen_length
        self.steps = steps
        self.block_size = block_size
        self.use_cache = use_cache
        self.if_cache_position = if_cache_position
        self.threshold = threshold
        self.delay_commit = delay_commit

        self.model = LLaDAModelLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=torch_dtype
        ).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

    @torch.no_grad()
    def generate_once(self, system_prompt, user_prompt):
        m = []
        if system_prompt:
            m.append({"role": "system", "content": system_prompt})
        m.append({"role": "user", "content": user_prompt})
        chat_prompt = self.tokenizer.apply_chat_template(
            m, add_generation_prompt=True, tokenize=False
        )
        input_ids = self.tokenizer(chat_prompt)["input_ids"]
        prompt = torch.tensor(input_ids, device=self.device).unsqueeze(0)

        if self.use_cache:
            if self.if_cache_position:
                if self.delay_commit:
                    out, nfe = generate_with_prefix_cache(
                        self.model, prompt,
                        steps=self.steps, gen_length=self.gen_length, block_length=self.block_size,
                        temperature=0., remasking="low_confidence",
                        threshold=self.threshold, delay_commit=True
                    )
                else:
                    out, nfe = generate_with_dual_cache(
                        self.model, prompt,
                        steps=self.steps, gen_length=self.gen_length, block_length=self.block_size,
                        temperature=0., remasking="low_confidence",
                        threshold=self.threshold, delay_commit=False
                    )
            else:
                out, nfe = generate_with_prefix_cache(
                    self.model, prompt,
                    steps=self.steps, gen_length=self.gen_length, block_length=self.block_size,
                    temperature=0., remasking="low_confidence",
                    threshold=self.threshold, delay_commit=self.delay_commit
                )
        else:
            out, nfe = generate(
                self.model, prompt,
                steps=self.steps, gen_length=self.gen_length, block_length=self.block_size,
                temperature=0., remasking="low_confidence", threshold=self.threshold
            )

        answer = self.tokenizer.batch_decode(
            out[:, prompt.shape[1]:], skip_special_tokens=True
        )[0]
        return answer, nfe
