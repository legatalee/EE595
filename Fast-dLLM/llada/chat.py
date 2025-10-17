# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA


# Emma rides her bike at 15 kilometers per hour for 3 hours, then at 10 kilometers per hour for 2 hours. How far does she travel in total?

# A car travels 90 kilometers at 60 km/h, then another 150 kilometers at 75 km/h. How long does the trip take in total?

# Worker A can complete a task in 6 hours, and Worker B can complete the same task in 8 hours. If they work together for 3 hours, how much of the task will be completed?

# python llada/chat.py \
#   --gen_length 128 --steps 128 --block_size 32 \
#   --use_cache --if_cache_position \
#   --threshold 0.9

# python llada/chat.py \
#   --gen_length 128 --steps 128 --block_size 32 \
#   --use_cache --if_cache_position \
#   --threshold 0.0 --delay_commit


import torch
import argparse

from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM

import warnings
warnings.filterwarnings("ignore", message="Please use the new API settings to control TF32 behavior", category=UserWarning)

def chat(args):
    device = 'cuda'
    model = LLaDAModelLM.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True
    )

    gen_length = args.gen_length
    steps = args.steps
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    conversation_num = 0
    prompt = None

    while True:
        user_input = input("Enter your question: ")

        # Always answer in English
        m = [
            {"role": "system", "content": "You are a helpful AI assistant. Always respond in English only."},
            {"role": "user", "content": user_input},
        ]
        chat_prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(chat_prompt)['input_ids']
        input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)

        if conversation_num == 0 or prompt is None:
            prompt = input_ids
        else:
            # concatenate without duplicating BOS token
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)

        print(
            f'use cache: {args.use_cache} '
            f'use cache position: {args.if_cache_position} '
            f'delay commit: {args.delay_commit} '
            f'threshold: {args.threshold} '
            f'block size: {args.block_size}'
        )

        if args.use_cache:
            if args.if_cache_position:
                if args.delay_commit:
                    print(" DEBUG | Used prefix-cache with delay")
                    out, nfe = generate_with_prefix_cache(
                        model,
                        prompt,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=args.block_size,
                        temperature=0.,
                        remasking='low_confidence',
                        threshold=args.threshold,
                        delay_commit=True,   # delayed cache
                    )
                else:
                    # dual cache
                    out, nfe = generate_with_dual_cache(
                        model,
                        prompt,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=args.block_size,
                        temperature=0.,
                        remasking='low_confidence',
                        threshold=args.threshold,
                        delay_commit=False,
                    )
            else:
                # prefix cache
                out, nfe = generate_with_prefix_cache(
                    model,
                    prompt,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=args.block_size,
                    temperature=0.,
                    remasking='low_confidence',
                    threshold=args.threshold,
                    delay_commit=args.delay_commit,
                )
        else:
            out, nfe = generate(
                model,
                prompt,
                steps=steps,
                gen_length=gen_length,
                block_length=args.block_size,
                temperature=0.,
                remasking='low_confidence',
                threshold=args.threshold,
            )


        answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=True)[0]
        print(f"Bot's reply: {answer}")
        print(f"Number of forward passes: {nfe}")

        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1
        print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--use_cache", action="store_true")
    parser.add_argument("--if_cache_position", action="store_true")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--delay_commit", action="store_true", help="Enable 1-step delayed KV-cache commit")

    args = parser.parse_args()
    chat(args)
