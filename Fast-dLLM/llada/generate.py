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

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@ torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             remasking='low_confidence', mask_id=126336, threshold=None, factor=None):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = (x == mask_id)
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length:] = 0
            if factor is None:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, None, factor)
            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
    return x, nfe


@torch.no_grad()
def generate_with_prefix_cache(
    model,
    prompt,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float = None,
    factor: float = None,
    delay_commit: bool = False,
):
    """
    Prefix-cache 방식의 생성기.
    - delay_commit=True 이면, '모델 입력 버퍼(x_model)'에는 직전 스텝 확정 토큰만 반영하여
      KV-cache 갱신을 한 스텝 늦춘다(= 지연 커밋). 반면, 실제 시퀀스 x 는 즉시 갱신한다.
    - delay_commit=False 이면, x 와 x_model 을 항상 동기화(즉시 커밋)한다.

    동작 개요
    1) 전체 시퀀스로 1회 호출하여 PKV를 만든다.
    2) 블록 시작점 이전 PKV만 남기도록 잘라낸다.
    3) 각 스텝에서
       - delay_commit=True: 직전 스텝에서 확정한 위치만 x_model 에 반영하고 model() 호출
       - delay_commit=False: x_model = x 동기화 후 model() 호출
       모델 출력으로 현재 스텝의 토큰을 선택하여 x 에 즉시 반영.
    """
    device = model.device if hasattr(model, "device") else prompt.device

    # [MASK]로 채운 출력 버퍼 초기화 후 프롬프트 복사
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        bstart = prompt.shape[1] + num_block * block_length
        bend = bstart + block_length

        # 블록 내에서 스텝당 이동할 토큰 수 사전 계산
        block_mask_index = (x[:, bstart:bend] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # 모델 입력 버퍼(x_model). delay_commit=True 일 때만 의미를 갖는다.
        x_model = x.clone()

        # 지연 커밋 대기 위치(다음 스텝에서 x_model에 반영)
        pending_commit_mask_full = torch.zeros_like(x, dtype=torch.bool, device=device)

        # (1) 전체 시퀀스로 1회 호출하여 PKV 생성
        out0 = model(x_model, use_cache=True)
        past_key_values = out0.past_key_values
        nfe += 1

        # (2) 스텝 0: 블록 영역만 유효하도록 마스크 후 선택
        global_mask = (x == mask_id)
        global_mask[:, bend:] = 0

        if factor is None:
            x0, transfer_index = get_transfer_index(
                out0.logits, temperature, remasking, global_mask, x,
                num_transfer_tokens[:, 0] if threshold is None else None, threshold
            )
        else:
            x0, transfer_index = get_transfer_index_dynamic(
                out0.logits, temperature, remasking, global_mask, x, None, factor
            )

        # 실제 시퀀스 x 는 즉시 갱신
        x[transfer_index] = x0[transfer_index]
        nfe += 1

        # (3) 블록 이전까지만 PKV 유지되도록 잘라내기
        #     이후 호출은 bstart 이후의 토큰만 입력으로 주되, PKV는 프리픽스를 보존한다.
        new_past_key_values = []
        for i in range(len(past_key_values)):
            new_past_key_values.append(())
            for j in range(len(past_key_values[i])):
                new_past_key_values[i] += (past_key_values[i][j][:, :, :bstart],)
        past_key_values = new_past_key_values

        # (4) 지연 커밋 처리
        if delay_commit:
            # 방금 확정한 위치는 다음 스텝 호출 전에 x_model 에 반영할 예정
            pending_commit_mask_full |= transfer_index
        else:
            # 즉시 커밋: x_model 을 x 로 동기화
            x_model = x.clone()

        # (5) 남은 스텝 반복
        step_idx = 1
        while True:
            if (x[:, bstart:bend] == mask_id).sum().item() == 0:
                break

            # 호출 직전: 지연 커밋 보류분을 x_model 에 반영하여 이번 호출에서 캐시에 들어가게 함
            if delay_commit and pending_commit_mask_full.any():
                x_model[pending_commit_mask_full] = x[pending_commit_mask_full]
                pending_commit_mask_full.zero_()

            nfe += 1

            # 블록 이후 부분만 입력. PKV는 프리픽스를 보존한 상태로 전달.
            local_input = x_model[:, bstart:]
            local_mask = (x[:, bstart:] == mask_id)
            local_mask[:, block_length:] = 0  # 현재 블록 이외는 무효

            logits = model(local_input, past_key_values=past_key_values, use_cache=True).logits

            # 현재 스텝 선택
            if factor is None:
                x0, transfer_local = get_transfer_index(
                    logits, temperature, remasking, local_mask,
                    x[:, bstart:], num_transfer_tokens[:, step_idx] if threshold is None else None, threshold
                )
            else:
                x0, transfer_local = get_transfer_index_dynamic(
                    logits, temperature, remasking, local_mask,
                    x[:, bstart:], None, factor
                )

            # 로컬 → 전체 인덱스
            transfer_full = torch.zeros_like(x, dtype=torch.bool, device=device)
            transfer_full[:, bstart:] = transfer_local

            # 실제 시퀀스 x 는 즉시 갱신
            x[:, bstart:][transfer_local] = x0[transfer_local]

            if delay_commit:
                # 다음 스텝 호출 전에 x_model 로 커밋
                pending_commit_mask_full |= transfer_full
            else:
                # 즉시 커밋
                x_model[:, bstart:][transfer_local] = x0[transfer_local]

            step_idx += 1

    return x, nfe


@torch.no_grad()
def generate_with_dual_cache(
    model,
    prompt,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float = None,
    factor: float = None,
    delay_commit: bool = False,
):
    """
    Dual-cache 방식의 생성기.

    - Dual cache 는 내부적으로 '왼쪽부터 연속(prefix)'인 구간 교체를 가정한다.
    - LLaDA/MDM의 임의 위치 리빌 + 지연 커밋 조합과는 구조적으로 어긋나므로,
      delay_commit=True 는 지원하지 않는다. (RoPE/KV 정렬 문제, 반복/파편/에러 유발)

    - delay_commit=False 일 때만 원본 동작을 보존한다.
    - delay_commit=True 로 들어오면 AssertionError 를 발생시켜 사용을 막는다.
    """
    assert not delay_commit, "Dual cache + delay_commit 조합은 지원하지 않습니다. Prefix cache를 사용하십시오."

    device = model.device if hasattr(model, "device") else prompt.device
    B = prompt.shape[0]

    # [MASK]로 채운 출력 버퍼 초기화 후 프롬프트 복사
    x = torch.full((B, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    nfe = 0

    for num_block in range(num_blocks):
        bstart = prompt.shape[1] + num_block * block_length
        bend = bstart + block_length

        # 블록 내에서 스텝당 이동할 토큰 수 사전 계산(원본 로직)
        block_mask_index = (x[:, bstart:bend] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        # (1) 전체 시퀀스로 1회 호출하여 PKV 생성
        out0 = model(x, use_cache=True)
        past_key_values = out0.past_key_values
        logits0 = out0.logits
        nfe += 1

        # (2) 스텝 0: 블록 이후는 무효화하여 선택
        global_mask = (x == mask_id)
        global_mask[:, bend:] = 0

        if factor is None:
            x0, transfer_first = get_transfer_index(
                logits0, temperature, remasking, global_mask, x,
                num_transfer_tokens[:, 0] if threshold is None else None, threshold
            )
        else:
            x0, transfer_first = get_transfer_index_dynamic(
                logits0, temperature, remasking, global_mask, x, None, factor
            )

        # 실제 시퀀스 x 즉시 갱신
        x[transfer_first] = x0[transfer_first]
        nfe += 1

        # (3) 남은 스텝: 원본 구현을 그대로 보존
        i = 1
        # 전체 길이와 동일한 replace_position 마스크를 쓰되, 블록 구간만 True
        replace_position_full = torch.zeros_like(x, dtype=torch.bool, device=device)
        replace_position_full[:, bstart:bend] = True

        while True:
            if (x[:, bstart:bend] == mask_id).sum().item() == 0:
                break

            # 블록 입력 + 전체 길이 replace_position (원본 모델의 기대 형태)
            out = model(
                x[:, bstart:bend],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position_full
            )
            logits = out.logits  # 원본 경로: PKV는 내부에서 처리(재할당 불필요)

            mask_local = (x[:, bstart:bend] == mask_id)
            if factor is None:
                x0, transfer_local = get_transfer_index(
                    logits, temperature, remasking, mask_local,
                    x[:, bstart:bend],
                    num_transfer_tokens[:, i] if threshold is None and i < steps_per_block else None,
                    threshold
                )
            else:
                x0, transfer_local = get_transfer_index_dynamic(
                    logits, temperature, remasking, mask_local,
                    x[:, bstart:bend], None, factor
                )

            # 실제 시퀀스 x 즉시 갱신
            x[:, bstart:bend][transfer_local] = x0[transfer_local]
            nfe += 1
            i += 1

    return x, nfe



def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index

def get_transfer_index_dynamic(logits, temperature, remasking, mask_index, x, num_transfer_tokens, factor=1):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l
    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(confidence.shape[0]):
        ns=list(range(1,num_transfer_tokens[j]+1))
        es=[factor/(n+1) for n in ns]
        threshs=[1-e for e in es]

        # at least one token is transferred
        threshs[0]=-1
        sorted_confidence=torch.sort(confidence[j][mask_index[j]],dim=-1,descending=True)[0]
        assert len(sorted_confidence)==len(threshs)
        for top_i in range(len(threshs)):
            if sorted_confidence[top_i]<threshs[top_i]:
                break

        if top_i == 0 or top_i == len(threshs)-1:
            top_i+=1

        _, select_index = torch.topk(confidence[j], k=top_i)
        transfer_index[j, select_index] = True

    return x0, transfer_index

def main():
    device = 'cuda'

    model = LLaDAModelLM.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [{"role": "user", "content": prompt}, ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate_with_dual_cache(model, input_ids, steps=128, gen_length=128, block_length=32, temperature=0., remasking='low_confidence')
    print(tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])

if __name__ == '__main__':
    main()
