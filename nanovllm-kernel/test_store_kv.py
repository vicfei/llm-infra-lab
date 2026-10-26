#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""store_kvcache_kernel 独立单测：对照 torch 参考实现验证散射写正确性。

跑法（vllm 容器内，自带 triton）:
    docker run --rm --gpus all -v /mnt/d/nano-vllm:/nv -v /mnt/d/models:/models \
      --entrypoint bash vllm/vllm-openai:v0.29.0 -c "python3 /nv/test_store_kv.py"
"""
import torch
import sys
import types

# 单测只测 kernel，不跑 attention forward——mock 掉 flash_attn（vllm 容器无此包名）
fa = types.ModuleType("flash_attn")
fa.flash_attn_varlen_func = None
fa.flash_attn_with_kvcache = None
sys.modules["flash_attn"] = fa

sys.path.insert(0, "/nv")
from nanovllm.layers.attention import store_kvcache  # 被测 kernel 的 Python 包装

torch.manual_seed(0)
dev = "cuda"

N, H, Dh = 1000, 8, 128          # 1000 token、8 头、128 维
D = H * Dh
num_blocks, block_size = 64, 16   # cache: 64 块 × 16 槽
cache_slots = num_blocks * block_size

key = torch.randn(N, H, Dh, device=dev)
value = torch.randn(N, H, Dh, device=dev)
k_cache = torch.zeros(num_blocks, block_size, D, device=dev)  # 真实形状:3维 paged
v_cache = torch.zeros(num_blocks, block_size, D, device=dev)

# 随机槽表：允许重复分配（真实场景同 token 只占一槽，这里随机即可）、5% padding(-1)
slots = torch.randperm(cache_slots, device=dev)[:N].contiguous()  # 唯一槽（真实语义）
pad = torch.rand(N, device=dev) < 0.05
slots[pad] = -1

store_kvcache(key, value, k_cache, v_cache, slots)
torch.cuda.synchronize()

# torch 参考实现：按槽散射（后写覆盖，与 kernel 同语义）
mask = slots != -1
k_ref = torch.zeros_like(k_cache).view(-1, D)
v_ref = torch.zeros_like(v_cache).view(-1, D)
k_ref[slots[mask]] = key[mask].reshape(-1, D)
v_ref[slots[mask]] = value[mask].reshape(-1, D)
k_ref = k_ref.view_as(k_cache); v_ref = v_ref.view_as(v_cache)

assert torch.equal(k_cache, k_ref), "K cache mismatch"
assert torch.equal(v_cache, v_ref), "V cache mismatch"
print("store_kvcache_kernel: PASS (%d tokens, %d padding, %d slots)" %
      (N, int(pad.sum()), cache_slots))

# 附：性能对照——kernel vs torch scatter（学 kernel 为什么要手写）
import time
slots2 = torch.randint(0, cache_slots, (N,), device=dev)
key_big = torch.randn(N * 20, H, Dh, device=dev)
kc2 = torch.zeros(num_blocks*4, block_size, D, device=dev)
vc2 = torch.zeros(num_blocks*4, block_size, D, device=dev)
sl2 = torch.randint(0, num_blocks*4*block_size, (N * 20,), device=dev)

def bench(fn, n=200):
    fn(); torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1e6  # us

t_kernel = bench(lambda: store_kvcache(key_big, key_big, kc2, vc2, sl2))
def torch_ref():
    m = sl2 != -1
    kc2.view(-1, D)[sl2[m]] = key_big[m].reshape(-1, D)
t_torch = bench(torch_ref)
print("perf: triton kernel %.1f us vs torch scatter %.1f us (%.1fx)" %
      (t_kernel, t_torch, t_torch / t_kernel))
