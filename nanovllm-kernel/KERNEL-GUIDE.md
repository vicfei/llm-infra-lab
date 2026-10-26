# nano-vllm Kernel 学习导读（源码 clone 自 GeeeekExplorer/nano-vllm，HEAD bb823b3 2026-04-26，1450 行）

## 一、Kernel 全景：三种实现策略分层

nano-vllm 的价值在于它展示了"一个推理引擎到底哪些部分需要手写 kernel"：

| 层 | 文件 | 实现策略 | 为什么 |
|---|---|---|---|
| **KV 写入 paged cache** | layers/attention.py | **自写 Triton kernel（40 行）** | 散射写（slot_mapping）模式特殊，现成算子覆盖不了 |
| **Attention 计算** | layers/attention.py | 调 flash-attn 库（varlen/kvcache 两接口） | FlashAttention 是深水区，工程上永远该借力 |
| RoPE / LayerNorm / Linear | layers/rotary_embedding.py 等 | 纯 torch 原生算子 | torch 已够快，不需要手写 |
| Sampler | layers/sampler.py | `@torch.compile` + Gumbel 采样技巧 | 编译器优化替代手写 |
| 引擎调度 | engine/（block_manager 120 行等） | 纯 Python | PagedAttention 的"页管理"大脑，kernel 的服务对象 |

## 二、唯一自写 kernel：store_kvcache_kernel（学习主战场）

```python
@triton.jit
def store_kvcache_kernel(key_ptr, key_stride, ..., D: tl.constexpr):
    idx = tl.program_id(0)          # 一个 program 处理一个 token 的 KV
    slot = tl.load(slot_mapping_ptr + idx)   # 该 token 的 KV 该落到 cache 哪个槽
    if slot == -1: return           # padding token
    key_offsets = idx * key_stride + tl.arange(0, D)   # 源：连续的 K
    cache_offsets = slot * D + tl.arange(0, D)         # 目的：按槽散射
    tl.store(k_cache_ptr + cache_offsets, tl.load(key_ptr + key_offsets))
```

**读它的三个问题**（读懂=理解 PagedAttention 的 KV 侧）：
1. `slot_mapping` 从哪来、为什么一个 token 一个槽？（答案在 engine/block_manager.py 的块分配 + engine/sequence.py）
2. 为什么 D = num_heads*head_dim 展平存？（GQA 下 KV 头数少于 Q 头，cache 布局只关心 KV）
3. 它和 flash_attn_with_kvcache 的 block_table 参数怎么衔接？（写入的物理布局=读取的寻址依据）

## 三、调用层语义：flash-attn 两接口（PagedAttention 的读取侧）

- **prefill**：`flash_attn_varlen_func(..., cu_seqlens_q/k, block_table=...)`——变长批处理（cu_seqlens 累计序列长前缀），block_table 非空时=prefix cache 命中，从 cache 读历史 KV
- **decode**：`flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens, block_table)`——每步只有 1 个新 token，全部历史 KV 从 paged cache 读

对照 vLLM：这两条路就是 vLLM 里 prefill/decode 分离的雏形。

## 四、四站学习路径（结合本机环境）

1. **跑通**：vllm 容器内跑 bench.py（qwen3-0.6b）——容器自带 torch/triton/flash_attn，零安装
2. **单测 kernel**：把 store_kvcache_kernel 拷出来独立跑（随机 K/V + 槽表，对照 torch.scatter 验证正确性）——Triton 入门最短路径
3. **改 kernel 实验**：比如把"每 program 处理一个 token"改成"处理 4 个 token"（grid 分块），看性能变化——体会 Triton 的并行度调优
4. **读引擎**：block_manager.py（120 行）的分配/释放逻辑 + scheduler.py 的 continuous batching——理解 kernel 之上的调度世界

## 五、本机跑法备忘

- 源码：./nanovllm/，模型：<你的模型目录>/Qwen3-0.6B
- 容器：vllm/vllm-openai:v0.29.0（--gpus all，VLLM_WSL2_ENABLE_PIN_MEMORY=1 可选）
- Triton kernel 修改后即时生效（JIT），无需重编 torch
