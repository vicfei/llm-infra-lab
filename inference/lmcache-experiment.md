# LMCache × vLLM 前缀命中实验（lmcache 0.5.4 / vLLM v0.29.0）

场景：量化部署（INT8, 4B 模型 @ 8GB 卡）KV 池紧张时的 RAG/多轮前缀命中收益。

## 集成配方

```bash
pip install lmcache   # 容器内即时安装
vllm serve <model> ... \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
# 环境变量：LMCACHE_MAX_LOCAL_CPU_SIZE=8   # CPU 缓存层大小 (GB)
```

坑：`kv_role` 合法值是 `kv_producer` / `kv_both` / `kv_consumer`——写 "producer" 会被
pydantic 拒（vLLM 0.29 校验），单机读写场景用 `kv_both`。Connector 已原生内置于 vLLM 0.29
（`vllm/.../kv_connector/v1/lmcache_connector.py`），装上 lmcache 包即生效。

## 实验设计

负载：`vllm bench serve --dataset-name random --random-prefix-len 1024 --random-input-len 1280
--random-output-len 64 --num-prompts 32 --seed 42`（1024 token 共享前缀 + 256 独特，两轮同题，
第二轮测命中收益）。对照组=同服务不带 kv-transfer-config。

## 结果（TTFT mean / median）

| | R1 冷 | R2 热 | 热轮降幅 |
|---|---|---|---|
| 基线（vLLM 自带 prefix cache） | 8470 / 8702ms | 8216 / 8484ms | **-3%（形同虚设）** |
| LMCache (CPU 层 8G) | 9163 / 9373ms | **5446 / 5741ms** | **-41%** |

## 判读

1. **基线失效根因**：INT8 档 KV 池仅 ~7.7K tokens，32 请求 × 1280 输入 ≈ 41K——前缀在轮内
   就被逐出，第二轮无缓存可命中。**vLLM 自带 prefix cache 在 KV 池紧张时名存实亡**。
2. **LMCache 增量价值**：CPU 内存层把有效 KV 容量扩大一个量级，热轮 TTFT 比基线热轮快
   **34%**（P99 同降 34%）；代价是冷轮写入开销约 +8%。
3. **适用边界**：KV 池紧张的部署（小卡/量化档）+ 前缀命中型负载（RAG 系统提示词、多轮对话）
   收益最大；KV 富余的大显存部署收益有限。
4. **注意**：CPU 层随进程存亡（重启即失），跨重启/跨实例需 disk 或远端后端（未在本实验范围）。
