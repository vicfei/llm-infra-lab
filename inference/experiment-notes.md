# 量化与推理优化实验笔记（Qwen3-4B-Instruct-2507 @ RTX 4060 8GB）

完整对比矩阵（vLLM v0.29.0，条件：4 并发 × 256 tok × 8 请求；压测为 vllm bench serve 32 请求全并发）：

## 性能与资源

| 方案 | 权重显存 | 服务就绪 | 输出吞吐 | 压测吞吐(32并发) | TTFT 中位 | 显存峰值 | SM 利用率 |
|---|---|---|---|---|---|---|---|
| BF16 直跑 | 7.6G | ❌ 装不下（空闲仅 ~6.9G） | — | — | — | — | — |
| BF16 + cpu-offload 4G | 4G | 390s | 7 tok/s | — | — | — | — |
| INT8 (W8A8) | 4.2G | 135s | 203 tok/s | ~380 tok/s | ~4.4s | 6987MB | 89% |
| W4A16 | 2.5G | 120s | 324.8 tok/s | — | — | — | — |

（对照组：Qwen3.5-0.8B BF16 压测 1463.5 tok/s / TTFT 799ms / 显存 5945MB / SM 95%——5 倍参数的 4B-INT8 单流延迟仅慢 1.7 倍）

## 关键洞察

1. **W4A16 反直觉提速**：量化不是牺牲速度换显存——权重从 7.6G 压到 2.5G 后，省出的显存全部变成 KV cache，并发容量翻倍（8 请求全并发 vs INT8 分两轮），总吞吐反超 INT8 六成。
2. **offload 只是保命**：WSL2 下 PCIe 往返搬运 4G 权重，吞吐坍缩 46 倍；原生 Linux 上损耗会小得多，但仍只适合应急。
3. **显存构成决定并发容量**：0.8B（1.6G 权重+4G KV）吞吐是 4B-INT8（4.2G 权重+7.7K tokens KV）的 3.9 倍，主因是并发容量而非算力。

## 量化脚本用法（inference/quant/quant_w8a8.py）

llmcompressor 0.14 API（与旧文档差异大，七轮排坑定稿）：

```bash
pip install llmcompressor   # 包名无连字符；不是 llm-compressor
python quant_w8a8.py INT8   # 或 W4A16
```

坑位清单（0.14 实测）：
1. `from llmcompressor import oneshot`（顶层导出，非 `.transformers`）
2. `oneshot(model=路径字符串, ...)` 直接收路径，无需先 from_pretrained
3. scheme 合法名：`INT8` / `W4A16` / `W8A8` / `FP8` / `NVFP4`（`W8A8_INT8` 不是合法名）
4. 校准集以 `datasets.Dataset.from_list([{...}])` 对象直传最稳（"custom"+目录路径的语义另有坑）
5. 镜像内是 `python3`（无 `python`）
6. 一次性容器每次需重新 pip install
7. 本地校准集规避 HF 数据集下载（json: `[{"text": "..."}]` 即可）

## 服务起停配方

```bash
docker run -d --gpus all -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 \
  -v $MODELS_DIR:/models -p 8000:8000 --entrypoint bash \
  vllm/vllm-openai:v0.29.0 -c \
  "vllm serve /models/Qwen3-4B-Instruct-2507-W8A8-INT8 --served-model-name qwen3-4b \
   --gpu-memory-utilization 0.85 --max-model-len 4096"
```
