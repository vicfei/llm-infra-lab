# llm-infra-lab

LLM 基础设施 / 推理优化 / 评测的实验工具箱——来自一台 RTX 4060 Laptop 8GB (WSL2) 上的实战沉淀。

## 板块导航

| 目录 | 内容 | 亮点 |
|---|---|---|
| `infra/` | WSL2 上跑 vLLM/CUDA 的环境配方 | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 这一隐藏开关的定位过程 |
| `inference/` | 量化工具链 + 基准测试 | llmcompressor 0.14 新 API 七轮排坑定稿脚本；8GB 卡上 4B 模型的完整方案矩阵 |
| `eval/` | lm-eval 评测管线 + 结果 | 概率型任务接 vLLM 的正确姿势；三模型速度×质量对比数据 |
| `nanovllm-kernel/` | nano-vllm kernel 学习材料 | 唯一自写 Triton kernel 的独立单测（含 2.1x 性能对照） |

## 核心结论速览（详见各板块笔记）

**8GB 卡跑 4B 模型（Qwen3-4B-Instruct-2507, vLLM v0.29）：**

| 方案 | 显存 | 输出吞吐 | GSM8K 质量 |
|---|---|---|---|
| BF16 直跑 | 7.6G | ❌ 装不下 | — |
| BF16 + cpu-offload | 4G | 7 tok/s（慢 46 倍，仅保命） | — |
| **INT8 (W8A8)** | **4.2G** | **203 tok/s** | **88%（与原版 90% 无显著差异）→ 推荐档** |
| W4A16 | 2.5G | 324 tok/s（并发翻倍） | 74%（数学任务真实退化）→ 吞吐档 |

## 环境与工具版本

- vLLM v0.29.0（官方镜像）/ llmcompressor 0.14.0.1 / lm-eval + datasets<3
- CUDA 12.0（apt nvidia-cuda-toolkit）/ Triton（vLLM 镜像自带）
- 模型：Qwen3-4B-Instruct-2507（ModelScope 下载）、Qwen3-0.6B

## 使用姿势

每个目录内的笔记都是"可直接照抄的命令 + 踩坑定位过程"。量化脚本对任意 HF 格式模型通用（改 `SRC` 路径与 `scheme` 即可）。
