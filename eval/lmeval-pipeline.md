# lm-eval × vLLM 评测管线配方（概率型任务 + 生成型任务）

实测：lm-eval + vLLM v0.29 服务，Qwen3-4B 三档量化对比。

## 正确调用形态（五坑全记录）

```bash
lm_eval --model local-completions \
  --model_args model=qwen3-4b,tokenizer=/models/Qwen3-4B-Instruct-2507,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=8 \
  --tasks mmlu_high_school_mathematics,mmlu_world_religions,gsm8k \
  --limit 50 --batch_size 8 --output_path /out/int8 --seed 42
```

五坑：
1. **概率型任务（MMLU 等 loglikelihood 类）不能用 chat-completions 后端**（报 NotImplementedError），必须 `local-completions`
2. `base_url` 必须是**完整端点**（含 `/v1/completions`），不是 API 根——传 API 根会报 404
3. 客户端侧要能拿到 tokenizer：`tokenizer=` 指本地模型目录（否则去 HF 拉而卡死；国内网络可加 `HF_ENDPOINT=https://hf-mirror.com`）
4. datasets 必须降级 `<3`（3.x 移除了数据集脚本支持）；部分旧式数据集（如 cmmlu）脚本内**硬编码 HF 直链**，镜像变量不生效，直接换任务
5. API 后端需 `pip install 'lm-eval[api]'`（tenacity 等依赖）

## BF16 基线的跑法（显存装不下原版时）

lm-eval 的 `hf` 后端 + accelerate 逐层卸载（不依赖 UVA，与 vLLM 的 offload 限制无关）：

```bash
lm_eval --model hf \
  --model_args pretrained=/models/Qwen3-4B-Instruct-2507,dtype=bfloat16,device_map=auto,parallelize=False \
  --tasks ... --limit 50 --batch_size 1 --seed 42
```

4B 模型 @ 8G 卡实测单前向 ~2.5s，50 题×3 任务约 1 小时（生成型任务最慢）。

## 结果（Qwen3-4B-Instruct-2507，50 条/任务，seed 42）

| 任务 | BF16 | INT8 | W4A16 |
|---|---|---|---|
| GSM8K 5-shot strict | 0.90 | **0.88** | 0.74 |
| GSM8K flexible | 0.84 | **0.84** | 0.52 |
| MMLU 高中数学 0-shot | 0.52 | **0.56** | 0.48 |
| MMLU 世界宗教 0-shot | 0.86 | **0.82** | 0.84 |

判读（50 条 stderr ±5~7%）：**INT8 三任务与原版无显著差异（质量无损档）**；W4A16 数学生成任务真实退化（-16~-32 点），知识记忆类无损——符合量化损伤"数值敏感先掉"的典型分布。

**样本量方法论**：50 条/任务可分辨 10 点级差异（崩溃检测），要精细到 1~2 点需 500+ 条——先小样本排雷、看方向，再决定是否加量。

## 压测资源采样（评测另一半）

```bash
# 压测期间每秒采样，事后取 util>=50% 的连续窗口算"压测窗口均值"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 1
```

经验：vLLM 预分配 KV 池 → 显存峰值在启动即达且压测中平稳（容量规划看启动占用即可）；全程均值会被 idle 稀释，必须按窗口过滤。
