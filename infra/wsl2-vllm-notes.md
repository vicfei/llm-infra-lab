# WSL2 上跑 vLLM 与 CUDA 的环境笔记

实测环境：RTX 4060 Laptop 8GB / WSL2 (内核 6.6.87) / 驱动 617.42 / Ubuntu 24.04。

## 1. 关键开关：VLLM_WSL2_ENABLE_PIN_MEMORY=1（最重要的一条）

**症状**：vLLM（v0.29 实测）在 WSL2 上启动即失败，报 `RuntimeError: UVA is not available`——无论量化还是 offload 路径都可能触发。

**根因链**（源码实锚 `vllm/platforms/cuda.py` 的 `is_pin_memory_available`）：
- WSL2 上 pinned memory **默认关闭**，即使内核 ≥ 4.19.121（已支持）；
- 需要显式设 `VLLM_WSL2_ENABLE_PIN_MEMORY=1` 才开启；
- UVA 检查实质 = pinned memory 可用性检查，报错信息完全不提这个开关。

**用法**：任何 vLLM 服务容器都带上：
```bash
docker run --gpus all -e VLLM_WSL2_ENABLE_PIN_MEMORY=1 ... vllm/vllm-openai:v0.29.0 ...
```

**附带风险须知**（vLLM 原文警告）：WSL2 上 offload + CUDA graphs 需要 pinned memory，Windows WDDM 默认硬顶物理内存的 50%，过量使用可能导致宿主无响应需硬重启。

## 2. cpu-offload 的利用率预检坑

`--gpu-memory-utilization` 的预检在 offload 生效**之前**执行：若 0.9×8G=7.2G > 当前空闲显存（WSL2 环境自身约占 1G，空闲通常 ~6.9G），直接拒启。**offload 场景预检线要降到 0.85 以下**。

## 3. CUDA 编程环境（apt 路线，最省事）

```bash
sudo apt install nvidia-cuda-toolkit build-essential   # CUDA 12.0 + 系统 gcc
nvcc -arch=sm_89 demo.cu -o demo && ./demo             # Ada 架构实测编译执行通过
```

注意：若存在用户级自定义 gcc（如独立前缀安装的 gcc-14），与系统 gcc 经 PATH 切换并存无冲突；nvcc 配系统 gcc 实测编译通过，遇版本投诉加 `-allow-unsupported-compiler`。

WSL2 写 CUDA 的已知限制：不支持 CUDA 统一内存（managed memory）；常规 kernel（显存/内存拷贝/共享内存）完全不受影响。

## 4. GPU 精度能力速查（Ada sm_89 vs Volta sm_70）

| 精度 | RTX 40xx (sm_89) | V100 (sm_70) |
|---|---|---|
| FP16/BF16 | ✅/✅ | ✅/❌ |
| FP8 | ✅（Ada 首代 FP8 Tensor Core） | ❌ |
| INT8/INT4 权重 | ✅ | INT8 ✅（W4 kernel 支持差） |
| FP4 | ❌（Blackwell 专属） | ❌ |

量化选型含义：Ada 卡可走 FP8/INT8/W4A16 全家；V100 只推荐 INT8 路线。
