#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qwen3-4B 显存方案基准测试客户端。

用法:
    python3 bench.py <api_base> [并发数] [每请求max_tokens] [请求数]

测: 吞吐(tokens/s, 生成侧)、平均延迟、TTFT 不测(简化)。
固定 prompt 约 500 token 输入 + 纯生成, 用 /v1/completions。
"""
import sys
import time
import json
import urllib.request
import concurrent.futures as cf

API = sys.argv[1].rstrip("/")
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
NREQ = int(sys.argv[4]) if len(sys.argv) > 4 else CONC * 2

PROMPT = ("请详细说明向量数据库的索引原理，包括 HNSW 图的构建过程、"
          "层级跳表的搜索策略、参数 m/ef_construction/ef_search 的影响，"
          "以及与 IVF-PQ 的对比权衡。请分点论述，每点展开说明。") * 4


def one(i):
    body = json.dumps({
        "model": "qwen3-4b", "prompt": PROMPT, "max_tokens": MAXTOK,
        "temperature": 0.7, "ignore_eos": True,
    }).encode()
    req = urllib.request.Request(API + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ct = d.get("usage", {}).get("completion_tokens", 0)
    return dt, ct


def main():
    print("bench: conc=%d maxtok=%d nreq=%d" % (CONC, MAXTOK, NREQ))
    t0 = time.time()
    with cf.ThreadPoolExecutor(CONC) as ex:
        rs = list(ex.map(one, range(NREQ)))
    wall = time.time() - t0
    toks = sum(c for _, c in rs)
    lats = sorted(d for d, _ in rs)
    print("wall=%.1fs  total_out_tokens=%d" % (wall, toks))
    print("throughput=%.1f tok/s (output)" % (toks / wall))
    print("latency avg=%.1fs p50=%.1fs max=%.1fs" % (
        sum(lats) / len(lats), lats[len(lats) // 2], lats[-1]))
    for d, c in rs:
        print("  req: %.1fs %d tok (%.1f tok/s)" % (d, c, c / d if d else 0))


if __name__ == "__main__":
    main()
