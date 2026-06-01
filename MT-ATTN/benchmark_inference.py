#!/usr/bin/env python3
"""
benchmark_inference.py — MT-PreamCNN latency/throughput benchmark.

Measures inference latency (mean, P95, P99) and throughput for both CNN and
Attn multi-task models across multiple backends:
  pytorch_cpu   : torch.no_grad(), CPU only
  pytorch_gpu   : torch.no_grad(), CUDA (if available)
  ort_cpu       : ONNX Runtime CPU (requires onnxruntime)
  tensorrt      : TensorRT FP16 engine (requires tensorrt, pycuda)

Usage (x86, all backends):
  python3 benchmark_inference.py --backend pytorch_cpu pytorch_gpu ort_cpu

Usage (Jetson, ORT + TRT):
  python3 benchmark_inference.py --backend ort_cpu tensorrt --onnx_dir ~/rf_stream/onnx

Output:
  rf_stream/benchmark_results_{hostname}.json
  (or --out_json path)

The ONNX files are exported from PyTorch checkpoints on x86 first, then copied
to Jetson.  On Jetson, only ort_cpu and tensorrt backends are available.
"""

import argparse, json, os, platform, socket, time, sys
import numpy as np

# Workaround: known Conv1D segfault on some GPU drivers (same fix as train_multitask_v2.py)
try:
    import torch
    torch.backends.cudnn.enabled = False
except ImportError:
    pass

# ── Constants ──────────────────────────────────────────────────────────────────
PREAM_LEN   = 800
PREAM_FLOAT = PREAM_LEN * 2  # 1600 floats per sample

ARCHS  = ["cnn", "attn"]
BATCHES = [1, 8, 32, 64]
WARMUP  = 50
REPEATS = 200

CKPT_DIR = "rf_stream/multitask_model_v2"
ONNX_DIR = "rf_stream/onnx"


# ── Inline model definitions (copied from train_multitask_v2.py) ───────────────
def _build_torch_models():
    import torch
    import torch.nn as nn

    class TemporalSelfAttention(nn.Module):
        def __init__(self, d_model=128, nhead=4, dropout=0.1):
            super().__init__()
            self.attn    = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            self.norm    = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        def forward(self, x):
            xt = x.permute(0, 2, 1)
            a, _ = self.attn(xt, xt, xt)
            xt = self.norm(xt + self.dropout(a))
            return xt.permute(0, 2, 1)

    class MultiTaskPreamCNN(nn.Module):
        def __init__(self, embed_dim=256, tasks=("gate", "mod", "snr")):
            super().__init__()
            self.tasks   = tasks
            self.encoder = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=31, padding=15),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(32, 64, kernel_size=15, padding=7),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8),
            )
            self.proj = nn.Sequential(nn.Linear(128 * 8, embed_dim), nn.ReLU(), nn.Dropout(0.3))
            if "gate" in tasks: self.gate_head = nn.Linear(embed_dim, 1)
            if "mod"  in tasks: self.mod_head  = nn.Linear(embed_dim, 1)
            if "snr"  in tasks:
                self.snr_head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            B = x.size(0)
            x2 = x.view(B, PREAM_LEN, 2).permute(0, 2, 1).contiguous()
            emb = self.proj(self.encoder(x2).view(B, -1))
            out = {}
            if "gate" in self.tasks: out["gate"] = self.gate_head(emb).squeeze(-1)
            if "mod"  in self.tasks: out["mod"]  = self.mod_head(emb).squeeze(-1)
            if "snr"  in self.tasks: out["snr"]  = self.snr_head(emb).squeeze(-1)
            return out

    class MultiTaskPreamCNNAttn(nn.Module):
        def __init__(self, embed_dim=256, tasks=("gate", "mod", "snr"), nhead=4, attn_dropout=0.1):
            super().__init__()
            self.tasks = tasks
            self.cnn = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=31, padding=15),
                nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(32, 64, kernel_size=15, padding=7),
                nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
                nn.Conv1d(64, 128, kernel_size=7, padding=3),
                nn.BatchNorm1d(128), nn.ReLU(),
            )
            self.attn_block = TemporalSelfAttention(d_model=128, nhead=nhead, dropout=attn_dropout)
            self.pool = nn.AdaptiveAvgPool1d(8)
            self.proj = nn.Sequential(nn.Linear(128 * 8, embed_dim), nn.ReLU(), nn.Dropout(0.3))
            if "gate" in tasks: self.gate_head = nn.Linear(embed_dim, 1)
            if "mod"  in tasks: self.mod_head  = nn.Linear(embed_dim, 1)
            if "snr"  in tasks:
                self.snr_head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            B = x.size(0)
            x2   = x.view(B, PREAM_LEN, 2).permute(0, 2, 1).contiguous()
            feat = self.cnn(x2)
            feat = self.attn_block(feat)
            feat = self.pool(feat)
            emb  = self.proj(feat.reshape(B, -1))
            out  = {}
            if "gate" in self.tasks: out["gate"] = self.gate_head(emb).squeeze(-1)
            if "mod"  in self.tasks: out["mod"]  = self.mod_head(emb).squeeze(-1)
            if "snr"  in self.tasks: out["snr"]  = self.snr_head(emb).squeeze(-1)
            return out

    return MultiTaskPreamCNN, MultiTaskPreamCNNAttn


# ── Timing helpers ─────────────────────────────────────────────────────────────

def _measure_latencies(fn, batches, warmup=WARMUP, repeats=REPEATS):
    """Run fn(batch_size) → list[float ms] for each batch size."""
    results = {}
    for bs in batches:
        # warmup
        for _ in range(warmup):
            fn(bs)
        # measure
        lats = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn(bs)
            lats.append((time.perf_counter() - t0) * 1e3)  # ms
        lats = np.array(lats)
        results[bs] = {
            "mean_ms":  float(np.mean(lats)),
            "p50_ms":   float(np.percentile(lats, 50)),
            "p95_ms":   float(np.percentile(lats, 95)),
            "p99_ms":   float(np.percentile(lats, 99)),
            "std_ms":   float(np.std(lats)),
            "throughput_sps": float(bs / (np.mean(lats) / 1e3)),
        }
        print(f"    bs={bs:2d}: {results[bs]['mean_ms']:6.2f}ms ± {results[bs]['std_ms']:.2f}ms  "
              f"({results[bs]['throughput_sps']:.0f} sps)")
    return results


def _gpu_measure_latencies(fn_gpu, batches, device, warmup=WARMUP, repeats=REPEATS):
    """CUDA-event timing for GPU benchmarks."""
    import torch
    results = {}
    for bs in batches:
        for _ in range(warmup):
            fn_gpu(bs)
        torch.cuda.synchronize(device)

        events_start = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        events_end   = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for i in range(repeats):
            events_start[i].record()
            fn_gpu(bs)
            events_end[i].record()
        torch.cuda.synchronize(device)

        lats = np.array([s.elapsed_time(e) for s, e in zip(events_start, events_end)])
        results[bs] = {
            "mean_ms":  float(np.mean(lats)),
            "p50_ms":   float(np.percentile(lats, 50)),
            "p95_ms":   float(np.percentile(lats, 95)),
            "p99_ms":   float(np.percentile(lats, 99)),
            "std_ms":   float(np.std(lats)),
            "throughput_sps": float(bs / (np.mean(lats) / 1e3)),
        }
        print(f"    bs={bs:2d}: {results[bs]['mean_ms']:6.2f}ms ± {results[bs]['std_ms']:.2f}ms  "
              f"({results[bs]['throughput_sps']:.0f} sps)")
    return results


# ── Load checkpoint ────────────────────────────────────────────────────────────

def load_checkpoint(arch, ckpt_dir=CKPT_DIR):
    """Load the mixed+phaseaug checkpoint; fall back to mixed if not found."""
    import torch
    MultiTaskPreamCNN, MultiTaskPreamCNNAttn = _build_torch_models()

    for suffix in ["_mixed_phaseaug", "_mixed"]:
        path = os.path.join(ckpt_dir, f"mt_preamcnn_{arch}{suffix}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location="cpu")
            embed_dim = ckpt.get("embed_dim", 256)
            tasks     = ckpt.get("tasks", ("gate", "mod", "snr"))
            model = (MultiTaskPreamCNN if arch == "cnn" else MultiTaskPreamCNNAttn)(
                embed_dim=embed_dim, tasks=tasks)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  Loaded {path} ({n_params:,} params)")
            return model, path
    raise FileNotFoundError(f"No checkpoint found for arch={arch} in {ckpt_dir}")


# ── ONNX export ────────────────────────────────────────────────────────────────

def _patch_adaptive_pool(module):
    """Replace AdaptiveAvgPool1d(8) with AvgPool1d(kernel=7,stride=6).

    CNN encoder produces 50 time steps (800 → MaxPool4 → 200 → MaxPool4 → 50).
    AdaptiveAvgPool1d(8) on 50 doesn't export to ONNX (50/8 is not integer).
    AvgPool1d(kernel=7, stride=6): floor((50-7)/6)+1 = 8 exactly.
    """
    import torch.nn as nn
    for name, child in list(module.named_children()):
        if isinstance(child, nn.AdaptiveAvgPool1d) and child.output_size in (8, (8,)):
            setattr(module, name, nn.AvgPool1d(kernel_size=7, stride=6))
        else:
            _patch_adaptive_pool(child)


def export_onnx(arch, ckpt_dir=CKPT_DIR, onnx_dir=ONNX_DIR):
    import torch
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, f"mt_preamcnn_{arch}.onnx")
    if os.path.exists(onnx_path):
        print(f"  ONNX already exists: {onnx_path}")
        return onnx_path

    model, _ = load_checkpoint(arch, ckpt_dir)
    # Patch AdaptiveAvgPool1d for ONNX compatibility (50→8 via fixed kernel)
    _patch_adaptive_pool(model)
    dummy = torch.zeros(1, PREAM_FLOAT)

    # Wrap model to return a single tensor tuple (ONNX doesn't handle dicts)
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        def forward(self, x):
            o = self.m(x)
            return o.get("gate", torch.zeros(x.size(0))), \
                   o.get("mod",  torch.zeros(x.size(0))), \
                   o.get("snr",  torch.zeros(x.size(0)))

    wrapper = OnnxWrapper(model)
    wrapper.eval()
    torch.onnx.export(
        wrapper, dummy, onnx_path,
        input_names=["preamble"],
        output_names=["gate_logit", "mod_logit", "snr_norm"],
        dynamic_axes={"preamble": {0: "batch"}, "gate_logit": {0: "batch"},
                      "mod_logit": {0: "batch"}, "snr_norm": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Exported ONNX: {onnx_path} ({os.path.getsize(onnx_path)/1024:.0f} KB)")
    return onnx_path


# ── TensorRT engine build ──────────────────────────────────────────────────────

def build_trt_engine(onnx_path, fp16=True):
    """Build TRT engine from ONNX with FP16 mode; cache as .trt file."""
    import tensorrt as trt

    trt_path = onnx_path.replace(".onnx", "_fp16.trt" if fp16 else ".trt")
    if os.path.exists(trt_path):
        print(f"  TRT engine cached: {trt_path}")
    else:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, logger)
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print("TRT parse error:", parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * (1 << 20))
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("  TRT: FP16 mode enabled")

        # Dynamic batch profile
        profile = builder.create_optimization_profile()
        profile.set_shape("preamble",
                          min=(1,  PREAM_FLOAT),
                          opt=(8,  PREAM_FLOAT),
                          max=(64, PREAM_FLOAT))
        config.add_optimization_profile(profile)

        print("  Building TRT engine (this may take 1-2 minutes)...")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TRT engine build failed")
        with open(trt_path, "wb") as f:
            f.write(serialized)
        print(f"  TRT engine saved: {trt_path} ({os.path.getsize(trt_path)/1024:.0f} KB)")

    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine


# ── TensorRT runner (ctypes / Jetson unified memory — no pycuda) ──────────────

def run_tensorrt_ctypes(arch, onnx_dir=ONNX_DIR):
    """TRT FP16 inference via ctypes + CUDA runtime (works on Jetson unified memory)."""
    import ctypes, tensorrt as trt

    onnx_path = os.path.join(onnx_dir, f"mt_preamcnn_{arch}.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    print(f"\n[tensorrt_ctypes] {arch}  TRT={trt.__version__}")
    engine = build_trt_engine(onnx_path, fp16=True)
    context = engine.create_execution_context()

    # Load CUDA runtime via ctypes
    libcuda = ctypes.CDLL("libcudart.so.12", use_errno=True)
    stream_p = ctypes.c_void_p()
    ret = libcuda.cudaStreamCreate(ctypes.byref(stream_p))
    if ret != 0:
        raise RuntimeError(f"cudaStreamCreate failed: {ret}")
    stream_handle = stream_p.value

    def cudaSync():
        libcuda.cudaStreamSynchronize(ctypes.c_void_p(stream_handle))

    rng = np.random.default_rng(0)
    results = {}

    for bs in BATCHES:
        context.set_input_shape("preamble", (bs, PREAM_FLOAT))

        # Allocate numpy arrays (Jetson unified memory — same address on CPU & GPU)
        inp      = np.ascontiguousarray(rng.standard_normal((bs, PREAM_FLOAT)).astype(np.float32))
        out_gate = np.empty((bs,), dtype=np.float32)
        out_mod  = np.empty((bs,), dtype=np.float32)
        out_snr  = np.empty((bs,), dtype=np.float32)

        context.set_tensor_address("preamble",   inp.ctypes.data)
        context.set_tensor_address("gate_logit", out_gate.ctypes.data)
        context.set_tensor_address("mod_logit",  out_mod.ctypes.data)
        context.set_tensor_address("snr_norm",   out_snr.ctypes.data)

        def fn_trt():
            context.execute_async_v3(stream_handle=stream_handle)
            cudaSync()

        # Warmup
        for _ in range(WARMUP):
            fn_trt()

        lats = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fn_trt()
            lats.append((time.perf_counter() - t0) * 1e3)

        lats = np.array(lats)
        results[bs] = {
            "mean_ms":  float(np.mean(lats)),
            "p50_ms":   float(np.percentile(lats, 50)),
            "p95_ms":   float(np.percentile(lats, 95)),
            "p99_ms":   float(np.percentile(lats, 99)),
            "std_ms":   float(np.std(lats)),
            "throughput_sps": float(bs / (np.mean(lats) / 1e3)),
        }
        print(f"    bs={bs:2d}: {results[bs]['mean_ms']:6.2f}ms ± {results[bs]['std_ms']:.2f}ms  "
              f"({results[bs]['throughput_sps']:.0f} sps)")

    libcuda.cudaStreamDestroy(ctypes.c_void_p(stream_handle))
    return results


def run_trtexec(arch, onnx_dir=ONNX_DIR,
                trtexec_bin="/usr/src/tensorrt/bin/trtexec"):
    """Parse trtexec output for latency at each batch size (fallback for no-pycuda)."""
    import subprocess, re

    onnx_path = os.path.join(onnx_dir, f"mt_preamcnn_{arch}.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")
    if not os.path.exists(trtexec_bin):
        raise FileNotFoundError(f"trtexec not found: {trtexec_bin}")

    print(f"\n[trtexec] {arch}")
    results = {}
    for bs in BATCHES:
        cmd = [
            trtexec_bin,
            f"--onnx={onnx_path}",
            "--fp16",
            f"--minShapes=preamble:1x{PREAM_FLOAT}",
            f"--optShapes=preamble:{bs}x{PREAM_FLOAT}",
            f"--maxShapes=preamble:64x{PREAM_FLOAT}",
            "--iterations=200", "--warmUp=2000",
            "--avgRuns=200", "--separateProfileRun",
            f"--shapes=preamble:{bs}x{PREAM_FLOAT}",
        ]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            text = out.stdout + out.stderr
            # Extract mean/median latency from trtexec output
            # Format: "Latency: min = X ms, max = Y ms, mean = Z ms, median = W ms"
            lat_match = re.search(
                r"Latency: min = ([\d.]+) ms, max = ([\d.]+) ms, mean = ([\d.]+) ms, median = ([\d.]+) ms",
                text)
            p99_match = re.search(r"99th percentile = ([\d.]+) ms", text)
            tput_match = re.search(r"Throughput: ([\d.]+) qps", text)
            if lat_match:
                mean_ms = float(lat_match.group(3))
                p50_ms  = float(lat_match.group(4))
                p99_ms  = float(p99_match.group(1)) if p99_match else mean_ms
                tput    = float(tput_match.group(1)) if tput_match else bs / (mean_ms / 1e3)
                results[bs] = {
                    "mean_ms": mean_ms, "p50_ms": p50_ms, "p99_ms": p99_ms,
                    "p95_ms": p99_ms,   "std_ms": 0.0,
                    "throughput_sps": tput,
                }
                print(f"    bs={bs:2d}: {mean_ms:6.2f}ms  ({tput:.0f} qps)")
            else:
                print(f"    bs={bs:2d}: PARSE FAILED (no latency line found)")
                print(text[-1000:])
                results[bs] = {"error": "parse_failed"}
        except subprocess.TimeoutExpired:
            print(f"    bs={bs:2d}: TIMEOUT")
            results[bs] = {"error": "timeout"}
        except Exception as e:
            print(f"    bs={bs:2d}: ERROR {e}")
            results[bs] = {"error": str(e)}
    return results


# ── Backend runners ────────────────────────────────────────────────────────────

def run_pytorch_cpu(arch, ckpt_dir=CKPT_DIR):
    import torch
    print(f"\n[pytorch_cpu] {arch}")
    model, _ = load_checkpoint(arch, ckpt_dir)
    model.eval()
    rng = np.random.default_rng(0)

    def fn(bs):
        x = torch.from_numpy(rng.standard_normal((bs, PREAM_FLOAT)).astype(np.float32))
        with torch.no_grad():
            model(x)

    return _measure_latencies(fn, BATCHES)


def run_pytorch_gpu(arch, ckpt_dir=CKPT_DIR):
    import torch
    if not torch.cuda.is_available():
        print(f"  [pytorch_gpu] CUDA not available, skipping")
        return {}
    device = torch.device("cuda")
    print(f"\n[pytorch_gpu] {arch}  device={torch.cuda.get_device_name(0)}")
    model, _ = load_checkpoint(arch, ckpt_dir)
    model.eval().to(device)
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(0)

    def fn(bs):
        x = torch.from_numpy(rng.standard_normal((bs, PREAM_FLOAT)).astype(np.float32)).to(device)
        with torch.no_grad():
            model(x)

    return _gpu_measure_latencies(fn, BATCHES, device)


def run_ort_cpu(arch, onnx_dir=ONNX_DIR):
    import onnxruntime as ort
    onnx_path = os.path.join(onnx_dir, f"mt_preamcnn_{arch}.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}  (run --export_onnx first)")
    print(f"\n[ort_cpu] {arch}  ORT={ort.__version__}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(onnx_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)

    def fn(bs):
        x = rng.standard_normal((bs, PREAM_FLOAT)).astype(np.float32)
        sess.run(None, {"preamble": x})

    return _measure_latencies(fn, BATCHES)


def run_tensorrt(arch, onnx_dir=ONNX_DIR):
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401

    onnx_path = os.path.join(onnx_dir, f"mt_preamcnn_{arch}.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX not found: {onnx_path}")

    print(f"\n[tensorrt] {arch}  TRT={trt.__version__}")
    engine  = build_trt_engine(onnx_path, fp16=True)
    context = engine.create_execution_context()

    rng = np.random.default_rng(0)
    results = {}

    for bs in BATCHES:
        # Set dynamic input shape
        context.set_input_shape("preamble", (bs, PREAM_FLOAT))

        # Allocate device buffers
        input_np  = rng.standard_normal((bs, PREAM_FLOAT)).astype(np.float32)
        out_gate  = np.empty((bs,), dtype=np.float32)
        out_mod   = np.empty((bs,), dtype=np.float32)
        out_snr   = np.empty((bs,), dtype=np.float32)

        d_in   = cuda.mem_alloc(input_np.nbytes)
        d_gate = cuda.mem_alloc(out_gate.nbytes)
        d_mod  = cuda.mem_alloc(out_mod.nbytes)
        d_snr  = cuda.mem_alloc(out_snr.nbytes)

        bindings = [int(d_in), int(d_gate), int(d_mod), int(d_snr)]
        stream   = cuda.Stream()

        def fn_trt(bs_inner=bs):
            cuda.memcpy_htod_async(d_in, input_np, stream)
            context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(out_gate, d_gate, stream)
            stream.synchronize()

        # Warmup
        for _ in range(WARMUP):
            fn_trt()

        # Measure with CPU timer (GPU is synchronous via stream.synchronize())
        lats = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fn_trt()
            lats.append((time.perf_counter() - t0) * 1e3)

        lats = np.array(lats)
        results[bs] = {
            "mean_ms":  float(np.mean(lats)),
            "p50_ms":   float(np.percentile(lats, 50)),
            "p95_ms":   float(np.percentile(lats, 95)),
            "p99_ms":   float(np.percentile(lats, 99)),
            "std_ms":   float(np.std(lats)),
            "throughput_sps": float(bs / (np.mean(lats) / 1e3)),
        }
        print(f"    bs={bs:2d}: {results[bs]['mean_ms']:6.2f}ms ± {results[bs]['std_ms']:.2f}ms  "
              f"({results[bs]['throughput_sps']:.0f} sps)")

        for d in [d_in, d_gate, d_mod, d_snr]:
            d.free()

    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", nargs="+",
                    choices=["pytorch_cpu", "pytorch_gpu", "ort_cpu",
                             "tensorrt", "tensorrt_ctypes", "trtexec"],
                    default=["pytorch_cpu"],
                    help="Backends to benchmark")
    ap.add_argument("--archs",   nargs="+", default=["cnn", "attn"])
    ap.add_argument("--ckpt_dir",  default=CKPT_DIR)
    ap.add_argument("--onnx_dir",  default=ONNX_DIR)
    ap.add_argument("--export_onnx", action="store_true",
                    help="Export ONNX files from PyTorch checkpoints (x86 only)")
    ap.add_argument("--out_json",  default=None,
                    help="Output JSON path (default: benchmark_results_<host>.json)")
    args = ap.parse_args()

    hostname = socket.gethostname()
    out_json = args.out_json or f"rf_stream/benchmark_results_{hostname}.json"

    results = {
        "hostname": hostname,
        "platform": platform.platform(),
        "python":   sys.version,
        "warmup":   WARMUP,
        "repeats":  REPEATS,
        "batch_sizes": BATCHES,
        "pream_float": PREAM_FLOAT,
        "backends":  {},
    }

    # Gather CPU / GPU info
    try:
        import torch
        results["torch_version"] = torch.__version__
        results["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        results["torch_version"] = "not_installed"
        results["cuda_available"] = False

    try:
        import onnxruntime as ort
        results["ort_version"] = ort.__version__
    except ImportError:
        results["ort_version"] = "not_installed"

    try:
        import tensorrt as trt
        results["trt_version"] = trt.__version__
    except ImportError:
        results["trt_version"] = "not_installed"

    # Export ONNX if requested
    if args.export_onnx:
        print("\n=== Exporting ONNX models ===")
        for arch in args.archs:
            try:
                export_onnx(arch, ckpt_dir=args.ckpt_dir, onnx_dir=args.onnx_dir)
            except Exception as e:
                print(f"  ERROR exporting {arch}: {e}")

    # Run benchmarks
    for backend in args.backend:
        results["backends"][backend] = {}
        for arch in args.archs:
            print(f"\n{'='*60}")
            try:
                if backend == "pytorch_cpu":
                    r = run_pytorch_cpu(arch, ckpt_dir=args.ckpt_dir)
                elif backend == "pytorch_gpu":
                    r = run_pytorch_gpu(arch, ckpt_dir=args.ckpt_dir)
                elif backend == "ort_cpu":
                    r = run_ort_cpu(arch, onnx_dir=args.onnx_dir)
                elif backend == "tensorrt":
                    r = run_tensorrt(arch, onnx_dir=args.onnx_dir)
                elif backend == "tensorrt_ctypes":
                    r = run_tensorrt_ctypes(arch, onnx_dir=args.onnx_dir)
                elif backend == "trtexec":
                    r = run_trtexec(arch, onnx_dir=args.onnx_dir)
                else:
                    r = {}
                # Convert batch_size keys to strings for JSON
                results["backends"][backend][arch] = {str(k): v for k, v in r.items()}
            except Exception as e:
                print(f"  ERROR [{backend}][{arch}]: {e}")
                results["backends"][backend][arch] = {"error": str(e)}

    # Save results
    os.makedirs(os.path.dirname(out_json) if os.path.dirname(out_json) else ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out_json}")

    # Print summary table
    print("\n=== SUMMARY (batch=1, mean latency ms) ===")
    print(f"{'Backend':<20} {'CNN':>10} {'Attn':>10}")
    print("-" * 42)
    for backend in args.backend:
        row = results["backends"].get(backend, {})
        cnn_lat  = row.get("cnn",  {}).get("1", {}).get("mean_ms", float("nan"))
        attn_lat = row.get("attn", {}).get("1", {}).get("mean_ms", float("nan"))
        print(f"{backend:<20} {cnn_lat:>10.2f} {attn_lat:>10.2f}")


if __name__ == "__main__":
    main()
