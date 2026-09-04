"""Export TimesFM 3.0 to ONNX so it can run in a browser via ONNX Runtime Web.

Why the graph is shaped the way it is
-------------------------------------
`TimesFM3Torch.decode` computes its patch indices with Python integer
arithmetic derived from the context length (`num_context_patches = context //
input_patch_len`). Those indices are baked in as constants when tracing, so a
graph exported with a dynamic context axis only ever works at the length it was
traced at.

Rather than rewrite the reference implementation, the export fixes the context
to a window of `--window` points and adds the model's own `mask` input. Series
shorter than the window are left-padded and masked off, which is exactly what
the mask is for -- output is provably independent of the padding values.

The horizon is likewise fixed at export time. Asking for a longer horizon than
you need and truncating costs about 0.2% against a natively-sized call.

Usage:
    python scripts/export_onnx.py --out web/models
    python scripts/export_onnx.py --out web/models --quantize int8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def cumprod_binary(x: torch.Tensor, dim: int, **_) -> torch.Tensor:
    """ONNX-exportable stand-in for torch.cumprod on 0/1 tensors.

    ONNX has CumSum but no CumProd at any opset. For a binary tensor the running
    product is 1 exactly while no zero has been seen, so
    cumprod(x) == (cumsum(x == 0) == 0). TimesFM only calls cumprod on boolean
    patch masks, making this substitution exact rather than approximate.
    """
    return (torch.cumsum((x == 0).to(torch.int32), dim=dim) == 0).to(x.dtype)


class PortableRMSNorm(nn.Module):
    """nn.RMSNorm rebuilt from primitive ops.

    ONNX only gained RMSNormalization in opset 23, newer than browser runtimes
    reliably support. Mul/ReduceMean/Rsqrt exist everywhere and the arithmetic
    is identical.
    """

    def __init__(self, src: nn.RMSNorm):
        super().__init__()
        self.ndims = len(src.normalized_shape)
        # nn.RMSNorm treats eps=None as finfo(dtype).eps.
        self.eps = src.eps if src.eps is not None else torch.finfo(torch.float32).eps
        if src.elementwise_affine:
            self.weight = nn.Parameter(src.weight.detach().clone())
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-self.ndims, 0))
        normed = x * torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + self.eps)
        return normed if self.weight is None else normed * self.weight


def replace_rms_norm(module: nn.Module) -> int:
    """Swap every nn.RMSNorm in the tree for the portable equivalent."""
    swapped = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.RMSNorm):
            setattr(module, name, PortableRMSNorm(child))
            swapped += 1
        else:
            swapped += replace_rms_norm(child)
    return swapped


class DecodeGraph(nn.Module):
    """(context (1,1,W), mask (1,W)) -> quantiles (1,1,H,9).

    `mask` is True at padded positions, matching the model's own convention.
    """

    def __init__(self, model: nn.Module, horizon: int):
        super().__init__()
        self.model = model
        self.horizon = horizon

    def forward(self, context: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.model.decode(context, horizon=self.horizon, mask=mask.bool())


def sample_batch(window: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    series = (np.sin(np.arange(window) * 2 * np.pi / 24) * 10 + 100
              + rng.normal(0, 0.5, window)).astype(np.float32)
    return torch.tensor(series)[None, None, :], torch.zeros(1, window, dtype=torch.bool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="google/timesfm-3.0-pytorch")
    ap.add_argument("--out", type=Path, default=Path("web/models"))
    ap.add_argument("--window", type=int, default=512, help="fixed context window")
    ap.add_argument("--horizon", type=int, default=256, help="fixed max horizon")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--quantize", choices=["none", "int8"], default="none")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import timesfm

    print(f"loading {args.checkpoint} ...")
    core = timesfm.TimesFM3Forecaster.from_pretrained(
        args.checkpoint, device="cpu").model.eval()

    context, mask = sample_batch(args.window)
    with torch.no_grad():
        before = DecodeGraph(core, args.horizon).eval()(context, mask)
    swapped = replace_rms_norm(core)
    graph = DecodeGraph(core, args.horizon).eval()
    with torch.no_grad():
        reference = graph(context, mask)
    drift = (before - reference).abs().max().item()
    print(f"replaced {swapped} RMSNorm layers, output drift {drift:.3e}")
    if drift > 1e-4:
        raise SystemExit(f"RMSNorm replacement altered the model ({drift:.3e})")

    target = args.out / "timesfm3-decode.onnx"
    original = torch.cumprod
    torch.cumprod = cumprod_binary  # only in effect while tracing
    try:
        t0 = time.time()
        torch.onnx.export(
            graph, (context, mask), str(target), dynamo=True,
            opset_version=args.opset,
            input_names=["context", "mask"], output_names=["quantiles"],
        )
        print(f"exported in {time.time() - t0:.0f}s -> {target}")
    finally:
        torch.cumprod = original

    if args.quantize == "int8":
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantized = args.out / "timesfm3-decode-int8.onnx"
        # Two things matter for accuracy, both measured rather than assumed:
        #   * per-channel + reduce_range - plain dynamic int8 pushes grid-load
        #     sMAPE from 1.9% to 6.0%; per-channel holds it at 3.1%.
        #   * keeping the output head in fp32 - quantizing it collapses p10-p90
        #     coverage from 92% to 30%, because the quantile spread is produced
        #     almost entirely there. Excluding four nodes costs 17 MB and
        #     restores coverage to 95%.
        exported = onnx.load(str(target)).graph
        head = [n.name for n in exported.node if n.op_type in ("MatMul", "Gemm")][-4:]
        print(f"quantizing to int8, keeping the output head in fp32: {head}")
        quantize_dynamic(str(target), str(quantized), weight_type=QuantType.QInt8,
                         per_channel=True, reduce_range=True, nodes_to_exclude=head)
        target = quantized

    total = sum(f.stat().st_size for f in args.out.glob("*.onnx*")) / 1e6
    print(f"artefacts in {args.out}: {total:.0f} MB")

    (args.out / "config.json").write_text(json.dumps({
        "window": args.window,
        "horizon": args.horizon,
        "quantiles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "median_index": 4,
        "file": target.name,
        # The browser keys its cache on this, so re-exporting invalidates it.
        "bytes": target.stat().st_size,
    }, indent=2) + "\n")

    # Parity against eager PyTorch at several real context lengths.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(1)
    full = (np.sin(np.arange(4096) * 2 * np.pi / 24) * 10 + 100
            + rng.normal(0, 0.5, 4096)).astype(np.float32)
    print(f"\n{'real ctx':>9}{'max abs diff':>15}{'relative':>11}")
    for used in (64, 128, 256, 512):
        used = min(used, args.window)
        pad = args.window - used
        x = np.concatenate([np.zeros(pad, np.float32), full[-used:]])[None, None, :]
        mk = np.concatenate([np.ones(pad, bool), np.zeros(used, bool)])[None, :]
        got = sess.run(None, {"context": x, "mask": mk})[0]
        with torch.no_grad():
            want = graph(torch.tensor(x), torch.tensor(mk)).numpy()
        diff = np.abs(got - want).max()
        print(f"{used:>9}{diff:>15.3e}{diff / np.abs(want).mean():>11.2e}")


if __name__ == "__main__":
    main()
