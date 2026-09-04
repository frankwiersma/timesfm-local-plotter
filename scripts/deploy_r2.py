"""Upload the exported weights and the wasm runtime to an R2 bucket.

Wrangler refuses single uploads over 300 MiB and the int8 model is 337 MiB, so
the model is split into shards. The browser fetches them in parallel and
concatenates, which is both a way around the limit and a faster first load.

Writes web/deploy.json, which the page reads to decide where its assets live.
Without that file the page falls back to local paths, so development is
unaffected.

Usage:
    python scripts/deploy_r2.py --bucket timesfm-plotter \
        --public-url https://pub-xxxx.r2.dev
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

SHARD_BYTES = 100 * 1024 * 1024  # comfortably under wrangler's 300 MiB ceiling


def put(bucket: str, key: str, path: Path, content_type: str) -> None:
    subprocess.run(
        ["wrangler", "r2", "object", "put", f"{bucket}/{key}",
         "--file", str(path), "--content-type", content_type, "--remote"],
        check=True, capture_output=True, text=True)
    print(f"  uploaded {key}  ({path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--public-url", required=True, help="r2.dev or custom domain, no trailing slash")
    ap.add_argument("--models", type=Path, default=Path("web/models"))
    ap.add_argument("--vendor", type=Path, default=Path("web/vendor"))
    ap.add_argument("--out", type=Path, default=Path("web/deploy.json"))
    args = ap.parse_args()

    base = args.public_url.rstrip("/")
    cfg = json.loads((args.models / "config.json").read_text())
    model = args.models / cfg["file"]
    total = model.stat().st_size
    shards = math.ceil(total / SHARD_BYTES)
    print(f"model {model.name}: {total / 1e6:.1f} MB -> {shards} shard(s)")

    names = []
    with tempfile.TemporaryDirectory() as tmp, model.open("rb") as fh:
        for i in range(shards):
            name = f"{cfg['file']}.{i:02d}"
            part = Path(tmp) / name
            part.write_bytes(fh.read(SHARD_BYTES))
            put(args.bucket, f"models/{name}", part, "application/octet-stream")
            names.append(name)

    for wasm in sorted(args.vendor.glob("*.wasm")):
        put(args.bucket, f"vendor/{wasm.name}", wasm, "application/wasm")
    for mjs in sorted(args.vendor.glob("*.mjs")):
        put(args.bucket, f"vendor/{mjs.name}", mjs, "text/javascript")

    args.out.write_text(json.dumps({
        "modelBase": f"{base}/models/",
        "wasmBase": f"{base}/vendor/",
        "shards": names,
        "bytes": total,
        "window": cfg["window"],
        "horizon": cfg["horizon"],
        "quantiles": cfg["quantiles"],
        "median_index": cfg["median_index"],
    }, indent=2) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
