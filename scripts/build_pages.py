"""Assemble the Cloudflare Pages bundle.

Only small, static files go to Pages. The model shards and the 26.5 MiB wasm
runtime live in R2 because Pages caps a single asset at 25 MiB.

Usage:
    python scripts/build_pages.py            # then: wrangler pages deploy dist
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

PAGES_MAX = 25 * 1024 * 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("dist"))
    args = ap.parse_args()

    web = Path("web")
    if not (web / "deploy.json").exists():
        raise SystemExit("web/deploy.json missing — run scripts/deploy_r2.py first")

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "vendor").mkdir(parents=True)

    # browser.html becomes the site root
    shutil.copy(web / "browser.html", args.out / "index.html")
    for name in ("demo-series.json", "deploy.json"):
        shutil.copy(web / name, args.out / name)
    # only the loader script; the wasm it pulls at runtime is served from R2
    shutil.copy(web / "vendor" / "ort.all.min.js", args.out / "vendor" / "ort.all.min.js")

    (args.out / "_headers").write_text(
        "/vendor/*\n  Cache-Control: public, max-age=31536000, immutable\n"
        "/demo-series.json\n  Cache-Control: public, max-age=3600\n")

    oversized = [f for f in args.out.rglob("*") if f.is_file() and f.stat().st_size > PAGES_MAX]
    if oversized:
        raise SystemExit("over the 25 MiB Pages limit: "
                         + ", ".join(f"{f} ({f.stat().st_size/1e6:.0f} MB)" for f in oversized))

    total = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    for f in sorted(args.out.rglob("*")):
        if f.is_file():
            print(f"  {f.stat().st_size/1e6:>8.2f} MB  {f.relative_to(args.out)}")
    print(f"  {total/1e6:>8.2f} MB  total — all under the 25 MiB per-asset limit")


if __name__ == "__main__":
    main()
