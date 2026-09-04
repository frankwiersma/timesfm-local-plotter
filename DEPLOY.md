# Hosting it for free on Cloudflare

The browser build is entirely static — page, weights, wasm runtime, nothing
else. No container, no function, no origin server. That makes hosting a
bandwidth question rather than a compute one.

**Live:** <https://timesfm-plotter.pages.dev>

## The number that decides the host

A first-time visitor downloads **382 MB**:

| Asset | Size |
| --- | --: |
| Model, int8, 512-window | 353.3 MB |
| ONNX Runtime wasm | 27.8 MB |
| ONNX Runtime loader | 0.8 MB |
| Page + demo series | 0.1 MB |

Repeat visits cost **nothing** — the weights sit in the Cache API. So the bill
tracks *unique* visitors, not pageviews.

## Why not Netlify

| Plan | Bandwidth | First-time visitors/month |
| --- | --: | --: |
| Free (2026 credit system, ~15 GB) | 15 GB | **39** |
| Free (legacy, pre-Sep 2025) | 100 GB | 262 |
| Pro, $20/mo | 1 TB | 2,618 |

Overage runs about $0.13/GB, so roughly **$0.05 per visitor** — 1,000 visitors
is about $50. On the current free plan, running out of credits stops the site
serving rather than auto-recharging.

## Why Cloudflare, and why Pages alone is not enough

**Pages caps a single asset at 25 MiB.** That rules out the 353 MB model and
also the 26.5 MiB wasm runtime. Cloudflare's own docs point at R2 for this.

So the split is: **Pages serves the site, R2 serves the big files.**

| R2 dimension | Free per month | This project uses |
| --- | --: | --: |
| Storage | 10 GB | 0.38 GB |
| **Egress** | **unlimited, free** | any |
| Class B (read) operations | 10 million | ~6 per new visitor |

Egress is the whole reason this works: R2 charges nothing for data transfer at
any volume. At roughly six reads per new visitor (four model shards, the wasm,
its loader), the 10 million free Class B operations cover about **1.6 million
first-time visitors a month**. Everything here lands in the free tier, with no
Workers Paid plan and nothing container-shaped.

## Deploying it yourself

```bash
python scripts/export_onnx.py --out web/models --quantize int8
./scripts/fetch_runtime.sh

wrangler r2 bucket create timesfm-plotter
wrangler r2 bucket dev-url enable timesfm-plotter
wrangler r2 bucket cors set timesfm-plotter --file scripts/r2-cors.json

python scripts/deploy_r2.py --bucket timesfm-plotter \
    --public-url https://pub-<your-hash>.r2.dev
python scripts/build_pages.py
wrangler pages deploy dist --project-name timesfm-plotter
```

`deploy_r2.py` writes `web/deploy.json`, which tells the page where its assets
live. Without that file the page falls back to `web/models` and `web/vendor`, so
local development is unaffected by any of this.

## Two things worth knowing

**Weights are sharded, and not only to dodge a limit.** Wrangler refuses uploads
over 300 MiB and the model is 337 MiB, so `deploy_r2.py` splits it into 100 MB
parts. The page fetches them concurrently and concatenates, which also makes the
first load faster than one long transfer. Sharded weights are normal practice —
it is how large checkpoints ship on the Hugging Face Hub.

**`r2.dev` is rate-limited and not meant for production.** It is fine for a
demo. For real traffic, attach a custom domain to the bucket and point
`--public-url` at it:

```bash
wrangler r2 bucket domain add timesfm-plotter --domain models.example.com
```

## Measured on the live deployment

Page from Pages, weights from R2, cross-origin:

| | Local | Deployed |
| --- | --: | --: |
| Forecast latency | 336 ms | **329 ms** |
| grid-load sMAPE | 1.83% | 1.83% |
| p10–p90 coverage | 96.9% | 96.9% |
| Skill vs naive | +76.0% | +76.0% |

Identical, as it should be — the same graph runs in the same runtime either way.

## Possible next step: wasm threads

Inference is single-threaded. `ort.env.wasm.numThreads > 1` needs COOP/COEP
headers, which Pages can set via `_headers`. Everything cross-origin would then
need to satisfy COEP — the R2 assets already come back CORS-clean, but the
Google Fonts stylesheet would have to be self-hosted first. Worth doing if 330 ms
is not fast enough; it is the largest remaining win.
