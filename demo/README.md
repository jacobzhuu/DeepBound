# DeepBound Frontend Demo

This demo serves real model predictions and metrics from the DeepBound finetuned checkpoint.

## Prerequisites

- Checkpoint: `checkpoints/finetune_msvs_funcbound_64/checkpoint_best.pt`
- Data: `data-raw/msvs_funcbound_64_bap_test/` with `truth_labeled_code/` and `ida_labeled_code/`
- Python environment with DeepBound + fairseq + torch installed

## Run the backend

From the repo root:

```bash
python demo/server.py
```

Optional overrides:

```bash
python demo/server.py \
  --host 0.0.0.0 --port 8000 \
  --checkpoint-dir checkpoints/finetune_msvs_funcbound_64 \
  --checkpoint-file checkpoint_best.pt \
  --data-bin data-bin/funcbound_msvs_64 \
  --data-root data-raw/msvs_funcbound_64_bap_test \
  --user-dir finetune_tasks \
  --device auto \
  --max-tokens 512
```

## Run the frontend

```bash
cd demo
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. The Vite dev server proxies `/api` to `http://localhost:8000`.

If you need a custom API base, set `VITE_API_BASE` when running Vite.

## One-click start (recommended for remote access)

```bash
./demo/start_demo.sh
```

The script prints a URL like `http://<server-ip>:5173`. Open it from your Mac browser.
