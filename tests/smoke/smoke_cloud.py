"""Smoke-test the foundation models on the cheapest vast.ai GPU, then destroy the instance.

Rents one GPU under the price cap, copies the code (no data, no weights), installs the
packages, runs Chronos-Bolt small, Chronos-2, TimesFM 2.5 and Kronos-small on synthetic
series, fetches the results to tests/smoke/out/cloud-smoke/<run-id>/ and destroys the instance
in every case; the instance also powers itself off after the time limit.

Usage: .venv/bin/python tests/smoke/smoke_cloud.py [--max-price 0.15] [--max-minutes 30]
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd

from src import LAB_DIR
from src.cloud import (
    cheapest_offer,
    create_instance,
    destroy_instance,
    ensure_ssh_key,
    rsync,
    running_instances,
    ssh,
    wait_for_ssh,
)

logger = logging.getLogger("cloud_smoke")

IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
MODELS = "chronos-bolt-small chronos-2 timesfm-2.5 kronos-small"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
REMOTE = "/root/anonymous-benchmark"
"""The lab's folder on the instance."""
SETUP = (
    f"set -e; cd {REMOTE}; curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null; "
    "export PATH=$HOME/.local/bin:$PATH; uv venv -q --python 3.12 .venv; "
    f"uv pip install -q --python .venv/bin/python torch --index-url {TORCH_INDEX}; "
    "uv pip install -q --python .venv/bin/python chronos-forecasting==2.3.2 "
    "'timesfm[torch]==3.0.2' einops safetensors huggingface_hub pandas pyarrow psutil"
)
SMOKE = (
    f"cd {REMOTE} && "
    "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > gpu.csv; "
    "SMOKE_SYNTHETIC=1 PYTHONPATH=. "
    f".venv/bin/python tests/smoke/smoke_models.py {MODELS} > smoke.log 2>&1; echo done"
)
# Leading "/" anchors a pattern at the lab root, so "/data" skips the data folder but keeps
# the src/data package.
EXCLUDES = [
    "/.venv", "/.venv-moirai", "/.hf", "/data", "/tests/smoke/out", "/experiments/*/results",
    "/experiments/*/forecasts", "/.git", "/.tmp", "/paper", "__pycache__", "node_modules",
    "/vendor/Kronos/examples", "/vendor/Kronos/figures", "/vendor/Kronos/webui",
    "/vendor/Kronos/finetune*",
]  # fmt: skip


def main() -> None:
    """Rent, test, fetch, destroy, and record time and cost."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--max-price", type=float, default=0.15)
    parser.add_argument("--max-minutes", type=int, default=30)
    arguments = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    run_id = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H%M")
    folder = LAB_DIR / "tests" / "smoke" / "out" / "cloud-smoke" / run_id
    folder.mkdir(parents=True)
    ensure_ssh_key(Path.home() / ".ssh" / "id_ed25519.pub")
    offer = cheapest_offer(max_price=arguments.max_price, min_vram_gb=12, min_cuda=12.8)
    started = time.monotonic()
    instance = create_instance(offer, image=IMAGE, disk_gb=40, guard_minutes=arguments.max_minutes)
    status = "failed"
    try:
        host, port = wait_for_ssh(instance, timeout_seconds=600)
        rsync(f"{LAB_DIR}/", f"root@{host}:{REMOTE}/", port, excludes=EXCLUDES)
        left = arguments.max_minutes * 60 - (time.monotonic() - started)
        setup = ssh(host, port, SETUP, timeout_seconds=max(left - 300, 60))
        logger.info("setup exit %d %s", setup.returncode, setup.stderr[-300:])
        left = arguments.max_minutes * 60 - (time.monotonic() - started)
        smoke = ssh(host, port, SMOKE, timeout_seconds=max(left - 60, 60))
        logger.info("smoke exit %d", smoke.returncode)
        for name in ("smoke.log", "gpu.csv", "tests/smoke/out/smoke_test.jsonl"):
            rsync(f"root@{host}:{REMOTE}/{name}", str(folder) + "/", port, excludes=[])
        status = "completed"
    finally:
        destroy_instance(instance)
        minutes = (time.monotonic() - started) / 60
        record = {
            "run_id": run_id, "status": status, "instance_id": instance, "offer": offer.id,
            "gpu": offer.gpu, "price_per_hour": offer.price_per_hour, "minutes": round(minutes, 1),
            "cost": round(offer.price_per_hour * minutes / 60, 4), "location": offer.location,
            "instances_left": len(running_instances()),
        }  # fmt: skip
        (folder / "meta.json").write_text(json.dumps(record, indent=2))
        logger.info("%s", record)


if __name__ == "__main__":
    main()
