"""Write data/manifest.json from the bars already on disk (while a download still runs).

Usage: .venv/bin/python scripts/refresh_manifest.py
"""

import logging

from src import LAB_DIR
from src.data import load_data_config, refresh_manifest


def main() -> None:
    """Describe and check every built instrument and write the manifest."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config = load_data_config(LAB_DIR / "configs" / "data.toml", data_dir=LAB_DIR / "data")
    refresh_manifest(config)


if __name__ == "__main__":
    main()
