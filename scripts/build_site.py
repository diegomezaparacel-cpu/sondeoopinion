from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import polars as pl
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yml")
    ap.add_argument("--data-parquet", default="data/paracel_mentions.parquet")
    ap.add_argument("--site-dir", default="docs")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    project = cfg.get("project") or {}

    site_dir = Path(args.site_dir)
    (site_dir / "data").mkdir(parents=True, exist_ok=True)

    parquet_path = Path(args.data_parquet)
    if not parquet_path.exists():
        (site_dir / "data" / "latest.json").write_text("[]\n", encoding="utf-8")
        meta = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "query": None,
            "n_items": 0,
            "note": "Dataset parquet not found."
        }
        (site_dir / "data" / "meta.json").write_text(pl.DataFrame([meta]).write_json(), encoding="utf-8")
        return

    df = pl.read_parquet(parquet_path)

    max_items = int(project.get("max_items_dashboard", 2000))
    # Orden: más recientes arriba
    if "published_at" in df.columns:
        df = df.sort("published_at", descending=True)

    # Campos mínimos para el front
    cols = [c for c in [
        "published_at","source","domain","topic","sentiment_label","sentiment_score",
        "title","extracted_title","snippet","url"
    ] if c in df.columns]

    df_out = df.select(cols).head(max_items)

    (site_dir / "data" / "latest.json").write_text(df_out.write_json(row_oriented=True), encoding="utf-8")

    # Meta
    first_query = None
    if "query" in df.columns and df.height > 0:
        first_query = df.select(pl.col("query").drop_nulls().head(1)).to_series().to_list()
        first_query = first_query[0] if first_query else None

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "query": first_query,
        "n_items": int(df_out.height),
        "max_items_dashboard": max_items,
    }
    (site_dir / "data" / "meta.json").write_text(pl.DataFrame([meta]).write_json(), encoding="utf-8")


if __name__ == "__main__":
    main()
