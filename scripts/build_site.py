from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import polars as pl
import yaml


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _queries_from_cfg(cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    q = cfg.get("queries") or []
    out: List[Dict[str, str]] = []
    for item in q:
        if isinstance(item, dict):
            name = str(item.get("name", "query"))
            query = str(item.get("query", "")).strip()
            if query:
                out.append({"name": name, "query": query})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yml")
    ap.add_argument("--data-parquet", default="data/paracel_mentions.parquet")
    ap.add_argument("--site-dir", default="docs")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    project = cfg.get("project") or {}
    max_items = int(project.get("max_items_dashboard", 2000))

    site_dir = Path(args.site_dir)
    data_dir = site_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = Path(args.data_parquet)

    # Meta base (siempre)
    qcfg = _queries_from_cfg(cfg)
    meta: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "queries": qcfg,
        "n_items": 0,
        "max_items_dashboard": max_items,
        "by_source": {},
        "by_topic": {},
        "by_sentiment": {},
        "note": None,
    }

    if not parquet_path.exists():
        (data_dir / "latest.json").write_text("[]\n", encoding="utf-8")
        meta["note"] = "Dataset parquet not found. Run scripts/run_daily.py then scripts/build_site.py (or GitHub Actions)."
        (data_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    df = pl.read_parquet(parquet_path)

    if df.is_empty():
        (data_dir / "latest.json").write_text("[]\n", encoding="utf-8")
        meta["note"] = "Dataset is empty (0 mentions). Consider simplifying queries and widening days_back."
        (data_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    # Orden: más recientes arriba
    if "published_at" in df.columns:
        df = df.sort("published_at", descending=True)

    cols = [c for c in [
        "published_at","source","domain","topic","sentiment_label","sentiment_score",
        "title","extracted_title","snippet","url","query_name","query"
    ] if c in df.columns]

    df_out = df.select(cols).head(max_items)
    (data_dir / "latest.json").write_text(df_out.write_json(row_oriented=True), encoding="utf-8")

    meta["n_items"] = int(df_out.height)

    # Resúmenes rápidos para el front
    def _count(col: str) -> Dict[str, int]:
        if col not in df.columns:
            return {}
        s = df.group_by(col).agg(pl.len().alias("n")).sort("n", descending=True)
        return {str(r[0]): int(r[1]) for r in s.iter_rows()}

    meta["by_source"] = _count("source")
    meta["by_topic"] = _count("topic")
    meta["by_sentiment"] = _count("sentiment_label")

    (data_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
