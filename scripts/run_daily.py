from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import polars as pl
import yaml

from paracel_monitor.pipeline import collect_all, mentions_to_df, merge_with_existing


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compile_topic_patterns(cfg: Dict[str, Any]) -> List[Tuple[str, re.Pattern]]:
    topics = (cfg.get("taxonomy") or {}).get("topics") or []
    out: List[Tuple[str, re.Pattern]] = []
    for t in topics:
        topic = str(t.get("topic", "NA"))
        pattern = str(t.get("pattern", ".*"))
        try:
            out.append((topic, re.compile(pattern, re.IGNORECASE)))
        except re.error:
            out.append((topic, re.compile(".*", re.IGNORECASE)))
    return out


def write_summaries(df: pl.DataFrame, out_dir: Path) -> None:
    by_source = df.group_by("source").agg(pl.len().alias("n")).sort("n", descending=True)
    by_domain = (
        df.filter(pl.col("domain").is_not_null())
          .group_by("domain")
          .agg(pl.len().alias("n"))
          .sort("n", descending=True)
          .head(200)
    )
    by_day = (
        df.with_columns(pl.col("published_at").dt.date().alias("day"))
          .group_by("day")
          .agg(pl.len().alias("n"))
          .sort("day")
    )
    by_topic = df.group_by("topic").agg(pl.len().alias("n")).sort("n", descending=True)
    by_sent = df.group_by("sentiment_label").agg(pl.len().alias("n"), pl.mean("sentiment_score").alias("avg_score")).sort("n", descending=True)

    by_source.write_csv(out_dir / "summary__by_source.csv")
    by_domain.write_csv(out_dir / "summary__by_domain.csv")
    by_day.write_csv(out_dir / "summary__by_day.csv")
    by_topic.write_csv(out_dir / "summary__by_topic.csv")
    by_sent.write_csv(out_dir / "summary__by_sentiment.csv")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yml")
    ap.add_argument("--out-data-dir", default="data")
    ap.add_argument("--days-back", type=int, default=None)
    ap.add_argument("--extract-text", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    cfg = load_config(Path(args.config))

    out_dir = Path(args.out_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    project = cfg.get("project") or {}
    days_back = int(args.days_back if args.days_back is not None else project.get("days_back", 180))
    max_records_gdelt = int(project.get("max_records_gdelt", 250))

    sources = cfg.get("sources") or {}
    use_gdelt = bool(sources.get("gdelt", True))
    use_google = bool(sources.get("google_news_rss", True))
    rss_feeds = sources.get("rss_feeds") or []

    queries = cfg.get("queries") or []
    mentions = collect_all(
        queries=queries,
        use_gdelt=use_gdelt,
        use_google_news_rss=use_google,
        rss_feeds=rss_feeds,
        days_back=days_back,
        max_records_gdelt=max_records_gdelt,
    )

    topic_patterns = compile_topic_patterns(cfg)

    sent_cfg = cfg.get("sentiment") or {}
    do_sentiment = bool(sent_cfg.get("enabled", True))
    positive_terms = sent_cfg.get("positive") or []
    negative_terms = sent_cfg.get("negative") or []

    df_new = mentions_to_df(
        mentions=mentions,
        topic_patterns=topic_patterns,
        do_extract_text=bool(args.extract_text),
        do_sentiment=do_sentiment,
        positive_terms=positive_terms,
        negative_terms=negative_terms,
    )

    parquet_path = out_dir / "paracel_mentions.parquet"
    csv_path = out_dir / "paracel_mentions.csv"

    df_all = merge_with_existing(str(parquet_path), df_new)

    # Esquema mínimo (para asegurar artefactos aunque no existan menciones)
    schema_cols = [
        ("query_name", pl.Utf8),
        ("query", pl.Utf8),
        ("source", pl.Utf8),
        ("title", pl.Utf8),
        ("extracted_title", pl.Utf8),
        ("url", pl.Utf8),
        ("domain", pl.Utf8),
        ("published_at", pl.Datetime(time_zone="UTC")),
        ("snippet", pl.Utf8),
        ("text", pl.Utf8),
        ("topic", pl.Utf8),
        ("sentiment_label", pl.Utf8),
        ("sentiment_score", pl.Float64),
        ("url_sha256", pl.Utf8),
    ]

    if df_all.is_empty():
        df_all = pl.DataFrame({c: [] for c, _ in schema_cols}).with_columns([
            pl.col("published_at").cast(pl.Datetime(time_zone="UTC")),
            pl.col("sentiment_score").cast(pl.Float64),
        ])
        (out_dir / ".empty_run").write_text("No data collected.\n", encoding="utf-8")
    else:
        if (out_dir / ".empty_run").exists():
            try:
                (out_dir / ".empty_run").unlink()
            except Exception:
                pass

    # Persistencia siempre (aunque sea vacío)
    df_all.write_parquet(parquet_path)
    df_all.write_csv(csv_path)

    # Resúmenes solo si hay datos
    if not df_all.is_empty():
        write_summaries(df_all, out_dir)


if __name__ == "__main__":
    main()
