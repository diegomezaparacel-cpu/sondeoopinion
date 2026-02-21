from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import polars as pl
import requests
import feedparser
import trafilatura
from dateutil import parser as dtparser
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
}


@dataclass(frozen=True)
class Mention:
    query_name: str
    query: str
    source: str
    title: Optional[str]
    url: str
    published_at: Optional[datetime]
    domain: Optional[str]
    snippet: Optional[str]
    raw: Optional[dict]


class HTTPError(RuntimeError):
    pass


def _safe_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = dtparser.parse(str(value))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _domain_from_url(url: str) -> Optional[str]:
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lower().strip()
        return netloc if netloc else None
    except Exception:
        return None


def _fingerprint(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8", errors="ignore")).hexdigest()


@retry(
    wait=wait_exponential(multiplier=0.8, min=1, max=12),
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type((requests.RequestException, HTTPError)),
)
def http_get(url: str, params: Optional[dict] = None, timeout: int = 30) -> requests.Response:
    resp = requests.get(url, params=params, headers=DEFAULT_HEADERS, timeout=timeout)
    if resp.status_code >= 400:
        raise HTTPError(f"HTTP {resp.status_code}")
    return resp


def fetch_gdelt(query_name: str, query: str, start_utc: datetime, end_utc: datetime, max_records: int = 250) -> List[Mention]:
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params: Dict[str, Any] = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": int(max_records),
        "sort": "HybridRel",
        "startdatetime": start_utc.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end_utc.strftime("%Y%m%d%H%M%S"),
    }
    r = http_get(base_url, params=params)
    payload = r.json()
    articles = payload.get("articles", []) or []
    out: List[Mention] = []
    for a in articles:
        url = a.get("url")
        if not url:
            continue
        out.append(
            Mention(
                query_name=query_name,
                query=query,
                source="gdelt",
                title=a.get("title"),
                url=url,
                published_at=_safe_dt(a.get("seendate") or a.get("published")),
                domain=_domain_from_url(url),
                snippet=a.get("snippet") or a.get("summary"),
                raw=a,
            )
        )
    return out


def build_google_news_rss_url(query: str, hl: str = "es-419", gl: str = "PY", ceid: str = "PY:es-419") -> str:
    from urllib.parse import quote_plus
    q = quote_plus(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"



def _sanitize_google_news_query(query: str) -> str:
    # Google News RSS es sensible a operadores. Normalizamos a una consulta "suave".
    q = query
    q = q.replace("(", " ").replace(")", " ")
    q = re.sub(r'\bOR\b', ' ', q, flags=re.IGNORECASE)
    q = re.sub(r'\bAND\b', ' ', q, flags=re.IGNORECASE)
    q = q.replace('"', ' ')
    q = re.sub(r'\s+', ' ', q).strip()
    return q


def fetch_google_news_rss(query_name: str, query: str) -> List[Mention]:
    rss_url = build_google_news_rss_url(_sanitize_google_news_query(query))
    d = feedparser.parse(rss_url)
    out: List[Mention] = []
    for e in d.entries or []:
        url = getattr(e, "link", None)
        if not url:
            continue
        title = getattr(e, "title", None)
        published = _safe_dt(getattr(e, "published", None)) if hasattr(e, "published") else None
        summary = getattr(e, "summary", None)
        out.append(
            Mention(
                query_name=query_name,
                query=query,
                source="google_news_rss",
                title=title,
                url=url,
                published_at=published,
                domain=_domain_from_url(url),
                snippet=summary,
                raw={"rss": rss_url},
            )
        )
    return out


def fetch_rss_feed(feed_url: str, query_name: str, query: str) -> List[Mention]:
    d = feedparser.parse(feed_url)
    out: List[Mention] = []
    for e in d.entries or []:
        url = getattr(e, "link", None)
        if not url:
            continue
        title = getattr(e, "title", None)
        summary = getattr(e, "summary", None)
        published = None
        if hasattr(e, "published"):
            published = _safe_dt(getattr(e, "published", None))
        elif hasattr(e, "updated"):
            published = _safe_dt(getattr(e, "updated", None))

        out.append(
            Mention(
                query_name=query_name,
                query=query,
                source="rss",
                title=title,
                url=url,
                published_at=published,
                domain=_domain_from_url(url),
                snippet=summary,
                raw={"feed": feed_url},
            )
        )
    return out


def extract_article(url: str, max_chars: int = 120_000) -> Tuple[Optional[str], Optional[str]]:
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None, None
        result = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
            output_format="json",
        )
        if not result:
            return None, None
        j = json.loads(result)
        text = j.get("text")
        title = j.get("title")
        if isinstance(text, str):
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > max_chars:
                text = text[:max_chars]
        else:
            text = None
        if isinstance(title, str):
            title = re.sub(r"\s+", " ", title).strip()
        else:
            title = None
        return text, title
    except Exception:
        return None, None


def classify_topic(text: str, topic_patterns: List[Tuple[str, re.Pattern]]) -> str:
    for topic, pat in topic_patterns:
        if pat.search(text):
            return topic
    return "NA"


def score_sentiment(text: str, positive_terms: List[str], negative_terms: List[str]) -> Tuple[str, float]:
    t = text.lower()
    pos = 0
    neg = 0
    for w in positive_terms:
        if w and w.lower() in t:
            pos += 1
    for w in negative_terms:
        if w and w.lower() in t:
            neg += 1
    if pos == 0 and neg == 0:
        return "NA", 0.0
    score = (pos - neg) / float(pos + neg)
    if score > 0.15:
        return "POSITIVE", score
    if score < -0.15:
        return "NEGATIVE", score
    return "NEUTRAL", score


def mentions_to_df(
    mentions: List[Mention],
    topic_patterns: List[Tuple[str, re.Pattern]],
    do_extract_text: bool,
    do_sentiment: bool,
    positive_terms: List[str],
    negative_terms: List[str],
) -> pl.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in mentions:
        text = None
        extracted_title = None
        if do_extract_text:
            text, extracted_title = extract_article(m.url)

        blob = " ".join([x for x in [m.title, extracted_title, m.snippet, text] if isinstance(x, str) and x.strip()])
        topic = classify_topic(blob, topic_patterns) if blob else "NA"

        sent_label = "NA"
        sent_score = 0.0
        if do_sentiment and blob:
            sent_label, sent_score = score_sentiment(blob, positive_terms, negative_terms)

        rows.append(
            {
                "query_name": m.query_name,
                "query": m.query,
                "source": m.source,
                "title": m.title,
                "extracted_title": extracted_title,
                "url": m.url,
                "domain": m.domain,
                "published_at": m.published_at,
                "snippet": m.snippet,
                "text": text,
                "topic": topic,
                "sentiment_label": sent_label,
                "sentiment_score": float(sent_score),
                "url_sha256": _fingerprint(m.url),
            }
        )
    df = pl.DataFrame(rows)
    if df.is_empty():
        return df
    df = df.with_columns(
        [
            pl.col("published_at").cast(pl.Datetime(time_zone="UTC")),
            pl.col("domain").cast(pl.Utf8).str.to_lowercase(),
            pl.col("topic").cast(pl.Utf8),
            pl.col("sentiment_label").cast(pl.Utf8),
            pl.col("sentiment_score").cast(pl.Float64),
        ]
    )
    return df


def merge_with_existing(existing_path: str, new_df: pl.DataFrame) -> pl.DataFrame:
    if new_df.is_empty():
        if existing_path and pl.Path(existing_path).exists():
            return pl.read_parquet(existing_path)
        return new_df
    try:
        p = pl.Path(existing_path)
        if p.exists():
            old = pl.read_parquet(existing_path)
            if old.is_empty():
                return new_df
            combined = pl.concat([old, new_df], how="vertical_relaxed")
            combined = combined.unique(subset=["url_sha256"], keep="first")
            return combined
    except Exception:
        return new_df
    return new_df


def collect_all(
    queries: List[Dict[str, str]],
    use_gdelt: bool,
    use_google_news_rss: bool,
    rss_feeds: List[str],
    days_back: int,
    max_records_gdelt: int,
) -> List[Mention]:
    start = datetime.now(timezone.utc) - timedelta(days=int(days_back))
    end = datetime.now(timezone.utc)

    out: List[Mention] = []
    for q in queries:
        qname = q.get("name", "query")
        qstr = q.get("query", "")
        if not qstr:
            continue
        if use_gdelt:
            out.extend(fetch_gdelt(qname, qstr, start, end, max_records=max_records_gdelt))
        if use_google_news_rss:
            out.extend(fetch_google_news_rss(qname, qstr))
        for feed in rss_feeds or []:
            out.extend(fetch_rss_feed(feed, qname, qstr))

    seen: set[str] = set()
    uniq: List[Mention] = []
    for m in out:
        fp = _fingerprint(m.url)
        if fp in seen:
            continue
        seen.add(fp)
        uniq.append(m)
    return uniq
