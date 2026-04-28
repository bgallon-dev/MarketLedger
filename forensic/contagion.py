"""
Sector Contagion Tracer — forensic module.

Tracks how risk-factor language propagates across SEC 10-K filings within a
sector. Companies that disclose a risk cluster before peers score as "Leaders";
those that adopt language after peers are "Followers".

Uses SEC EDGAR full-text search (no API key required). Caches Item 1A text and
MinHash signatures in the existing SQLite DB. Activated by --contagion CLI flag.

Public API:
    run_contagion_scan(candidates_df, prefetched=None, verbose=True,
                       max_workers=None) -> pd.DataFrame
"""

import hashlib
import html
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────

EDGAR_USER_AGENT = "MarketLedger/1.0 research@example.com"
MINHASH_N = 128
SHINGLE_SIZE = 5
JACCARD_THRESHOLD = 0.35
CACHE_MAX_AGE_DAYS = 90
CIK_MAP_REFRESH_DAYS = 30

DB_PATH = Path(__file__).parent.parent / "database" / "financial_data.db"
CIK_MAP_PATH = Path(__file__).parent.parent / "database" / "sec_cik_map.json"

# ── Rate limiting ──────────────────────────────────────────────────────────────
# EDGAR requires ≤ 10 req/sec and a User-Agent header.

_EDGAR_SEMAPHORE = threading.Semaphore(10)
_EDGAR_LAST_CALL_LOCK = threading.Lock()
_EDGAR_LAST_CALL_TIME: float = 0.0


# ── SQLite cache ───────────────────────────────────────────────────────────────

def _ensure_contagion_table() -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sec_filings (
                ticker       TEXT NOT NULL,
                cik          TEXT NOT NULL,
                filed_date   TEXT NOT NULL,
                accession_no TEXT NOT NULL,
                item_1a_text TEXT,
                item_7_text  TEXT,
                minhash_json TEXT,
                fetched_at   TEXT NOT NULL,
                PRIMARY KEY (ticker, filed_date)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sec_filings_ticker ON sec_filings(ticker)"
        )
        # Migrations: add columns for databases created before these existed
        for col_sql in (
            "ALTER TABLE sec_filings ADD COLUMN item_7_text TEXT",
            "ALTER TABLE sec_filings ADD COLUMN form_type TEXT DEFAULT '10-K'",
        ):
            try:
                conn.execute(col_sql)
            except Exception:
                pass  # Column already exists
        conn.commit()
    finally:
        conn.close()


def _get_cached_filing(ticker: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM sec_filings WHERE ticker=? ORDER BY filed_date DESC LIMIT 1",
            (ticker.upper(),),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None
    try:
        age = (datetime.now() - datetime.fromisoformat(row["fetched_at"])).days
    except Exception:
        age = CACHE_MAX_AGE_DAYS + 1
    if age > CACHE_MAX_AGE_DAYS:
        return None
    # Skip entries with no extracted text at all (pre-20-F cache misses)
    cols = row.keys()
    item_7 = row["item_7_text"] if "item_7_text" in cols else None
    if row["item_1a_text"] is None and row["minhash_json"] is None and item_7 is None:
        return None

    minhash = json.loads(row["minhash_json"]) if row["minhash_json"] else None
    return {
        "ticker": row["ticker"],
        "cik": row["cik"],
        "filed_date": row["filed_date"],
        "accession_no": row["accession_no"],
        "form_type": row["form_type"] if "form_type" in cols else "10-K",
        "item_1a_text": row["item_1a_text"],
        "item_7_text": item_7,
        "minhash": minhash,
        "sic": None,
    }


def _get_cached_filings_n(ticker: str, n: int = 5) -> List[Dict[str, Any]]:
    """Return up to n cached filings for ticker, most recent first, within cache TTL."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT * FROM sec_filings WHERE ticker=? ORDER BY filed_date DESC LIMIT ?",
            (ticker.upper(), n),
        ).fetchall()
    finally:
        conn.close()

    results = []
    for row in rows:
        try:
            age = (datetime.now() - datetime.fromisoformat(row["fetched_at"])).days
        except Exception:
            age = CACHE_MAX_AGE_DAYS + 1
        if age > CACHE_MAX_AGE_DAYS:
            continue
        cols = row.keys()
        item_7 = row["item_7_text"] if "item_7_text" in cols else None
        # Skip entries with no extracted text at all (pre-20-F cache misses)
        if row["item_1a_text"] is None and row["minhash_json"] is None and item_7 is None:
            continue
        minhash = json.loads(row["minhash_json"]) if row["minhash_json"] else None
        results.append({
            "ticker": row["ticker"],
            "cik": row["cik"],
            "filed_date": row["filed_date"],
            "accession_no": row["accession_no"],
            "form_type": row["form_type"] if "form_type" in cols else "10-K",
            "item_1a_text": row["item_1a_text"],
            "item_7_text": item_7,
            "minhash": minhash,
        })
    return results


def _save_filing_to_cache(
    ticker: str,
    cik: str,
    filed_date: str,
    accession_no: str,
    item_1a_text: Optional[str],
    minhash: Optional[List[int]],
    item_7_text: Optional[str] = None,
    form_type: str = "10-K",
) -> None:
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO sec_filings
                (ticker, cik, filed_date, accession_no, item_1a_text,
                 item_7_text, minhash_json, fetched_at, form_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ticker.upper(),
                cik,
                filed_date,
                accession_no,
                item_1a_text,
                item_7_text,
                json.dumps(minhash) if minhash is not None else None,
                datetime.now().isoformat(),
                form_type,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ── EDGAR HTTP layer ───────────────────────────────────────────────────────────

def _edgar_get(url: str, timeout: int = 20) -> bytes:
    global _EDGAR_LAST_CALL_TIME
    with _EDGAR_SEMAPHORE:
        with _EDGAR_LAST_CALL_LOCK:
            gap = time.monotonic() - _EDGAR_LAST_CALL_TIME
            if gap < 0.1:
                time.sleep(0.1 - gap)
            _EDGAR_LAST_CALL_TIME = time.monotonic()
        req = urllib.request.Request(url, headers={"User-Agent": EDGAR_USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()


def _load_cik_map() -> Dict[str, str]:
    if CIK_MAP_PATH.exists():
        try:
            age = (datetime.now() - datetime.fromtimestamp(
                CIK_MAP_PATH.stat().st_mtime
            )).days
            if age < CIK_MAP_REFRESH_DAYS:
                with open(CIK_MAP_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass

    raw = _edgar_get("https://www.sec.gov/files/company_tickers.json")
    data = json.loads(raw)
    mapping: Dict[str, str] = {}
    for entry in data.values():
        ticker = str(entry.get("ticker", "")).strip().upper()
        cik_int = int(entry.get("cik_str", 0))
        if ticker and cik_int:
            mapping[ticker] = f"{cik_int:010d}"

    try:
        CIK_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CIK_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f)
    except Exception:
        pass
    return mapping


def _fetch_latest_10k_meta(cik: str) -> Optional[Dict[str, str]]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        raw = _edgar_get(url)
    except Exception:
        return None
    try:
        data = json.loads(raw)
    except Exception:
        return None

    sic = str(data.get("sic", "") or "")
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    documents = filings.get("primaryDocument", [])

    for form, date, acc, doc in zip(forms, dates, accessions, documents):
        if form in ("10-K", "20-F"):
            return {
                "filed_date": date,
                "accession_no": acc,
                "document": doc,
                "sic": sic,
            }
    return None


def _extract_item_1a_from_text(text: str) -> Optional[str]:
    # 10-K: Item 1A through Item 1B/2; 20-F: Item 3D through Item 3E/Item 4
    # Take the longest match to skip the short TOC entry and capture the body section.
    for pattern in (
        re.compile(r"ITEM\s*1A[\s\S]*?(?=ITEM\s*1B|ITEM\s*2)", re.IGNORECASE),
        re.compile(r"ITEM\s*3\.?D[\s\S]*?(?=ITEM\s*3\.?E|ITEM\s*4)", re.IGNORECASE),
    ):
        matches = [m for m in pattern.finditer(text) if len(m.group(0)) > 200]
        if matches:
            return max(matches, key=lambda m: len(m.group(0))).group(0)[:200_000]
    return None


def _extract_item_7_from_text(text: str) -> Optional[str]:
    # 10-K: Item 7 MD&A through Item 7A/8; 20-F: Item 5 Operating Review through Item 6
    # Take the longest match to skip short TOC entries and capture the body section.
    for pattern in (
        re.compile(
            r"ITEM\s*7[\s.]+MANAGEMENT[‘’S\s]+DISCUSSION[\s\S]*?(?=ITEM\s*7A\s*\.|ITEM\s*8\s*\.)",
            re.IGNORECASE,
        ),
        re.compile(r"ITEM\s*5[\s.]+OPERATING[\s\S]*?(?=ITEM\s*6\s*\.)", re.IGNORECASE),
    ):
        matches = [m for m in pattern.finditer(text) if len(m.group(0)) > 200]
        if matches:
            return max(matches, key=lambda m: len(m.group(0))).group(0)[:200_000]
    return None


def _fetch_and_strip_doc(cik: str, accession_no: str, document: str) -> Optional[str]:
    """Fetch a filing document from EDGAR and strip HTML tags. Returns plain text."""
    acc_nodashes = accession_no.replace("-", "")
    cik_int = int(cik)
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_int}/{acc_nodashes}/{document}"
    )
    try:
        raw_bytes = _edgar_get(url)
        text = raw_bytes.decode("utf-8", errors="replace")
    except Exception:
        return None
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _fetch_item_1a(cik: str, accession_no: str, document: str) -> Optional[str]:
    text = _fetch_and_strip_doc(cik, accession_no, document)
    if text is None:
        return None
    return _extract_item_1a_from_text(text)


def _fetch_n_10k_metas(cik: str, n: int = 5) -> List[Dict[str, str]]:
    """Return metadata for the last n 10-K/20-F annual filings for a CIK."""
    return _fetch_filing_metas(cik, form_types=("10-K", "20-F"), n=n)


def _fetch_filing_metas(
    cik: str, form_types: Tuple[str, ...], n: int = 5
) -> List[Dict[str, str]]:
    """Return metadata for the last n filings matching any of form_types."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        raw = _edgar_get(url)
        data = json.loads(raw)
    except Exception:
        return []

    sic = str(data.get("sic", "") or "")
    filings = data.get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    documents = filings.get("primaryDocument", [])

    results: List[Dict[str, str]] = []
    for form, date, acc, doc in zip(forms, dates, accessions, documents):
        if form in form_types:
            results.append({
                "filed_date": date, "accession_no": acc,
                "document": doc, "sic": sic, "form_type": form,
            })
            if len(results) >= n:
                break
    return results


def _extract_10q_mda(text: str) -> Optional[str]:
    """Extract Item 2 (MD&A) from a 10-Q filing's plain text."""
    pattern = re.compile(
        r"ITEM\s*2[\s.]+MANAGEMENT[''S\s]+DISCUSSION[\s\S]*?(?=ITEM\s*[3-9]\s*[\.\s])",
        re.IGNORECASE,
    )
    matches = [m for m in pattern.finditer(text) if len(m.group(0)) > 200]
    if matches:
        return max(matches, key=lambda m: len(m.group(0))).group(0)[:200_000]
    return None


def _extract_10q_risk_changes(text: str) -> Optional[str]:
    """Extract Item 1A (Changes to Risk Factors) from a 10-Q filing."""
    pattern = re.compile(
        r"ITEM\s*1A[\s.]+(?:CHANGES\s+TO\s+)?RISK\s+FACTORS[\s\S]*?(?=ITEM\s*[2-9]\s*[\.\s])",
        re.IGNORECASE,
    )
    matches = [m for m in pattern.finditer(text) if len(m.group(0)) > 100]
    if matches:
        return max(matches, key=lambda m: len(m.group(0))).group(0)[:50_000]
    return None


def fetch_filing_history(
    ticker: str,
    n_filings: int = 5,
    verbose: bool = False,
    include_10q: bool = True,
    n_10q: int = 4,
) -> List[Dict[str, Any]]:
    """
    Public API for the credibility scorer and discovery pipeline.

    Returns up to n_filings annual (10-K/20-F) dicts per ticker, each containing
    item_1a_text and item_7_text, fetching from EDGAR as needed and caching in
    SQLite. Most-recent filing first.

    When include_10q=True, also fetches up to n_10q quarterly (10-Q) filings and
    stores them in sec_filings with form_type='10-Q'. These are not returned in
    the primary list (which remains annual only) but are available via
    _get_cached_filings_n with form_type filtering.
    """
    ticker = ticker.strip().upper()
    _ensure_contagion_table()

    cached = _get_cached_filings_n(ticker, n_filings)
    cached_dates = {c["filed_date"] for c in cached}
    if len(cached) >= n_filings:
        # Still fetch 10-Qs if requested and not yet cached
        if include_10q:
            _fetch_10q_filings(ticker, n_10q, cached_dates, verbose)
        return cached

    try:
        cik_map = _load_cik_map()
    except Exception as exc:
        if verbose:
            print(f"  [credibility] CIK map load failed for {ticker}: {exc}")
        return cached

    cik = _resolve_cik(ticker, cik_map)
    if not cik:
        if verbose:
            print(f"  [credibility] No CIK for {ticker}")
        return cached

    try:
        metas = _fetch_n_10k_metas(cik, n=n_filings)
    except Exception as exc:
        if verbose:
            print(f"  [credibility] EDGAR meta error {ticker}: {exc}")
        return cached

    for meta in metas:
        if meta["filed_date"] in cached_dates:
            continue
        if verbose:
            print(f"  [credibility] Fetching {ticker} filing {meta['filed_date']}...")
        text = _fetch_and_strip_doc(cik, meta["accession_no"], meta["document"])
        if text is not None:
            item_1a = _extract_item_1a_from_text(text)
            item_7 = _extract_item_7_from_text(text)
        else:
            item_1a = item_7 = None
        minhash = _minhash(_shingles(item_1a)) if item_1a else None
        _save_filing_to_cache(
            ticker, cik, meta["filed_date"], meta["accession_no"],
            item_1a, minhash, item_7_text=item_7, form_type="10-K",
        )
        cached_dates.add(meta["filed_date"])

    if include_10q:
        _fetch_10q_filings(ticker, n_10q, cached_dates, verbose, cik=cik)

    return _get_cached_filings_n(ticker, n_filings)


def _fetch_10q_filings(
    ticker: str,
    n: int,
    already_cached_dates: set,
    verbose: bool = False,
    cik: Optional[str] = None,
) -> None:
    """Fetch up to n recent 10-Q filings and cache them. Silently skips on errors."""
    if cik is None:
        try:
            cik = _resolve_cik(ticker, _load_cik_map())
        except Exception:
            return
    if not cik:
        return

    # Check how many 10-Q rows we already have in cache
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        existing_q_dates = {
            row["filed_date"]
            for row in conn.execute(
                "SELECT filed_date FROM sec_filings WHERE ticker=? AND form_type='10-Q'",
                (ticker.upper(),),
            ).fetchall()
        }
    finally:
        conn.close()

    if len(existing_q_dates) >= n:
        return

    try:
        metas = _fetch_filing_metas(cik, form_types=("10-Q",), n=n)
    except Exception:
        return

    for meta in metas:
        if meta["filed_date"] in already_cached_dates or meta["filed_date"] in existing_q_dates:
            continue
        if verbose:
            print(f"  [contagion] Fetching {ticker} 10-Q {meta['filed_date']}...")
        try:
            text = _fetch_and_strip_doc(cik, meta["accession_no"], meta["document"])
            if text is not None:
                item_1a = _extract_10q_risk_changes(text)
                item_7 = _extract_10q_mda(text)
            else:
                item_1a = item_7 = None
        except Exception as exc:
            if verbose:
                print(f"  [contagion] 10-Q fetch error {ticker}: {exc}")
            continue
        # No MinHash for 10-Q (Item 1A changes section is too sparse for meaningful similarity)
        _save_filing_to_cache(
            ticker, cik, meta["filed_date"], meta["accession_no"],
            item_1a, None, item_7_text=item_7, form_type="10-Q",
        )
        existing_q_dates.add(meta["filed_date"])


# ── MinHash ────────────────────────────────────────────────────────────────────

def _shingles(text: str, k: int = SHINGLE_SIZE) -> set:
    words = re.findall(r"[a-z]+", text.lower())
    if len(words) < k:
        return set()
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def _minhash(shingle_set: set, n: int = MINHASH_N) -> List[int]:
    if not shingle_set:
        return [2**32 - 1] * n
    shingle_bytes = [" ".join(s).encode("utf-8") for s in shingle_set]
    signature = []
    for i in range(n):
        salt = i.to_bytes(4, "big")
        min_val = 2**32
        for sb in shingle_bytes:
            h = int(hashlib.sha256(salt + sb).hexdigest(), 16) % (2**32)
            if h < min_val:
                min_val = h
        signature.append(min_val)
    return signature


def _jaccard_from_minhash(sig_a: List[int], sig_b: List[int]) -> float:
    if not sig_a or not sig_b or len(sig_a) != len(sig_b):
        return 0.0
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


# ── Per-ticker fetch ───────────────────────────────────────────────────────────

def _resolve_cik(ticker: str, cik_map: Dict[str, str]) -> Optional[str]:
    cik = cik_map.get(ticker)
    if cik:
        return cik
    # Fallback for tickers with dots (e.g. BRK.B → BRK-B or BRKB)
    for variant in [ticker.replace(".", "-"), ticker.replace(".", "")]:
        cik = cik_map.get(variant)
        if cik:
            return cik
    return None


def _fetch_filing_for_ticker(
    ticker: str,
    cik_map: Dict[str, str],
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    ticker = ticker.strip().upper()

    cached = _get_cached_filing(ticker)
    if cached is not None:
        return cached

    cik = _resolve_cik(ticker, cik_map)
    if not cik:
        if verbose:
            print(f"  [contagion] No CIK for {ticker}")
        return None

    try:
        meta = _fetch_latest_10k_meta(cik)
    except Exception as exc:
        if verbose:
            print(f"  [contagion] EDGAR meta error {ticker}: {exc}")
        return None

    if meta is None:
        if verbose:
            print(f"  [contagion] No 10-K found for {ticker}")
        return None

    try:
        # Fetch the document once and extract both Item 1A and Item 7
        doc_text = _fetch_and_strip_doc(cik, meta["accession_no"], meta["document"])
        item_1a = _extract_item_1a_from_text(doc_text) if doc_text else None
        item_7 = _extract_item_7_from_text(doc_text) if doc_text else None
    except Exception as exc:
        if verbose:
            print(f"  [contagion] Filing fetch error {ticker}: {exc}")
        item_1a = item_7 = None

    minhash = _minhash(_shingles(item_1a)) if item_1a else None

    _save_filing_to_cache(
        ticker, cik, meta["filed_date"], meta["accession_no"],
        item_1a, minhash, item_7_text=item_7,
    )

    return {
        "ticker": ticker,
        "cik": cik,
        "filed_date": meta["filed_date"],
        "accession_no": meta["accession_no"],
        "item_1a_text": item_1a,
        "item_7_text": item_7,
        "minhash": minhash,
        "sic": meta.get("sic", ""),
    }


# ── Sector grouping ────────────────────────────────────────────────────────────

def _build_sector_groups(
    tickers: List[str],
    prefetched_sectors: Dict[str, Dict[str, Optional[str]]],
    filing_map: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for ticker in tickers:
        sic = (filing_map.get(ticker) or {}).get("sic") or ""
        if sic:
            key = f"SIC_{sic}"
        else:
            yf_info = prefetched_sectors.get(ticker) or {}
            sector = (yf_info.get("sector") or "").strip()
            key = sector if sector else "__no_sector__"
        groups.setdefault(key, []).append(ticker)
    return groups


# ── Risk cluster detection ─────────────────────────────────────────────────────

def _find_risk_clusters(
    filing_data: List[Dict[str, Any]],
    threshold: float = JACCARD_THRESHOLD,
) -> List[List[str]]:
    valid = [f for f in filing_data if f.get("minhash")]
    n = len(valid)
    if n < 2:
        return [[f["ticker"]] for f in valid]

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard_from_minhash(valid[i]["minhash"], valid[j]["minhash"]) >= threshold:
                union(i, j)

    comp: Dict[int, List[str]] = defaultdict(list)
    for i, f in enumerate(valid):
        comp[find(i)].append(f["ticker"])
    return list(comp.values())


# ── Scoring ────────────────────────────────────────────────────────────────────

def _count_novel_shingles(
    own_minhash: List[int],
    peer_minhashes: List[List[int]],
    bucket_size: int = 500,
) -> int:
    if not peer_minhashes:
        return len(own_minhash) // bucket_size
    novel = sum(
        1 for i, v in enumerate(own_minhash)
        if not any(v == peer[i] for peer in peer_minhashes)
    )
    return novel // bucket_size


def _insufficient_data_row(ticker: str) -> Dict[str, Any]:
    return {
        "Ticker": ticker,
        "Contagion_Leadership_Score": None,
        "Contagion_Disclosure_Rank": None,
        "Contagion_Peer_Count": None,
        "Contagion_Novel_Risks": None,
        "Contagion_Risk_Label": "Insufficient Data",
        "Contagion_Filing_Date": None,
    }


def _score_ticker_in_sector(
    ticker: str,
    sector_tickers: List[str],
    filing_map: Dict[str, Optional[Dict[str, Any]]],
) -> Dict[str, Any]:
    filing = filing_map.get(ticker)
    if filing is None or not filing.get("minhash"):
        return _insufficient_data_row(ticker)

    peer_filings = [
        filing_map[t]
        for t in sector_tickers
        if t != ticker and (filing_map.get(t) or {}).get("minhash")
    ]

    if not peer_filings:
        return {
            "Ticker": ticker,
            "Contagion_Leadership_Score": 50.0,
            "Contagion_Disclosure_Rank": 1,
            "Contagion_Peer_Count": 0,
            "Contagion_Novel_Risks": _count_novel_shingles(filing["minhash"], []),
            "Contagion_Risk_Label": "Isolated",
            "Contagion_Filing_Date": filing["filed_date"],
        }

    all_filings = [filing] + peer_filings
    clusters = _find_risk_clusters(all_filings)
    ticker_clusters = [c for c in clusters if ticker in c and len(c) > 1]
    peer_set = {t for c in ticker_clusters for t in c if t != ticker}

    rank_sum = 0.0
    rank_count = 0
    latest_cluster_rank = 1
    latest_cluster_date = ""

    for cluster in ticker_clusters:
        ordered = sorted(
            cluster,
            key=lambda t: (
                (filing_map.get(t) or {}).get("filed_date", "9999-99-99"),
                t,
            ),
        )
        n = len(ordered)
        rank = ordered.index(ticker) + 1
        rank_sum += 100 * (n - rank + 1) / n
        rank_count += 1

        cluster_date = (filing_map.get(ticker) or {}).get("filed_date", "")
        if cluster_date >= latest_cluster_date:
            latest_cluster_date = cluster_date
            latest_cluster_rank = rank

    leadership_score = rank_sum / rank_count if rank_count > 0 else 0.0

    peer_minhashes = [f["minhash"] for f in peer_filings if f and f.get("minhash")]
    novel_risks = _count_novel_shingles(filing["minhash"], peer_minhashes)

    if not peer_set:
        label = "Isolated"
    elif leadership_score >= 70:
        label = "Leader"
    else:
        label = "Follower"

    return {
        "Ticker": ticker,
        "Contagion_Leadership_Score": round(leadership_score, 1),
        "Contagion_Disclosure_Rank": latest_cluster_rank,
        "Contagion_Peer_Count": len(peer_set),
        "Contagion_Novel_Risks": novel_risks,
        "Contagion_Risk_Label": label,
        "Contagion_Filing_Date": filing["filed_date"],
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def run_contagion_scan(
    candidates_df: pd.DataFrame,
    prefetched: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """
    Run Sector Contagion Tracer on the candidates DataFrame.

    Fetches (or loads from cache) the most recent 10-K Item 1A for each
    ticker via SEC EDGAR, computes MinHash similarity, clusters companies
    by shared risk language within each sector, then assigns leadership scores.

    Returns a DataFrame with Ticker + 6 Contagion_* columns, left-merged to
    preserve the original row order from candidates_df.
    """
    if "Ticker" not in candidates_df.columns:
        raise ValueError("candidates_df must contain a 'Ticker' column")
    if candidates_df.empty:
        return candidates_df[["Ticker"]].copy()

    _ensure_contagion_table()

    prefetched = prefetched or {}
    sectors_map: Dict[str, Dict[str, Optional[str]]] = prefetched.get("sectors") or {}
    tickers = [str(t).strip().upper() for t in candidates_df["Ticker"].dropna().unique()]

    if verbose:
        print(f"  [contagion] Loading CIK map...")
    try:
        cik_map = _load_cik_map()
    except Exception as exc:
        if verbose:
            print(f"  [contagion] Could not load CIK map: {exc}. Returning Insufficient Data.")
        return pd.DataFrame([_insufficient_data_row(t) for t in tickers])

    workers = max(1, min(int(max_workers or 0) or min(8, os.cpu_count() or 4), 8))

    if verbose:
        print(
            f"  [contagion] Fetching/caching 10-K filings for {len(tickers)} tickers "
            f"({workers} workers)..."
        )

    filing_map: Dict[str, Optional[Dict[str, Any]]] = {}
    if workers == 1 or len(tickers) == 1:
        for ticker in tickers:
            try:
                filing_map[ticker] = _fetch_filing_for_ticker(ticker, cik_map, verbose)
            except Exception as exc:
                if verbose:
                    print(f"  [contagion] Fetch failed {ticker}: {exc}")
                filing_map[ticker] = None
    else:
        futures = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for ticker in tickers:
                futures[executor.submit(
                    _fetch_filing_for_ticker, ticker, cik_map, verbose
                )] = ticker
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                filing_map[ticker] = future.result()
            except Exception as exc:
                if verbose:
                    print(f"  [contagion] Fetch failed {ticker}: {exc}")
                filing_map[ticker] = None

    sector_groups = _build_sector_groups(tickers, sectors_map, filing_map)

    results = []
    for sector_key, sector_tickers in sector_groups.items():
        for ticker in sector_tickers:
            try:
                results.append(_score_ticker_in_sector(ticker, sector_tickers, filing_map))
            except Exception as exc:
                if verbose:
                    print(f"  [contagion] Score failed {ticker}: {exc}")
                results.append(_insufficient_data_row(ticker))

    if not results:
        return candidates_df[["Ticker"]].copy()

    result_df = pd.DataFrame(results)
    merged = candidates_df[["Ticker"]].merge(result_df, on="Ticker", how="left")

    if verbose:
        counts = result_df["Contagion_Risk_Label"].value_counts()
        print(
            f"  [contagion] Done: "
            + ", ".join(f"{v} {k}" for k, v in counts.items())
        )

    return merged
