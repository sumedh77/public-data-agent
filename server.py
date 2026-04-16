"""
Public Intelligence Agent
Natural language questions → Parallel AI + Exa + Firecrawl → answer with full source evidence.
No internal data. Public internet only.
"""
import asyncio
import datetime
import json
import math
import re
import time
import sqlite3
import os
from typing import Optional
from typing import AsyncGenerator

import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
EXA_API_KEY       = os.environ.get("EXA_API_KEY", "")
EXA_URL           = os.environ.get("EXA_URL", "https://api.exa.ai/search")
FIRECRAWL_KEY     = os.environ.get("FIRECRAWL_KEY", "")
FIRECRAWL_URL     = os.environ.get("FIRECRAWL_URL", "https://api.firecrawl.dev/v1")
PARALLEL_AI_KEY   = os.environ.get("PARALLEL_AI_KEY", "")
PARALLEL_AI_URL   = os.environ.get("PARALLEL_AI_URL", "https://api.parallel.ai")
# SEC EFTS (free official full-text search) — no API key needed
SEC_EFTS_URL      = os.environ.get("SEC_EFTS_URL", "https://efts.sec.gov/LATEST/search-index")
SEC_USER_AGENT    = os.environ.get("SEC_USER_AGENT", "PublicDataAgent contact@publicdataagent.com")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# ── SQLite Setup for Monitoring & Costs ──────────────────────────────────────
DB_PATH = "metrics.db"
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                query TEXT,
                tools_used TEXT,
                llm_cost REAL,
                exa_cost REAL,
                pai_cost REAL,
                sec_cost REAL,
                fc_cost REAL,
                search_cost REAL,
                total_cost REAL,
                distinct_domains INTEGER,
                num_results INTEGER
            )
        ''')
        # Migrate older databases that may be missing new columns
        existing = {row[1] for row in conn.execute("PRAGMA table_info(queries)")}
        for col, typ in [
            ("exa_cost",          "REAL"),
            ("pai_cost",          "REAL"),
            ("sec_cost",          "REAL"),
            ("fc_cost",           "REAL"),
            ("distinct_domains",  "INTEGER"),
            ("fc_pages_scraped",  "INTEGER"),   # how many pages Firecrawl actually returned content for
            ("pai_extract_cost",  "REAL"),      # PAI Extract fallback cost (separate from fc_cost)
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE queries ADD COLUMN {col} {typ} DEFAULT 0")
init_db()

# Exa pricing (docs: deep-reasoning = $15/1000 requests = $0.015 per call;
#  additional results beyond 10 = $0.001 each; highlights = $0.001 per page)
COST_EXA_BASE   = 0.015   # deep-reasoning per request
COST_EXA_RESULT = 0.001   # per result page for highlights
COST_PAI_BASE = 0.010
COST_PAI_RESULT = 0.0002
COST_SEC_BASE = 0.0      # Free — SEC EFTS API
COST_SEC_RESULT = 0.0    # Free — SEC EFTS API
COST_FC_SUCCESS = 0.002
# Claude 3 Haiku pricing — used for fast/cheap calls (routing, intent, follow-ups)
COST_HAIKU_IN_PER_1M  = 0.25
COST_HAIKU_OUT_PER_1M = 1.25
# Claude 3.5 Sonnet pricing — used for synthesis and signal extraction (quality matters)
COST_SONNET_IN_PER_1M  = 3.00
COST_SONNET_OUT_PER_1M = 15.00

# ── Model preference order (best → fallback) ──────────────────────────────────
# The agent tries the best model first. If the API key doesn't have access
# (404 not_found_error), it automatically retries with the fallback model.
_QUALITY_MODELS = [
    ("claude-3-5-sonnet-20241022", COST_SONNET_IN_PER_1M, COST_SONNET_OUT_PER_1M),
    ("claude-3-5-haiku-20241022",  0.80,  4.00),   # newer Haiku — often accessible
    ("claude-3-haiku-20240307",    COST_HAIKU_IN_PER_1M, COST_HAIKU_OUT_PER_1M),
]

async def _anthropic_call_with_fallback(
    client: httpx.AsyncClient,
    *,
    system: str,
    user_content: str,
    max_tokens: int,
    timeout: float = 90,
    extra_headers: dict | None = None,
) -> tuple[str, float]:
    """
    Try _QUALITY_MODELS in order. Returns (response_text, cost).
    Falls back to the next model on 404 (model not found) or 529 (overloaded).
    Raises on any other HTTP error.
    """
    headers_base = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    if extra_headers:
        headers_base.update(extra_headers)

    last_err: Exception | None = None
    for model, cost_in, cost_out in _QUALITY_MODELS:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers_base,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": user_content}],
                },
                timeout=timeout,
            )
            if resp.status_code in (404, 529):
                print(f"Model {model} returned {resp.status_code}, trying next…")
                last_err = httpx.HTTPStatusError(resp.text, request=resp.request, response=resp)
                continue
            resp.raise_for_status()
            data    = resp.json()
            text    = data.get("content", [{}])[0].get("text", "")
            usage   = data.get("usage", {})
            cost    = (usage.get("input_tokens", 0) / 1_000_000) * cost_in \
                    + (usage.get("output_tokens", 0) / 1_000_000) * cost_out
            print(f"LLM call succeeded with model: {model}")
            return text, cost
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (404, 529):
                print(f"Model {model} not accessible ({e.response.status_code}), trying next…")
                last_err = e
                continue
            raise
    raise RuntimeError(f"All models failed. Last error: {last_err}")

def log_query_metrics(query: str, tools_used: dict, llm_cost: float,
                      exa_cost: float, pai_cost: float, sec_cost: float, fc_cost: float,
                      distinct_domains: int, num_results: int,
                      fc_pages_scraped: int = 0, pai_extract_cost: float = 0.0):
    try:
        search_cost = exa_cost + pai_cost + sec_cost + fc_cost + pai_extract_cost
        total_cost = llm_cost + search_cost
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute('''
                INSERT INTO queries (
                    query, tools_used, llm_cost, exa_cost, pai_cost, sec_cost,
                    fc_cost, search_cost, total_cost, distinct_domains, num_results,
                    fc_pages_scraped, pai_extract_cost
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                query, json.dumps(tools_used), llm_cost,
                exa_cost, pai_cost, sec_cost, fc_cost,
                search_cost, total_cost, distinct_domains, num_results,
                fc_pages_scraped, pai_extract_cost,
            ))
    except Exception as e:
        print("Failed to log metrics:", e)

# ── Domains to DROP entirely — genuinely zero signal content ─────────────────
# linkedin.com intentionally NOT excluded: LinkedIn posts (user testimonials,
# thought leadership, product mentions) are high-quality B2B signals.
# Exa indexes them and returns highlights directly — Firecrawl will fail on
# login-gated URLs but the snippet evidence is already sufficient.
_MEDIA_DOMAINS = {
    "twitter.com", "x.com",
    "instagram.com", "tiktok.com", "pinterest.com",
    "hackernews.com", "news.ycombinator.com",
    # NOTE: medium.com, substack.com, linkedin.com intentionally NOT excluded
}

# ── Platform-specific domain sets for targeted queries ───────────────────────
_PLATFORM_DOMAINS: dict[str, list[str]] = {
    "linkedin":  ["linkedin.com"],
    "reddit":    ["reddit.com"],
    "github":    ["github.com"],
    "twitter":   ["twitter.com", "x.com"],
    "producthunt": ["producthunt.com"],
    "hackernews": ["news.ycombinator.com"],
    "medium":    ["medium.com"],
    "substack":  ["substack.com"],
}

# ── Domains where Firecrawl will fail (login-gated) — use snippet only ────────
_SNIPPET_ONLY_DOMAINS = {"linkedin.com", "twitter.com", "x.com", "reddit.com"}

# ── Press-wire / news-wire / aggregator domains ──────────────────────────────
# Articles from these domains are ABOUT a company, not FROM the company.
# The company name must be extracted from the article content, not the URL.
_NEWS_WIRE_DOMAINS = {
    "prnewswire.com", "businesswire.com", "globenewswire.com",
    "accesswire.com", "prnews.io", "einpresswire.com",
    "prweb.com", "newswire.com", "apnews.com", "marketwatch.com",
    "finance.yahoo.com", "investing.com", "seekingalpha.com",
    "businessinsider.com", "entrepreneur.com", "inc.com",
    # Aggregators / directories — the subject is inside the article
    "stocktitan.net", "ycombinator.com", "crunchbase.com",
    "tracxn.com", "pitchbook.com", "cbinsights.com",
    "wellfound.com", "angel.co", "f6s.com",
    "techcrunch.com", "venturebeat.com", "sifted.eu",
    "wired.com", "zdnet.com", "theverge.com",
    "forbes.com", "bloomberg.com", "reuters.com", "cnbc.com",
    "ft.com", "infocapital.com", "theregister.com",
    # Additional news / media outlets
    "thenextweb.com", "infoq.com", "devops.com", "sdtimes.com",
    "helpnetsecurity.com", "darkreading.com", "securityweek.com",
    "cio.com", "computerworld.com", "itpro.co.uk", "techtarget.com",
    "informationweek.com", "networkworld.com", "eweek.com",
    "readwrite.com", "siliconangle.com", "betakit.com",
    "hbr.org", "mckinsey.com", "gartner.com",
    "morningstar.com", "barrons.com", "wsj.com", "nytimes.com",
    "washingtonpost.com", "theguardian.com", "bbc.co.uk", "bbc.com",
    "news.google.com", "msn.com", "yahoo.com",
}

# ── Entity resolution ─────────────────────────────────────────────────────────
_STRIP_SUFFIXES = re.compile(
    r'\s+(inc|llc|corp|ltd|co|group|technologies|solutions|systems|platform|ai)\.?$',
    re.IGNORECASE,
)

def _normalize_entity(name: str) -> str:
    return _STRIP_SUFFIXES.sub('', name.lower().strip()).strip()

class EntityResolver:
    def __init__(self):
        self._domain_map: dict[str, str] = {}
        self._name_map:   dict[str, str] = {}

    def register(self, canonical: str, names: list[str] = [], domains: list[str] = []):
        for d in domains: self._domain_map[d.lower()] = canonical
        for n in names:   self._name_map[_normalize_entity(n)] = canonical

    def resolve(self, name: Optional[str], domain: Optional[str]) -> Optional[str]:
        if domain and domain.lower() in self._domain_map:
            return self._domain_map[domain.lower()]
        if name:
            hit = self._name_map.get(_normalize_entity(name))
            if hit: return hit
        return domain

    def learn(self, canonical: str, name: str):
        if canonical and name:
            self._name_map[_normalize_entity(name)] = canonical

entity_resolver = EntityResolver()
entity_resolver.register("openai.com",     names=["OpenAI", "Open AI"])
entity_resolver.register("anthropic.com",  names=["Anthropic", "Anthropic AI"])
entity_resolver.register("langchain.com",  names=["LangChain", "LangChain Inc"])
entity_resolver.register("notion.so",      names=["Notion", "Notion Labs"])
entity_resolver.register("vercel.com",     names=["Vercel"])
entity_resolver.register("linear.app",     names=["Linear"])
entity_resolver.register("huggingface.co", names=["Hugging Face", "HuggingFace"])
entity_resolver.register("databricks.com", names=["Databricks"])
entity_resolver.register("mistral.ai",     names=["Mistral", "Mistral AI"])
entity_resolver.register("cohere.com",     names=["Cohere"])

# ── Utilities ─────────────────────────────────────────────────────────────────

def _resolve_company_domain(company_name: str, url_domain: str) -> str:
    """
    Determine the true company domain from a (company_name, url_domain) pair.

    Rule: if the company slug matches the root of the URL domain, the URL IS
    the company's own site → keep it.  Otherwise the URL is a 3rd-party
    media/news site reporting *about* the company → infer domain from name.

    Examples:
      ("Accoil",      "accoil.com")         → "accoil.com"       (own site)
      ("Lightfield",  "martechseries.com")  → "lightfield.com"   (media site)
      ("Lightfield",  "globenewswire.com")  → "lightfield.com"   (press wire)
      ("Gong",        "gong.io")            → "gong.io"          (own site)
      ("ChurnZero",   "churnzero.com")      → "churnzero.com"    (own site)
    """
    if not company_name:
        return url_domain
    slug = re.sub(r'[^a-z0-9]', '', company_name.lower())
    if not slug:
        return url_domain
    # Root domain component: "martechseries" from "martechseries.com"
    domain_root = url_domain.split('.')[0]
    # Company owns the domain when slug and domain_root are the same word
    # (allow prefix overlap only when both are ≥ 4 chars to avoid false matches)
    min_len = max(4, len(slug) - 2)
    if (domain_root == slug
            or (len(slug) >= min_len and domain_root.startswith(slug))
            or (len(domain_root) >= min_len and slug.startswith(domain_root))):
        return url_domain
    # 3rd-party reporting site → infer the company's own domain
    inferred = infer_company_domain(company_name)
    return inferred if inferred else url_domain

# Subdomains that are hosting prefixes, not the company name
_HOSTING_SUBDOMAINS = re.compile(
    r'^(?:www|blog|blogs|news|press|media|insights|resources|learn|'
    r'academy|docs|help|support|app|go|get|try|community|forum|'
    r'engineering|tech|developers|dev|api|status|about)\.'
)

def extract_domain_from_url(url: str) -> str:
    """Return the root company domain, stripping hosting subdomains like blog.oliv.ai → oliv.ai."""
    m = re.search(r'https?://([^/]+)', url)
    if not m:
        return ""
    host = m.group(1).lower()
    # Always strip www.
    host = re.sub(r'^www\.', '', host)
    # Strip hosting subdomains (blog., engineering., etc.) — only if what remains still has a dot
    stripped = _HOSTING_SUBDOMAINS.sub('', host)
    return stripped if '.' in stripped else host

def extract_company_from_title(title: str) -> str:
    """
    Extract the subject company name from a news headline.

    Handles patterns:
      - "Gong Raises $250M…"                 (company at start + verb)
      - "onPhase Appoints Heather as CCO"     (appointment verb at start)
      - "Emeka Iheme Joins Finys as CCSO"     (Person Joins Company)
      - "Industry first by ChurnZero gives…" (by Company pattern)
    """
    # Pattern 1: Company-first headlines — wide verb list including appointments.
    # Lowercase first char allowed (e.g. onPhase, eBay).
    # (?!\w) after alternation prevents "Names?" matching "Named" as a prefix.
    m = re.match(
        r'^([A-Za-z][A-Za-z0-9][A-Za-z0-9\s\.\-]{0,28}?)'
        r'(?:\s+(?:Raises?|Announces?|Closes?|Secures?|Launches?|Lands?|Gets?'
        r'|Completes?|Unveils?|Expands?|Names?|Partners?|Publishes?|Releases?'
        r'|Shares?|Introduces?|Appoints?|Hires?|Promotes?|Welcomes?|Selects?'
        r'|Acquires?|Merges?|Reports?|Achieves?|Wins?|Receives?|Integrates?'
        r'|Deploys?|Adopts?|Migrates?|Signs?|Earns?|Surpasses?|Hits?|Reaches?'
        r'|Grows?|Cuts?|Opens?|Adds?|Builds?|Creates?|Delivers?'
        r'|Doubles?|Triples?|Backs?|Funds?|Invests?'
        r'|Strengthens?|Enhances?|Improves?|Simplifies?|Transforms?)(?!\w)'
        r'|:|\s+\$|\s+has\s|\s+is\s|\s+will\s)',
        title,
    )
    if m:
        cand = m.group(1).strip()
        # Guard: reject if it looks like a person name (two Title-case words) + verb is "Named/Appointed/Hired"
        words = cand.split()
        if len(words) == 2 and all(w[0].isupper() for w in words):
            # Two-word title-case → likely a person; skip Pattern 1, try others
            pass
        else:
            return cand

    # Pattern 2: "Person Joins/Appointed-to Company as Role"
    # Only triggers on verbs that explicitly introduce the company (joins/appointed to/hired at)
    m2 = re.search(
        r'\b(?:joins?\s+|appointed\s+(?:at|to)\s+|hired\s+(?:at|by)\s+|named\s+to\s+|promoted\s+to\s+)'
        r'([A-Z][A-Za-z][A-Za-z0-9\s\.\-]{0,25}?)'
        r'(?:\s+as\s|\s+to\s|\s*,|\s*$)',
        title, re.IGNORECASE,
    )
    if m2:
        cand = m2.group(1).strip()
        # skip role/generic words
        skip = {'the', 'a', 'an', 'its', 'their', 'chief', 'new', 'head', 'vice',
                'senior', 'president', 'director', 'officer', 'manager'}
        first_word = cand.split()[0].lower() if cand else ''
        if first_word not in skip and len(cand) > 2:
            return cand

    # Pattern 3: "… by Company …" — e.g. "Industry first by ChurnZero gives…"
    m3 = re.search(
        r'\bby\s+([A-Z][A-Za-z0-9][A-Za-z0-9\s\.\-]{0,25}?)'
        r'(?=\s+(?:gives?|launches?|announces?|unveils?|releases?|introduces?|adds?|enables?))',
        title, re.IGNORECASE,
    )
    if m3:
        return m3.group(1).strip()

    return ""


def infer_company_domain(company_name: str) -> str:
    """
    Infer the most likely domain from a company name.
    "Gong" → "gong.com", "ChurnZero" → "churnzero.com"
    Strips common legal suffixes before appending .com.
    """
    if not company_name:
        return ""
    # Lowercase, keep only alphanumeric, drop spaces
    slug = re.sub(r'[^a-z0-9]', '', company_name.lower())
    # Strip common suffixes (only if stripping leaves ≥3 chars)
    for sfx in ('technologies', 'technology', 'software', 'solutions',
                'systems', 'platform', 'group', 'inc', 'llc', 'corp',
                'ltd', 'labs', 'ai', 'io', 'co'):
        if slug.endswith(sfx) and len(slug) - len(sfx) >= 3:
            slug = slug[:-len(sfx)]
            break
    return (slug + ".com") if slug else ""

def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"

# ── Stop words for keyword extraction ─────────────────────────────────────────
_STOP_WORDS = {
    'how', 'many', 'have', 'recently', 'are', 'the', 'a', 'an', 'is', 'was',
    'were', 'that', 'which', 'what', 'who', 'when', 'where', 'why', 'and',
    'or', 'not', 'about', 'in', 'on', 'to', 'for', 'of', 'with', 'do', 'did',
    'has', 'had', 'been', 'be', 'by', 'from', 'at', 'this', 'their', 'its',
    'can', 'could', 'would', 'should', 'will', 'any', 'some', 'all',
}

# Extra words that describe search INTENT but are not topic keywords.
# Stripping these gives the actual subject the user wants to find.
_INTENT_WORDS = {
    'find', 'finding', 'show', 'list', 'give', 'get', 'want', 'need',
    'companies', 'company', 'startups', 'startup', 'businesses', 'business',
    'published', 'publishing', 'publish', 'wrote', 'write', 'written', 'writing',
    'blog', 'blogs', 'post', 'posts', 'article', 'articles', 'content',
    'announcement', 'announcements', 'changelog', 'changelogs',
    'saas', 'b2b', 'enterprise', 'series', 'stage', 'based',
    'entire', 'full', 'complete', 'all', 'every', 'possible',
    'days', 'months', 'weeks', 'year', 'years', 'last', 'recent', 'latest',
    'new', 'old', 'past', '180', '90', '60', '30', '365',
    'please', 'can', 'could', 'would', 'should',
}

# ── Query Intent Parser ───────────────────────────────────────────────────────
# Detects requested result count and entity type from the user's question.
# Drives numResults on Exa and max_results on Parallel AI, plus startup filtering.

_STARTUP_SIGNALS = {
    "startup", "startups", "seed", "early-stage", "early stage",
    "pre-seed", "series a", "series b", "founded", "new company",
    "young company", "small company", "bootstrapped", "vc-backed",
}
_ENTERPRISE_SIGNALS = {
    "enterprise", "fortune 500", "large company", "big company",
    "publicly traded", "public company", "corporation", "conglomerate",
}

# Regex that matches startup signals inside scraped page content
_STARTUP_CONTENT_RE = re.compile(
    r'\b(seed\s+round|series\s+[abc]|early[- ]stage|founded\s+in\s+20\d\d|'
    r'we\s+raised|pre[- ]seed|angel\s+round|y\s*combinator|'
    r'\d+\s+employees?|small\s+team|our\s+team\s+of\s+\d+)\b',
    re.IGNORECASE,
)

def _parse_query_intent(question: str) -> dict:
    """
    Extract from the user's question:
      - requested_count  : int       — explicit number of results (0 = unspecified)
      - entity_type      : str       — "startup" | "enterprise" | "company"
      - startup_filter   : bool      — True when post-scrape startup signal check needed
      - target_platform  : str|None  — "linkedin" | "reddit" | "github" | etc.
      - include_domains  : list[str] — domain allowlist to pass to Exa includeDomains
    """
    q_lower = question.lower()

    # Count: "300 companies", "50 startups", "10 posts", "find 100 results"
    count_m = re.search(
        r'\b(\d{1,4})\s*(?:companies|startups?|results?|firms?|organizations?|posts?|articles?|examples?)\b',
        q_lower,
    )
    if not count_m:
        count_m = re.search(r'\bfind\s+(\d{1,4})\b', q_lower)
    requested_count = int(count_m.group(1)) if count_m else 0

    is_startup    = any(sig in q_lower for sig in _STARTUP_SIGNALS)
    is_enterprise = any(sig in q_lower for sig in _ENTERPRISE_SIGNALS)
    entity_type   = "startup" if is_startup else ("enterprise" if is_enterprise else "company")

    # Platform detection — "linkedin posts", "reddit threads", "github issues"
    target_platform: str | None = None
    include_domains: list[str]  = []
    for platform, domains in _PLATFORM_DOMAINS.items():
        if platform in q_lower:
            target_platform = platform
            include_domains = domains
            break

    return {
        "requested_count":  requested_count,
        "entity_type":      entity_type,
        "startup_filter":   is_startup,
        "target_platform":  target_platform,
        "include_domains":  include_domains,
    }

def _is_startup_content(content: str) -> bool:
    """Return True if scraped page contains at least one startup signal."""
    return bool(_STARTUP_CONTENT_RE.search(content[:3000]))


def _extract_keywords(text: str) -> list[str]:
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words if w not in _STOP_WORDS and len(w) > 2]

def _extract_topic_keywords(text: str) -> list[str]:
    """Like _extract_keywords but also strips intent/meta words, leaving only topic signal."""
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words
            if w not in _STOP_WORDS
            and w not in _INTENT_WORDS
            and len(w) > 2
            and not w.isdigit()]

# ── Step 1: Derive search queries ─────────────────────────────────────────────
def derive_search_queries(question: str, entity_type: str = "company") -> list[str]:
    """
    Generate 10 targeted search queries from the question.

    Key insight: separate INTENT words ("find companies that published blogs about")
    from TOPIC words ("CRM migration tech stack overhaul").  Then build:
      - First-person company-blog queries ("we migrated from CRM" "how we switched")
      - Topic + year queries for recency
      - Neural/semantic queries for Exa
      - Specific tool-name queries when detected
    """
    q = question.replace("?", "").strip()

    # Full keyword set (for evidence scoring — keep everything)
    all_kws   = _extract_keywords(q)
    # Topic-only keywords (for building search queries — strips "find companies published blog")
    topic_kws = _extract_topic_keywords(q)

    # Top 5 topic words as the core search phrase
    core = " ".join(topic_kws[:5]) if topic_kws else " ".join(all_kws[:5])

    # Detect year hints to append
    cur_year = datetime.date.today().year
    years    = f"{cur_year} {cur_year - 1}"

    queries: list[str] = []

    # 1. Raw question — best for neural search engines
    queries.append(q)

    # 2. Core topic + recent years — clean keyword search
    queries.append(f"{core} blog post {years}")

    # 3. First-person narrative — how a company writes about its OWN experience
    queries.append(f'"we {topic_kws[0]}" company engineering blog {cur_year}' if topic_kws else core)

    # 4. "How we" story pattern — very common in company engineering/growth blogs
    queries.append(f'"how we" {core} blog {cur_year}')

    # 5. Lessons learned / retrospective pattern
    queries.append(f'{core} "lessons learned" OR "how we did it" blog {cur_year}')

    # 6. Announcement / launch pattern
    queries.append(f'{core} announcement launch company blog {years}')

    # 7. Engineering blog pattern — targets engineering.company.com style
    queries.append(f'{core} engineering blog post {years}')

    # 8. Case study / story pattern
    queries.append(f'{core} case study company story {years}')

    # 9. Condensed 3-word core — broader catch-all
    short_core = " ".join(topic_kws[:3]) if len(topic_kws) >= 3 else core
    queries.append(f'{short_core} startup company {cur_year}')

    # 10. Changelog / product update pattern
    queries.append(f'{core} changelog product update {years}')

    # 11. Technical migration / switch pattern
    queries.append(f'{core} "technical migration" OR "migrated to" OR "switched to" {cur_year}')

    # 12. Partnership / Integration pattern
    queries.append(f'{core} partnership OR integrated with OR integration {cur_year}')

    # 13. Executive / Shareholder news
    queries.append(f'{core} "executive summary" OR "shareholder letter" OR "investor relations" {cur_year}')

    # 14. New hire / Team expansion (LinkedIn style)
    queries.append(f'{core} "welcoming" OR "new hire" OR "joined the team" {cur_year}')

    # 15. Comparison / Alternative pattern
    queries.append(f'{core} "alternative to" OR "comparison" OR "vs" blog {cur_year}')

    # ── Entity-type query injection ──────────────────────────────────────────────
    # Prepend highly targeted entity-specific queries so they rank first (Exa
    # and PAI both receive queries in order — top queries carry more weight).
    if entity_type == "startup":
        queries = [
            f'{core} startup seed early-stage company {cur_year}',
            f'{core} YC "series A" OR "seed round" OR "pre-seed" {cur_year}',
            f'"we are a startup" {core} blog {cur_year}',
        ] + queries
    elif entity_type == "enterprise":
        queries = [
            f'{core} enterprise "Fortune 500" OR "large company" OR "publicly traded" {cur_year}',
            f'{core} corporation "investor relations" OR "annual report" {cur_year}',
        ] + queries

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique = []
    for qq in queries:
        if qq.strip() and qq not in seen:
            seen.add(qq)
            unique.append(qq)

    return unique[:15]

# ── Step 2a: Parallel AI Search (agentic mode) ────────────────────────────────
async def search_parallel_ai(
    client: httpx.AsyncClient,
    queries: list[str],
    requested_count: int = 0,
) -> list[dict]:
    """
    Single agentic-mode call to Parallel AI Search.
    Sends the top 5 most targeted queries as search_queries[] — agentic mode
    reasons across all of them and returns richer, deduplicated results.

    Field mapping fix (PAI actual response):
      publish_date  → date
      excerpts[]    → snippet  (array, not a string)
    """
    if not queries:
        return []

    # Top 5 most targeted queries for agentic mode
    top_queries = queries[:5]

    # Scale max_results to what the user asked for (cap at 100)
    max_results = min(max(requested_count, 20), 100) if requested_count > 0 else 20

    # The raw question (first query) is the objective
    objective = queries[0]

    try:
        resp = await client.post(
            f"{PARALLEL_AI_URL}/v1beta/search",
            headers={"x-api-key": PARALLEL_AI_KEY, "Content-Type": "application/json"},
            json={
                "mode":           "agentic",
                "search_queries": top_queries,
                "max_results":    max_results,
                "objective":      objective,
                "excerpts":       {"max_chars_per_result": 2000},
            },
            timeout=90,   # agentic mode is deeper — needs more time than fast mode
        )
        resp.raise_for_status()
        out = []
        for r in resp.json().get("results", []):
            url   = r.get("url", "")
            title = r.get("title", "") or r.get("name", "")
            src_domain = extract_domain_from_url(url)
            co_name    = extract_company_from_title(title)

            # PAI returns excerpts as an array — join first 2 for richer snippet
            excerpts = r.get("excerpts") or []
            snippet  = " … ".join(excerpts[:2]) if excerpts else (
                r.get("excerpt") or r.get("snippet") or ""
            )

            out.append({
                "url":                url,
                "domain":             src_domain,
                "news_source_domain": src_domain,
                "title":              title,
                # PAI returns publish_date, with date/publishedDate as fallbacks
                "date":               (r.get("publish_date") or r.get("date") or r.get("publishedDate") or "")[:10],
                "snippet":            snippet[:500],
                "company_name":       co_name,
                "company_domain":     _resolve_company_domain(co_name, src_domain),
                "source":             "parallel_ai",
            })
        return out
    except Exception as e:
        print(f"PAI agentic search error: {e}")
        return []

# ── Step 2b: Exa Search (deep-reasoning model) ───────────────────────────────
# Domains excluded globally — no signal value, just noise
_EXA_EXCLUDE_DOMAINS = [
    "youtube.com", "instagram.com", "facebook.com", "tiktok.com",
    "twitter.com", "x.com", "pinterest.com", "reddit.com",
]

async def search_exa(
    client: httpx.AsyncClient,
    queries: list[str],
    max_days: int = 180,
    requested_count: int = 0,
    include_domains: list[str] | None = None,
) -> list[dict]:
    """
    Exa deep-reasoning search using additionalQueries to bundle all queries into
    ONE API call (instead of N parallel calls) — reduces cost by ~75%.

    Per Exa docs (March 2026):
    - additionalQueries: pass all query variations alongside the main query for
      more comprehensive coverage in a single request.
    - deep-reasoning: $0.015 per request (vs $0.005 old estimate)
    - outputSchema: used ONLY for structured JSON extraction — NOT passed here
      since we want individual result pages with highlights, not synthesized output.
    - maxCharacters: 1500 highlights per result for richer signal extraction.
    - startPublishedDate / endPublishedDate: still the correct date-filter params.

    For large requested_count (> 100): run a second pass with queries[:3] to
    collect additional coverage (Exa caps at 100 per call).
    """
    if not queries:
        return []

    EXA_MAX_PER_CALL = 100
    today      = datetime.date.today()
    start_date = (today - datetime.timedelta(days=max_days)).isoformat()
    end_date   = today.isoformat()

    num_results = min(max(requested_count, 40), EXA_MAX_PER_CALL) if requested_count > 0 else EXA_MAX_PER_CALL

    def _parse_results(raw: list[dict]) -> list[dict]:
        out = []
        for r in raw:
            url        = r.get("url", "")
            title      = r.get("title", "")
            highlights = r.get("highlights") or []
            rich_snippet = " … ".join(h for h in highlights if h)[:1000]
            src_domain = extract_domain_from_url(url)
            co_name    = extract_company_from_title(title)
            out.append({
                "url":                url,
                "domain":             src_domain,
                "news_source_domain": src_domain,
                "title":              title,
                "date":               (r.get("publishedDate") or "")[:10],
                "snippet":            rich_snippet,
                "company_name":       co_name,
                "company_domain":     _resolve_company_domain(co_name, src_domain),
                "source":             "exa",
                "_highlights":        highlights,
            })
        return out

    async def _call(primary_query: str, additional: list[str], inc_domains: list[str] | None = None) -> list[dict]:
        payload: dict = {
            "query":              primary_query,
            "type":               "deep-reasoning",
            "numResults":         num_results,
            "startPublishedDate": start_date + "T00:00:00.000Z",
            "endPublishedDate":   end_date   + "T23:59:59.999Z",
            "contents": {
                "highlights": {
                    "maxCharacters": 1500,
                    "query": primary_query,
                }
            },
        }
        # Platform-specific: use includeDomains (can't combine with excludeDomains)
        if inc_domains:
            payload["includeDomains"] = inc_domains
        else:
            payload["excludeDomains"] = _EXA_EXCLUDE_DOMAINS

        # additionalQueries bundles all remaining queries into the same call
        if additional:
            payload["additionalQueries"] = additional

        try:
            resp = await client.post(
                EXA_URL,
                headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
                json=payload,
                timeout=90,  # deep-reasoning: 10–60s per docs
            )
            resp.raise_for_status()
            return _parse_results(resp.json().get("results", []))
        except Exception as e:
            print(f"Exa deep-reasoning error: {e}")
            return []

    # Pass ALL queries in ONE call using additionalQueries
    # This replaces N parallel calls → single request, ~75% cost reduction
    # include_domains (e.g. ["linkedin.com"]) is forwarded to Exa's includeDomains
    tasks = [_call(queries[0], queries[1:], inc_domains=include_domains or [])]

    # Second pass for large counts (> 100 requested)
    if requested_count > EXA_MAX_PER_CALL and len(queries) >= 3:
        tasks.append(_call(queries[1], queries[2:] + queries[:1], inc_domains=include_domains or []))

    batches = await asyncio.gather(*tasks)
    flat    = [r for b in batches for r in b]
    seen: set[str] = set()
    return [r for r in flat if r["url"] not in seen and not seen.add(r["url"])]  # type: ignore[func-returns-value]

# ── SEC Query Builder ─────────────────────────────────────────────────────────
# SEC EFTS is full-text search over filing documents. Blog-style phrases like
# "companies that published engineering blog posts" will NEVER match SEC filings.
# This function extracts financial/corporate keywords from the user's question.
_SEC_FINANCIAL_TERMS = {
    "acquisition", "merger", "ipo", "s-1", "10-k", "8-k", "10-q", "form d",
    "revenue", "earnings", "quarterly", "annual", "fiscal", "filing", "sec",
    "nasdaq", "nyse", "public offering", "shares", "stock", "equity",
    "fundraising", "series a", "series b", "series c", "venture", "funding",
    "bankruptcy", "chapter 11", "restructuring", "spinoff", "divestiture",
    "cybersecurity incident", "breach", "material event", "material weakness",
    "executive compensation", "proxy statement", "def 14a",
}

_SEC_FORM_TO_TERMS = {
    "8-K":  ["material event", "current report"],
    "S-1":  ["initial public offering", "IPO", "registration statement"],
    "10-K": ["annual report", "fiscal year"],
    "10-Q": ["quarterly report", "quarterly earnings"],
}

def _build_sec_queries(question: str, forms: str) -> list[str]:
    """
    Build 3 SEC-appropriate search queries from the user's question.
    Strips intent/blog framing, extracts financial signal words.
    Never wraps in quotes — SEC EFTS is keyword search, not exact phrase.
    """
    q_lower = question.lower()

    # 1. Extract any company/ticker mentions (simple heuristic: Title-case tokens)
    company_tokens = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b', question)
    # filter out common non-company title-case words
    skip = {'Find', 'Which', 'What', 'Show', 'List', 'Give', 'Get', 'Companies',
            'Company', 'SaaS', 'How', 'Who', 'Where', 'When', 'The', 'Last', 'Recent'}
    company_tokens = [t for t in company_tokens if t not in skip][:2]

    # 2. Extract matching financial terms
    matched_terms = [t for t in _SEC_FINANCIAL_TERMS if t in q_lower]

    # 3. Extract 2-4 core topic words (non-stopword, non-intent)
    topic_words = _extract_topic_keywords(question)[:4]

    queries: list[str] = []

    # Query 1: Company names + financial terms if detected
    if company_tokens and matched_terms:
        queries.append(f"{' '.join(company_tokens)} {matched_terms[0]}")
    elif matched_terms:
        queries.append(" ".join(matched_terms[:3]))
    else:
        queries.append(" ".join(topic_words[:3]))

    # Query 2: Form-type specific language
    form_list = [f.strip() for f in forms.split(",")]
    form_terms = []
    for f in form_list:
        form_terms.extend(_SEC_FORM_TO_TERMS.get(f, []))
    if form_terms:
        core = " ".join(topic_words[:2]) if topic_words else " ".join(matched_terms[:2])
        queries.append(f"{core} {form_terms[0]}" if core else form_terms[0])

    # Query 3: Pure topic keywords — broadest fallback
    if topic_words:
        queries.append(" ".join(topic_words[:4]))

    # Deduplicate and cap at 3 (SEC rate-limits at 10 req/s; we respect it)
    seen: set[str] = set()
    unique = []
    for qq in queries:
        qq = qq.strip()
        if qq and qq not in seen:
            seen.add(qq)
            unique.append(qq)
    return unique[:3]


# ── Step 2d: SEC EFTS Full-Text Search (FREE official API) ────────────────────
# Uses the SEC's own EFTS (Electronic Full-Text Search) at efts.sec.gov.
# Surfaces 8-K material events, S-1 IPO filings, 10-K annual reports, etc.
# No API key required. Rate limit: 10 req/sec. Must send User-Agent header.
async def search_sec_edgar(
    client: httpx.AsyncClient,
    queries: list[str],
    forms: str = "8-K,S-1,10-K,10-Q",
    max_days: int = 365,
) -> list[dict]:
    """
    Search SEC filings full-text via the free EFTS API (efts.sec.gov).
    Returns URLs pointing directly to the filing documents on sec.gov.
    """
    start_date = (datetime.date.today() - datetime.timedelta(days=max_days)).isoformat()
    end_date = datetime.date.today().isoformat()
    form_types_list = [f.strip() for f in forms.split(",")]

    def _parse_display_name(display_name: str) -> tuple[str, str, str]:
        """Parse 'IRON MOUNTAIN INC  (IRM)  (CIK 0001020569)' → (name, ticker, cik)."""
        # Extract CIK
        cik_m = re.search(r'\(CIK\s+(\d+)\)', display_name)
        cik = cik_m.group(1) if cik_m else ""
        # Remove CIK part
        without_cik = re.sub(r'\s*\(CIK\s+\d+\)', '', display_name).strip()
        # Extract ticker(s)
        ticker_m = re.search(r'\(([A-Z0-9,\s\-]+)\)\s*$', without_cik)
        ticker = ticker_m.group(1).split(",")[0].strip() if ticker_m else ""
        # Company name is everything before the ticker parens
        co_name = re.sub(r'\s*\([A-Z0-9,\s\-]+\)\s*$', '', without_cik).strip()
        # Title-case the all-caps SEC name for readability
        if co_name == co_name.upper() and len(co_name) > 3:
            co_name = co_name.title()
        return co_name, ticker, cik

    def _build_filing_url(cik: str, adsh: str, doc_id: str) -> str:
        """Construct the direct filing URL from EFTS metadata."""
        # doc_id format: "0001020569-25-000040:irm-20241231.htm"
        filename = doc_id.split(":", 1)[1] if ":" in doc_id else ""
        if not filename or not cik or not adsh:
            return ""
        adsh_nodash = adsh.replace("-", "")
        # Strip leading zeros from CIK for the URL path
        cik_stripped = cik.lstrip("0") or "0"
        return f"https://www.sec.gov/Archives/edgar/data/{cik_stripped}/{adsh_nodash}/{filename}"

    async def _one(q: str) -> list[dict]:
        try:
            # NOTE: No quote-wrapping — SEC EFTS is keyword search over filings.
            # Exact phrase matching ("...") is far too strict for financial documents.
            params = {
                "q": q,
                "forms": ",".join(form_types_list),
                "dateRange": "custom",
                "startdt": start_date,
                "enddt": end_date,
            }
            resp = await client.get(
                SEC_EFTS_URL,
                params=params,
                headers={"User-Agent": SEC_USER_AGENT},
                timeout=20,
            )
            resp.raise_for_status()
            out = []

            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])[:8]
            for hit in hits:
                doc_id = hit.get("_id", "")
                src = hit.get("_source", {})

                # Parse company info from display_names
                display_names = src.get("display_names", [])
                raw_display = display_names[0] if display_names else ""
                co_name, ticker, cik = _parse_display_name(raw_display)
                if not cik:
                    ciks = src.get("ciks", [])
                    cik = ciks[0] if ciks else ""

                adsh = src.get("adsh", "")
                form_type = src.get("form", "filing")
                file_date = src.get("file_date", "")
                file_desc = src.get("file_description", "")

                # Build direct filing URL
                doc_url = _build_filing_url(cik, adsh, doc_id)
                if not doc_url:
                    continue

                snippet = " | ".join(filter(None, [
                    f"{form_type} filing",
                    file_desc,
                    f"Filed {file_date}",
                    f"Ticker: {ticker}" if ticker else "",
                ]))

                out.append({
                    "url":            doc_url,
                    "domain":         "sec.gov",
                    "title":          f"{form_type}: {co_name} ({file_date})",
                    "date":           file_date,
                    "snippet":        snippet[:500],
                    "company_name":   co_name,
                    "company_domain": infer_company_domain(co_name),
                    "source":         "sec_edgar",
                    "_highlights":    [snippet] if snippet else [],
                    # SEC metadata for compatibility
                    "_sec_form":      form_type,
                    "_sec_cik":       cik,
                })
            return out
        except Exception as e:
            print(f"SEC EFTS API Error: {e}")
            return []

    # Build SEC-specific financial queries from the original question.
    # We reconstruct the question from the first entry in `queries` (the raw question).
    # This replaces the old approach of passing blog-style phrases to SEC.
    original_question = queries[0] if queries else ""
    sec_queries = _build_sec_queries(original_question, forms)
    print(f"SEC EFTS queries: {sec_queries}")
    batches = await asyncio.gather(*[_one(q) for q in sec_queries])
    flat = [r for b in batches for r in b]
    seen: set[str] = set()
    return [r for r in flat if r["url"] not in seen and not seen.add(r["url"])]  # type: ignore[func-returns-value]


# ── Step 2c: Claude Web Search ────────────────────────────────────────────────
# Uses Anthropic's native web_search tool — runs AFTER Exa.
# Returns results in the same shape as search_exa() / search_parallel_ai().
CLAUDE_WEB_SEARCH_MODEL = "claude-3-5-sonnet-20241022"

async def search_claude_web(
    client: httpx.AsyncClient,
    queries: list[str],
    max_days: int = 180,
) -> list[dict]:
    """
    Run the top 3 queries through Claude's built-in web search tool.
    Claude autonomously decides which URLs to fetch and surfaces structured results.

    Waterfall position: runs AFTER Exa, BEFORE PAI.
    Only the top 3 queries are sent — Claude web search is slower and costlier
    than Exa, so we use it for depth, not breadth.
    """
    if not ANTHROPIC_API_KEY:
        return []

    out: list[dict] = []

    async def _one(q: str) -> list[dict]:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key":        ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type":      "application/json",
                    "anthropic-beta":    "web-search-2025-03-05",
                },
                json={
                    "model":      CLAUDE_WEB_SEARCH_MODEL,
                    "max_tokens": 2000,
                    "tools": [
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "max_uses": 3,
                        }
                    ],
                    "messages": [
                        {
                            "role":    "user",
                            "content": (
                                f"Search the web for: {q}\n\n"
                                f"Return a list of relevant companies/articles published in the last {max_days} days. "
                                f"For each result include: URL, title, company name, and a 1-2 sentence excerpt."
                            ),
                        }
                    ],
                },
                timeout=45,
            )
            resp.raise_for_status()
            data    = resp.json()
            results = []

            # Walk content blocks — extract tool_result blocks which contain search hits
            for block in data.get("content", []):
                if block.get("type") == "tool_result":
                    for item in (block.get("content") or []):
                        if isinstance(item, dict) and item.get("type") == "text":
                            # Parse any URLs mentioned in the text block
                            urls_found = re.findall(r'https?://[^\s\)\]"\']+', item.get("text", ""))
                            for url in urls_found[:10]:
                                src_domain = extract_domain_from_url(url)
                                results.append({
                                    "url":                url,
                                    "domain":             src_domain,
                                    "news_source_domain": src_domain,
                                    "title":              "",
                                    "date":               "",
                                    "snippet":            item.get("text", "")[:400],
                                    "company_name":       src_domain.split(".")[0].replace("-", " ").title(),
                                    "company_domain":     src_domain,
                                    "source":             "claude_web",
                                    "_highlights":        [],
                                })
                # Also capture the assistant's final text response — it often lists companies
                elif block.get("type") == "text":
                    text = block.get("text", "")
                    # Extract any URLs from the narrative
                    urls_found = re.findall(r'https?://[^\s\)\]"\'<>]+', text)
                    for url in urls_found[:15]:
                        src_domain = extract_domain_from_url(url)
                        if src_domain and src_domain not in _MEDIA_DOMAINS:
                            # Find title near the URL in the text
                            pos     = text.find(url)
                            context = text[max(0, pos-150):pos+200]
                            results.append({
                                "url":                url,
                                "domain":             src_domain,
                                "news_source_domain": src_domain,
                                "title":              "",
                                "date":               "",
                                "snippet":            context.strip()[:400],
                                "company_name":       src_domain.split(".")[0].replace("-", " ").title(),
                                "company_domain":     src_domain,
                                "source":             "claude_web",
                                "_highlights":        [],
                            })
            return results

        except Exception as e:
            print(f"Claude web search error for '{q[:60]}': {e}")
            return []

    # Run top 3 queries in parallel
    batches = await asyncio.gather(*[_one(q) for q in queries[:3]])
    flat = [r for b in batches for r in b]
    seen: set[str] = set()
    return [r for r in flat if r["url"] not in seen and not seen.add(r["url"])]  # type: ignore[func-returns-value]


# ── Step 2d: Parallel AI FindAll API ─────────────────────────────────────────
# FindAll is an async company-discovery API — NOT a search API.
# Use it when the user is asking for a LIST of COMPANIES matching criteria,
# as opposed to articles/blog posts (which the search API handles better).
FINDALL_BASE   = "https://api.parallel.ai/v1beta/findall"
FINDALL_BETA   = "findall-2025-09-15"
FINDALL_HEADERS = {
    "x-api-key":      "",   # filled at runtime from PARALLEL_AI_KEY
    "Content-Type":   "application/json",
    "parallel-beta":  FINDALL_BETA,
}

# Heuristic: queries that want a list of companies → FindAll
# Queries about articles/blog posts → agentic search
_FINDALL_TRIGGERS = re.compile(
    r'\b(find|list|show|give me|identify|which)\b.{0,40}'
    r'\b(companies|startups?|firms?|organizations?|businesses?|vendors?)\b',
    re.IGNORECASE,
)
_ARTICLE_SIGNALS = re.compile(
    r'\b(blog post|article|news|published|engineering blog|post|wrote|writing|press release|announcement)\b',
    re.IGNORECASE,
)

def _should_use_findall(question: str) -> bool:
    """
    Return True when the user clearly wants a company list, not articles.
    If the query mentions both companies AND blog posts, prefer the search API
    (it returns richer content-level evidence).
    """
    wants_companies = bool(_FINDALL_TRIGGERS.search(question))
    wants_articles  = bool(_ARTICLE_SIGNALS.search(question))
    return wants_companies and not wants_articles


def _build_findall_payload(question: str, requested_count: int, entity_type: str) -> dict:
    """Convert the user's question into a FindAll /runs request payload."""
    # Generator tier: use 'core' for standard queries, 'pro' for large counts
    generator  = "pro" if requested_count > 100 else "core"
    match_limit = max(requested_count, 50) if requested_count > 0 else 50

    # Build match_conditions from the question's key intent phrases
    topic_kws = _extract_topic_keywords(question)
    core      = " ".join(topic_kws[:6])

    return {
        "objective":        question,
        "entity_type":      entity_type,       # "startup", "enterprise", or "company"
        "match_conditions": [
            {
                "name":        "Matches user objective",
                "description": question,
            },
            {
                "name":        "Core topic match",
                "description": f"Company is related to: {core}",
            },
        ],
        "generator":   generator,
        "match_limit": match_limit,
    }


async def search_parallel_findall(
    client: httpx.AsyncClient,
    question: str,
    requested_count: int = 0,
    entity_type: str = "company",
) -> list[dict]:
    """
    Async company-discovery via Parallel AI FindAll API.

    Flow:
      1. POST /ingest  — get structured schema from objective
      2. POST /runs    — start async discovery job → findall_id
      3. Poll GET /runs/{id} until status == "completed" (max 90s)
      4. GET /runs/{id}/result — fetch matched candidates

    Returns results in the same shape as search_exa() / search_parallel_ai().
    """
    headers = {**FINDALL_HEADERS, "x-api-key": PARALLEL_AI_KEY}

    # ── Step 1: Ingest — get structured schema ────────────────────────────────
    try:
        ingest_resp = await client.post(
            f"{FINDALL_BASE}/ingest",
            headers=headers,
            json={"objective": question},
            timeout=20,
        )
        ingest_resp.raise_for_status()
        print(f"FindAll /ingest OK: {ingest_resp.json().get('schema_id', 'no-id')}")
    except Exception as e:
        print(f"FindAll /ingest failed: {e}")
        return []

    # ── Step 2: Start async discovery run ────────────────────────────────────
    payload = _build_findall_payload(question, requested_count, entity_type)
    try:
        run_resp = await client.post(
            f"{FINDALL_BASE}/runs",
            headers=headers,
            json=payload,
            timeout=20,
        )
        run_resp.raise_for_status()
        findall_id = run_resp.json().get("id") or run_resp.json().get("findall_id")
        if not findall_id:
            print("FindAll /runs returned no id")
            return []
        print(f"FindAll /runs started: {findall_id}")
    except Exception as e:
        print(f"FindAll /runs failed: {e}")
        return []

    # ── Step 3: Poll until completed (max 90s, 5s intervals) ─────────────────
    max_wait   = 90
    poll_every = 5
    waited     = 0
    while waited < max_wait:
        await asyncio.sleep(poll_every)
        waited += poll_every
        try:
            status_resp = await client.get(
                f"{FINDALL_BASE}/runs/{findall_id}",
                headers=headers,
                timeout=10,
            )
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status      = status_data.get("status", "")
            metrics     = status_data.get("metrics", {})
            print(f"FindAll poll [{waited}s]: status={status} matched={metrics.get('matched_candidates',0)}")
            if status in ("completed", "done", "finished"):
                break
            if status in ("failed", "error"):
                print(f"FindAll job failed: {status_data}")
                return []
        except Exception as e:
            print(f"FindAll poll error: {e}")

    # ── Step 4: Fetch results ─────────────────────────────────────────────────
    try:
        result_resp = await client.get(
            f"{FINDALL_BASE}/runs/{findall_id}/result",
            headers=headers,
            timeout=20,
        )
        result_resp.raise_for_status()
        candidates = result_resp.json().get("candidates", [])
    except Exception as e:
        print(f"FindAll /result failed: {e}")
        return []

    # ── Map candidates → standard result shape ────────────────────────────────
    out: list[dict] = []
    for c in candidates:
        if c.get("match_status") != "matched":
            continue
        url        = c.get("url", "")
        name       = c.get("name", "")
        desc       = c.get("description", "")
        src_domain = extract_domain_from_url(url) or infer_company_domain(name)
        # basis = list of citations with reasoning
        basis      = c.get("basis") or []
        snippet    = basis[0].get("reasoning", desc) if basis else desc

        out.append({
            "url":                url or f"https://{src_domain}",
            "domain":             src_domain,
            "news_source_domain": src_domain,
            "title":              name,
            "date":               "",
            "snippet":            snippet[:500],
            "company_name":       name,
            "company_domain":     src_domain,
            "source":             "parallel_findall",
            "_highlights":        [snippet] if snippet else [],
        })

    print(f"FindAll returned {len(out)} matched candidates")
    return out


# ── Step 3: Scrape URLs with Firecrawl (Parallel AI Extract fallback) ─────────
# Returns: (content_map, fc_success_count, pai_extract_success_count)
async def scrape_urls(client: httpx.AsyncClient, urls: list[str]) -> tuple[dict[str, str], int, int]:
    sem = asyncio.Semaphore(8)

    # Sentinel tags to know which backend actually served content
    _FC_TAG  = "__fc__"
    _PAI_TAG = "__pai__"

    async def _one(url: str) -> tuple[str, str, str]:
        """Returns (url, content, backend_tag) — backend_tag is _FC_TAG or _PAI_TAG or ''."""
        async with sem:
            # ── Firecrawl first ──
            try:
                resp = await client.post(
                    f"{FIRECRAWL_URL}/scrape",
                    headers={"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"},
                    json={"url": url, "formats": ["markdown"]},
                    timeout=20,
                )
                if resp.status_code < 400:
                    data = resp.json()
                    md = (data.get("data") or {}).get("markdown") or data.get("markdown") or ""
                    if md.strip():
                        return url, md[:4000], _FC_TAG
                elif resp.status_code in (401, 403):
                    print(f"⚠️  Firecrawl auth error {resp.status_code} — key may be expired. "
                          f"Renew at https://firecrawl.dev → falling back to PAI Extract.")
                else:
                    print(f"Firecrawl {resp.status_code} for {url}: {resp.text[:100]}")
            except Exception as e:
                print(f"Firecrawl exception for {url}: {e}")

            # ── Fallback: Parallel AI Extract ──
            try:
                resp = await client.post(
                    f"{PARALLEL_AI_URL}/v1beta/extract",
                    headers={"x-api-key": PARALLEL_AI_KEY, "Content-Type": "application/json"},
                    json={"url": url},
                    timeout=20,
                )
                if resp.status_code < 400:
                    data = resp.json()
                    text = data.get("content") or data.get("markdown") or data.get("text") or ""
                    if text.strip():
                        return url, text[:4000], _PAI_TAG
            except Exception as e:
                print(f"PAI Extract exception for {url}: {e}")

            return url, "", ""

    triples = await asyncio.gather(*[_one(u) for u in urls])

    content_map:   dict[str, str] = {}
    fc_count  = 0
    pai_count = 0
    for url, content, backend in triples:
        if content.strip():
            content_map[url] = content
            if backend == _FC_TAG:
                fc_count += 1
            elif backend == _PAI_TAG:
                pai_count += 1

    print(f"scrape_urls: {fc_count} FC hits, {pai_count} PAI-Extract hits, "
          f"{len(urls) - fc_count - pai_count} misses out of {len(urls)} URLs")
    return content_map, fc_count, pai_count

# ── Step 4: Extract evidence from scraped page (text-based, no external API) ──
_URL_CONTENT_TYPE_HINTS = [
    (r'/blog/', 'blog post'), (r'/posts?/', 'blog post'), (r'/news/', 'announcement'),
    (r'/press/', 'announcement'), (r'/whitepaper', 'whitepaper'),
    (r'/case-stud', 'case study'), (r'/guide', 'guide'), (r'/tutorial', 'guide'),
    (r'/playbook', 'guide'), (r'/docs/', 'documentation'), (r'/research/', 'research'),
    (r'/engineering/', 'blog post'), (r'/insights/', 'blog post'),
]

def _detect_content_type(url: str, content: str) -> str:
    url_lower = url.lower()
    for pattern, ctype in _URL_CONTENT_TYPE_HINTS:
        if re.search(pattern, url_lower):
            return ctype
    # Heuristic from content
    c = content[:500].lower()
    if any(x in c for x in ['published', 'posted', 'author', 'written by', 'min read']):
        return 'blog post'
    if any(x in c for x in ['press release', 'announces', 'today announced']):
        return 'announcement'
    return 'article'

def _extract_date_from_content(content: str) -> str:
    # Look for ISO or human-readable dates near the top of the content
    m = re.search(r'\b(20(?:2[3-9]|[3-9]\d)[-/]\d{2}[-/]\d{2})\b', content[:2000])
    if m:
        return m.group(1).replace('/', '-')
    m = re.search(
        r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+\d{1,2},?\s+(20(?:2[3-9]|[3-9]\d))\b',
        content[:2000], re.IGNORECASE,
    )
    if m:
        return m.group(0)
    return ""

# ── Recency check & sanity evaluator ─────────────────────────────────────────
_MONTH_MAP = {
    'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
    'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,
}

def _parse_date(date_str: str) -> Optional[datetime.date]:
    """Best-effort parse of ISO or human-readable date strings."""
    if not date_str:
        return None
    s = date_str.strip()[:20]
    # ISO: 2025-03-07  or  2025/03/07
    m = re.match(r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', s)
    if m:
        try:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    # "March 7 2025" or "March 7, 2025"
    m = re.match(
        r'(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+(\d{1,2}),?\s+(\d{4})', s, re.IGNORECASE)
    if m:
        mo = _MONTH_MAP.get(m.group(1).lower()[:3])
        if mo:
            try:
                return datetime.date(int(m.group(3)), mo, int(m.group(2)))
            except ValueError:
                pass
    return None

def _days_ago(date_str: str) -> Optional[int]:
    """Return how many days ago the date is, or None if unparseable."""
    d = _parse_date(date_str)
    if d is None:
        return None
    return (datetime.date.today() - d).days

def sanity_check_evidence(records: list[dict], max_days: int = 180) -> tuple[list[dict], list[dict]]:
    """
    Evaluate each evidence record for:
      1. Recency  — date must be within max_days (if a date is present)
      2. Minimum confidence — must be ≥ 0.3
      3. Non-empty excerpt

    Returns (passed, rejected) where rejected items include a 'sanity_fail' reason.
    Records with NO date are kept but flagged with 'date_unverified'.
    """
    passed:   list[dict] = []
    rejected: list[dict] = []

    for rec in records:
        reasons: list[str] = []

        # ── Confidence floor ──
        if float(rec.get("confidence", 0)) < 0.3:
            reasons.append(f"confidence too low ({rec.get('confidence', 0):.0%})")

        # ── Empty excerpt ──
        if not rec.get("key_excerpt", "").strip():
            reasons.append("no key excerpt")

        # ── Date recency ──
        date_str = rec.get("date") or rec.get("published_date") or ""
        age = _days_ago(date_str)
        if age is None:
            # Strict mode: user asked for a tight window (≤ 90d) and this is a
            # snippet-only result (no full scrape) with no verifiable date → reject.
            # For scraped pages we give benefit of the doubt (API date filter covered it).
            if max_days <= 90 and rec.get("_from_snippet"):
                reasons.append(f"no date — rejected in strict {max_days}d window")
            else:
                rec["date_unverified"] = True   # no date found — keep but flag
        elif age > max_days:
            reasons.append(f"published {age}d ago (limit: {max_days}d) — date: {date_str[:10]}")

        if reasons:
            rec["sanity_fail"] = "; ".join(reasons)
            rejected.append(rec)
        else:
            rec.pop("sanity_fail", None)
            passed.append(rec)

    return passed, rejected

# ── Person / role signal extractor ───────────────────────────────────────────
# Matches titles like "VP of Engineering", "Head of Sales", "Senior Director"
_PERSON_TITLE_RE = re.compile(
    r'(?:(?:Senior|Sr\.?|Junior|Jr\.?|Principal|Staff|Lead|Chief|Global|'
    r'Regional|Associate|Founding|Co-)\s+)?'
    r'(?:VP|Vice President|President|CEO|CTO|CPO|CMO|CFO|COO|CISO|CRO|'
    r'Director|Head|Manager|Engineer|Architect|Scientist|Analyst|'
    r'Founder|Partner|Fellow|Advisor)\s+(?:of\s+|at\s+)?[\w\s&,]{2,40}',
    re.IGNORECASE,
)
# Matches action verbs that indicate who did what
_ACTION_VERB_RE = re.compile(
    r'\b(published|posted|wrote|authored|shared|announced|presented|'
    r'released|launched|introduced|revealed|detailed|explained|described)\b',
    re.IGNORECASE,
)

def _extract_precise_signal(url: str, content: str, company_name: str, best_para: str) -> str:
    """
    Build a precise, human-readable signal string from scraped content.

    Priority order:
    1. Person title + action + subject  ("VP of Eng at Acme published a deep-dive on…")
    2. Company + action + subject       ("Acme launched their new agentic pipeline…")
    3. Verbatim excerpt from best paragraph (fallback)

    Always ends with the proof URL so the reader can click straight through.
    """
    search_text = content[:3000]

    # Try to find a person title near an action verb
    person_match = _PERSON_TITLE_RE.search(search_text)
    action_match = _ACTION_VERB_RE.search(search_text)

    if person_match and action_match:
        person = person_match.group(0).strip()
        action = action_match.group(1).lower()
        # Grab up to 120 chars after the action verb as the subject
        action_pos  = action_match.end()
        subject_raw = search_text[action_pos:action_pos + 150].strip()
        subject     = re.sub(r'\s+', ' ', subject_raw.split('\n')[0])[:120]
        return (
            f"{person} at {company_name} {action}: \"{subject}\" — "
            f"Source: {url}"
        )

    # No person found — use company + action
    if action_match:
        action    = action_match.group(1).lower()
        action_pos = action_match.end()
        subject_raw = search_text[action_pos:action_pos + 150].strip()
        subject     = re.sub(r'\s+', ' ', subject_raw.split('\n')[0])[:120]
        return (
            f"{company_name} {action}: \"{subject}\" — "
            f"Source: {url}"
        )

    # Fallback: verbatim best paragraph excerpt
    excerpt = best_para[:200].strip() if best_para else ""
    if excerpt:
        return f"{company_name}: \"{excerpt}\" — Source: {url}"

    return f"Content from {company_name} — Source: {url}"


def extract_evidence(url: str, content: str, question: str, hint_date: str = "") -> dict:
    """
    Text-based evidence extraction — no external API.
    Scores paragraphs by keyword overlap with the question,
    returns the most relevant verbatim excerpt + a precise signal string.
    """
    if not content.strip():
        return {"url": url, "relevant": False}

    domain = extract_domain_from_url(url)
    if domain in _MEDIA_DOMAINS:
        return {"url": url, "relevant": False}

    # Use TOPIC keywords for scoring (not full question keywords).
    question_kws = set(_extract_keywords(question))
    topic_kws    = set(_extract_topic_keywords(question))
    score_kws    = topic_kws if len(topic_kws) >= 2 else question_kws
    if not score_kws:
        return {"url": url, "relevant": False}

    # Split into paragraphs (prefer natural paragraph breaks)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}|\. {2,}', content) if len(p.strip()) > 50]
    if not paragraphs:
        paragraphs = [content[i:i+400] for i in range(0, min(len(content), 4000), 300)]

    best_para  = ""
    best_score = 0
    for para in paragraphs[:80]:
        para_words = set(_extract_keywords(para))
        overlap = len(score_kws & para_words)
        if overlap > best_score:
            best_score = overlap
            best_para  = para[:500]

    effective_denom = min(len(score_kws), 5)
    confidence   = round(min(0.95, 0.65 + (best_score / max(effective_denom, 1) * 0.3)), 2)
    company_name = domain.split('.')[0].replace('-', ' ').title()

    # Build the precise signal — replaces the old generic summary
    signal = _extract_precise_signal(url, content, company_name, best_para)

    # Best date: content-extracted first, search API hint as fallback
    content_date = _extract_date_from_content(content)
    best_date    = content_date or hint_date

    return {
        "url":            url,
        "relevant":       True,
        "company_name":   company_name,
        "company_domain": domain,
        "content_type":   _detect_content_type(url, content),
        "published_date": best_date,
        "key_excerpt":    best_para,
        "signal":         signal,          # ← precise, proof-linked signal
        "proof_url":      url,             # ← always present, always clickable
        "confidence":     confidence,
    }

# ── Step 4b: Extract True Subject Entities (Phase 4 AI Extractor) ─────────
async def _extract_subjects_batch(
    client: httpx.AsyncClient,
    batch: list[dict],
    question: str,
    revops_context: dict | None = None,
) -> tuple[dict[str, dict], float]:
    """
    Process a single batch of up to 15 snippets through Claude.
    revops_context (optional) carries the buyer persona + signal_focus from
    analyze_revops_intent() so the signal is shaped for the right audience.
    """
    persona      = (revops_context or {}).get("buyer_persona", "RevOps professional")
    signal_focus = (revops_context or {}).get("signal_focus", "the specific action taken and by whom")

    prompt = f"""You are an elite GTM intelligence analyst writing for a {persona}.
The user asked: "{question}"
Signal focus: {signal_focus}

I will provide an array of web snippets. Each snippet has an "id" and a "url". Your job:
1. Identify the ACTUAL subject company performing the core action — NOT the news outlet or aggregator.
2. Write a highly precise `aligned_signal` (1-2 sentences) that directly answers the user's query.
   - ALWAYS name the specific person and their title/role if mentioned (e.g. "Jane Smith, VP of Sales at Acme").
   - ALWAYS state the specific action (published, announced, launched, hired, raised, etc.).
   - Include concrete details: dollar amounts, tool names, team sizes, dates — whatever is present.
   - End with: "Proof: [url]" so the reader can verify immediately.
   - BAD: "This article discusses agentic systems."
   - GOOD: "Marcus Lee, Head of Platform Engineering at Acme Corp, published a post detailing how they replaced LangChain with a custom multi-agent orchestration layer — Proof: https://acme.com/blog/agents"
3. Extract `person_name` and `person_title` if any individual is named as the actor (else empty string).

Rules:
- If prnewswire/businesswire publishes "Acme Raises $10M", subject = "Acme", domain = "acme.com"
- If it's a company's own blog/post, subject = that company
- If it's a generic roundup with no single acting company, output "unknown" for company/domain
- NEVER use a news wire or aggregator as subject_company

Return ONLY valid JSON:
{{
  "extractions": [
    {{
      "id": "0",
      "subject_company": "Acme Corp",
      "subject_domain": "acme.com",
      "person_name": "Marcus Lee",
      "person_title": "Head of Platform Engineering",
      "aligned_signal": "Marcus Lee, Head of Platform Engineering at Acme Corp, published a post on replacing LangChain with a custom multi-agent layer — Proof: https://acme.com/blog/agents"
    }}
  ]
}}

Snippets:
{json.dumps(batch)}"""

    try:
        raw_text, ai_cost = await _anthropic_call_with_fallback(
            client,
            system=(
                "Output only valid JSON. "
                "Never use news wires or aggregators as subject_company. "
                "Always name the specific person and title if present. "
                "aligned_signal must end with 'Proof: [url]'. "
                "Include all concrete specifics: dollar amounts, tool names, team sizes, percentages."
            ),
            user_content=prompt,
            max_tokens=3000,
            timeout=45,
        )
        json_match = re.search(r'\{.*\}', raw_text.replace('\n', ''), re.DOTALL)
        res = json.loads(json_match.group(0)) if json_match else json.loads(raw_text)
        return {str(item["id"]): item for item in res.get("extractions", []) if "id" in item}, ai_cost
    except Exception as e:
        print("Subject Entity Extraction failed via AI:", e)
        return {}, 0.0

async def extract_subjects_with_ai(
    client: httpx.AsyncClient,
    evidence_list: list[dict],
    question: str,
    revops_context: dict | None = None,
) -> tuple[dict[str, dict], float]:
    """
    Uses Claude to isolate the ACTUAL company + person acting in the snippet.
    Processes in batches of 15 to avoid output truncation.
    Returns (extractions_dict keyed by URL, llm_cost).

    Each value in the dict contains:
      subject_company, subject_domain, person_name, person_title, aligned_signal
    """
    if not evidence_list:
        return {}, 0.0

    # Assign temp IDs — Claude sometimes mutates raw URLs, IDs are stable
    for i, e in enumerate(evidence_list[:60]):
        e["_temp_id"] = str(i)

    all_payloads = [
        {
            "id":      e["_temp_id"],
            "url":     e.get("url", ""),
            "snippet": e.get("key_excerpt") or e.get("snippet") or "",
        }
        for e in evidence_list[:60]
    ]

    BATCH_SIZE = 15
    batches = [all_payloads[i:i+BATCH_SIZE] for i in range(0, len(all_payloads), BATCH_SIZE)]

    results = await asyncio.gather(*[
        _extract_subjects_batch(client, batch, question, revops_context=revops_context)
        for batch in batches
    ])

    merged_by_id: dict[str, dict] = {}
    total_cost = 0.0
    for batch_result, batch_cost in results:
        merged_by_id.update(batch_result)
        total_cost += batch_cost

    # Remap by URL + apply negative filtering (drop news wires as subjects)
    cleaned: dict[str, dict] = {}
    for e in evidence_list[:60]:
        _id = e["_temp_id"]
        if _id in merged_by_id:
            data        = merged_by_id[_id]
            subj_domain = (data.get("subject_domain") or "").lower().strip()
            if subj_domain and subj_domain not in _NEWS_WIRE_DOMAINS and subj_domain not in _MEDIA_DOMAINS:
                cleaned[e["url"]] = data

    return cleaned, total_cost

# ── Step 5: Synthesize final answer ──────────────────────────────────────────
async def synthesize_answer(
    client: httpx.AsyncClient,
    question: str,
    evidence_by_company: dict,
    revops_context: dict | None = None,
) -> tuple[str, float]:
    """
    Uses Claude 3.5 Sonnet to produce a tiered, Opus-quality intelligence report.
    Each entry includes the specific person + action + Signal Type + Outreach Angle.
    """
    if not evidence_by_company:
        return f"No direct public evidence found for: {question}", 0.0

    persona      = (revops_context or {}).get("buyer_persona", "RevOps professional")
    use_case     = (revops_context or {}).get("use_case", "B2B intelligence gathering")
    signal_focus = (revops_context or {}).get("signal_focus", "specific action taken and by whom")

    # Sort by max confidence, pick top companies with richest signals
    sorted_companies = sorted(
        evidence_by_company.items(),
        key=lambda kv: max((ev.get("confidence", 0) for ev in kv[1]), default=0),
        reverse=True,
    )

    companies_data = []
    for domain, items in sorted_companies[:20]:  # Top 20 companies
        ev_list = []
        for ev in items[:5]:  # Top 5 signals per company
            ev_list.append({
                "signal":       ev.get("signal") or ev.get("key_excerpt", "")[:300],
                "person":       f"{ev.get('person_name', '')} {ev.get('person_title', '')}".strip(),
                "date":         ev.get("published_date") or ev.get("date", ""),
                "proof_url":    ev.get("proof_url") or ev.get("url", ""),
                "source":       ev.get("news_source_domain", ""),
                "content_type": ev.get("content_type", ""),
                "confidence":   ev.get("confidence", 0.5),
            })
        companies_data.append({
            "company":  items[0].get("company_name") or domain,
            "domain":   domain,
            "signals":  ev_list,
        })

    prompt = f"""You are a senior GTM intelligence analyst writing for a {persona}.
Goal: {use_case}
Signal focus: {signal_focus}

QUERY: "{question}"

EVIDENCE ({len(companies_data)} companies):
{json.dumps(companies_data, indent=2)}

---

Write a concise intelligence briefing in markdown. STRICT LIMIT: under 200 words total.

Format:
**[1-line pattern summary]**

**Top signals:**
- **Company** — Person, Title: what they did + why it matters. [Source](proof_url) · Signal: type · Hook: outreach angle
- repeat for top 3-5 entries only (highest confidence, named person preferred)

**Pattern:** [1 sentence on the broader trend across all results]

RULES:
- Under 200 words — no exceptions
- Only include companies from the evidence
- Each bullet = 1 line max
- Always include the proof URL as a clickable markdown link
- If signal ends with "Proof: [url]" use that URL
- No headers beyond the ones shown, no Tier sections, no "How to prioritize" section
- Return ONLY the briefing. No preamble."""

    try:
        answer, cost = await _anthropic_call_with_fallback(
            client,
            system=(
                "You are an elite B2B GTM intelligence analyst. "
                "Write concise intelligence briefings strictly under 200 words. "
                "Each bullet: company, person+title, specific action, proof URL, signal type, outreach hook — all on one line. "
                "Never exceed 200 words. Never write vague summaries."
            ),
            user_content=prompt,
            max_tokens=4000,
            timeout=90,
        )
        return answer, cost
    except Exception as e:
        print(f"Synthesis failed: {e}")
        return f"An error occurred while synthesizing: {str(e)}", 0.0

# ── Claude-only direct knowledge answer ──────────────────────────────────────
async def answer_from_claude_knowledge(client: httpx.AsyncClient, question: str) -> tuple[str, float]:
    """
    For factual / well-known entity questions Claude can answer without web search.
    Returns (answer_text, llm_cost).
    """
    if not ANTHROPIC_API_KEY:
        return "Claude API key not configured.", 0.0
    try:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 1024,
                "system": (
                    "You are a knowledgeable B2B technology research assistant. "
                    "Answer the user's question concisely and accurately from your training knowledge. "
                    "Format your answer in clean markdown. Be specific and helpful."
                ),
                "messages": [{"role": "user", "content": question}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        cost = (usage.get("input_tokens", 0) / 1_000_000) * COST_HAIKU_IN_PER_1M \
             + (usage.get("output_tokens", 0) / 1_000_000) * COST_HAIKU_OUT_PER_1M
        answer = data.get("content", [{}])[0].get("text", "")
        return answer, cost
    except Exception as e:
        print(f"Claude knowledge answer failed: {e}")
        return f"Unable to answer: {e}", 0.0


# ── Step 5b: Generate contextual follow-up suggestions ───────────────────────
async def generate_follow_ups(
    client: httpx.AsyncClient,
    question: str,
    top_companies: list[str],
    top_signals: list[str] | None = None,
) -> list[dict]:
    """Uses Claude Haiku to generate 3 highly contextual follow-up questions grounded in actual results."""
    if not ANTHROPIC_API_KEY:
        return []
    companies_str = ", ".join(top_companies[:6]) if top_companies else "the discovered companies"
    signals_str   = "\n".join(f"- {s}" for s in (top_signals or [])[:5]) if top_signals else ""

    prompt = f"""You are a senior RevOps intelligence analyst. A user just ran this B2B research query and received results.

ORIGINAL QUERY: "{question}"

TOP COMPANIES FOUND: {companies_str}
{f"ACTUAL SIGNALS FOUND:{chr(10)}{signals_str}" if signals_str else ""}

Generate exactly 3 follow-up research questions that are the NATURAL NEXT STEP given these specific results.
Rules:
1. Each follow-up MUST be directly derived from the actual companies or signals found — NEVER generic
2. Vary the angle: e.g. hiring intent → funding check → competitive displacement, or drill deeper into the same theme
3. Each follow-up should reference at least 1-2 of the actual companies or a specific technology/signal found
4. Keep the label under 55 chars and add a relevant emoji
5. The "query" field should be the full question the user would actually type

Return ONLY a JSON array of exactly 3 objects:
[
  {{"label": "emoji Short label here", "query": "Full follow-up question referencing {companies_str.split(',')[0].strip() if companies_str != 'the discovered companies' else 'the companies'}..."}},
  {{"label": "emoji Short label here", "query": "..."}},
  {{"label": "emoji Short label here", "query": "..."}}
]"""

    try:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 600,
                "system": "Output only valid JSON array. No explanation. No markdown fences.",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = data.get("content", [{}])[0].get("text", "[]")
        match = re.search(r'\[.*\]', raw.replace('\n', ' '), re.DOTALL)
        return json.loads(match.group(0)) if match else []
    except Exception as e:
        print(f"Follow-up generation failed: {e}")
        return []


# ── Main pipeline ─────────────────────────────────────────────────────────────
def _infer_max_days(question: str) -> int:
    """
    Parse a recency window from the question text.
    Returns the number of days — used both to filter search APIs (startPublishedDate)
    and to reject stale evidence in sanity_check_evidence().
    """
    q = question.lower()
    # Explicit day count: "last 90 days", "past 60 days"
    m = re.search(r'(\d+)\s*day', q)
    if m:
        return int(m.group(1))
    # Explicit month count: "last 3 months", "past 6 months"
    m = re.search(r'(\d+)\s*month', q)
    if m:
        return int(m.group(1)) * 30
    # Named windows
    if 'last week' in q:                        return 7
    if 'last month' in q:                       return 30
    if 'last quarter' in q:                     return 90
    if 'last year' in q or 'this year' in q:    return 365
    if 'last 6' in q or 'recent' in q or 'lately' in q:
        return 180
    return 90   # default: 3 months

async def analyze_revops_intent(client: httpx.AsyncClient, question: str) -> tuple[dict, float]:
    """
    Phase -1: Understand WHY the user is asking this question before doing anything else.

    A RevOps person asking "VPs who posted about AI agents" wants a prospect list.
    A founder asking the same wants competitive intelligence.
    The output columns, signal focus, and extraction strategy all change accordingly.

    Always-present columns: company_name, company_domain.
    All other columns are dynamic based on the use case.

    Returns (context_dict, llm_cost) where context_dict contains:
      - buyer_persona   : who is asking (SDR, AE, RevOps, Founder, Analyst…)
      - use_case        : what they will DO with the answer (prospect, track, hire…)
      - signal_focus    : what to look for in scraped content
      - output_columns  : ordered list of columns for the response table
      - search_modifier : extra keywords to inject into queries
    """
    prompt = f"""You are a senior RevOps strategist. A user has asked the following question to a B2B intelligence tool:

"{question}"

Your job is to understand the underlying business rationale — WHY would someone ask this — and define the ideal output structure.

Think step by step:
1. Who is the most likely persona asking this? (SDR, AE, RevOps, Marketing, Founder, Analyst, Recruiter…)
2. What is their use case? (finding prospects, tracking competitors, identifying buying signals, sourcing candidates, market research…)
3. What specific signal in the content would actually be valuable to them?
4. What output columns would make this immediately actionable? (company_name and company_domain are ALWAYS included)
5. What extra search modifier words would improve result quality for this persona?

Return ONLY valid JSON:
{{
  "buyer_persona": "SDR at a B2B SaaS company",
  "use_case": "Identifying warm prospects who are actively thinking about AI agent infrastructure",
  "signal_focus": "The specific person who published the post, their title, what they said about the technology, and whether their company is likely a buyer",
  "output_columns": ["company_name", "company_domain", "person_name", "person_title", "signal", "proof_url", "published_date"],
  "search_modifier": "engineering blog post published written"
}}"""

    try:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": ANTHROPIC_API_KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model":      "claude-3-haiku-20240307",
                "max_tokens": 600,
                "system":     "You are a JSON-only RevOps strategist. Output nothing but valid JSON.",
                "messages":   [{"role": "user", "content": prompt}],
            },
            timeout=20,
        )
        resp.raise_for_status()
        data     = resp.json()
        usage    = data.get("usage", {})
        llm_cost = (usage.get("input_tokens", 0) / 1_000_000) * COST_HAIKU_IN_PER_1M \
                 + (usage.get("output_tokens", 0) / 1_000_000) * COST_HAIKU_OUT_PER_1M
        raw_text = data.get("content", [{}])[0].get("text", "{}")
        match    = re.search(r'\{.*\}', raw_text.replace('\n', ''), re.DOTALL)
        ctx      = json.loads(match.group(0)) if match else json.loads(raw_text)

        # Guarantee company_name + company_domain are always first two columns
        cols = ctx.get("output_columns", [])
        for must_have in ["company_domain", "company_name"]:
            if must_have not in cols:
                cols.insert(0, must_have)
        ctx["output_columns"] = cols

        return ctx, llm_cost

    except Exception as e:
        print(f"RevOps intent analysis failed: {e}")
        # Safe defaults — always works even if Claude is down
        return {
            "buyer_persona":   "RevOps professional",
            "use_case":        "general intelligence gathering",
            "signal_focus":    "the specific action taken and by whom",
            "output_columns":  ["company_name", "company_domain", "signal", "proof_url", "published_date"],
            "search_modifier": "",
        }, 0.0


async def analyze_intent_and_routing(client: httpx.AsyncClient, question: str) -> tuple[dict, str, list[str], float]:
    """Uses Claude to clean typos, generate search queries, and route to tools."""
    prompt = f"""You are the intelligence routing brain for a public data AI agent.
The user asked: "{question}"

Instructions:
1. "corrected_query": Fix ANY typos, spelling mistakes, and grammatical errors in the user's question, while preserving their exact intent.
2. "search_queries": Generate an array of exactly 4 specialized search queries to find this data on Google/Exa.
   - PLATFORM RULE: If the user explicitly asks for content from a specific platform (linkedin, reddit, github, twitter, producthunt, etc.), ALL 4 queries MUST be tailored to surface posts/threads from that platform. For LinkedIn, write queries as if someone would post them: first-person testimonial style, mention of the product/topic, personal experience. Example for "linkedin posts about Sumble": ["Sumble sales intelligence linkedin", "using Sumble for prospecting linkedin post", "Sumble GTM tool review linkedin", "Sumble account intelligence user experience"]
   - If NOT platform-specific: 1 direct keyword, 2 semantic company-signal queries, 1 first-person blog style.
3. "routing": Decide which data tools are needed to answer this.
   - "claude_only": Set to TRUE if the question is a FACTUAL KNOWLEDGE question that Claude can answer directly without web search (e.g. "What does Stripe do?", "Does GitHub have an API?", "What is gRPC?", "Who is the CEO of Salesforce?"). Set FALSE for any query asking to FIND, LIST, DISCOVER, or TRACK companies / signals / articles / posts.
   - "exa": Web search — use for general news, company blogs, product launches, tech content, social posts. Set false if claude_only is true.
   - "parallel_ai": Deep search — use for technical deep-dives, developer tooling, architecture. Set false if claude_only is true OR if the query is platform-specific (linkedin/reddit/github) — Exa handles those better.
   - "sec_api": SEC EDGAR search — use ONLY for explicit SEC filings (10-K, 8-K), financial earnings, M&A, revenue. Set false if claude_only is true.

Return ONLY a valid JSON object matching this exact structure:
{{
  "corrected_query": "...",
  "search_queries": ["...", "...", "...", "..."],
  "routing": {{"claude_only": false, "exa": true, "parallel_ai": true, "sec_api": false}}
}}"""
    try:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-haiku-20240307",
                "max_tokens": 400,
                "system": "You are a JSON-only tool routing system. Output nothing but valid JSON.",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
        raw_text = data.get("content", [{}])[0].get("text", "{}")
        
        usage = data.get("usage", {})
        in_tokens = usage.get("input_tokens", 0)
        out_tokens = usage.get("output_tokens", 0)
        cost = (in_tokens / 1_000_000) * COST_SONNET_IN_PER_1M + (out_tokens / 1_000_000) * COST_SONNET_OUT_PER_1M
        
        json_match = re.search(r'\{.*\}', raw_text.replace('\n', ''), re.DOTALL)
        if json_match:
            res = json.loads(json_match.group(0))
        else:
            res = json.loads(raw_text)
            
        routing = res.get("routing", {})
        claude_only = routing.get("claude_only", False)
        r_dict = {
            "claude_only":  claude_only,
            "exa":          False if claude_only else routing.get("exa", True),
            "parallel_ai":  False if claude_only else routing.get("parallel_ai", True),
            "sec_api":      False if claude_only else routing.get("sec_api", False),
        }

        # Hardcode SEC routing if the prompt explicitly mentions SEC keywords
        q_lower = question.lower()
        if any(kw in q_lower for kw in ["sec", "10-k", "10-q", "8-k", "earnings", "s-1"]):
            r_dict["sec_api"] = True
            r_dict["claude_only"] = False

        return r_dict, res.get("corrected_query", question), res.get("search_queries", []), cost
    except httpx.HTTPStatusError as e:
        print(f"LLM routing HTTP failed: {e.response.text}")
        q_lower = question.lower()
        use_sec_fallback = any(kw in q_lower for kw in ["sec", "10-k", "10-q", "8-k", "earnings", "s-1"])
        return {"claude_only": False, "exa": True, "parallel_ai": True, "sec_api": use_sec_fallback}, question, [], 0.0
    except Exception as e:
        print("LLM routing failed. Defaulting to regex queries.", e)
        q_lower = question.lower()
        use_sec_fallback = any(kw in q_lower for kw in ["sec", "10-k", "10-q", "8-k", "earnings", "s-1"])
        return {"claude_only": False, "exa": True, "parallel_ai": True, "sec_api": use_sec_fallback}, question, [], 0.0

async def run_ask_pipeline(question: str) -> AsyncGenerator[str, None]:
    """
    5-phase SSE pipeline:
    1. Derive search queries from question
    2. Search: Parallel AI Search + Exa in parallel
    3. Scrape top URLs: Firecrawl + Parallel AI Extract fallback
    4. Extract evidence per page + sanity-check dates
    5. Synthesize answer + return grouped evidence with proofs
    """
    t0 = time.time()
    max_days = _infer_max_days(question)

    # ── Parse query intent (count + entity type + platform) ───────────────────
    intent           = _parse_query_intent(question)
    requested_count  = intent["requested_count"]
    entity_type      = intent["entity_type"]
    startup_filter   = intent["startup_filter"]
    target_platform  = intent.get("target_platform")
    include_domains  = intent.get("include_domains", [])

    yield sse("start", {"question": question, "msg": "Starting public research pipeline…"})

    limits = httpx.Limits(max_connections=12, max_keepalive_connections=6)
    async with httpx.AsyncClient(verify=False, limits=limits, headers={"Connection": "close"}) as client:

        # ── Phase -1: RevOps Intent Analysis ─────────────────────────────────
        yield sse("phase", {"phase": "revops_intent", "status": "running",
                             "msg": "Understanding the business rationale behind your query…"})
        revops_context, revops_cost = await analyze_revops_intent(client, question)
        llm_cost = revops_cost
        yield sse("phase", {
            "phase":          "revops_intent",
            "status":         "done",
            "buyer_persona":  revops_context.get("buyer_persona"),
            "use_case":       revops_context.get("use_case"),
            "signal_focus":   revops_context.get("signal_focus"),
            "output_columns": revops_context.get("output_columns"),
            "msg": (
                f"Persona: {revops_context.get('buyer_persona')} · "
                f"Use case: {revops_context.get('use_case')}"
            ),
        })

        # ── Phase 0: Intent & Routing via Claude ──
        yield sse("phase", {"phase": "routing", "status": "running", "msg": "Analyzing intent with Claude..."})
        routing_decisions, raw_clean_question, ai_queries, routing_cost = await analyze_intent_and_routing(client, question)
        llm_cost += routing_cost
        
        # Globally update the question variable to the typo-free version so downstream string-matching tools don't fail!
        question = raw_clean_question if isinstance(raw_clean_question, str) else question
        
        use_claude_only = routing_decisions.get("claude_only", False)
        use_exa = routing_decisions.get("exa", True) and not use_claude_only
        use_pai = routing_decisions.get("parallel_ai", True) and not use_claude_only
        use_sec = routing_decisions.get("sec_api", False) and not use_claude_only

        route_mode = "Claude Knowledge (no web search)" if use_claude_only else f"Exa={use_exa}, PAI={use_pai}, SEC={use_sec}"
        route_msg = f"Routing: {route_mode} (LLM so far: ${llm_cost:.4f})"
        yield sse("phase", {"phase": "routing", "status": "done", "msg": route_msg})

        # ── Claude-Only Fast Path ─────────────────────────────────────────────
        if use_claude_only:
            yield sse("phase", {"phase": "synthesis", "status": "running",
                                 "msg": "Answering from Claude's knowledge base (no web search needed)…"})
            answer, answer_cost = await answer_from_claude_knowledge(client, question)
            llm_cost += answer_cost
            elapsed = round(time.time() - t0, 1)
            yield sse("phase", {"phase": "synthesis", "status": "done",
                                 "msg": f"Answer ready from Claude knowledge (${answer_cost:.4f})"})
            yield sse("done", {
                "question":         question,
                "answer":           answer,
                "companies":        [],
                "total_companies":  0,
                "total_articles":   0,
                "elapsed":          elapsed,
                "follow_ups":       [],
                "revops_context":   revops_context,
                "pipeline_stats":   {"claude_only": True},
                "sanity_report":    {},
                "search_queries_used": [],
            })
            log_query_metrics(question, routing_decisions, llm_cost, 0, 0, 0, 0, 0, 0)
            return

        # ── Phase 1: Derive queries ──
        yield sse("phase", {"phase": "query_generation", "status": "running",
                             "msg": "Generating search queries from your question…"})
        
        # Fallback to python regex generator if Claude failed to output list
        if not ai_queries or not isinstance(ai_queries, list):
            queries = derive_search_queries(question, entity_type=entity_type)
        else:
            queries = ai_queries

        # Surface intent to the frontend
        if requested_count > 0 or entity_type != "company" or target_platform:
            yield sse("phase", {
                "phase":            "intent",
                "requested_count":  requested_count,
                "entity_type":      entity_type,
                "startup_filter":   startup_filter,
                "target_platform":  target_platform,
                "include_domains":  include_domains,
                "msg": (
                    f"Intent detected — "
                    + (f"targeting {requested_count} results · " if requested_count else "")
                    + (f"platform: {target_platform} · " if target_platform else "")
                    + f"entity type: {entity_type}"
                ),
            })

        yield sse("phase", {"phase": "query_generation", "status": "done",
                             "queries": queries, "msg": f"Generated {len(queries)} search queries"})

        # ── Phase 2: Search (Round 1) ──
        active_sources = []
        if use_pai: active_sources.append("Parallel AI")
        if use_exa: active_sources.append("Exa")
        if use_sec: active_sources.append("SEC API")
        
        yield sse("phase", {"phase": "search", "status": "running",
                             "msg": f"Round 1 — Searching {' + '.join(active_sources)} with {len(queries)} queries…"})

        async def _noop() -> list: return []

        # Detect if FindAll is appropriate for this query
        use_findall = use_pai and _should_use_findall(question)

        # ── Waterfall: Exa → Claude web search → PAI (if needed) ──────────────
        # Phase A: Exa + Claude web search run in parallel (both are retrieval-first).
        # PAI (FindAll or agentic) runs in Phase A too for SEC/company queries,
        # then Round 2 only fires if Phase A total is thin (< 50 unique domains).

        pai_task = (
            search_parallel_findall(client, question, requested_count=requested_count, entity_type=entity_type)
            if use_findall
            else search_parallel_ai(client, queries, requested_count=requested_count)
            if use_pai
            else _noop()
        )

        pai_results, exa_results, claude_web_results, sec_results = await asyncio.gather(
            pai_task,
            search_exa(client, queries, max_days=max_days, requested_count=requested_count,
                       include_domains=include_domains or None)                             if use_exa else _noop(),
            search_claude_web(client, queries, max_days=max_days)                          if use_exa and not include_domains else _noop(),
            search_sec_edgar(client, queries, max_days=max_days)                           if use_sec else _noop(),
        )

        # Log which source FindAll was used
        if use_findall:
            print(f"FindAll mode active — returned {len(pai_results)} company candidates")

        def _merge_dedup(lists: list[list[dict]]) -> list[dict]:
            seen: set[str] = set()
            out: list[dict] = []
            for r in [item for lst in lists for item in lst]:
                if r.get("url") and r["url"] not in seen:
                    seen.add(r["url"])
                    out.append(r)
            return out

        all_results = _merge_dedup([pai_results, exa_results, claude_web_results, sec_results])

        # ── Round 2: targeted re-search if first pass is thin ──
        round2_pai: list[dict] = []
        round2_exa: list[dict] = []
        round2_queries: list[str] = []   # always defined so cost logging is safe
        unique_domains_r1 = {r.get("domain", "") for r in all_results if r.get("domain")}

        if len(unique_domains_r1) < 50:
            topic_kws = _extract_topic_keywords(question)
            core2     = " ".join(topic_kws[:4]) if topic_kws else " ".join(_extract_keywords(question)[:4])
            cur_year  = datetime.date.today().year
            round2_queries = [
                f'{core2} "we switched" OR "we migrated" OR "we moved" company blog {cur_year}',
                f'{core2} engineering blog post lessons learned {cur_year}',
                f'{core2} startup company announcement {cur_year}',
                f'"how we" {core2} {cur_year}',
                f'{core2} site:medium.com OR site:substack.com {cur_year}',
            ]
            active_r2 = []
            if use_pai: active_r2.append("Parallel AI")
            if use_exa: active_r2.append("Exa")
            yield sse("phase", {"phase": "search", "status": "running",
                                 "msg": f"Round 2 — only {len(unique_domains_r1)} domains, re-searching {' + '.join(active_r2) or 'no providers'}…"})
            # Respect routing decisions — don't force-enable providers that were routed off
            round2_pai, round2_exa = await asyncio.gather(
                search_parallel_ai(client, round2_queries, requested_count=requested_count) if use_pai else _noop(),
                search_exa(client, round2_queries, max_days=max_days, requested_count=requested_count,
                           include_domains=include_domains or None)                         if use_exa else _noop(),
            )
            all_results = _merge_dedup([all_results, round2_pai, round2_exa])

        # Prefer company blogs/filings over media — float to top
        def _is_media(r: dict) -> bool:
            return r.get("domain", "") in _MEDIA_DOMAINS
        all_results.sort(key=_is_media)

        total_pai       = len(pai_results) + len(round2_pai)
        total_exa       = len(exa_results) + len(round2_exa)
        total_claude    = len(claude_web_results)
        total_sec       = len(sec_results)
        rounds_run      = 2 if (round2_pai or round2_exa) else 1
        pai_mode_label  = "FindAll" if use_findall else "PAI Agentic"
        yield sse("phase", {
            "phase": "search", "status": "done",
            "msg": (
                f"Found {len(all_results)} unique URLs across {rounds_run} round(s) — "
                f"Exa: {total_exa} · Claude Web: {total_claude} · "
                f"{pai_mode_label}: {total_pai} · SEC: {total_sec}"
            ),
            "parallel_ai_count":  total_pai,
            "exa_count":          total_exa,
            "claude_web_count":   total_claude,
            "sec_api_count":      total_sec,
            "findall_mode":       use_findall,
            "unique_urls":        len(all_results),
            "search_rounds":      rounds_run,
        })

        # ── Phase 3: Snippet-first scoring (free — no API call) ──
        # Score ALL results via snippets/highlights before scraping anything.
        # This lets us rank every URL by relevance and only Firecrawl the best ones.
        question_kws  = set(_extract_keywords(question))
        topic_kws_set = set(_extract_topic_keywords(question))
        score_set     = topic_kws_set if len(topic_kws_set) >= 2 else question_kws

        snippet_evidence: list[dict] = []
        for r in all_results:
            if r.get("domain") in _MEDIA_DOMAINS:
                continue
            highlights = r.get("_highlights") or []
            snippet    = r.get("snippet", "")
            rich_text  = " … ".join(highlights) if highlights else snippet
            if not rich_text:
                continue
            snippet_words = set(_extract_keywords(rich_text))
            overlap = len(score_set & snippet_words)
            if overlap >= 0:
                effective_denom = min(len(score_set), 5)
                # Base confidence 0.65 from neural search relevance. Lexical hits add bonus.
                conf = round(min(0.85, 0.65 + (overlap / max(effective_denom, 1) * 0.3)), 2)
                src_dom  = r.get("domain", "")
                is_wire  = src_dom in _NEWS_WIRE_DOMAINS
                co_name  = r.get("company_name") or ""
                if not co_name and not is_wire:
                    # Only derive company name from domain if it's NOT a news wire
                    co_name = src_dom.split('.')[0].replace('-', ' ').title()
                # For news wires / aggregators, leave co_name blank —
                # the AI extractor will fill it in Phase 4.
                # For SEC filings, company_domain is pre-resolved from filing metadata;
                # don't attempt to re-resolve from "sec.gov" URL domain.
                co_domain = r.get("company_domain") or (
                    infer_company_domain(co_name) if src_dom == "sec.gov"
                    else (_resolve_company_domain(co_name, src_dom) if not is_wire else "")
                )
                # Use SEC-specific content type for filings
                content_type = (
                    f"SEC {r.get('_sec_form','filing')}" if r.get("source") == "sec_edgar"
                    else _detect_content_type(r["url"], rich_text)
                )
                snippet_evidence.append({
                    "url":                r["url"],
                    "relevant":           True,
                    "company_name":       co_name,
                    "company_domain":     co_domain,
                    "news_source_domain": src_dom,
                    "content_type":       content_type,
                    "published_date":     r.get("date", ""),
                    "key_excerpt":        rich_text[:600],
                    "summary":            f"Search result snippet from {src_dom} — matched {overlap} keywords",
                    "confidence":         conf,
                    "_from_snippet":      True,
                })

        # Sort snippets by confidence DESC — top ones get full Firecrawl treatment
        snippet_evidence.sort(key=lambda x: x["confidence"], reverse=True)

        # ── Phase 3b: Firecrawl — scrape top N by snippet confidence ─────────
        # Scale the cap to match the user's requested count (min 40, max 120).
        # Skip login-gated domains (LinkedIn, Reddit, Twitter) — Exa highlights
        # are already the full signal; Firecrawl will 403/timeout on these.
        FIRECRAWL_CAP  = min(max(requested_count, 40), 120) if requested_count > 0 else 40
        urls_to_scrape = [
            se["url"] for se in snippet_evidence[:FIRECRAWL_CAP]
            if extract_domain_from_url(se["url"]) not in _SNIPPET_ONLY_DOMAINS
        ]
        snippet_only_count = sum(
            1 for se in snippet_evidence[:FIRECRAWL_CAP]
            if extract_domain_from_url(se["url"]) in _SNIPPET_ONLY_DOMAINS
        )
        yield sse("phase", {"phase": "scraping", "status": "running",
                             "msg": f"Deep-scraping {len(urls_to_scrape)} pages via Firecrawl"
                                    + (f" · {snippet_only_count} platform posts using snippet directly" if snippet_only_count else "") + "…"})
        scraped, fc_pages, pai_extract_pages = await scrape_urls(client, urls_to_scrape)
        pai_extract_cost = COST_PAI_RESULT * pai_extract_pages  # PAI Extract per-page cost

        # ── Startup filter: drop scraped pages with no startup signals ──────────
        # Only applied when the user explicitly asked for startups.
        # We keep pages that either (a) contain a startup signal, or (b) couldn't
        # be scraped — snippet-only results pass through and are filtered later.
        if startup_filter and scraped:
            before = len(scraped)
            scraped = {url: content for url, content in scraped.items()
                       if _is_startup_content(content)}
            dropped = before - len(scraped)
            if dropped:
                print(f"Startup filter: dropped {dropped} non-startup pages, kept {len(scraped)}")

        yield sse("phase", {"phase": "scraping", "status": "done",
                             "msg": (f"Scraped {len(scraped)} pages "
                                     f"(Firecrawl: {fc_pages}, PAI-Extract fallback: {pai_extract_pages}) "
                                     f"— {len(snippet_evidence) - len(urls_to_scrape)} scored from snippets")})

        # ── Phase 4: Evidence extraction + sanity check ──
        url_meta      = {r["url"]: r for r in all_results}
        url_hint_date = {r["url"]: r.get("date", "") for r in all_results}

        pages = [(url, content) for url, content in scraped.items()]
        yield sse("phase", {"phase": "extraction", "status": "running",
                             "msg": f"Extracting & sanity-checking {len(snippet_evidence)} results (date filter: last {max_days}d)…"})

        # Run full content extraction on scraped pages
        loop = asyncio.get_event_loop()
        extractions = await asyncio.gather(*[
            loop.run_in_executor(None, extract_evidence, u, c, question, url_hint_date.get(u, ""))
            for u, c in pages
        ])

        # Merge: Firecrawl full-content extractions take priority over snippets for same URL
        relevant_raw = [e for e in extractions if e.get("relevant") and e.get("key_excerpt")]
        scraped_covered = {e.get("url") for e in relevant_raw}
        for se in snippet_evidence:
            if se["url"] not in scraped_covered:
                relevant_raw.append(se)

        # ── Sanity check: enforce recency + quality ──
        relevant, rejected = sanity_check_evidence(relevant_raw, max_days=max_days)
        date_unverified    = [r for r in relevant if r.get("date_unverified")]

        # ── Intelligence Layer: Subject Entity Extraction ──
        yield sse("phase", {"phase": "extraction", "status": "running", "msg": "Using AI to extract the true subject companies, persons, and precise signals…"})
        ai_entities, entity_extraction_cost = await extract_subjects_with_ai(
            client, relevant, question, revops_context=revops_context
        )
        llm_cost += entity_extraction_cost

        # Group by canonical company domain
        evidence_by_company: dict[str, list[dict]] = {}
        for ev in relevant:
            meta       = url_meta.get(ev["url"], {})
            url_domain = extract_domain_from_url(ev["url"])

            # Use AI Extractor as the absolute source of truth if available (except for SEC filings)
            is_sec = meta.get("source") == "sec_edgar"
            ai_data = ai_entities.get(ev["url"], {}) if not is_sec else {}
            
            ai_company = ai_data.get("subject_company", "")
            if ai_company:
                if ai_company.lower() == "unknown":
                    co_name = "unknown"
                    co_domain = "unknown"
                else:
                    co_name = ai_company
                    co_domain = ai_data.get("subject_domain") or _resolve_company_domain(co_name, url_domain)
            else:
                co_name = meta.get("company_name") or ev.get("company_name") or ""
                co_domain = meta.get("company_domain") or _resolve_company_domain(co_name, url_domain)

            # News source = the URL's actual domain (the page we fetched)
            news_src = url_domain

            # Fall back: if co_name still empty, derive from resolved domain
            if not co_name or co_name == "unknown":
                co_name = co_domain.split('.')[0].replace('-', ' ').title() if co_domain and co_domain != "unknown" else "unknown"

            # ── Fix 1: Skip entries where we still can't identify the company ──
            # "unknown" company/domain means it's a generic roundup or news outlet
            # with no identifiable subject — never show these as a company row.
            if co_name == "unknown" or co_domain == "unknown" or not co_name:
                continue
            # Also skip if the subject domain is a news wire / aggregator
            if url_domain in _NEWS_WIRE_DOMAINS and (not ai_data or not ai_data.get("subject_company")):
                continue

            domain    = co_domain or url_domain
            canonical = entity_resolver.resolve(co_name, domain) or domain
            if co_name:
                entity_resolver.learn(canonical, co_name)

            aligned_signal = ai_data.get("aligned_signal") or ev.get("signal", "")
            person_name    = ai_data.get("person_name", "")
            person_title   = ai_data.get("person_title", "")

            # signal = AI-crafted precise statement (with proof URL baked in)
            # proof_url = always the raw URL for direct click-through
            record = {
                "url":                ev["url"],
                "proof_url":          ev["url"],
                "title":              meta.get("title", ev.get("url", "")),
                "date":               ev.get("published_date") or meta.get("date", ""),
                "published_date":     ev.get("published_date") or meta.get("date", ""),
                "content_type":       ev.get("content_type", ""),
                "key_excerpt":        ev.get("key_excerpt", ""),
                # signal = the precise, human-readable statement (AI > regex fallback)
                "signal":             aligned_signal if aligned_signal else ev.get("signal", ev.get("key_excerpt", "")[:200]),
                "person_name":        person_name,
                "person_title":       person_title,
                "confidence":         float(ev.get("confidence", 0.5)),
                "discovered_via":     meta.get("source", "unknown"),
                "snippet":            meta.get("snippet", ""),
                "company_name":       co_name,
                "company_domain":     co_domain,
                "news_source_domain": news_src,
            }
            if record["confidence"] >= 0.70:
                evidence_by_company.setdefault(canonical, []).append(record)

        # Sort each company's evidence by confidence DESC
        for items in evidence_by_company.values():
            items.sort(key=lambda x: x["confidence"], reverse=True)

        yield sse("phase", {
            "phase": "extraction", "status": "done",
            "msg": (
                f"Found {len(relevant)} pieces of evidence across {len(evidence_by_company)} companies "
                f"({len(rejected)} rejected by sanity check, {len(date_unverified)} with unverified dates)"
            ),
            "relevant_pages":   len(relevant),
            "companies_found":  len(evidence_by_company),
            "sanity_rejected":  len(rejected),
            "date_unverified":  len(date_unverified),
            "max_days_filter":  max_days,
        })

        # ── Phase 5: Synthesize answer ──
        yield sse("phase", {"phase": "synthesis", "status": "running",
                             "msg": "Synthesizing tiered intelligence report with Claude Sonnet 3.5…"})
        answer, synthesis_cost = await synthesize_answer(
            client, question, evidence_by_company, revops_context=revops_context
        )
        llm_cost += synthesis_cost
        yield sse("phase", {"phase": "synthesis", "status": "done", "msg": f"Intelligence report ready (${synthesis_cost:.4f})"})

        # ── Final result ──
        elapsed = round(time.time() - t0, 1)

        # Build the companies list for the response
        companies_list = []
        for domain, items in sorted(
            evidence_by_company.items(),
            key=lambda kv: max(i["confidence"] for i in kv[1]),
            reverse=True,
        ):
            companies_list.append({
                "company_name":   items[0].get("company_name") or domain,
                "company_domain": domain,
                "evidence_count": len(items),
                "evidence":       items,
                # Bubble up the best signal + person for the top-level row
                "signal":         items[0].get("signal", ""),
                "proof_url":      items[0].get("proof_url", items[0].get("url", "")),
                "person_name":    items[0].get("person_name", ""),
                "person_title":   items[0].get("person_title", ""),
                "published_date": items[0].get("published_date", ""),
            })

        # Generate dynamic follow-ups grounded in actual results
        top_company_names = [c["company_name"] for c in companies_list[:6]]
        top_signals       = [c.get("signal", "")[:120] for c in companies_list[:5] if c.get("signal")]
        follow_ups = await generate_follow_ups(client, question, top_company_names, top_signals=top_signals)

        yield sse("done", {
            "question":         question,
            "answer":           answer,
            "companies":        companies_list,
            "total_companies":  len(companies_list),
            "total_articles":   sum(len(c["evidence"]) for c in companies_list),
            "elapsed":          elapsed,
            "follow_ups":       follow_ups,
            # RevOps intent context — frontend uses output_columns to render the table
            "revops_context": {
                "buyer_persona":  revops_context.get("buyer_persona"),
                "use_case":       revops_context.get("use_case"),
                "signal_focus":   revops_context.get("signal_focus"),
                "output_columns": revops_context.get("output_columns"),
            },
            "pipeline_stats": {
                "queries_generated":    len(queries),
                "urls_discovered":      len(all_results),
                "pages_scraped":        len(scraped),
                "pages_with_evidence":  len(relevant),
                "exa_urls":             len(exa_results),
                "claude_web_urls":      len(claude_web_results),
                "parallel_ai_urls":     len(pai_results),
                "findall_mode":         use_findall,
            },
            "sanity_report": {
                "max_days_filter":   max_days,
                "passed":            len(relevant),
                "rejected":          len(rejected),
                "date_unverified":   len(date_unverified),
                "rejection_reasons": [r.get("sanity_fail","") for r in rejected[:10]],
            },
            "search_queries_used": queries,
        })

        
        # Log to DB — Dynamic Per-Provider Costs
        # Exa: now 1 call (+ 1 for Round 2 if triggered) via additionalQueries
        total_exa_calls = (1 if use_exa else 0) + (1 if round2_exa else 0)
        total_q_pai = (len(queries) if use_pai else 0) + (len(round2_queries) if round2_pai else 0)
        # SEC now uses _build_sec_queries → always 3 queries max
        total_q_sec = 3 if use_sec else 0

        # Exact cost = Base rate per request + rate per result page
        exa_cost = (COST_EXA_BASE * total_exa_calls) + (COST_EXA_RESULT * len(exa_results))
        pai_cost = (COST_PAI_BASE * total_q_pai) + (COST_PAI_RESULT * len(pai_results))
        sec_cost = (COST_SEC_BASE * total_q_sec) + (COST_SEC_RESULT * len(sec_results))
        fc_cost  = COST_FC_SUCCESS * fc_pages  # only count true Firecrawl successes
        distinct_domains = len(evidence_by_company)

        log_query_metrics(
            question, routing_decisions, llm_cost,
            exa_cost, pai_cost, sec_cost, fc_cost,
            distinct_domains, len(relevant),
            fc_pages_scraped=fc_pages,
            pai_extract_cost=pai_extract_cost,
        )

# ══════════════════════════════════════════════════════════════════════════════
# CLOSED-SOURCE SIGNAL PIPELINE
# No LLM calls — pure ClickHouse + Exa queries, template-based output.
# ══════════════════════════════════════════════════════════════════════════════

import re as _re

CH_URL  = "http://20.185.50.64:8123/"
CH_USER = "reo_readonly_user"
CH_PASS = "fghjkvbnwe4567DFGH"

# ── Pydantic models ───────────────────────────────────────────────────────────
class ICPParams(BaseModel):
    tech_stack:    list[str] = ["kubernetes", "terraform", "go"]
    min_engineers: int       = 50
    verticals:     list[str] = ["fintech", "dev-infra"]
    stage:         str       = "series-b+"

class ClosedSourceRequest(BaseModel):
    query:         str
    icp:           ICPParams  = ICPParams()
    playbook_type: Optional[str] = None
    signal_type:   Optional[str] = None

# ── ClickHouse helper ─────────────────────────────────────────────────────────
async def ch_query(client: httpx.AsyncClient, sql: str) -> list[dict]:
    try:
        r = await client.post(
            CH_URL,
            params={"user": CH_USER, "password": CH_PASS, "default_format": "JSON"},
            content=sql,
            timeout=20,
        )
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception:
        return []

# ── Step 1: Seed ──────────────────────────────────────────────────────────────
async def seed_accounts(client: httpx.AsyncClient, icp: ICPParams) -> list[dict]:
    slugs_sql = ", ".join(f"'{s.lower()}'" for s in icp.tech_stack)
    # org_jobs_inter is the ClickHouse-native jobs table; filter junk + mega-enterprises
    sql = f"""
SELECT company_domain, any(company_name) AS company_name,
       countIf(is_active = 1) AS open_roles,
       any(company_employee_count) AS emp_count
FROM org_jobs_inter
WHERE hasAny(all_skills_slugs, [{slugs_sql}])
  AND is_active = 1
  AND job_posted_date >= now() - INTERVAL 60 DAY
  AND is_known_domain = 1
  AND company_employee_count BETWEEN 50 AND 500
  AND company_domain NOT IN (
      'unknown.com','dice.com','indeed.com','linkedin.com','glassdoor.com',
      'tcs.com','infosys.com','wipro.com','cognizant.com','capgemini.com',
      'accenture.com','deloitte.com','kpmg.com','pwc.com','ey.com',
      'careersatagoda.com', 'vodafone.com', 'nvidia.com',
      'hcltech.com','genpact.com','cgi.com','ntt.com','services.global.ntt',
      'hackajob.com','free-work.com','mygwork.com','efinancialcareers.com',
      'globallogic.com','ciandt.com','boozallen.com','boozallen.co',
      'leidos.com','saic.com','luxoft.com','epam.com','thoughtworks.com'
  )
GROUP BY company_domain
HAVING open_roles >= 2
ORDER BY open_roles DESC
LIMIT 150
"""
    rows = await ch_query(client, sql)
    return [{"domain": r["company_domain"],
             "name":   r.get("company_name", ""),
             "open_roles": int(r.get("open_roles", 0))}
            for r in rows if r.get("company_domain")]

# ── Step 2: Intent ────────────────────────────────────────────────────────────
async def enrich_profiles(client: httpx.AsyncClient, domains: list[str]) -> dict[str, dict]:
    if not domains:
        return {}
    dom_sql = ", ".join(f"'{d}'" for d in domains[:200])
    # Real column: position_title (not job_title); seniority values: Director/VP/CXO/Head/Executive
    sql = f"""
SELECT company_domain,
       countIf(is_current = 1) AS total_tech,
       countIf(is_current = 1 AND start_date >= today() - INTERVAL 180 DAY) AS new_hires_6m,
       countIf(is_current = 1 AND seniority IN
           ('Director','VP','CXO','Head','Executive','Senior IC','Staff/Principal IC')) AS senior_count
FROM profile_positions_enriched
WHERE company_domain IN ({dom_sql})
GROUP BY company_domain
"""
    rows = await ch_query(client, sql)
    return {r["company_domain"]: {
        "total_tech":   int(r.get("total_tech", 0)),
        "new_hires_6m": int(r.get("new_hires_6m", 0)),
        "senior_count": int(r.get("senior_count", 0)),
    } for r in rows}

async def enrich_snapshots(client: httpx.AsyncClient, domains: list[str]) -> dict[str, dict]:
    if not domains:
        return {}
    dom_sql = ", ".join(f"'{d}'" for d in domains[:200])
    sql = f"""
SELECT company_domain,
       argMax(ai_ml_count, snapshot_month) AS ai_ml_now,
       argMin(ai_ml_count, snapshot_month) AS ai_ml_6m,
       (argMax(ai_ml_count, snapshot_month) - argMin(ai_ml_count, snapshot_month)) AS abs_change
FROM company_tech_monthly_snapshots
WHERE company_domain IN ({dom_sql})
  AND snapshot_month >= today() - INTERVAL 6 MONTH
GROUP BY company_domain
"""
    rows = await ch_query(client, sql)
    out: dict[str, dict] = {}
    for r in rows:
        now_ = int(r.get("ai_ml_now", 0))
        ago_ = int(r.get("ai_ml_6m", 0))
        delta = int(r.get("abs_change", 0))
        pct = round(delta * 100.0 / ago_, 1) if ago_ > 0 else 0.0
        out[r["company_domain"]] = {
            "ai_ml_now":  now_,
            "ai_ml_6m":   ago_,
            "abs_change": delta,
            "growth_pct": pct if (now_ >= 5 and delta >= 2) else 0.0,
        }
    return out

# ── Step 3: Triggers ──────────────────────────────────────────────────────────
async def find_senior_hires(client: httpx.AsyncClient, domains: list[str]) -> dict[str, dict]:
    if not domains:
        return {}
    dom_sql = ", ".join(f"'{d}'" for d in domains[:200])
    # Use real column 'position_title'; real seniority values: VP/CXO/Head/Director/Executive
    sql = f"""
SELECT company_domain,
       any(full_name)      AS exec_name,
       any(position_title) AS exec_title,
       max(start_date)     AS hired_date
FROM profile_positions_enriched
WHERE company_domain IN ({dom_sql})
  AND is_current = 1
  AND seniority IN ('VP','CXO','Head','Director','Executive')
  AND lower(position_title) NOT LIKE '%advisor%'
  AND start_date >= today() - INTERVAL 90 DAY
GROUP BY company_domain
"""
    rows = await ch_query(client, sql)
    out: dict[str, dict] = {}
    for r in rows:
        if not r.get("exec_name"):
            continue
        hired = r.get("hired_date", "")
        days_ago = "recently"
        if hired:
            try:
                from datetime import date as _date
                d = _date.fromisoformat(str(hired)[:10])
                days_ago = f"{(_date.today() - d).days} days ago"
            except Exception:
                pass
        out[r["company_domain"]] = {
            "name": r["exec_name"], "title": r["exec_title"], "hired_days": days_ago
        }
    return out

async def exa_news_domains(client: httpx.AsyncClient, terms: list[str]) -> set[str]:
    query = " ".join(terms[:6])
    try:
        r = await client.post(
            "https://api.exa.ai/search",
            headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
            json={"query": query, "category": "news", "numResults": 20,
                  "highlights": {"numSentences": 1, "highlightsPerUrl": 1}},
            timeout=10,
        )
        r.raise_for_status()
        return {extract_domain_from_url(i.get("url", ""))
                for i in r.json().get("results", [])
                if i.get("url")} - {""}
    except Exception:
        return set()

# ── Step 4: Contacts ──────────────────────────────────────────────────────────
async def best_contacts(client: httpx.AsyncClient, domains: list[str]) -> dict[str, dict]:
    if not domains:
        return {}
    dom_sql = ", ".join(f"'{d}'" for d in domains)
    # Use position_title (not job_title); filter by tech_functions array; real seniority values
    sql = f"""
SELECT company_domain,
       any(full_name)      AS name,
       any(position_title) AS title,
       any(start_date)     AS start_date,
       any(seniority)      AS seniority
FROM profile_positions_enriched
WHERE company_domain IN ({dom_sql})
  AND is_current = 1
  AND (
      lower(position_title) LIKE '%vp engineering%' OR 
      lower(position_title) LIKE '%head of platform%' OR 
      lower(position_title) LIKE '%cto%' OR 
      lower(position_title) LIKE '%director of engineering%' OR
      (seniority IN ('VP','CXO','Head','Director','Executive','Senior IC','Staff/Principal IC')
       AND hasAny(tech_functions, ['Engineering Leadership','Cloud & Infrastructure',
           'DevOps Platform & Reliability','Backend Engineering','Full Stack',
           'Artificial Intelligence & Machine Learning','Infrastructure & IT Operations']))
  )
GROUP BY company_domain
"""
    rows = await ch_query(client, sql)
    out: dict[str, dict] = {}
    for r in rows:
        d = r.get("company_domain", "")
        if d in out or not r.get("name"):
            continue
        sd = r.get("start_date", "")
        cstart = "Active role"
        if sd:
            try:
                from datetime import date as _date
                dt = _date.fromisoformat(str(sd)[:10])
                months = (_date.today().year - dt.year)*12 + (_date.today().month - dt.month)
                cstart = f"{months}mo tenure" if months > 3 else f"{(_date.today()-dt).days}d tenure"
            except Exception:
                pass
        out[d] = {"name": r["name"], "title": r["title"], "cstart": cstart}
    return out

# ── Templates (zero LLM cost) ─────────────────────────────────────────────────
_EXEC_KW = ("cto","vp engineering","vp of engineering","head of platform",
            "head of engineering","chief technology","vp infrastructure","chief architect",
            "chief executive","chief product","vp product","head of product")

def _is_exec(title: str) -> bool:
    t = title.lower()
    return any(x in t for x in _EXEC_KW)

def _domain_to_name(domain: str) -> str:
    n = domain.split(".")[0]
    n = _re.sub(r'([a-z])([A-Z])', r'\1 \2', n).replace("-", " ").replace("_", " ")
    return n.title()

def _make_insight(co: dict) -> str:
    snap = co.get("snap", {})
    prof = co.get("prof", {})
    parts: list[str] = []
    if snap.get("growth_pct", 0) > 0:
        parts.append(f"AI/ML team grew {snap['growth_pct']:.0f}% in 6mo "
                     f"({snap['ai_ml_6m']}→{snap['ai_ml_now']})")
    if prof.get("new_hires_6m", 0) > 0:
        parts.append(f"{prof['new_hires_6m']} new tech hires in 6mo")
    if co.get("hire"):
        h = co["hire"]
        parts.append(f"New {h['title']} ({h['name']}) joined {h['hired_days']}")
    if prof.get("total_tech", 0) > 0:
        parts.append(f"{prof['total_tech']} total tech employees")
    if co.get("open_roles", 0) > 0:
        parts.append(f"{co['open_roles']} open roles matching ICP tech stack")
    return ". ".join(parts[:3]) + "." if parts else "ICP tech stack match detected."

def _make_angle(co: dict) -> str:
    snap = co.get("snap", {})
    prof = co.get("prof", {})
    if co.get("hire") and _is_exec(co["hire"].get("title", "")):
        h = co["hire"]
        return (f"New {h['title']} joined {h['hired_days']} — "
                "prime window to present before tooling decisions lock in.")
    if snap.get("growth_pct", 0) >= 20:
        return (f"AI/ML team up {snap['growth_pct']:.0f}% — tooling complexity is "
                "outpacing the team. Lead with consolidation story.")
    if co.get("in_news"):
        return ("Recent news signal = active budget cycle. "
                "Lead with how peers at this stage reduce infra overhead.")
    if prof.get("new_hires_6m", 0) >= 10:
        return (f"{prof['new_hires_6m']} new hires in 6mo means onboarding pain. "
                "Lead with developer velocity and ramp time.")
    return "ICP stack match — reach out with a use case tailored to their tech profile."

def _detect_sigs(co: dict) -> list[str]:
    sigs = ["tech_shop"]
    if co.get("open_roles", 0) >= 3:
        sigs.append("tech_hiring")
    if co.get("snap", {}).get("growth_pct", 0) > 0:
        sigs.append("tech_scaling")
    if co.get("prof", {}).get("new_hires_6m", 0) >= 5:
        sigs.append("team_scaling")
    if co.get("prof", {}).get("total_tech", 0) >= 50:
        sigs.append("headcount_fit")
    if co.get("hire"):
        sigs.append("senior_hire")
    return list(dict.fromkeys(sigs))

def _score(co: dict, icp: ICPParams) -> int:
    snap = co.get("snap", {})
    prof = co.get("prof", {})
    s  = min(40, co.get("open_roles", 0) * 2)
    s += min(25, int(snap.get("growth_pct", 0)))
    s += min(20, prof.get("new_hires_6m", 0) * 2)
    s += 15 if co.get("in_news") else 0
    if prof.get("total_tech", 0) >= icp.min_engineers:
        s += 5
    if co.get("hire") and _is_exec(co["hire"].get("title", "")):
        s = min(100, s + 10)
    return min(100, s)

# ── Pipeline ──────────────────────────────────────────────────────────────────
async def run_closed_source_pipeline(
    query: str, icp: ICPParams,
    playbook_type: Optional[str], signal_type: Optional[str],
) -> AsyncGenerator[str, None]:
    import time as _time
    t0 = _time.time()
    yield sse("start", {"msg": "Initialising closed-source signal pipeline…"})

    async with httpx.AsyncClient(verify=False, timeout=20) as client:

        # Phase 1 — SEED
        yield sse("phase", {"phase": "seed", "status": "running",
                             "msg": f"Querying org_jobs for stack: {', '.join(icp.tech_stack[:3])}…"})
        seed = await seed_accounts(client, icp)
        domains = [c["domain"] for c in seed]
        yield sse("phase", {"phase": "seed", "status": "done",
                             "msg": f"Found {len(seed)} candidate accounts", "count": len(seed)})

        # Phase 2 — INTENT
        yield sse("phase", {"phase": "intent", "status": "running",
                             "msg": f"Analysing headcount trends across {len(domains)} accounts…"})
        prof_map, snap_map = await asyncio.gather(
            enrich_profiles(client, domains),
            enrich_snapshots(client, domains),
        )
        filtered = []
        for c in seed:
            p = prof_map.get(c["domain"], {})
            s = snap_map.get(c["domain"], {})
            if (p.get("new_hires_6m", 0) > 0 or s.get("growth_pct", 0) > 0
                    or p.get("total_tech", 0) >= icp.min_engineers
                    or c.get("open_roles", 0) >= 2):
                filtered.append({**c, "prof": p, "snap": s})
        if not filtered:
            filtered = [{**c, "prof": prof_map.get(c["domain"], {}),
                         "snap": snap_map.get(c["domain"], {})} for c in seed[:50]]
        yield sse("phase", {"phase": "intent", "status": "done",
                             "msg": f"{len(filtered)} accounts show active momentum"})

        # Phase 3 — TRIGGER
        yield sse("phase", {"phase": "trigger", "status": "running",
                             "msg": "Scanning for senior hires (last 90d) + news signals…"})
        trig_doms = [c["domain"] for c in filtered]
        exa_terms = icp.tech_stack[:3] + icp.verticals[:2] + ["Series B", "funding"]
        hire_map, news_doms = await asyncio.gather(
            find_senior_hires(client, trig_doms),
            exa_news_domains(client, exa_terms),
        )
        for c in filtered:
            c["hire"]    = hire_map.get(c["domain"])
            c["in_news"] = c["domain"] in news_doms
        yield sse("phase", {"phase": "trigger", "status": "done",
                             "msg": f"{len(hire_map)} leadership changes, "
                                    f"{sum(1 for c in filtered if c['in_news'])} news signals"})

        # Phase 4 — SCORE
        yield sse("phase", {"phase": "score", "status": "running",
                             "msg": "Computing composite signal scores…"})
        for c in filtered:
            c["score"] = _score(c, icp)
        top = sorted(filtered, key=lambda x: x["score"], reverse=True)[:25]
        yield sse("phase", {"phase": "score", "status": "done",
                             "msg": f"Top {len(top)} accounts ranked by composite score"})

        # Phase 5 — ENRICH
        yield sse("phase", {"phase": "enrich", "status": "running",
                             "msg": "Finding best contacts per account…"})
        contact_map = await best_contacts(client, [c["domain"] for c in top])
        yield sse("phase", {"phase": "enrich", "status": "done",
                             "msg": "Contacts and outreach angles ready"})

        for c in top:
            d = c["domain"]
            ct = contact_map.get(d)
            if not ct and c.get("hire"):
                ct = {"name": c["hire"]["name"],
                      "title": c["hire"]["title"],
                      "cstart": c["hire"]["hired_days"]}
            ct = ct or {"name": "Engineering Lead",
                        "title": "Head of Engineering", "cstart": "Active role"}
            sigs = _detect_sigs(c)
            srcs: list[str] = []
            if c.get("open_roles", 0) > 0:         srcs.append("Jobs")
            if c.get("prof", {}).get("total_tech"): srcs.append("Headcount")
            if c.get("snap", {}).get("growth_pct"): srcs.append("Team growth")
            if c.get("hire"):                        srcs.append("Leadership")
            if c.get("in_news"):                     srcs.append("News")
            yield sse("result", {
                "name":    c.get("name") or _domain_to_name(d),
                "dom":     d,
                "sigs":    sigs,
                "score":   c["score"],
                "insight": _make_insight(c),
                "contact": ct["name"],
                "ctitle":  ct["title"],
                "cstart":  ct["cstart"],
                "angle":   _make_angle(c),
                "sources": srcs or ["Stack detection"],
            })

        elapsed = round(_time.time() - t0, 1)
        yield sse("done", {
            "total":            len(top),
            "elapsed":          elapsed,
            "high_score_count": sum(1 for c in top if c["score"] >= 80),
            "trigger_count":    sum(1 for c in top if c.get("hire") or c.get("in_news")),
        })


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Public Intelligence Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class AskRequest(BaseModel):
    question: str

class SecTestRequest(BaseModel):
    query: str
    forms: str = "8-K,S-1,10-K,10-Q"
    max_days: int = 365


@app.post("/api/ask")
async def ask_endpoint(body: AskRequest):
    return StreamingResponse(
        run_ask_pipeline(body.question),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/closed-source")
async def closed_source_endpoint(body: ClosedSourceRequest):
    return StreamingResponse(
        run_closed_source_pipeline(body.query, body.icp, body.playbook_type, body.signal_type),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/api/test-sec")
async def test_sec_endpoint(body: SecTestRequest):
    limits = httpx.Limits(max_connections=5)
    async with httpx.AsyncClient(verify=False, limits=limits) as client:
        results = await search_sec_edgar(client, [body.query], body.forms, body.max_days)
        return {"results": results}

# ── GitHub presence checker ───────────────────────────────────────────────────
GITHUB_RE     = re.compile(r'github\.com/[a-zA-Z0-9_\-]', re.IGNORECASE)
GITHUB_URL_RE = re.compile(r'https?://github\.com/[a-zA-Z0-9_\-/]+', re.IGNORECASE)

class GithubCheckRequest(BaseModel):
    domains: list[str]

async def _check_one_domain(client: httpx.AsyncClient, domain: str) -> dict:
    """Scrape a domain homepage and detect GitHub presence."""
    url = f"https://{domain}"
    try:
        r = await client.post(
            f"{FIRECRAWL_URL}/scrape",
            headers={"Authorization": f"Bearer {FIRECRAWL_KEY}", "Content-Type": "application/json"},
            json={
                "url": url,
                "formats": ["markdown", "links"],
                "onlyMainContent": False,
            },
            timeout=25,
        )
        data = r.json()
        if not data.get("success"):
            return {"domain": domain, "has_github": False, "github_url": None, "error": "scrape_failed"}

        content  = data.get("data", {})
        markdown = content.get("markdown", "")
        links    = content.get("links", [])

        github_links = [l for l in links if "github.com/" in l.lower()
                        and not any(x in l.lower() for x in ["github.com/login", "github.com/signup"])]

        text_hit = bool(GITHUB_RE.search(markdown))

        github_url = None
        if github_links:
            github_url = github_links[0]
        else:
            m = GITHUB_URL_RE.search(markdown)
            if m:
                github_url = m.group(0)

        return {
            "domain":     domain,
            "has_github": bool(github_links or text_hit),
            "github_url": github_url,
        }
    except Exception as e:
        return {"domain": domain, "has_github": False, "github_url": None, "error": str(e)}

@app.post("/api/check-github")
async def check_github_endpoint(body: GithubCheckRequest):
    """Check up to 30 domains in parallel for GitHub presence on their homepage."""
    domains = list(dict.fromkeys(body.domains))[:30]  # dedupe + cap at 30
    limits  = httpx.Limits(max_connections=10, max_keepalive_connections=5)
    async with httpx.AsyncClient(verify=False, limits=limits) as client:
        results = await asyncio.gather(*[
            _check_one_domain(client, d) for d in domains
        ])
    return {"results": list(results)}


@app.get("/api/dashboard-stats")
async def dashboard_stats():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("""SELECT COUNT(*) as cnt,
                COALESCE(SUM(total_cost),0) as spend,
                COALESCE(SUM(llm_cost),0) as llm,
                COALESCE(SUM(search_cost),0) as search,
                COALESCE(SUM(exa_cost),0) as exa,
                COALESCE(SUM(pai_cost),0) as pai,
                COALESCE(SUM(sec_cost),0) as sec,
                COALESCE(SUM(fc_cost),0) as fc,
                COALESCE(SUM(pai_extract_cost),0) as pai_extract,
                COALESCE(SUM(fc_pages_scraped),0) as fc_pages,
                COALESCE(SUM(distinct_domains),0) as domains
                FROM queries""")
            row = cur.fetchone()
            cur.execute("SELECT tools_used FROM queries")
            exa_count = pai_count = sec_count = 0
            for r in cur.fetchall():
                try:
                    t = json.loads(r["tools_used"])
                    if t.get("exa"): exa_count += 1
                    if t.get("parallel_ai"): pai_count += 1
                    if t.get("sec_api"): sec_count += 1
                except: pass
            cur.execute("SELECT COUNT(*) as cnt, COALESCE(SUM(total_cost),0) as spend FROM queries WHERE DATE(timestamp) = DATE('now')")
            today = cur.fetchone()
            return {
                "total_queries": row["cnt"],
                "total_spend": round(row["spend"], 4),
                "total_llm_cost": round(row["llm"], 4),
                "total_search_cost": round(row["search"], 4),
                "total_distinct_domains": int(row["domains"]),
                "today_queries": today["cnt"],
                "today_spend": round(today["spend"], 4),
                "tool_usage": {"exa": exa_count, "parallel_ai": pai_count, "sec_api": sec_count},
                "total_fc_pages_scraped": int(row["fc_pages"]),
                "tool_costs": {
                    "exa":         round(row["exa"], 4),
                    "pai":         round(row["pai"], 4),
                    "sec":         round(row["sec"], 4),
                    "fc":          round(row["fc"], 4),
                    "pai_extract": round(row["pai_extract"], 4),
                    "llm":         round(row["llm"], 4),
                },
            }
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/dashboard-logs")
async def dashboard_logs():
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute("SELECT * FROM queries ORDER BY id DESC LIMIT 50")
            rows = cur.fetchall()
            return {"logs": [dict(r) for r in rows]}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/health")
async def health():
    return {"status": "ok", "mode": "public_internet_only"}

# Serve static files
app.mount("/", StaticFiles(directory="public", html=True), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 4001))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
