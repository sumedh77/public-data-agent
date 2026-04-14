"""
Provider Comparison Evaluation
================================
Runs 8 test queries through Exa, Tavily, and Parallel AI independently,
then scores each provider on relevance/coverage and computes overlap metrics.

Output: provider_eval_results.json
"""
import asyncio
import datetime
import json
import re
import sys
from typing import Optional

import httpx

# ── Re-use config from server.py ──────────────────────────────────────────────
EXA_API_KEY     = "ee756ef9-ead3-4cf2-ac12-472f2b772d4b"
EXA_URL         = "https://api.exa.ai/search"
PARALLEL_AI_KEY = "aT2x_g3JX3rXyoDrZc5nS5RoaiQl29UEpSiF5z03"
PARALLEL_AI_URL = "https://api.parallel.ai"
TAVILY_API_KEY  = "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"
TAVILY_URL      = "https://api.tavily.com/search"

# ── Test queries ──────────────────────────────────────────────────────────────
TEST_QUERIES = [
    "company blogs about AI security breaches",
    "engineering blog posts about scaling infrastructure",
    "SaaS companies announcing upmarket move enterprise",
    "startup blog posts about SOC2 compliance journey",
    "product launch announcements developer tools 2024 2025",
    "fintech company blog about payment fraud detection",
    "engineering retrospectives on incident response",
    "company blog posts about migrating off AWS",
]

# ── Media domains to exclude (copied from server.py) ─────────────────────────
_MEDIA_DOMAINS = {
    "techcrunch.com", "bloomberg.com", "forbes.com", "reuters.com", "wsj.com",
    "ft.com", "wired.com", "venturebeat.com", "sifted.eu", "crunchbase.com",
    "businessinsider.com", "cnbc.com", "theverge.com", "zdnet.com",
    "infoq.com", "hackernews.com", "news.ycombinator.com",
    "linkedin.com", "twitter.com", "x.com",
    "prnewswire.com", "businesswire.com", "globenewswire.com",
    "accesswire.com", "apnews.com", "marketwatch.com",
    "finance.yahoo.com", "seeking alpha.com", "entrepreneur.com", "inc.com",
}

_HOSTING_SUBDOMAINS = re.compile(
    r'^(?:www|blog|blogs|news|press|media|insights|resources|learn|'
    r'academy|docs|help|support|app|go|get|try|community|forum|'
    r'engineering|tech|developers|dev|api|status|about)\.'
)

_STOP_WORDS = {
    'how', 'many', 'have', 'recently', 'are', 'the', 'a', 'an', 'is', 'was',
    'were', 'that', 'which', 'what', 'who', 'when', 'where', 'why', 'and',
    'or', 'not', 'about', 'in', 'on', 'to', 'for', 'of', 'with', 'do', 'did',
    'has', 'had', 'been', 'be', 'by', 'from', 'at', 'this', 'their', 'its',
    'can', 'could', 'would', 'should', 'will', 'any', 'some', 'all',
}
_INTENT_WORDS = {
    'find', 'finding', 'show', 'list', 'give', 'get', 'want', 'need',
    'companies', 'company', 'startups', 'startup', 'businesses', 'business',
    'published', 'publishing', 'publish', 'wrote', 'write', 'written', 'writing',
    'blog', 'blogs', 'post', 'posts', 'article', 'articles', 'content',
    'announcement', 'announcements', 'saas', 'b2b',
    'days', 'months', 'weeks', 'year', 'years', 'last', 'recent', 'latest',
    'new', 'past', '180', '90', '60', '30', '365',
}

_URL_CONTENT_TYPE_HINTS = [
    (r'/blog/', 'blog post'), (r'/posts?/', 'blog post'), (r'/news/', 'announcement'),
    (r'/press/', 'announcement'), (r'/whitepaper', 'whitepaper'),
    (r'/case-stud', 'case study'), (r'/guide', 'guide'), (r'/tutorial', 'guide'),
    (r'/engineering/', 'blog post'), (r'/insights/', 'blog post'),
    (r'/security/', 'security'), (r'/incident', 'incident report'),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_domain_from_url(url: str) -> str:
    m = re.search(r'https?://([^/]+)', url)
    if not m:
        return ""
    host = m.group(1).lower()
    host = re.sub(r'^www\.', '', host)
    stripped = _HOSTING_SUBDOMAINS.sub('', host)
    return stripped if '.' in stripped else host


def _extract_topic_keywords(text: str) -> list[str]:
    words = re.sub(r'[^\w\s]', '', text.lower()).split()
    return [w for w in words
            if w not in _STOP_WORDS and w not in _INTENT_WORDS
            and len(w) > 2 and not w.isdigit()]


def score_snippet_confidence(snippet: str, query: str) -> float:
    """Keyword overlap score between snippet and query topic keywords."""
    if not snippet:
        return 0.0
    topic_kws = set(_extract_topic_keywords(query))
    if not topic_kws:
        return 0.0
    snippet_words = set(re.sub(r'[^\w\s]', '', snippet.lower()).split())
    overlap = len(topic_kws & snippet_words)
    return min(overlap / max(len(topic_kws), 6) * 1.4, 0.95)


def detect_content_type(url: str) -> str:
    url_lower = url.lower()
    for pattern, ctype in _URL_CONTENT_TYPE_HINTS:
        if re.search(pattern, url_lower):
            return ctype
    return 'article'


def is_company_domain(domain: str) -> bool:
    return domain not in _MEDIA_DOMAINS


# ── Single-query search functions (each provider, 1 query) ───────────────────

async def _exa_single(client: httpx.AsyncClient, query: str, use_neural: bool) -> list[dict]:
    start_date = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()
    try:
        resp = await client.post(
            EXA_URL,
            headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
            json={
                "query": query,
                "type": "neural" if use_neural else "auto",
                "num_results": 20,
                "startPublishedDate": start_date + "T00:00:00.000Z",
                "contents": {"highlights": {"numSentences": 3, "highlightsPerUrl": 3}},
            },
            timeout=30,
        )
        resp.raise_for_status()
        out = []
        for r in resp.json().get("results", []):
            url = r.get("url", "")
            highlights = r.get("highlights") or []
            snippet = " … ".join(h for h in highlights if h)[:800]
            out.append({"url": url, "domain": extract_domain_from_url(url), "snippet": snippet})
        return out
    except Exception as e:
        print(f"  [exa error] {e}", file=sys.stderr)
        return []


async def _tavily_single(client: httpx.AsyncClient, query: str, topic: str) -> list[dict]:
    try:
        resp = await client.post(
            TAVILY_URL,
            headers={"Authorization": f"Bearer {TAVILY_API_KEY}", "Content-Type": "application/json"},
            json={
                "query": query,
                "search_depth": "advanced",
                "max_results": 20,
                "chunks_per_source": 3,
                "topic": topic,
                "time_range": "year",
                "include_answer": False,
                "include_images": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        out = []
        for r in resp.json().get("results", []):
            url = r.get("url", "")
            out.append({"url": url, "domain": extract_domain_from_url(url), "snippet": r.get("content", "")[:800]})
        return out
    except Exception as e:
        print(f"  [tavily error] {e}", file=sys.stderr)
        return []


async def _parallel_single(client: httpx.AsyncClient, query: str) -> list[dict]:
    try:
        resp = await client.post(
            f"{PARALLEL_AI_URL}/v1beta/search",
            headers={"x-api-key": PARALLEL_AI_KEY, "Content-Type": "application/json"},
            json={"objective": query, "mode": "fast", "excerpts": {"max_chars_per_result": 600}},
            timeout=30,
        )
        resp.raise_for_status()
        out = []
        for r in resp.json().get("results", []):
            url = r.get("url", "")
            snippet = (r.get("excerpt") or r.get("snippet") or "")[:500]
            out.append({"url": url, "domain": extract_domain_from_url(url), "snippet": snippet})
        return out
    except Exception as e:
        print(f"  [parallel error] {e}", file=sys.stderr)
        return []


async def fetch_provider_results(client: httpx.AsyncClient, query: str, provider: str) -> list[dict]:
    """Run 3 query variants through a single provider and deduplicate."""
    # Use 3 query formulations to give each provider a fair shot
    topic_kws = _extract_topic_keywords(query)
    core = " ".join(topic_kws[:4]) if topic_kws else query
    cur_year = datetime.date.today().year
    variants = [
        query,
        f"{core} blog post {cur_year}",
        f'"how we" {core} engineering blog',
    ]

    results: list[dict] = []
    if provider == "exa":
        tasks = [_exa_single(client, q, use_neural=(i % 2 == 1)) for i, q in enumerate(variants)]
        batches = await asyncio.gather(*tasks)
        results = [r for b in batches for r in b]
    elif provider == "tavily":
        tasks = [_tavily_single(client, q, topic=("news" if i % 2 == 0 else "general")) for i, q in enumerate(variants)]
        batches = await asyncio.gather(*tasks)
        results = [r for b in batches for r in b]
    elif provider == "parallel_ai":
        tasks = [_parallel_single(client, q) for q in variants]
        batches = await asyncio.gather(*tasks)
        results = [r for b in batches for r in b]

    # Deduplicate by URL
    seen: set[str] = set()
    deduped = []
    for r in results:
        if r["url"] and r["url"] not in seen:
            seen.add(r["url"])
            deduped.append(r)
    return deduped


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_provider_results(results: list[dict], query: str) -> dict:
    total = len(results)
    if total == 0:
        return {
            "total_urls": 0,
            "domains_found": 0,
            "company_first_count": 0,
            "company_first_rate": 0.0,
            "avg_confidence": 0.0,
            "content_types": {},
            "urls": [],
            "domains": [],
        }

    company_results = [r for r in results if is_company_domain(r["domain"])]
    confidences = [score_snippet_confidence(r["snippet"], query) for r in company_results]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    unique_domains = list({r["domain"] for r in company_results if r["domain"]})

    content_types: dict[str, int] = {}
    for r in company_results:
        ct = detect_content_type(r["url"])
        content_types[ct] = content_types.get(ct, 0) + 1

    return {
        "total_urls": total,
        "domains_found": len(unique_domains),
        "company_first_count": len(company_results),
        "company_first_rate": round(len(company_results) / total, 3) if total else 0.0,
        "avg_confidence": round(avg_conf, 3),
        "content_types": content_types,
        "urls": [r["url"] for r in results],
        "domains": unique_domains,
    }


def compute_overlap(provider_results: dict[str, dict]) -> dict:
    url_sets = {p: set(d["urls"]) for p, d in provider_results.items()}
    domain_sets = {p: set(d["domains"]) for p, d in provider_results.items()}
    providers = list(url_sets.keys())

    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 0.0
        return round(len(a & b) / len(a | b), 3)

    overlap = {}

    # Pairwise Jaccard on URLs
    for i in range(len(providers)):
        for j in range(i + 1, len(providers)):
            pa, pb = providers[i], providers[j]
            overlap[f"{pa}_vs_{pb}_url_jaccard"] = jaccard(url_sets[pa], url_sets[pb])
            overlap[f"{pa}_vs_{pb}_domain_jaccard"] = jaccard(domain_sets[pa], domain_sets[pb])

    # Marginal value of each provider (% new domains vs the union of others)
    marginal = {}
    all_domains = set()
    for d in domain_sets.values():
        all_domains |= d

    for p in providers:
        others = set()
        for op, od in domain_sets.items():
            if op != p:
                others |= od
        unique_to_p = domain_sets[p] - others
        marginal[p] = {
            "unique_domains": len(unique_to_p),
            "unique_domain_list": sorted(unique_to_p),
            "marginal_pct": round(len(unique_to_p) / max(len(all_domains), 1) * 100, 1),
        }
    overlap["marginal_value"] = marginal

    return overlap


def recommend(provider_scores: dict[str, dict], overlap: dict) -> str:
    """Simple heuristic recommendation based on marginal value."""
    mv = overlap.get("marginal_value", {})

    # Sort providers by composite score: avg_confidence × company_first_rate × domains_found
    def composite(p: str) -> float:
        s = provider_scores[p]
        return s["avg_confidence"] * s["company_first_rate"] * max(s["domains_found"], 1)

    ranked = sorted(provider_scores.keys(), key=composite, reverse=True)
    best = ranked[0]

    # Check if second provider adds meaningful unique domains (>= 2)
    second = ranked[1] if len(ranked) > 1 else None
    second_unique = mv.get(second, {}).get("unique_domains", 0) if second else 0

    if second_unique >= 2:
        third = ranked[2] if len(ranked) > 2 else None
        third_unique = mv.get(third, {}).get("unique_domains", 0) if third else 0
        if third_unique >= 2:
            return "all_three"
        return f"{best}+{second}"
    return f"{best}_only"


# ── Main evaluation loop ──────────────────────────────────────────────────────

async def evaluate_query(client: httpx.AsyncClient, query: str) -> dict:
    print(f"\n→ Query: {query!r}")

    # Run all 3 providers in parallel
    exa_raw, tavily_raw, parallel_raw = await asyncio.gather(
        fetch_provider_results(client, query, "exa"),
        fetch_provider_results(client, query, "tavily"),
        fetch_provider_results(client, query, "parallel_ai"),
    )

    print(f"  exa={len(exa_raw)} tavily={len(tavily_raw)} parallel={len(parallel_raw)} raw URLs")

    provider_scores = {
        "exa":         score_provider_results(exa_raw, query),
        "tavily":      score_provider_results(tavily_raw, query),
        "parallel_ai": score_provider_results(parallel_raw, query),
    }

    overlap = compute_overlap(provider_scores)
    rec = recommend(provider_scores, overlap)

    print(f"  domains: exa={provider_scores['exa']['domains_found']} "
          f"tavily={provider_scores['tavily']['domains_found']} "
          f"parallel={provider_scores['parallel_ai']['domains_found']} "
          f"→ {rec}")

    return {
        "query": query,
        "providers": provider_scores,
        "overlap": overlap,
        "recommendation": rec,
    }


async def main():
    print("=" * 60)
    print("Search Provider Comparison Evaluation")
    print(f"Queries: {len(TEST_QUERIES)} | Date: {datetime.date.today()}")
    print("=" * 60)

    results = []
    async with httpx.AsyncClient() as client:
        for query in TEST_QUERIES:
            result = await evaluate_query(client, query)
            results.append(result)

    # Build summary
    provider_names = ["exa", "tavily", "parallel_ai"]
    summary: dict = {
        "total_queries": len(results),
        "per_provider": {},
        "recommendations": {},
        "avg_pairwise_url_jaccard": {},
        "avg_pairwise_domain_jaccard": {},
    }

    for p in provider_names:
        avg_domains = sum(r["providers"][p]["domains_found"] for r in results) / len(results)
        avg_conf = sum(r["providers"][p]["avg_confidence"] for r in results) / len(results)
        avg_company_rate = sum(r["providers"][p]["company_first_rate"] for r in results) / len(results)
        avg_marginal = sum(r["overlap"]["marginal_value"][p]["marginal_pct"] for r in results) / len(results)
        summary["per_provider"][p] = {
            "avg_domains_found": round(avg_domains, 1),
            "avg_confidence": round(avg_conf, 3),
            "avg_company_first_rate": round(avg_company_rate, 3),
            "avg_marginal_pct": round(avg_marginal, 1),
        }

    # Recommendation counts
    rec_counts: dict[str, int] = {}
    for r in results:
        rec = r["recommendation"]
        rec_counts[rec] = rec_counts.get(rec, 0) + 1
    summary["recommendations"] = rec_counts

    # Average pairwise Jaccard
    pairs = [("exa", "tavily"), ("exa", "parallel_ai"), ("tavily", "parallel_ai")]
    for pa, pb in pairs:
        key = f"{pa}_vs_{pb}"
        avg_url_j = sum(r["overlap"].get(f"{key}_url_jaccard", 0) for r in results) / len(results)
        avg_dom_j = sum(r["overlap"].get(f"{key}_domain_jaccard", 0) for r in results) / len(results)
        summary["avg_pairwise_url_jaccard"][key] = round(avg_url_j, 3)
        summary["avg_pairwise_domain_jaccard"][key] = round(avg_dom_j, 3)

    output = {"results": results, "summary": summary}

    out_path = "/Users/sumedhbang/Public Data Agent/provider_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for p in provider_names:
        s = summary["per_provider"][p]
        print(f"{p:15} avg_domains={s['avg_domains_found']:5.1f}  "
              f"avg_conf={s['avg_confidence']:.3f}  "
              f"company_rate={s['avg_company_first_rate']:.2f}  "
              f"marginal={s['avg_marginal_pct']:.1f}%")

    print(f"\nRecommendations across {len(results)} queries:")
    for rec, count in sorted(rec_counts.items(), key=lambda x: -x[1]):
        print(f"  {rec}: {count} queries")

    print(f"\nAvg URL overlap (Jaccard):")
    for key, val in summary["avg_pairwise_url_jaccard"].items():
        print(f"  {key}: {val:.3f}")

    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
