"""Quick 3-way comparison for a single query across Exa, Parallel AI, and both combined."""
import asyncio, re
import httpx

EXA_API_KEY     = "ee756ef9-ead3-4cf2-ac12-472f2b772d4b"
EXA_URL         = "https://api.exa.ai/search"
PARALLEL_AI_KEY = "aT2x_g3JX3rXyoDrZc5nS5RoaiQl29UEpSiF5z03"
PARALLEL_AI_URL = "https://api.parallel.ai"

QUERY = "Find blog posts or case studies where companies wrote about replacing a legacy CRM with a modern alternative"

_MEDIA = {
    "techcrunch.com","bloomberg.com","forbes.com","reuters.com","wsj.com","ft.com",
    "wired.com","venturebeat.com","cnbc.com","theverge.com","zdnet.com",
    "hackernews.com","news.ycombinator.com","linkedin.com","twitter.com","x.com",
    "prnewswire.com","businesswire.com","globenewswire.com","apnews.com",
    "marketwatch.com","finance.yahoo.com","entrepreneur.com","inc.com",
}
_HOSTING = re.compile(r'^(?:www|blog|blogs|news|press|engineering|tech|developers|dev|docs|help|support|insights|resources)\.')

def domain(url):
    m = re.search(r'https?://([^/]+)', url)
    if not m: return ""
    h = re.sub(r'^www\.', '', m.group(1).lower())
    s = _HOSTING.sub('', h)
    return s if '.' in s else h

def is_company(d): return d and d not in _MEDIA

import datetime
START = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()

# 3 query variants to give each provider a fair shot
VARIANTS = [
    QUERY,
    'replacing legacy CRM modern alternative company blog "we switched" OR "we migrated" 2024 2025',
    '"how we" CRM migration replacement engineering blog 2024 2025',
]

async def exa(client, q, neural=False):
    try:
        r = await client.post(EXA_URL,
            headers={"x-api-key": EXA_API_KEY, "Content-Type": "application/json"},
            json={"query": q, "type": "neural" if neural else "auto", "num_results": 20,
                  "startPublishedDate": START + "T00:00:00.000Z",
                  "contents": {"highlights": {"numSentences": 3, "highlightsPerUrl": 3}}},
            timeout=30)
        r.raise_for_status()
        return [{"url": x.get("url",""), "title": x.get("title",""),
                 "snippet": " … ".join((x.get("highlights") or []))[:400], "src": "exa"}
                for x in r.json().get("results", [])]
    except Exception as e:
        print(f"  [exa err] {e}"); return []

async def parallel(client, q):
    try:
        r = await client.post(f"{PARALLEL_AI_URL}/v1beta/search",
            headers={"x-api-key": PARALLEL_AI_KEY, "Content-Type": "application/json"},
            json={"objective": q, "mode": "fast", "excerpts": {"max_chars_per_result": 600}},
            timeout=30)
        r.raise_for_status()
        return [{"url": x.get("url",""), "title": x.get("title","") or x.get("name",""),
                 "snippet": (x.get("excerpt") or x.get("snippet",""))[:400], "src": "parallel"}
                for x in r.json().get("results", [])]
    except Exception as e:
        print(f"  [parallel err] {e}"); return []

def dedup(results):
    seen = set(); out = []
    for r in results:
        if r["url"] and r["url"] not in seen:
            seen.add(r["url"]); out.append(r)
    return out

def summarise(label, results):
    company_results = [r for r in results if is_company(domain(r["url"]))]
    domains = sorted({domain(r["url"]) for r in company_results})
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  Total URLs: {len(results)}  |  Company domains: {len(domains)}")
    print(f"{'='*60}")
    for d in domains:
        # find the title for this domain
        title = next((r["title"] for r in company_results if domain(r["url"]) == d), "")
        print(f"  • {d:<40} {title[:55]}")
    return set(domains)

async def main():
    print(f"\nQuery: {QUERY!r}\n")
    async with httpx.AsyncClient() as c:
        # Run all fetches in parallel
        exa_tasks     = [exa(c, v, neural=(i%2==1)) for i, v in enumerate(VARIANTS)]
        parallel_tasks = [parallel(c, v) for v in VARIANTS]
        all_results = await asyncio.gather(*exa_tasks, *parallel_tasks)

    exa_raw      = dedup([r for b in all_results[:3] for r in b])
    parallel_raw = dedup([r for b in all_results[3:] for r in b])
    combined_raw = dedup(exa_raw + parallel_raw)

    exa_domains      = summarise("1. EXA ONLY", exa_raw)
    parallel_domains = summarise("2. PARALLEL AI ONLY", parallel_raw)
    combined_domains = summarise("3. BOTH COMBINED", combined_raw)

    only_exa      = exa_domains - parallel_domains
    only_parallel = parallel_domains - exa_domains
    overlap       = exa_domains & parallel_domains

    print(f"\n{'='*60}")
    print(f"  OVERLAP ANALYSIS")
    print(f"{'='*60}")
    print(f"  Exa only:         {len(only_exa)} unique domains")
    print(f"  Parallel only:    {len(only_parallel)} unique domains")
    print(f"  Shared (overlap): {len(overlap)} domains")
    if overlap: print(f"    → {', '.join(sorted(overlap))}")
    print(f"  Combined total:   {len(combined_domains)} domains")
    print(f"\n  Marginal value of adding Parallel to Exa: +{len(only_parallel)} domains ({len(only_parallel)/max(len(combined_domains),1)*100:.0f}% of total)")
    print(f"  Marginal value of adding Exa to Parallel: +{len(only_exa)} domains ({len(only_exa)/max(len(combined_domains),1)*100:.0f}% of total)")

asyncio.run(main())
