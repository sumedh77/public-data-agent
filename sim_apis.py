import asyncio
import httpx
import json
import os

EXA_API_KEY = "ee756ef9-ead3-4cf2-ac12-472f2b772d4b"
PARALLEL_AI_KEY = "aT2x_g3JX3rXyoDrZc5nS5RoaiQl29UEpSiF5z03"
FIRECRAWL_KEY = "fc-6d6b3a7613eb49e2aad46098f7beba34"

async def run():
    async with httpx.AsyncClient() as client:
        # Exa
        print("====== EXA SEARCH ======")
        p_exa = {
            "query": "recently funded startups building AI agent security layer",
            "type": "neural",
            "num_results": 2,
            "startPublishedDate": "2025-01-01T00:00:00.000Z",
            "contents": {
                "highlights": {"numSentences": 3, "highlightsPerUrl": 3}
            }
        }
        r_exa = await client.post("https://api.exa.ai/search", headers={"x-api-key": EXA_API_KEY}, json=p_exa)
        exa_data = r_exa.json()
        print(json.dumps(exa_data, indent=2)[:800], "...\n")
        
        target_url = None
        if exa_data.get("results"):
            target_url = exa_data["results"][0]["url"]

        # Parallel
        print("====== PARALLEL AI SEARCH ======")
        p_pai = {"objective": "how we built our AI agent security layer Series A", "mode": "fast", "excerpts": {"max_chars_per_result": 600}}
        r_pai = await client.post("https://api.parallel.ai/v1beta/search", headers={"x-api-key": PARALLEL_AI_KEY}, json=p_pai)
        pai_data = r_pai.json()
        print(json.dumps(pai_data, indent=2)[:800], "...\n")
        
        if not target_url and pai_data.get("results"):
            target_url = pai_data["results"][0]["url"]

        if target_url:
            print("====== FIRECRAWL SCRAPING ======")
            print(f"Scraping: {target_url}")
            p_fc = {"url": target_url, "formats": ["markdown"]}
            r_fc = await client.post("https://api.firecrawl.dev/v1/scrape", headers={"Authorization": f"Bearer {FIRECRAWL_KEY}"}, json=p_fc)
            fc_data = r_fc.json()
            md = (fc_data.get("data") or {}).get("markdown") or ""
            print(md[:700], "...\n")
            
asyncio.run(run())
