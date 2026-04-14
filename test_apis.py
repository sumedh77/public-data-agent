import asyncio
import httpx
import json

EXA_API_KEY = "ee756ef9-ead3-4cf2-ac12-472f2b772d4b"
PARALLEL_AI_KEY = "aT2x_g3JX3rXyoDrZc5nS5RoaiQl29UEpSiF5z03"

async def run():
    q = "how we secured our AI agents infrastructure"
    async with httpx.AsyncClient() as client:
        # Exa
        print("EXA REQUEST")
        p_exa = {
            "query": q,
            "type": "neural",
            "num_results": 1,
            "startPublishedDate": "2026-01-01T00:00:00.000Z",
            "contents": {
                "highlights": {"numSentences": 3, "highlightsPerUrl": 3}
            }
        }
        print(json.dumps(p_exa))
        print("EXA RESP")
        try:
            r = await client.post("https://api.exa.ai/search", headers={"x-api-key": EXA_API_KEY}, json=p_exa)
            print(json.dumps(r.json())[:800])
        except Exception as e:
            print(e)

        # Parallel
        print("\nPAI REQUEST")
        p_pai = {"objective": q, "mode": "fast", "excerpts": {"max_chars_per_result": 600}}
        print(json.dumps(p_pai))
        print("PAI RESP")
        try:
            r2 = await client.post("https://api.parallel.ai/v1beta/search", headers={"x-api-key": PARALLEL_AI_KEY}, json=p_pai)
            print(json.dumps(r2.json())[:800])
        except Exception as e:
            print(e)
            
asyncio.run(run())
