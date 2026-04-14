import asyncio
import json
import httpx
ANTHROPIC_API_KEY = "<REDACTED>"
async def run():
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-5-sonnet-20240620",
                    "max_tokens": 1000,
                    "messages": [{"role": "user", "content": "Just say hi."}]
                },
                timeout=60,
            )
            print(resp.status_code)
            print(resp.text)
        except Exception as e:
            print(e)
asyncio.run(run())
