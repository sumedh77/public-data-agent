import asyncio
import httpx
import json

ANTHROPIC_API_KEY = "<REDACTED>"

async def run():
    client = httpx.AsyncClient()
    question = "Find me recently funded companies posting about building a security layer for AI agents."
    
    companies_data = [
        {
          "domain": "certiv.com",
          "name": "Certiv",
          "signals": [
            {
              "source": "prnewswire.com",
              "date": "2026-03-16",
              "signal": "Certiv Emerges from Stealth to Launch the First Runtime Assurance Layer for AI Agents",
              "url": "https://www.prnewswire.com/news-releases/certiv-emerges-from-stealth-to-launch-the-first-runtime-assurance-layer-for-ai-agents-302713956.html"
            }
          ]
        },
        {
          "domain": "lakera.ai",
          "name": "Lakera AI",
          "signals": [
            {
              "source": "lakera.ai",
              "date": "2026-02-12",
              "signal": "Lakera Secures $20M Series A to Build Enterprise Security Layer for Autonomous AI Agents",
              "url": "https://www.lakera.ai/news/series-a-funding-ai-agent-security"
            }
          ]
        }
    ]

    prompt = f"""You are a senior intelligence analyst. Synthesize a brief summary of the evidence below.

QUERY: {question}

EVIDENCE DATA:
{json.dumps(companies_data, indent=2)}

INSTRUCTIONS:
1. Provide a VERY BRIEF 2-3 sentence summary of the key findings.
2. Group trends if applicable (e.g., "Several fintech startups have recently...").
3. Use a professional, objective tone, but keep it extremely short. Do not list companies exhaustively.
4. Return ONLY the report text. No intro/outro."""

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
                "max_tokens": 250,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=20,
        )
        data = resp.json()
        print(data["content"][0]["text"])
    except Exception as e:
        print("Error:", e)
        
    await client.aclose()

asyncio.run(run())
