# Public Intelligence Agent

Natural language questions → Parallel AI + Exa + Firecrawl → answer with full source evidence.

No internal data. Public internet only.

## Setup

1. Clone the repo
2. Copy `.env.example` to `.env` and fill in your API keys
3. Install dependencies: `pip install -r requirements.txt`
4. Run: `python server.py`

## Environment Variables

| Variable | Description |
|---|---|
| `EXA_API_KEY` | Exa search API key |
| `FIRECRAWL_KEY` | Firecrawl scraping API key |
| `PARALLEL_AI_KEY` | Parallel AI key |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key |
| `PORT` | Port to run server on (default: 4001) |

## Deploy

Deploy on Railway — connect your GitHub repo and add env vars in the Railway dashboard.
