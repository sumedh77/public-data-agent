import { useState } from "react";

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <div className="mb-8">
    <h2 className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3 border-b border-gray-100 pb-2">
      {title}
    </h2>
    {children}
  </div>
);

const ApiCard = ({
  name,
  emoji,
  color,
  role,
  how,
  cost,
  data,
}: {
  name: string;
  emoji: string;
  color: string;
  role: string;
  how: string;
  cost: string;
  data: string;
}) => (
  <div className={`rounded-lg border ${color} p-4 flex flex-col gap-2`}>
    <div className="flex items-center gap-2">
      <span className="text-xl">{emoji}</span>
      <span className="font-bold text-sm text-gray-800">{name}</span>
    </div>
    <div className="text-xs text-gray-600 leading-relaxed">
      <span className="font-semibold text-gray-700">Role: </span>{role}
    </div>
    <div className="text-xs text-gray-600 leading-relaxed">
      <span className="font-semibold text-gray-700">How: </span>{how}
    </div>
    <div className="text-xs text-gray-600 leading-relaxed">
      <span className="font-semibold text-gray-700">What it finds: </span>{data}
    </div>
    <div className="mt-1 text-xs font-mono bg-white/70 border border-gray-200 rounded px-2 py-1 text-gray-700">
      💰 {cost}
    </div>
  </div>
);

const Step = ({ n, title, desc }: { n: string; title: string; desc: string }) => (
  <div className="flex gap-3 mb-4">
    <div className="w-6 h-6 rounded-full bg-gray-800 text-white text-xs font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
      {n}
    </div>
    <div>
      <div className="text-sm font-semibold text-gray-800">{title}</div>
      <div className="text-xs text-gray-500 mt-0.5 leading-relaxed">{desc}</div>
    </div>
  </div>
);

const Tag = ({ children, color = "bg-gray-100 text-gray-600" }: { children: React.ReactNode; color?: string }) => (
  <span className={`inline-block text-xs px-2 py-0.5 rounded font-medium mr-1.5 mb-1.5 ${color}`}>{children}</span>
);

export default function App() {
  const [page, setPage] = useState(0);

  return (
    <div className="min-h-screen bg-[#fafaf8] font-sans text-gray-900">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-8 py-5 flex items-center justify-between sticky top-0 z-10">
        <div className="flex items-center gap-3">
          <div className="w-7 h-7 rounded bg-gray-900 flex items-center justify-center">
            <span className="text-white text-xs font-bold">R</span>
          </div>
          <div>
            <div className="font-bold text-sm text-gray-900">Public Data Agent</div>
            <div className="text-xs text-gray-400">Reo.Dev · Technical Overview</div>
          </div>
        </div>
        <div className="flex gap-1 bg-gray-100 p-1 rounded-lg">
          {["Overview & Architecture", "APIs, Cost & Data Types"].map((label, i) => (
            <button
              key={i}
              onClick={() => setPage(i)}
              className={`text-xs px-3 py-1.5 rounded-md font-medium transition-all ${
                page === i ? "bg-white text-gray-900 shadow-sm" : "text-gray-500 hover:text-gray-700"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="text-xs text-gray-400 font-mono">v1.0 · Apr 2026</div>
      </div>

      <div className="max-w-4xl mx-auto px-8 py-8">
        {page === 0 && (
          <>
            {/* Hero */}
            <div className="mb-10">
              <div className="flex items-start justify-between mb-3">
                <h1 className="text-2xl font-bold text-gray-900 leading-tight">
                  Public Intelligence Agent
                </h1>
                <span className="text-xs font-mono bg-green-50 text-green-700 border border-green-200 px-2 py-1 rounded mt-1">
                  ● Live · localhost:4001
                </span>
              </div>
              <p className="text-sm text-gray-600 leading-relaxed max-w-2xl">
                A Python-based research agent that accepts a natural language question, fires 10 search queries
                in parallel across <strong>Exa</strong> and <strong>Parallel AI</strong>,
                scrapes the top results via <strong>Firecrawl</strong>, and synthesises a sourced answer —
                entirely from public internet data, no internal database involved.
              </p>

              <div className="flex flex-wrap gap-2 mt-4">
                <Tag color="bg-blue-50 text-blue-700">FastAPI + Python</Tag>
                <Tag color="bg-purple-50 text-purple-700">Server-Sent Events (streaming)</Tag>
                <Tag color="bg-orange-50 text-orange-700">Async / asyncio</Tag>
                <Tag color="bg-gray-100 text-gray-600">Public internet only</Tag>
                <Tag color="bg-green-50 text-green-700">No LLM · deterministic pipeline</Tag>
              </div>
            </div>

            {/* Architecture flow */}
            <Section title="How It Works — 5-Step Pipeline">
              <Step
                n="1"
                title="Query Derivation"
                desc='Your single question is expanded into 10 targeted search queries. Intent words like "find companies that published blog about" are stripped away, leaving only topic keywords (e.g. "agentic AI security breach"). Queries are shaped as first-person narratives ("we migrated"), "how we" patterns, engineering blog queries, and changelog patterns — covering every angle a company might publish about a topic.'
              />
              <Step
                n="2"
                title="Parallel Search — Exa + Parallel AI (simultaneous)"
                desc="Both search APIs are called at the same time (asyncio.gather). Exa alternates between neural and keyword search modes for diversity. Parallel AI runs fast-mode extraction. Each returns up to 20 results per query → potentially 400 raw URLs collected and de-duplicated."
              />
              <Step
                n="3"
                title="Smart Filtering & De-duplication"
                desc="Media aggregators (TechCrunch, Bloomberg, Forbes, Reuters, WSJ) are excluded. Press-wire sources (PRNewswire, BusinessWire) are kept but the company is inferred from the headline, not the wire domain. A domain resolver ensures 'blog.acme.com' → 'acme.com', and entity aliases (OpenAI, Open AI → openai.com) are normalised."
              />
              <Step
                n="4"
                title="Firecrawl Scraping (8 concurrent)"
                desc="Top candidate URLs are scraped in parallel (semaphore of 8). Firecrawl converts each page to clean Markdown — stripping nav, ads, and boilerplate. If Firecrawl fails, Parallel AI Extract is used as a fallback. Up to 4,000 characters of content per page are kept for evidence."
              />
              <Step
                n="5"
                title="Evidence Extraction & Streaming Response"
                desc="The agent scores each result, extracts dates and content types (blog post, announcement, case study, whitepaper), and streams the final answer back to the frontend via Server-Sent Events — so results appear in real-time as they are found, not all at once."
              />
            </Section>

            {/* Architecture diagram */}
            <Section title="System Architecture">
              <div className="bg-gray-900 rounded-xl p-5 text-xs font-mono text-green-300 leading-relaxed overflow-x-auto">
                <div className="text-gray-400 mb-2">// Public Data Agent — Request Flow</div>
                <div>
                  <span className="text-yellow-300">User Question</span>
                  <span className="text-gray-400"> ──▶ </span>
                  <span className="text-blue-300">Query Expander</span>
                  <span className="text-gray-400"> ──▶ </span>
                  <span className="text-gray-300">10 search queries</span>
                </div>
                <div className="mt-2 ml-4">
                  <span className="text-gray-400">├── </span><span className="text-purple-300">Exa API</span>
                  <span className="text-gray-500">  (neural + keyword, 20 results/query, 180-day filter)</span>
                </div>
                <div className="ml-4">
                  <span className="text-gray-400">└── </span><span className="text-orange-300">Parallel AI</span>
                  <span className="text-gray-500"> (fast mode, 600 char excerpts)</span>
                </div>
                <div className="mt-2 text-gray-400">  ↓ de-duplicate + filter media domains + entity resolve</div>
                <div>
                  <span className="text-red-300">Firecrawl Scraper</span>
                  <span className="text-gray-400"> ──▶ </span>
                  <span className="text-gray-300">Markdown content (4k chars/page, 8 concurrent)</span>
                </div>
                <div className="mt-1 ml-4 text-gray-500">└── Parallel AI Extract (fallback)</div>
                <div className="mt-2 text-gray-400">  ↓ score + date extract + content type classify</div>
                <div>
                  <span className="text-green-300">SSE Stream</span>
                  <span className="text-gray-400"> ──▶ </span>
                  <span className="text-gray-300">Frontend (real-time results as they arrive)</span>
                </div>
              </div>
            </Section>

            {/* What it can find */}
            <Section title="What Kind of Information Can Be Scraped">
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Company Blog Posts", desc: "Engineering, growth, product blogs from any company's own domain", tag: "blog post" },
                  { label: "Press Announcements", desc: "Funding rounds, product launches, partnerships, expansions", tag: "announcement" },
                  { label: "Case Studies", desc: "How companies solved problems, migrations, platform changes", tag: "case study" },
                  { label: "Security Incident Reports", desc: "CVEs, breach disclosures, post-mortems, vulnerability research", tag: "security" },
                  { label: "Whitepapers & Research", desc: "Technical deep-dives, industry reports, benchmarks", tag: "whitepaper" },
                  { label: "Changelogs & Release Notes", desc: "Product updates, feature announcements, deprecations", tag: "changelog" },
                  { label: "Hiring & Leadership Signals", desc: "Job posts, org expansions, leadership appointments from news", tag: "signal" },
                  { label: "Competitor Intelligence", desc: "What rivals are building, announcing, or writing about", tag: "intel" },
                ].map((item) => (
                  <div key={item.label} className="bg-white border border-gray-100 rounded-lg p-3 flex flex-col gap-1">
                    <div className="flex items-center gap-2">
                      <span className="text-xs font-mono bg-gray-100 text-gray-500 px-1.5 py-0.5 rounded">{item.tag}</span>
                      <span className="text-sm font-semibold text-gray-800">{item.label}</span>
                    </div>
                    <p className="text-xs text-gray-500 leading-relaxed">{item.desc}</p>
                  </div>
                ))}
              </div>
            </Section>

            {/* Excluded sources */}
            <Section title="What It Intentionally Excludes">
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-xs text-amber-800 leading-relaxed">
                <strong>Media aggregators excluded:</strong> TechCrunch, Bloomberg, Forbes, Reuters, WSJ, FT, Wired, VentureBeat, CNBC, ZDNet, The Verge, Hacker News, LinkedIn, Twitter/X
                <br /><br />
                <strong>Why:</strong> These are articles <em>about</em> companies, not <em>from</em> companies. The agent is designed to surface what companies say in their own voice — their own blogs, docs, and announcements. When a press wire (PRNewswire, BusinessWire) is found, the company is inferred from the headline and their own domain is resolved instead.
                <br /><br />
                <strong>Note:</strong> Medium and Substack are NOT excluded — many companies publish engineering and growth blogs there.
              </div>
            </Section>
          </>
        )}

        {page === 1 && (
          <>
            <div className="mb-8">
              <h1 className="text-2xl font-bold text-gray-900 mb-2">APIs, Costs & Data Capabilities</h1>
              <p className="text-sm text-gray-500">Four external services power the agent. Each plays a distinct role in the pipeline.</p>
            </div>

            {/* API cards */}
            <Section title="The Three APIs">
              <div className="grid grid-cols-2 gap-4">
                <ApiCard
                  name="Exa"
                  emoji="🔍"
                  color="border-purple-100 bg-purple-50/40"
                  role="Primary semantic search engine. Finds company blog posts and articles by meaning, not just keywords."
                  how="Alternates between 'neural' (semantic embedding search) and 'keyword' (BM25-style) modes. Returns 20 results per query with 3 highlighted excerpts of 3 sentences each — rich extractable signals. Filtered to last 180 days by default."
                  data="Blog posts, engineering articles, product announcements, case studies, security disclosures, changelogs"
                  cost="~$5 per 1,000 searches. Free tier: 1,000 searches/month. At 10 queries/question → ~$0.05/question."
                />
                <ApiCard
                  name="Parallel AI"
                  emoji="⚡"
                  color="border-orange-100 bg-orange-50/40"
                  role="Fast-mode search + fallback web extractor. Provides a third independent index and backs up Firecrawl scraping."
                  how="Search: runs all 10 queries simultaneously in 'fast' mode with 600-char excerpts. Extract: used as a fallback when Firecrawl can't scrape a URL — pulls raw text/markdown directly from any public URL."
                  data="Web pages, company blogs, news articles, documentation — anything publicly accessible on the internet"
                  cost="Usage-based pricing. Search: ~$0.01–0.03/query. Extract: ~$0.005–0.01/URL. Estimated ~$0.10–0.30/question total."
                />
                <ApiCard
                  name="Firecrawl"
                  emoji="🔥"
                  color="border-red-100 bg-red-50/40"
                  role="Primary web scraper. Converts any public URL into clean Markdown, removing ads, navigation, and boilerplate."
                  how="Called for every candidate URL after search. POST to /v1/scrape with format:markdown. Returns up to 4,000 characters of clean content per page. Runs 8 concurrent scrapes (semaphore-limited). Parallel AI Extract is the fallback."
                  data="Full blog post body, article text, press release content, changelog entries, whitepaper sections, documentation"
                  cost="Free tier: 500 pages/month. Paid: ~$15/mo (3k pages), ~$50/mo (10k pages). At ~50 URLs/question → ~$0.25/question."
                />
              </div>
            </Section>

            {/* Cost summary */}
            <Section title="Estimated Cost Per Question">
              <div className="bg-white border border-gray-200 rounded-xl overflow-hidden">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-gray-100 bg-gray-50">
                      <th className="text-left px-4 py-3 font-semibold text-gray-600">API</th>
                      <th className="text-left px-4 py-3 font-semibold text-gray-600">Usage per Question</th>
                      <th className="text-left px-4 py-3 font-semibold text-gray-600">Unit Cost</th>
                      <th className="text-right px-4 py-3 font-semibold text-gray-600">Cost/Question</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ["Exa", "10 queries × 20 results", "~$0.005/search", "~$0.05"],
                      ["Parallel AI Search", "10 queries fast mode", "~$0.01–0.03/query", "~$0.10–0.30"],
                      ["Firecrawl Scrape", "~50 URLs scraped", "~$0.005/page", "~$0.25"],
                      ["Parallel AI Extract", "Fallback only (~5 URLs)", "~$0.005–0.01/URL", "~$0.02–0.05"],
                    ].map(([api, usage, unit, cost], i) => (
                      <tr key={i} className="border-b border-gray-50 hover:bg-gray-50/50">
                        <td className="px-4 py-3 font-medium text-gray-800">{api}</td>
                        <td className="px-4 py-3 text-gray-500">{usage}</td>
                        <td className="px-4 py-3 text-gray-500 font-mono">{unit}</td>
                        <td className="px-4 py-3 text-right font-mono font-semibold text-gray-800">{cost}</td>
                      </tr>
                    ))}
                    <tr className="bg-gray-900 text-white">
                      <td className="px-4 py-3 font-bold" colSpan={3}>Total Estimated Cost per Question</td>
                      <td className="px-4 py-3 text-right font-mono font-bold text-green-300">~$0.42–0.65</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p className="text-xs text-gray-400 mt-2 italic">* Costs are estimates and vary by API tier, volume discounts, and fallback usage. All 4 APIs offer free tiers for development.</p>
            </Section>

            {/* Data types matrix */}
            <Section title="Full Data Type Matrix — What Can Be Discovered">
              <div className="space-y-2">
                {[
                  {
                    category: "Company Blog Posts",
                    sources: ["Exa (neural)", "Parallel AI", "Firecrawl"],
                    examples: "Engineering posts, growth teardowns, product deep-dives, 'how we built X', migration stories",
                    signal: "High — first-party company voice",
                    color: "text-purple-700 bg-purple-50 border-purple-100",
                  },
                  {
                    category: "Security Breaches & CVEs",
                    sources: ["Exa (keyword)", "Parallel AI"],
                    examples: "Incident post-mortems, vulnerability disclosures, breach announcements, CVSS reports",
                    signal: "High — directly from security blogs",
                    color: "text-red-700 bg-red-50 border-red-100",
                  },
                  {
                    category: "Funding & Financials",
                    sources: ["Parallel AI", "PRNewswire (domain inferred)"],
                    examples: "Series A/B/C raises, M&A activity, revenue milestones, IPO filings",
                    signal: "Medium — press wires + company newsrooms",
                    color: "text-green-700 bg-green-50 border-green-100",
                  },
                  {
                    category: "Product Launches & Changelogs",
                    sources: ["Exa", "Firecrawl (changelog pages)"],
                    examples: "New feature announcements, deprecations, API changes, release notes",
                    signal: "High — from /blog, /news, /changelog paths",
                    color: "text-blue-700 bg-blue-50 border-blue-100",
                  },
                  {
                    category: "Hiring & Leadership Signals",
                    sources: ["Exa (keyword)", "Parallel AI"],
                    examples: "CXO appointments, team expansions, office openings, layoffs",
                    signal: "Medium — inferred from headline parsing",
                    color: "text-amber-700 bg-amber-50 border-amber-100",
                  },
                  {
                    category: "Technical Whitepapers & Research",
                    sources: ["Exa (neural)", "Firecrawl (/research, /docs)"],
                    examples: "Architecture papers, benchmark reports, developer guides, playbooks",
                    signal: "High — from /research, /whitepaper, /docs paths",
                    color: "text-gray-700 bg-gray-50 border-gray-100",
                  },
                ].map((row) => (
                  <div key={row.category} className={`border rounded-lg p-3 ${row.color}`}>
                    <div className="flex items-start justify-between gap-4">
                      <div>
                        <div className="font-semibold text-sm mb-1">{row.category}</div>
                        <div className="text-xs opacity-80 mb-1">{row.examples}</div>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {row.sources.map((s) => (
                            <span key={s} className="text-xs bg-white/70 border border-current/20 rounded px-1.5 py-0.5 opacity-70">
                              {s}
                            </span>
                          ))}
                        </div>
                      </div>
                      <div className="text-xs font-medium whitespace-nowrap opacity-80 text-right min-w-fit">
                        {row.signal}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </Section>

            {/* Tech stack */}
            <Section title="Tech Stack & Deployment">
              <div className="grid grid-cols-3 gap-3">
                {[
                  { label: "Runtime", value: "Python 3.11+ / FastAPI" },
                  { label: "Async Engine", value: "asyncio + httpx" },
                  { label: "Streaming", value: "Server-Sent Events (SSE)" },
                  { label: "Frontend", value: "Vanilla HTML/JS" },
                  { label: "Port", value: "localhost:4001" },
                  { label: "Concurrency", value: "8 parallel scrapes + N queries" },
                ].map((item) => (
                  <div key={item.label} className="bg-white border border-gray-100 rounded-lg px-3 py-2.5">
                    <div className="text-xs text-gray-400 mb-0.5">{item.label}</div>
                    <div className="text-sm font-semibold text-gray-800 font-mono">{item.value}</div>
                  </div>
                ))}
              </div>
            </Section>
          </>
        )}
      </div>

      {/* Footer */}
      <div className="border-t border-gray-100 mt-8 py-4 px-8 text-xs text-gray-400 flex justify-between">
        <span>Public Data Agent · Reo.Dev Internal Documentation</span>
        <span>Page {page + 1} of 2 · April 2026</span>
      </div>
    </div>
  );
}
