import { useState } from "react";

const PHASES = [
  {
    id: 1,
    label: "Phase 1",
    name: "Query Derivation",
    type: "free",
    execution: "Sequential",
    execColor: "bg-gray-100 text-gray-600",
    description: "Your question is expanded into 10 targeted search queries using pure Python regex — no API call.",
    apis: [],
    callCount: "0 API calls",
    costMin: 0,
    costMax: 0,
    detail: [
      "Strips intent words ('find companies that published...')",
      "Builds 10 query variations: first-person, how-we, lessons learned, changelog, announcement, engineering blog patterns",
      "100% free — runs locally in Python",
    ],
  },
  {
    id: 2,
    label: "Phase 2",
    name: "Parallel Search",
    type: "parallel",
    execution: "Parallel (asyncio.gather)",
    execColor: "bg-blue-100 text-blue-700",
    description: "Both search engines fire simultaneously. Each engine runs all 10 queries in parallel internally too.",
    apis: [
      {
        name: "Exa",
        color: "border-purple-200 bg-purple-50",
        badge: "bg-purple-100 text-purple-700",
        callsR1: 10,
        callsR2: 5,
        resultsPerCall: 20,
        unitCost: 0.005,
        unitLabel: "$0.005 / search",
        costR1: 0.05,
        costR2: 0.025,
        note: "Alternates neural ↔ keyword per query. 3 highlights × 3 sentences per result.",
        pricingTier: "~$5 / 1,000 searches",
        freeTier: "1,000 searches/month free",
      },
      {
        name: "Parallel AI Search",
        color: "border-orange-200 bg-orange-50",
        badge: "bg-orange-100 text-orange-700",
        callsR1: 10,
        callsR2: 5,
        resultsPerCall: 20,
        unitCost: 0.02,
        unitLabel: "$0.01–0.03 / query",
        costR1: 0.20,
        costR2: 0.10,
        note: "Fast mode. 600-char excerpts per result. Third independent index.",
        pricingTier: "~$0.01–0.03 / query",
        freeTier: "Usage-based, no free tier",
      },
    ],
    round2: {
      trigger: "Only fires if Round 1 returns < 30 unique domains",
      queries: 5,
      label: "Round 2 (conditional)",
    },
    costMin: 0.25,
    costMax: 0.50,
  },
  {
    id: 3,
    label: "Phase 3",
    name: "Scraping",
    type: "waterfall",
    execution: "Waterfall (after search)",
    execColor: "bg-amber-100 text-amber-700",
    description: "Runs AFTER search completes. Top 80 URLs scraped 8 at a time concurrently. Parallel AI Extract is the fallback if Firecrawl fails.",
    apis: [
      {
        name: "Firecrawl Scrape",
        color: "border-red-200 bg-red-50",
        badge: "bg-red-100 text-red-700",
        callsR1: 80,
        callsR2: 0,
        resultsPerCall: 1,
        unitCost: 0.005,
        unitLabel: "$0.005 / page",
        costR1: 0.40,
        costR2: 0,
        note: "Primary scraper. Converts any URL to clean Markdown. Runs 8 concurrent (semaphore). Max 4,000 chars/page.",
        pricingTier: "~$0.005 / page (paid tier)",
        freeTier: "500 pages/month free",
      },
      {
        name: "Parallel AI Extract",
        color: "border-orange-200 bg-orange-50",
        badge: "bg-orange-100 text-orange-700",
        callsR1: 10,
        callsR2: 0,
        resultsPerCall: 1,
        unitCost: 0.01,
        unitLabel: "$0.005–0.01 / URL",
        costR1: 0.10,
        costR2: 0,
        note: "FALLBACK ONLY — fires if Firecrawl returns empty or errors. Estimated ~10–20 fallback pages per question.",
        pricingTier: "~$0.005–0.01 / URL",
        freeTier: "Usage-based",
      },
    ],
    costMin: 0.40,
    costMax: 0.50,
  },
  {
    id: 4,
    label: "Phase 4–5",
    name: "Extract & Synthesize",
    type: "free",
    execution: "Sequential",
    execColor: "bg-gray-100 text-gray-600",
    description: "Pure Python — scores paragraphs by keyword overlap, extracts dates, classifies content type, groups by company. Zero API calls.",
    apis: [],
    callCount: "0 API calls",
    costMin: 0,
    costMax: 0,
    detail: [
      "Keyword scoring — no LLM, no AI inference",
      "Date parsing + sanity check (reject stale results)",
      "Group by company domain, sort by confidence",
      "Stream results to frontend via SSE",
    ],
  },
];

const execBadge: Record<string, string> = {
  "Parallel (asyncio.gather)": "bg-blue-100 text-blue-700 border-blue-200",
  "Waterfall (after search)": "bg-amber-100 text-amber-700 border-amber-200",
  "Sequential": "bg-gray-100 text-gray-500 border-gray-200",
};

export default function App() {
  const [expanded, setExpanded] = useState<number | null>(2);

  const totalMin = PHASES.reduce((s, p) => s + p.costMin, 0);
  const totalMax = PHASES.reduce((s, p) => s + p.costMax, 0);

  return (
    <div className="min-h-screen bg-[#f8f8f6] font-sans text-gray-900" style={{ fontFamily: "'DM Sans', system-ui, sans-serif" }}>
      {/* Header */}
      <div className="bg-white border-b border-gray-200 px-8 py-5">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-lg font-bold text-gray-900">Public Data Agent — Cost & Execution Model</h1>
            <p className="text-xs text-gray-400 mt-0.5">Exactly what fires, when, and what it costs per question</p>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-right">
              <div className="text-xs text-gray-400">Cost per question</div>
              <div className="text-xl font-bold text-gray-900">${totalMin.toFixed(2)}–${totalMax.toFixed(2)}</div>
              <div className="text-xs text-gray-400">Exa + Parallel AI only</div>
            </div>
            <div className="w-px h-10 bg-gray-200" />
            <div className="text-right">
              <div className="text-xs text-gray-400">API calls</div>
              <div className="text-xl font-bold text-gray-900">50–75</div>
            </div>
          </div>
        </div>
      </div>

      {/* Execution legend */}
      <div className="border-b border-gray-100 bg-white px-8 py-3">
        <div className="max-w-4xl mx-auto flex items-center gap-6 text-xs">
          <span className="text-gray-400 font-medium">Execution model:</span>
          {[
            { label: "Parallel — all fire at the same time", cls: "bg-blue-100 text-blue-700 border-blue-200" },
            { label: "Waterfall — waits for previous phase", cls: "bg-amber-100 text-amber-700 border-amber-200" },
            { label: "Sequential / Free — no API", cls: "bg-gray-100 text-gray-500 border-gray-200" },
          ].map((l) => (
            <span key={l.label} className={`border rounded px-2 py-0.5 font-medium ${l.cls}`}>{l.label}</span>
          ))}
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-8 py-8 space-y-3">

        {PHASES.map((phase) => (
          <div
            key={phase.id}
            className="bg-white border border-gray-200 rounded-xl overflow-hidden"
          >
            {/* Phase header */}
            <button
              className="w-full flex items-center gap-4 px-5 py-4 text-left hover:bg-gray-50/50 transition-colors"
              onClick={() => setExpanded(expanded === phase.id ? null : phase.id)}
            >
              <div className="w-8 h-8 rounded-lg bg-gray-900 text-white text-xs font-bold flex items-center justify-center flex-shrink-0">
                {phase.id === 4 ? "4–5" : phase.id}
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-sm text-gray-900">{phase.name}</span>
                  <span className={`text-xs border rounded px-2 py-0.5 font-medium ${execBadge[phase.execution]}`}>
                    {phase.execution}
                  </span>
                  {phase.type === "free" && (
                    <span className="text-xs bg-green-50 text-green-700 border border-green-200 rounded px-2 py-0.5 font-medium">Free</span>
                  )}
                  {phase.id === 3 && (
                    <span className="text-xs bg-red-50 text-red-600 border border-red-200 rounded px-2 py-0.5 font-medium">Fires after Phase 2</span>
                  )}
                </div>
                <p className="text-xs text-gray-500 mt-0.5">{phase.description}</p>
              </div>
              <div className="text-right flex-shrink-0">
                {phase.costMin === 0 ? (
                  <span className="text-sm font-bold text-green-600">$0.00</span>
                ) : (
                  <span className="text-sm font-bold text-gray-900">${phase.costMin.toFixed(2)}–${phase.costMax.toFixed(2)}</span>
                )}
              </div>
            </button>

            {/* Expanded detail */}
            {expanded === phase.id && (
              <div className="border-t border-gray-100 px-5 pb-5 pt-4">
                {phase.type === "free" && phase.detail && (
                  <ul className="space-y-1.5">
                    {phase.detail.map((d, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-gray-600">
                        <span className="text-green-500 font-bold mt-0.5">✓</span>
                        {d}
                      </li>
                    ))}
                  </ul>
                )}

                {phase.apis && phase.apis.length > 0 && (
                  <div className="space-y-3">
                    {/* Round 2 callout for search phase */}
                    {phase.round2 && (
                      <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-2.5 text-xs text-amber-800 mb-4">
                        <strong>⚡ Round 2 (conditional):</strong> {phase.round2.trigger}. If triggered, fires {phase.round2.queries} more queries to both engines simultaneously. Adds ~$0.10–0.20 extra.
                      </div>
                    )}

                    {phase.id === 3 && (
                      <div className="bg-gray-50 border border-gray-200 rounded-lg px-4 py-2.5 text-xs text-gray-600 mb-4">
                        <strong>Execution:</strong> Top 80 URLs from search → Firecrawl scrapes 8 at a time concurrently → if Firecrawl fails on a URL, Parallel AI Extract fires as fallback for that URL only.
                      </div>
                    )}

                    <div className="overflow-hidden rounded-lg border border-gray-200">
                      <table className="w-full text-xs">
                        <thead>
                          <tr className="bg-gray-50 border-b border-gray-200">
                            <th className="text-left px-4 py-2.5 font-semibold text-gray-500 uppercase tracking-wider text-[10px]">API</th>
                            <th className="text-center px-3 py-2.5 font-semibold text-gray-500 uppercase tracking-wider text-[10px]">Calls (Round 1)</th>
                            <th className="text-center px-3 py-2.5 font-semibold text-gray-500 uppercase tracking-wider text-[10px]">Calls (Round 2)</th>
                            <th className="text-center px-3 py-2.5 font-semibold text-gray-500 uppercase tracking-wider text-[10px]">Unit Price</th>
                            <th className="text-right px-4 py-2.5 font-semibold text-gray-500 uppercase tracking-wider text-[10px]">Cost R1</th>
                            <th className="text-right px-4 py-2.5 font-semibold text-gray-500 uppercase tracking-wider text-[10px]">Cost R2</th>
                          </tr>
                        </thead>
                        <tbody>
                          {phase.apis.map((api, i) => (
                            <tr key={i} className="border-b border-gray-50 last:border-0">
                              <td className="px-4 py-3">
                                <div className={`inline-flex items-center gap-1.5 border rounded px-2 py-1 ${api.color}`}>
                                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded ${api.badge}`}>{api.name}</span>
                                </div>
                                <div className="text-[10px] text-gray-400 mt-1 max-w-xs">{api.note}</div>
                                <div className="text-[10px] text-gray-400">{api.freeTier}</div>
                              </td>
                              <td className="px-3 py-3 text-center">
                                <span className="font-mono font-bold text-gray-800">{api.callsR1}</span>
                                {api.resultsPerCall > 1 && <div className="text-[10px] text-gray-400">× {api.resultsPerCall} results each</div>}
                              </td>
                              <td className="px-3 py-3 text-center">
                                {api.callsR2 > 0 ? (
                                  <span className="font-mono font-bold text-amber-600">{api.callsR2}</span>
                                ) : (
                                  <span className="text-gray-300">—</span>
                                )}
                              </td>
                              <td className="px-3 py-3 text-center font-mono text-gray-600">{api.unitLabel}</td>
                              <td className="px-4 py-3 text-right font-mono font-bold text-gray-900">${api.costR1.toFixed(2)}</td>
                              <td className="px-4 py-3 text-right font-mono font-bold text-amber-600">
                                {api.callsR2 > 0 ? `$${api.costR2.toFixed(2)}` : "—"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}

        {/* Total row */}
        <div className="bg-gray-900 text-white rounded-xl px-6 py-5 flex items-center justify-between">
          <div>
            <div className="font-bold text-sm">Total per Question</div>
            <div className="text-xs text-gray-400 mt-1">
              Round 1 only (if &gt;30 domains found): ~$0.65–0.85 &nbsp;·&nbsp; Round 2 triggered: ~$0.75–1.05
            </div>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold text-white">${totalMin.toFixed(2)}–${totalMax.toFixed(2)}</div>
            <div className="text-xs text-gray-400 mt-1">typical range · Round 2 adds ~$0.25–0.30</div>
          </div>
        </div>

        {/* Flow diagram */}
        <div className="bg-white border border-gray-200 rounded-xl p-6">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-4">Execution Timeline — What Fires When</h3>
          <div className="space-y-3">
            {[
              { time: "t=0ms", label: "Phase 1", desc: "Query derivation (Python, free, ~5ms)", color: "bg-gray-200", w: "w-8" },
              { time: "t=5ms", label: "Phase 2 START", desc: "Exa + Parallel AI both fire simultaneously", color: "bg-blue-400", w: "w-64" },
              { time: "t=5ms–3s", label: "⟵ all 3 engines running in parallel ⟶", desc: "", color: "", w: "" },
              { time: "t~3–8s", label: "Phase 2 END", desc: "All search results collected, de-duped", color: "bg-blue-200", w: "w-4" },
              { time: "t~3–8s", label: "Phase 3 START", desc: "Firecrawl scrapes top 80 URLs (8 at a time)", color: "bg-red-400", w: "w-48" },
              { time: "t~8–25s", label: "⟵ scraping 80 pages concurrently ⟶", desc: "", color: "", w: "" },
              { time: "t~25s", label: "Phase 4–5", desc: "Extract + synthesize (Python, free, ~1s)", color: "bg-gray-200", w: "w-8" },
            ].map((row, i) => (
              <div key={i} className="flex items-center gap-3 text-xs">
                <div className="w-20 text-gray-400 font-mono flex-shrink-0">{row.time}</div>
                {row.color ? (
                  <>
                    <div className={`h-5 rounded ${row.color} ${row.w} flex-shrink-0`} />
                    <div className="text-gray-700">
                      <span className="font-semibold">{row.label}</span>
                      {row.desc && <span className="text-gray-400"> — {row.desc}</span>}
                    </div>
                  </>
                ) : (
                  <div className="text-blue-500 font-medium">{row.label}</div>
                )}
              </div>
            ))}
          </div>

          <div className="mt-5 pt-4 border-t border-gray-100 grid grid-cols-3 gap-4 text-xs">
            <div className="bg-blue-50 border border-blue-100 rounded-lg p-3">
              <div className="font-bold text-blue-700 mb-1">🔀 Parallel (Phase 2)</div>
              <div className="text-gray-600">Exa and Parallel AI both fire at the same time. Within each engine, all 10 queries also fire in parallel. Total: 20 simultaneous HTTP requests.</div>
            </div>
            <div className="bg-amber-50 border border-amber-100 rounded-lg p-3">
              <div className="font-bold text-amber-700 mb-1">⬇ Waterfall (Phase 3)</div>
              <div className="text-gray-600">Firecrawl WAITS for search to finish. It needs the URLs. Runs 8 concurrent scrapes at a time. Parallel AI Extract fires only as fallback for failed pages.</div>
            </div>
            <div className="bg-green-50 border border-green-100 rounded-lg p-3">
              <div className="font-bold text-green-700 mb-1">⚡ Conditional Round 2</div>
              <div className="text-gray-600">If Round 1 finds fewer than 30 unique domains, both search engines fire again with 5 new queries. This doubles search cost but not scrape cost.</div>
            </div>
          </div>
        </div>

        {/* Free tier note */}
        <div className="bg-white border border-gray-200 rounded-xl p-5">
          <h3 className="text-xs font-bold uppercase tracking-widest text-gray-400 mb-3">Free Tiers — How Long Before You Pay</h3>
          <div className="grid grid-cols-4 gap-3">
            {[
              { api: "Exa", free: "1,000 searches/mo", questionsPerMonth: "~100 questions", color: "text-purple-700 bg-purple-50 border-purple-100" },
              { api: "Parallel AI", free: "No free tier", questionsPerMonth: "Pay from day 1", color: "text-orange-700 bg-orange-50 border-orange-100" },
              { api: "Firecrawl", free: "500 pages/mo", questionsPerMonth: "~6–10 questions", color: "text-red-700 bg-red-50 border-red-100" },
            ].map((row) => (
              <div key={row.api} className={`border rounded-lg p-3 ${row.color}`}>
                <div className="font-bold text-sm">{row.api}</div>
                <div className="text-xs mt-1">{row.free}</div>
                <div className="text-xs mt-1 font-medium">{row.questionsPerMonth}</div>
              </div>
            ))}
          </div>
          <p className="text-xs text-gray-400 mt-3">* "Questions per month" on free tier assumes 10 queries/question for Exa, 80 pages/question for Firecrawl.</p>
        </div>

      </div>
    </div>
  );
}
