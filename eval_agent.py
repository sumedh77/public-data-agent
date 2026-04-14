#!/usr/bin/env python3
"""
Batch evaluation of 50 SDR queries against the Public Data Agent at localhost:4001/api/ask
Parses SSE stream, captures final 'done' event, and produces a quality-scored report.
"""

import json
import time
import requests

BASE_URL = "http://localhost:4001/api/ask"

QUERIES = [
    "Which B2B SaaS companies published blog posts about scaling their sales team or hiring their first SDRs in the last 90 days?",
    "Which companies announced a new VP of Sales or CRO via press releases or LinkedIn in the last 60 days?",
    "Which SaaS companies recently published content about moving upmarket from SMB to mid-market or enterprise?",
    "Which dev tool companies published engineering blogs about building internal tooling to replace a third-party vendor in the last 6 months?",
    "Which companies raised a Series B or C in the last 90 days and published content about scaling their GTM motion?",
    "Which B2B SaaS companies are simultaneously hiring a Head of RevOps and a Sales Ops Manager right now?",
    "Which companies published content about outgrowing their current CRM or needing a new sales tech stack in the last 6 months?",
    "Which SaaS companies announced SOC 2 certification or enterprise security compliance in the last 90 days?",
    "Which SaaS companies published content about shifting from PLG to sales-led or hybrid GTM in the last 6 months?",
    "Which companies had their CEO or founder publicly write about aggressive revenue targets or doubling ARR in the last 90 days?",
    "Summarise everything Databricks has announced publicly in the last 90 days including blog posts and product launches",
    "What are Snowflake top 3 publicly communicated strategic priorities in the last 6 months?",
    "What has Atlassian leadership said publicly about cloud migration and enterprise expansion in the last 6 months?",
    "Find recent job postings from a target account that reveal what internal problems they are trying to solve",
    "What has the leadership team at Linear publicly written about product philosophy or engineering culture?",
    "Find the most recent blog post or announcement from Figma and extract a specific business initiative I can reference in a cold email",
    "What has Vercel published recently about their developer experience priorities?",
    "Which companies on my target list published content about AI adoption or LLM integration I can reference in emails?",
    "Find the latest changelog or product update from Intercom and identify a specific feature launch I can tie my outreach to",
    "What has HubSpot publicly said about their AI strategy in blogs or press releases in the last 90 days?",
    "Find blog posts or case studies where companies wrote about replacing a legacy CRM with a modern alternative",
    "Which companies published content about the ROI of investing in sales intelligence or prospecting tools?",
    "Find public content where RevOps leaders wrote about the cost of bad data or poor CRM hygiene",
    "Find case studies where companies described switching from point solutions to a platform approach",
    "Which companies published content about reducing their sales tech stack or consolidating vendors in the last 6 months?",
    "Find any public post-mortems or incident reports from Stripe in the last year that signal infrastructure pain points",
    "Find public content where engineers at a company wrote about pain points with their current data infrastructure",
    "Which companies published content about restructuring their sales process or adopting a new methodology in the last 90 days?",
    "Which fintech companies announced new enterprise partnerships or channel deals in the last 60 days?",
    "Find companies that published blogs about improving win rate or conversion rate or sales cycle length",
    "Which B2B SaaS companies published content about their outbound sales strategy or SDR playbook in the last 6 months?",
    "Find public announcements from pipeline companies about new product launches that could change buying timelines",
    "Which data infrastructure companies published content signalling budget planning or annual procurement cycles?",
    "Find SaaS companies that published year-end retrospectives mentioning technology investment priorities for next year",
    "Which companies recently published content about achieving profitability or extending runway?",
    "Which companies published content about fiscal year planning or procurement processes or vendor evaluation criteria?",
    "Which companies published content about cost reduction or headcount changes or budget cuts that might delay purchases?",
    "Which companies in my CRM have been acquired or merged or rebranded based on public announcements in the last 12 months?",
    "Which companies on my target list have shut down or raised a down round or announced significant layoffs in the last 6 months?",
    "Find recent press releases confirming current HQ location and company size for accounts in my territory",
    "Find the most recent funding status and employee count for a list of companies to update stale CRM records",
    "Find B2B SaaS companies in EMEA that published content about expanding their engineering team or opening a new office in the last 90 days",
    "Which healthcare tech companies published blogs about compliance automation or regulatory technology in the last 6 months?",
    "Which HR tech companies published content about expanding into Europe in the last 6 months?",
    "Find the top 10 blog posts by companies in my ICP about their biggest operational pain points in the last 90 days",
    "Which companies in my target segment recently published content signalling an active vendor evaluation or RFP process?",
    "Find public benchmarks about average SDR quota attainment or email reply rates or outbound conversion rates in the last 12 months",
    "Summarise the key product themes from all public Salesforce blog posts and changelogs in the last 6 months",
    "Find the current LinkedIn title and company for contacts I have not engaged with in 6 months to check for job changes",
    "Which target accounts recently published content signalling they are evaluating or have replaced a competitor product?",
]

def parse_sse_stream(response):
    events = []
    current_event = None
    current_data = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            if current_event and current_data:
                try:
                    data = json.loads(" ".join(current_data))
                    events.append((current_event, data))
                except json.JSONDecodeError:
                    events.append((current_event, {"raw": " ".join(current_data)}))
            current_event = None
            current_data = []
            continue
        if line.startswith("event:"):
            current_event = line[6:].strip()
        elif line.startswith("data:"):
            current_data.append(line[5:].strip())
    # Handle last event if no trailing newline
    if current_event and current_data:
        try:
            data = json.loads(" ".join(current_data))
            events.append((current_event, data))
        except json.JSONDecodeError:
            pass
    return events

def score_result(done_data):
    scores = {}
    total_companies = done_data.get("total_companies", 0)
    total_articles = done_data.get("total_articles", 0)
    elapsed = done_data.get("elapsed", 0)

    if total_companies == 0: scores["companies_found"] = 0
    elif total_companies >= 10: scores["companies_found"] = 10
    elif total_companies >= 5: scores["companies_found"] = 7
    elif total_companies >= 3: scores["companies_found"] = 5
    else: scores["companies_found"] = 3

    avg_ev = total_articles / total_companies if total_companies > 0 else 0
    if avg_ev >= 3: scores["evidence_depth"] = 10
    elif avg_ev >= 2: scores["evidence_depth"] = 7
    elif avg_ev >= 1: scores["evidence_depth"] = 5
    else: scores["evidence_depth"] = 0

    answer = done_data.get("answer", "")
    if "No direct public evidence" in answer: scores["answer_quality"] = 0
    elif total_companies > 0 and len(answer) > 80: scores["answer_quality"] = 8
    elif total_companies > 0: scores["answer_quality"] = 5
    else: scores["answer_quality"] = 2

    companies = done_data.get("companies", [])
    if companies:
        confs = [ev.get("confidence",0) for co in companies for ev in co.get("evidence",[])]
        scores["confidence"] = round((sum(confs)/len(confs))*10,1) if confs else 0
    else:
        scores["confidence"] = 0

    sanity = done_data.get("sanity_report", {})
    p, rej = sanity.get("passed",0), sanity.get("rejected",0)
    total_c = p + rej
    scores["sanity_pass_rate"] = round(p/total_c*10,1) if total_c > 0 else 5

    if elapsed <= 15: scores["speed"] = 10
    elif elapsed <= 30: scores["speed"] = 8
    elif elapsed <= 60: scores["speed"] = 5
    else: scores["speed"] = 3

    overall = (
        scores["companies_found"] * 0.30 +
        scores["evidence_depth"] * 0.20 +
        scores["answer_quality"] * 0.25 +
        scores["confidence"] * 0.15 +
        scores["sanity_pass_rate"] * 0.05 +
        scores["speed"] * 0.05
    )
    scores["overall"] = round(overall, 1)
    scores["relevance_label"] = "HIGH" if overall >= 7 else ("MEDIUM" if overall >= 4 else "LOW")
    return scores

def run_query(idx, query):
    print(f"\n[{idx+1:02d}/50] {query[:80]}...")
    result = {"query_idx": idx+1, "query": query, "status": "error", "error": None, "done_data": None, "scores": None, "elapsed": 0}
    start_time = time.time()
    try:
        resp = requests.post(BASE_URL, json={"question": query}, stream=True, timeout=180, headers={"Accept":"text/event-stream"})
        resp.raise_for_status()
        events = parse_sse_stream(resp)
        done_event = next((d for t,d in events if t=="done"), None)
        result["elapsed"] = round(time.time() - start_time, 1)
        if done_event:
            result["status"] = "success"
            result["done_data"] = done_event
            result["elapsed"] = done_event.get("elapsed", 0)
            result["scores"] = score_result(done_event)
            tc = done_event.get("total_companies",0)
            ta = done_event.get("total_articles",0)
            ov = result["scores"]["overall"]
            rel = result["scores"]["relevance_label"]
            print(f"  -> {tc} companies | {ta} articles | score={ov}/10 | {rel} | {result['elapsed']}s")
        else:
            result["status"] = "no_done_event"
            print(f"  -> No 'done' event received")
    except requests.exceptions.Timeout:
        result["status"] = "timeout"
        result["error"] = "Timeout (180s)"
        print(f"  -> TIMEOUT")
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"  -> ERROR: {e}")
    return result

def main():
    output_file = "/Users/sumedhbang/Public Data Agent/eval_results.json"
    print("=" * 70)
    print("PUBLIC DATA AGENT — 50-QUERY BATCH EVALUATION")
    print(f"Endpoint: {BASE_URL}")
    print("=" * 70)

    results = []
    for i, q in enumerate(QUERIES):
        r = run_query(i, q)
        results.append(r)
        time.sleep(1)

    # Summary
    success = [r for r in results if r["status"] == "success"]
    failed  = [r for r in results if r["status"] != "success"]
    high = [r for r in success if r["scores"]["relevance_label"] == "HIGH"]
    med  = [r for r in success if r["scores"]["relevance_label"] == "MEDIUM"]
    low  = [r for r in success if r["scores"]["relevance_label"] == "LOW"]

    avg_score = sum(r["scores"]["overall"] for r in success)/len(success) if success else 0
    avg_companies = sum(r["done_data"].get("total_companies",0) for r in success)/len(success) if success else 0
    avg_elapsed = sum(r["elapsed"] for r in success)/len(success) if success else 0

    summary = {
        "total_queries": len(results),
        "successful": len(success),
        "failed": len(failed),
        "avg_quality_score": round(avg_score,1),
        "high_relevance": len(high),
        "medium_relevance": len(med),
        "low_relevance": len(low),
        "avg_companies_per_query": round(avg_companies,1),
        "avg_response_time_s": round(avg_elapsed,1),
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Successful: {len(success)}/50  |  Failed: {len(failed)}/50")
    print(f"Avg Score:  {avg_score:.1f}/10")
    print(f"HIGH: {len(high)}  |  MEDIUM: {len(med)}  |  LOW: {len(low)}")
    print(f"Avg companies/query: {avg_companies:.1f}  |  Avg time: {avg_elapsed:.1f}s")
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
