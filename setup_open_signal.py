import re

with open('/Users/sumedhbang/Downloads/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Narrative Changes
content = content.replace("Reo.Dev", "Open Signal")
content = content.replace("Public Data", "Open Signal")
content = content.replace("SDR Intelligence Agent", "Open Signal Agent")
content = content.replace("Hi I am <em>Reon</em>", "Hi I am <em>Open Signal</em>")
content = content.replace("Reon will create", "Open Signal will create")
content = content.replace("Reon prompt", "Open Signal prompt")
content = content.replace("Reo agent library", "Open Signal agent library")
content = content.replace("reo-signals", "open-signals")

# Endpoint Logic
# We'll replace the runQ function completely.
runq_replacement = """
const API = '/api/ask';
const PHASE_MAP = {
  'phase0_routing': 'query',
  'phase1_query_generation': 'query',
  'phase2_search': 'search',
  'phase3_scraping': 'scrape',
  'phase4_extraction': 'extract',
  'phase5_synthesis': 'score'
};

async function runQ(q){
  if(run)return;run=true;
  document.getElementById('suggs').style.display='none';
  document.getElementById('sendBtn').disabled=true;
  addU(q);
  document.getElementById('empty').style.display='none';
  document.getElementById('pip').style.display='block';
  document.getElementById('resH').style.display='none';
  document.getElementById('tsc').style.display='none';
  document.getElementById('san').style.display='none';
  document.getElementById('tb').innerHTML='';
  ['query','search','scrape','extract','score'].forEach(rPh);
  sLg('Connecting…');
  addTh('th');

  try {
    const resp = await fetch(API, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({question: q}),
    });
    if (!resp.ok) throw new Error(`Server error ${resp.status}: ${await resp.text()}`);

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      const parts = buf.split('\\n\\n');
      buf = parts.pop();
      for (const part of parts) {
        const lines = part.trim().split('\\n');
        let event='message', dataStr='';
        for (const l of lines) {
          if (l.startsWith('event:')) event=l.slice(6).trim();
          if (l.startsWith('data:'))  dataStr=l.slice(5).trim();
        }
        if (dataStr) {
          const data = JSON.parse(dataStr);
          if (event === 'start') {
            sLg('Pipeline started…');
          } else if (event === 'phase') {
            const currPhase = PHASE_MAP[data.phase] || 'query';
            sPh(currPhase, data.status);
            sLg(data.msg || '');
            const e=document.getElementById('th');
            if(e && data.status==='running') e.querySelector('.fth-t').textContent=data.msg;
          } else if (event === 'done') {
            const e=document.getElementById('th');if(e)e.remove();
            ['query','search','scrape','extract','score'].forEach(p=>sPh(p,'done'));
            sLg(`Complete — ${data.total_companies||0} companies`);
            
            // Format the list of evidence items before rendering to match MOCK expectations
            renRes(data);
            
            // Render Answer synthesis from Claude
            let ans = data.answer || '';
            const n = aR.length || 0;
            const san = data.sanity_report||{};
            let msg = `<div class="fan">Open Signal Agent</div>
              <div style="margin-bottom:8px">${ans.replace(/\\n/g, '<br/>')}</div>
              <p style="margin-top:4px;font-size:11px;color:var(--tm)">${n} signals collected. ${san.rejected ? `🛡 ${san.rejected} stale results removed.` : ''}</p>`;
            addA(msg);
            
            if(n > 0) addFU();
            const el=document.getElementById('elap');
            if(data.elapsed) {
                el.textContent=`${data.elapsed}s`;
                el.style.display='inline';
            }
          }
        }
      }
    }
  } catch(err) {
    const e=document.getElementById('th');if(e)e.remove();
    sLg('Error: ' + err.message);
    addA(`<div class="fan">Error</div><p>${err.message}</p>`);
  }
  
  run=false;
  document.getElementById('sendBtn').disabled=false;
}
"""

runQ_pattern = re.compile(r'function runQ\(q\)\{.*?dl\);\n\}', re.DOTALL)
content = runQ_pattern.sub(runq_replacement, content)

with open('/Users/sumedhbang/Public Data Agent/public/open-signal.html', 'w', encoding='utf-8') as f:
    f.write(content)
print("Saved UI to public/open-signal.html")
