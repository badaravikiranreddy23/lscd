/* ============================================
   LSCD Frontend — main.js
   Handles API calls, state, and DOM updates
   ============================================ */

const LOADING_MESSAGES = [
  "Computing contextual BERT embeddings...",
  "Running K-Means sense induction...",
  "Calculating Jensen-Shannon Divergence...",
  "Generating Flan-T5 definitions...",
  "Assembling results..."
];

let loadingInterval = null;
let msgIndex = 0;

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
  fetchWords();
  document.getElementById('wordInput').addEventListener('keydown', e => {
    if (e.key === 'Enter') analyzeWord();
  });
});

async function fetchWords() {
  try {
    const res = await fetch('/api/words');
    const data = await res.json();
    renderChips('pandemicChips', data.pandemic, 'chip-pandemic');
    renderChips('controlChips',  data.control,   'chip-control');
  } catch {
    const fallback = { pandemic: ['mask','lockdown','variant','essential','bubble'], control: ['remote','viral','spread'] };
    renderChips('pandemicChips', fallback.pandemic, 'chip-pandemic');
    renderChips('controlChips',  fallback.control,   'chip-control');
  }
}

function renderChips(containerId, words, cls) {
  const el = document.getElementById(containerId);
  el.innerHTML = words.map(w =>
    `<button class="chip ${cls}" onclick="quickWord('${w}')">${w}</button>`
  ).join('');
}

function quickWord(w) {
  document.getElementById('wordInput').value = w;
  analyzeWord();
}

// ---- Main analysis ----
async function analyzeWord() {
  const word = document.getElementById('wordInput').value.trim().toLowerCase();
  if (!word) return;

  setLoading(true);
  hideError();
  hideResults();

  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ word })
    });

    const data = await res.json();

    if (!res.ok) {
      showError(data.error || 'Analysis failed.');
      return;
    }

    renderResults(word, data);

  } catch (err) {
    showError('Could not reach the server. Make sure app.py is running.');
  } finally {
    setLoading(false);
  }
}

// ---- Render results ----
function renderResults(word, data) {
  const score = data.score;

  // Score number + color
  const color = getScoreColor(score);
  const numEl = document.getElementById('scoreNumber');
  numEl.textContent = score.toFixed(2);
  numEl.style.color = color.accent;

  // Score bar
  setTimeout(() => {
    const fill = document.getElementById('scoreFill');
    fill.style.width = (score * 100).toFixed(1) + '%';
    fill.style.background = color.accent;
  }, 80);

  // Verdict badge
  const badge = document.getElementById('verdictBadge');
  badge.textContent = data.verdict;
  badge.className = 'verdict-badge ' + color.badgeClass;

  // Mode tag
  const modeEl = document.getElementById('modeTag');
  modeEl.textContent = data.mode === 'live' ? '⬤ Live pipeline' : '◌ Demo mode';

  // Clusters
  renderClusters(data.clusters);

  // Definitions
  renderDefs(data.old_definition, data.new_definition);

  // Paper card
  renderPaperCard(word, score, data.verdict);

  showResults();
}

function getScoreColor(score) {
  if (score >= 0.61) return { accent: '#d85a30', badgeClass: 'verdict-high' };
  if (score >= 0.31) return { accent: '#e8a232', badgeClass: 'verdict-mod' };
  return { accent: '#2db88a', badgeClass: 'verdict-low' };
}

function renderClusters(clusters) {
  if (!clusters || !clusters.length) {
    document.getElementById('clusterList').innerHTML = '<p style="color:var(--text-3);font-size:12px;">No cluster data.</p>';
    return;
  }

  const maxVal = Math.max(...clusters.flatMap(c => [c.pre, c.post]));
  const maxWidth = 160; // px

  document.getElementById('clusterList').innerHTML = clusters.map(c => {
    const preW  = Math.round((c.pre  / maxVal) * maxWidth);
    const postW = Math.round((c.post / maxVal) * maxWidth);
    return `
      <div class="cluster-row">
        <div class="cluster-name">${c.name}</div>
        <div class="bar-pair">
          <div class="bar-wrap">
            <div class="bar bar-pre"  style="width:${preW}px">${c.pre}%</div>
          </div>
          <div class="bar-wrap">
            <div class="bar bar-post" style="width:${postW}px">${c.post}%</div>
          </div>
        </div>
      </div>`;
  }).join('');
}

function renderDefs(oldDef, newDef) {
  const defs = [
    { def: oldDef, eraClass: 'def-era-pre' },
    { def: newDef, eraClass: 'def-era-post' }
  ];

  document.getElementById('defCards').innerHTML = defs.map(({ def, eraClass }) => {
    if (!def || !def.title) return '';
    const examples = (def.examples || []).map(e => `<p class="def-ex">"${e}"</p>`).join('');
    return `
      <div class="def-card">
        <p class="def-era ${eraClass}">${def.era || ''}</p>
        <p class="def-title">${def.title}</p>
        <p class="def-text">${def.text}</p>
        ${examples ? `<div class="def-examples">${examples}</div>` : ''}
      </div>`;
  }).join('');
}

function renderPaperCard(word, score, verdict) {
  let body = '';
  if (score >= 0.61) {
    body = `"${word}" shows a JSD score of ${score.toFixed(2)} — the original Aida & Bollegala paper would label this binary "1 (changed)" and stop there. Our graded score quantifies <em>how much</em> it changed, and the definitions above explain <em>how</em> it changed. Both flaws addressed in one pipeline.`;
  } else if (score >= 0.31) {
    body = `"${word}" shows moderate semantic drift (${score.toFixed(2)}). The original paper's binary output misses this nuance — it may label this "changed" or "not changed" depending on its threshold. Our continuous JSD score captures the partial shift, and the definitions show the emerging new sense without erasure of the old.`;
  } else {
    body = `"${word}" scores ${score.toFixed(2)} — low semantic change. This is our control word confirming the pipeline works: words that shouldn't shift don't. The original paper can detect this too, but cannot tell you it's a 0.${Math.round(score*100)} vs a 0.83 — our graded output makes this distinction explicit.`;
  }
  document.getElementById('paperBody').innerHTML = body;
}

// ---- UI state helpers ----
function setLoading(on) {
  const el   = document.getElementById('loadingState');
  const btn  = document.getElementById('analyzeBtn');
  const fill = document.getElementById('loadingFill');
  const txt  = document.getElementById('loadingText');

  if (on) {
    el.classList.add('active');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    fill.style.animation = 'none';
    fill.offsetHeight; // reflow
    fill.style.animation = 'progress 2.5s ease-in-out forwards';
    msgIndex = 0;
    txt.textContent = LOADING_MESSAGES[0];
    loadingInterval = setInterval(() => {
      msgIndex = (msgIndex + 1) % LOADING_MESSAGES.length;
      txt.textContent = LOADING_MESSAGES[msgIndex];
    }, 500);
  } else {
    el.classList.remove('active');
    btn.disabled = false;
    btn.textContent = 'Analyze';
    clearInterval(loadingInterval);
  }
}

function showResults() {
  document.getElementById('results').classList.add('visible');
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResults() {
  document.getElementById('results').classList.remove('visible');
}

function showError(msg) {
  const el = document.getElementById('errorState');
  el.classList.add('active');
  document.getElementById('errorMsg').textContent = msg;
}

function hideError() {
  document.getElementById('errorState').classList.remove('active');
}
