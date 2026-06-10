// ═══════════════════════════════════════════════════════════
// app.js — Qurio Observability Frontend
// ═══════════════════════════════════════════════════════════

const API = 'http://localhost:8080';
let currentSessionId = null;
let network = null;
let nodesDS = null;
let edgesDS = null;
let lastTrace = null;

// ── STARTUP ──────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  checkApiHealth();
  loadSidebarData();
  initGraph();
  setupTabNav();
  setupChatInput();
  setInterval(checkApiHealth, 15000);
});

// ── API HEALTH ───────────────────────────────────────────
async function checkApiHealth() {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  try {
    const r = await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(3000) });
    if (r.ok) {
      dot.className = 'status-dot online';
      txt.textContent = 'API Online';
    } else throw new Error();
  } catch {
    dot.className = 'status-dot offline';
    txt.textContent = 'API Offline';
  }
}

// ── TAB NAVIGATION ───────────────────────────────────────
function setupTabNav() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => switchTab(btn.dataset.tab));
  });
}

function switchTab(tab) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-view').forEach(v => v.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
  document.getElementById(`view-${tab}`).classList.add('active');

  if (tab === 'graph' && network) setTimeout(() => network.fit(), 100);
  if (tab === 'sessions') loadSessions();
  if (tab === 'retrieval') loadRetrievalTraces();
}

// ── SIDEBAR TOGGLE ───────────────────────────────────────
function toggleSection(id) {
  const section = document.getElementById(id);
  const body = section.querySelector('.section-body');
  const chevron = section.querySelector('.chevron');
  body.classList.toggle('open');
  chevron.classList.toggle('rotated');
}

// ── SIDEBAR DATA ─────────────────────────────────────────
async function loadSidebarData() {
  loadDbTables();
  loadKnowledgeBase();
}

async function loadDbTables() {
  try {
    const r = await fetch(`${API}/api/database/tables`);
    const data = await r.json();
    renderDbTables(data.tables || []);
  } catch {
    document.getElementById('db-loading').innerHTML = '<span style="color:var(--red);font-size:11px">Failed to load</span>';
  }
}

function renderDbTables(tables) {
  const loading = document.getElementById('db-loading');
  const container = document.getElementById('db-tables');
  const countEl = document.getElementById('db-count');
  loading.style.display = 'none';
  countEl.textContent = tables.length;

  container.innerHTML = tables.map(t => `
    <div class="db-table-item" onclick="openDbModal('${t.name}', ${t.row_count})">
      <div class="db-table-left">
        <div class="db-table-dot"></div>
        <span class="db-table-name">${t.name}</span>
      </div>
      <span class="db-table-count">${t.row_count.toLocaleString()} rows</span>
    </div>
    <div class="db-schema-cols">
      ${(t.columns || []).map(c => `<span class="db-col-tag ${c.pk ? 'pk' : ''}">${c.name}</span>`).join('')}
    </div>
  `).join('');

  // Auto-open the section
  const body = document.getElementById('section-db').querySelector('.section-body');
  const chevron = document.getElementById('section-db').querySelector('.chevron');
  body.classList.add('open');
  chevron.classList.add('rotated');
}

async function loadKnowledgeBase() {
  try {
    const r = await fetch(`${API}/api/knowledge-base`);
    const data = await r.json();
    renderKnowledgeBase(data.knowledge_base || {});
  } catch {
    document.getElementById('kb-loading').innerHTML = '<span style="color:var(--red);font-size:11px">Failed to load</span>';
  }
}

function renderKnowledgeBase(kb) {
  const loading = document.getElementById('kb-loading');
  const container = document.getElementById('kb-tree');
  const countEl = document.getElementById('kb-count');
  loading.style.display = 'none';

  const cats = Object.keys(kb);
  countEl.textContent = cats.length;

  container.innerHTML = cats.map((catKey, i) => {
    const cat = kb[catKey];
    const sections = cat.sections || [];
    return `
      <div class="kb-category">
        <div class="kb-category-header" onclick="toggleKbCat(this)">
          <span class="kb-category-name">${catKey.replace(/_/g, ' ')}</span>
          <span class="kb-section-count">${sections.length} sections</span>
        </div>
        <div class="kb-sections ${i === 0 ? 'open' : ''}">
          ${sections.map(s => `
            <div class="kb-section-item">
              <div class="kb-section-dot"></div>
              <div class="kb-section-info">
                <div class="kb-section-heading">${s.heading}</div>
                <div class="kb-section-id">${s.parent_id}</div>
              </div>
              <span class="kb-q-count">${(s.questions || []).length}q</span>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }).join('');

  const body = document.getElementById('section-kb').querySelector('.section-body');
  const chevron = document.getElementById('section-kb').querySelector('.chevron');
  body.classList.add('open');
  chevron.classList.add('rotated');
}

function toggleKbCat(header) {
  const sections = header.nextElementSibling;
  sections.classList.toggle('open');
}

// ── DB MODAL ─────────────────────────────────────────────
async function openDbModal(tableName, rowCount) {
  document.getElementById('db-modal-overlay').classList.add('open');
  document.getElementById('modal-table-name').textContent = tableName;
  document.getElementById('modal-table-info').textContent = `${rowCount.toLocaleString()} rows`;
  document.getElementById('modal-body').innerHTML = `
    <div class="loading-state"><div class="spinner"></div><p>Loading data...</p></div>`;

  try {
    const r = await fetch(`${API}/api/database/table/${tableName}?limit=100`);
    const data = await r.json();
    renderModalTable(data);
  } catch (e) {
    document.getElementById('modal-body').innerHTML = `<p style="color:var(--red)">Error: ${e.message}</p>`;
  }
}

function renderModalTable(data) {
  const { columns, rows, total_rows } = data;
  document.getElementById('modal-table-info').textContent =
    `${total_rows.toLocaleString()} rows total · showing ${rows.length}`;

  const headers = columns.map(c => `<th>${c}</th>`).join('');
  const bodyRows = rows.map(row => {
    const cells = columns.map(c => {
      let val = row[c] ?? '';
      let cls = '';
      if (c === 'direction') cls = val === 'debit' ? 'td-debit' : 'td-credit';
      return `<td class="${cls}">${val}</td>`;
    }).join('');
    return `<tr>${cells}</tr>`;
  }).join('');

  document.getElementById('modal-body').innerHTML = `
    <div class="modal-table-wrap">
      <table class="data-table">
        <thead><tr>${headers}</tr></thead>
        <tbody>${bodyRows}</tbody>
      </table>
    </div>`;
}

function closeDbModal() {
  document.getElementById('db-modal-overlay').classList.remove('open');
}

// ── CHAT ─────────────────────────────────────────────────
function setupChatInput() {
  const input = document.getElementById('chat-input');
  input.addEventListener('keydown', e => { if (e.key === 'Enter' && !e.shiftKey) handleChatSend(); });
  const gi = document.getElementById('graph-input');
  gi.addEventListener('keydown', e => { if (e.key === 'Enter') handleGraphSend(); });
}

function sendExample(text) {
  document.getElementById('chat-input').value = text;
  handleChatSend();
}

async function handleChatSend() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  await sendMessage(msg);
}

async function handleGraphSend() {
  const input = document.getElementById('graph-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  await sendMessage(msg);
}

async function sendMessage(msg) {
  const empty = document.getElementById('chat-empty');
  if (empty) empty.style.display = 'none';

  appendMessage('human', msg);
  const typingEl = appendTyping();
  const sendBtn = document.getElementById('chat-send');
  sendBtn.disabled = true;

  try {
    const r = await fetch(`${API}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg, session_id: currentSessionId })
    });

    typingEl.remove();

    if (!r.ok) {
      const err = await r.json().catch(() => ({ detail: 'Unknown error' }));
      appendMessage('ai', `⚠ Error: ${err.detail}`);
      return;
    }

    const data = await r.json();
    currentSessionId = data.session_id;
    document.getElementById('session-badge').textContent = 's_' + data.session_id.slice(0, 8);

    appendMessage('ai', data.ai_response);
    appendTraceCards(data);
    lastTrace = data;

    // Update graph
    animateGraphTrace(data.node_trace || []);
    document.getElementById('graph-empty')?.classList.add('hidden');

  } catch (e) {
    typingEl.remove();
    appendMessage('ai', `⚠ Cannot reach API. Is the server running at ${API}?`);
  } finally {
    sendBtn.disabled = false;
  }
}

function appendMessage(role, text) {
  const inner = document.getElementById('chat-inner');
  const div = document.createElement('div');
  div.className = `msg-row ${role}`;
  
  let formattedText = role === 'ai' ? formatAiMessage(text) : escHtmlAndBold(text);
  div.innerHTML = `<div class="msg-bubble">${formattedText}</div>`;
  
  inner.appendChild(div);
  scrollChat();
  return div;
}

function formatAiMessage(text) {
  let thinking = "";
  let response = text;

  // Extract thinking block and response block
  const thinkMatch = text.match(/\*\*THINKING:\*\*\s*([\s\S]*?)(?:\*\*RESPONSE:\*\*|$)/i);
  if (thinkMatch) {
    thinking = thinkMatch[1].trim();
    const respMatch = text.match(/\*\*RESPONSE:\*\*\s*([\s\S]*)$/i);
    if (respMatch) {
      response = respMatch[1].trim();
    } else {
      // Fallback: If the LLM forgot the **RESPONSE:** tag, it likely output the answer 
      // inside the thinking block. We render the raw text so the answer isn't hidden.
      thinking = "";
      response = text;
    }
  }

  let html = "";
  if (thinking) {
    html += `
      <div class="agent-thought-process">
        <div class="thought-header" onclick="this.parentElement.classList.toggle('open')">
          <div class="thought-header-left">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
            <span>Agent reasoned before responding</span>
          </div>
          <svg class="thought-chevron" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg>
        </div>
        <div class="thought-content">${escHtmlAndBold(thinking)}</div>
      </div>
    `;
  }
  
  if (response) {
    html += `<div class="agent-response">${escHtmlAndBold(response)}</div>`;
  }
  
  return html || escHtmlAndBold(text);
}

function appendTyping() {
  const inner = document.getElementById('chat-inner');
  const div = document.createElement('div');
  div.className = 'msg-row ai';
  div.innerHTML = `<div class="msg-bubble"><div class="typing-dots">
    <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
  </div></div>`;
  inner.appendChild(div);
  scrollChat();
  return div;
}

function appendTraceCards(data) {
  const inner = document.getElementById('chat-inner');
  const trace = document.createElement('div');
  trace.className = 'trace-block';

  let html = '';

  // 1. Intent card
  if (data.intent) {
    const intentColor = {
      chat: 'label-indigo', query_user_information: 'label-purple',
      search_knowledge_base: 'label-green'
    }[data.intent] || 'label-blue';
    const routerNode = (data.node_trace || []).find(n => n.node === 'classify_intent');
    const ms = routerNode ? `${routerNode.duration_ms}ms` : '';
    html += traceCard(
      'classify_intent', 'label-blue', data.intent, ms,
      `<div class="rp-kv-list">
        <div class="rp-kv"><span class="rp-key">intent</span><span class="rp-val ${intentColor}">${data.intent}</span></div>
        ${ms ? `<div class="rp-kv"><span class="rp-key">latency</span><span class="rp-val">${ms}</span></div>` : ''}
      </div>`
    );
  }

  // 2. Per node cards
  for (const node of (data.node_trace || [])) {
    if (node.node === 'classify_intent') continue;

    const color = nodeColor(node.node);
    const out = node.outgoing_state || {};

    // Tool call: extract SQL
    if (node.node === 'tool_node') {
      const toolContent = out.appended?.[0]?.content || '';
      let parsed = null;
      try { parsed = JSON.parse(toolContent); } catch {}
      const rows = parsed?.rows || [];
      const cols = rows.length > 0 ? Object.keys(rows[0]) : [];
      const sqlNode = (data.node_trace || []).find(n =>
        n.node === 'agent_execute' && n.outgoing_state?.appended?.[0]?.tool_calls?.length
      );
      const sql = sqlNode?.outgoing_state?.appended?.[0]?.tool_calls?.[0]?.args?.sql || '';

      html += traceCard('tool_node', color, `SQL · ${rows.length} rows`, `${node.duration_ms}ms`,
        `${sql ? `<div class="sql-block">${highlightSql(sql)}</div>` : ''}
        ${rows.length ? `<table class="data-table">
          <thead><tr>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead>
          <tbody>${rows.slice(0,10).map(r=>`<tr>${cols.map(c=>{
            let v = r[c] ?? ''; let cls = c==='direction'?(v==='debit'?'td-debit':'td-credit'):'';
            return `<td class="${cls}">${v}</td>`;
          }).join('')}</tr>`).join('')}</tbody>
        </table>` : ''}`
      );
      continue;
    }

    // Agent execute with tool calls
    if (node.node === 'agent_execute') {
      const toolCalls = out.appended?.[0]?.tool_calls || [];
      if (toolCalls.length > 0) {
        html += traceCard('agent_execute', color, `Tool Call: ${toolCalls[0].tool}`, `${node.duration_ms}ms`,
          `<div class="rp-kv-list">
            <div class="rp-kv"><span class="rp-key">tool</span><span class="rp-val" style="color:var(--amber)">${toolCalls[0].tool}</span></div>
            <div class="rp-kv"><span class="rp-key">sql</span><span class="rp-val">${toolCalls[0].args?.sql || ''}</span></div>
          </div>`
        );
      }
      continue;
    }

    // RAG node
    if (node.node === 'prompt_llm_search_knowledge_base') {
      const toolContent = out.appended?.[0]?.content || '';
      // Try to extract chunks if it's JSON or markdown
      let chunksHtml = `<div class="sql-block" style="color:var(--muted-bright);font-size:11.5px;max-height:260px;overflow-y:auto;white-space:pre-wrap;margin-top:6px;line-height:1.5;">${escHtml(toolContent)}</div>`;
      
      html += traceCard(node.node, color, 'RAG · Knowledge Base', `${node.duration_ms}ms`,
        `<div class="rp-kv-list">
          <div class="rp-kv"><span class="rp-key">latency</span><span class="rp-val">${node.duration_ms}ms</span></div>
          <div class="rp-kv"><span class="rp-key">msg_in</span><span class="rp-val">${node.incoming_state?.message_count || 0}</span></div>
        </div>
        ${chunksHtml}`
      );
      continue;
    }

    // Default card
    html += traceCard(node.node, color, node.node.replace(/_/g, ' '), `${node.duration_ms}ms`,
      `<div class="rp-kv-list">
        <div class="rp-kv"><span class="rp-key">duration</span><span class="rp-val">${node.duration_ms}ms</span></div>
        <div class="rp-kv"><span class="rp-key">msgs_in</span><span class="rp-val">${node.incoming_state?.message_count || 0}</span></div>
      </div>`
    );
  }

  trace.innerHTML = html;

  // Wire toggles
  trace.querySelectorAll('.trace-card-header').forEach(h => {
    h.addEventListener('click', () => {
      const body = h.nextElementSibling;
      const chev = h.querySelector('.trace-chevron');
      body.classList.toggle('hidden');
      chev.classList.toggle('open');
    });
  });

  inner.appendChild(trace);
  scrollChat();
}

function traceCard(node, colorClass, title, latency, bodyHtml) {
  return `
    <div class="trace-card">
      <div class="trace-card-header">
        <div class="trace-card-left">
          <span class="trace-node-label ${colorClass}">${node}</span>
          <span class="trace-title">${escHtml(title)}</span>
        </div>
        <div class="trace-card-right">
          <span class="trace-latency">${latency}</span>
          <svg class="trace-chevron open" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
        </div>
      </div>
      <div class="trace-card-body">${bodyHtml}</div>
    </div>`;
}

function nodeColor(name) {
  if (name.includes('classify')) return 'label-blue';
  if (name.includes('agent')) return 'label-purple';
  if (name.includes('tool')) return 'label-amber';
  if (name.includes('search_knowledge')) return 'label-green';
  if (name.includes('chat')) return 'label-indigo';
  return 'label-blue';
}

function scrollChat() {
  const s = document.getElementById('chat-scroll');
  s.scrollTop = s.scrollHeight;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\n/g, '<br>');
}

function escHtmlAndBold(str) {
  let s = escHtml(str);
  // Markdown bolding
  s = s.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  return s;
}

function highlightSql(sql) {
  const kw = ['SELECT','FROM','WHERE','ORDER BY','LIMIT','GROUP BY','HAVING','AND','OR','NOT','IN','LIKE','AS','JOIN','ON','COUNT','SUM','MAX','MIN','AVG','DISTINCT','INSERT','UPDATE','DELETE','DESC','ASC'];
  let s = escHtml(sql);
  kw.forEach(k => {
    s = s.replace(new RegExp(`\\b${k}\\b`, 'gi'),
      m => `<span class="sql-keyword">${m}</span>`);
  });
  return s;
}
