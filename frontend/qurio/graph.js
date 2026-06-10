// ═══════════════════════════════════════════════════════════
// graph.js — Vis.js Graph + Sessions + Retrieval panels
// ═══════════════════════════════════════════════════════════

// ── GRAPH DEFINITIONS ────────────────────────────────────
// Accurate nodes matching graph.py architecture
const GRAPH_NODES = [
  { id: 'START',      label: 'START\nUser Input',     color: '#FFD700', shape: 'ellipse', level: 0 },
  { id: 'classify',   label: 'classify_intent\nSemantic Router',  color: '#3B82F6', level: 1 },
  { id: 'agent',      label: 'agent_execute\nReAct · Qwen3.5',   color: '#8B5CF6', level: 2 },
  { id: 'tool',       label: 'tool_node\nSQLite Tool',           color: '#F59E0B', level: 3 },
  { id: 'rag',        label: 'prompt_llm_search_kb\nRAG · ChromaDB',  color: '#10B981', level: 2 },
  { id: 'chat',       label: 'prompt_llm_chat\nDirect LLM',      color: '#6366F1', level: 2 },
  { id: 'END',        label: 'END\nAI Response',      color: '#FFD700', shape: 'ellipse', level: 4 },
];

const GRAPH_EDGES = [
  { from: 'START',    to: 'classify',  label: 'user query' },
  { from: 'classify', to: 'agent',     label: 'query_db' },
  { from: 'classify', to: 'rag',       label: 'search_kb' },
  { from: 'classify', to: 'chat',      label: 'chat' },
  { from: 'agent',    to: 'tool',      label: 'tool_call', dashes: true },
  { from: 'tool',     to: 'agent',     label: 'result', dashes: true },
  { from: 'agent',    to: 'END',       label: 'answer' },
  { from: 'rag',      to: 'END',       label: 'answer' },
  { from: 'chat',     to: 'END',       label: 'answer' },
];

function makeNodeData(n, latency = null) {
  const base = n.color;

  let labelText = n.label;
  if (latency !== null) labelText += `\n⏱ ${latency}ms`;

  return {
    id: n.id,
    label: labelText,
    level: n.level,
    shape: n.shape || 'box',
    font: { color: '#E2E8F0', size: 11, face: 'JetBrains Mono, monospace', multi: false },
    color: {
      background: '#0E1117',
      border: base,
      highlight: { background: '#0E1117', border: '#FFD700' },
    },
    borderWidth: 2,
    margin: { top: 10, right: 14, bottom: 10, left: 14 },
  };
}

function initGraph() {
  const container = document.getElementById('graph-canvas');
  nodesDS = new vis.DataSet(GRAPH_NODES.map(n => makeNodeData(n)));
  edgesDS = new vis.DataSet(GRAPH_EDGES.map((e, i) => ({
    id: i,
    from: e.from,
    to: e.to,
    label: e.label || '',
    dashes: e.dashes || false,
    arrows: { to: { enabled: true, scaleFactor: 0.8 } },
    color: { color: '#2A3340', highlight: '#FFD700', hover: '#3B82F6' },
    font: { color: '#64748B', size: 10, face: 'Inter, sans-serif', align: 'middle', background: '#0E1117', strokeWidth: 0 },
    smooth: { enabled: true, type: 'cubicBezier', roundness: 0.5 },
    width: 2,
  })));

  network = new vis.Network(container, { nodes: nodesDS, edges: edgesDS }, {
    physics: false,
    layout: {
      hierarchical: {
        enabled: true,
        direction: 'LR',
        sortMethod: 'directed',
        levelSeparation: 220,
        nodeSpacing: 90,
        shakeTowards: 'roots',
      }
    },
    interaction: { hover: true, tooltipDelay: 300, zoomView: true, dragView: true },
  });

  network.once('afterDrawing', () => { network.fit({ animation: { duration: 500 } }); });

  network.on('click', params => {
    if (params.nodes.length > 0) {
      openNodePanel(params.nodes[0]);
    } else {
      closeRightPanel();
    }
  });
}

function graphFit() { network?.fit({ animation: { duration: 400 } }); }
function graphZoomIn() { if (network) network.moveTo({ scale: network.getScale() * 1.3, animation: { duration: 200 } }); }
function graphZoomOut() { if (network) network.moveTo({ scale: network.getScale() * 0.75, animation: { duration: 200 } }); }

// ── NODE PANEL ────────────────────────────────────────────
function openNodePanel(nodeId) {
  const nodeDef = GRAPH_NODES.find(n => n.id === nodeId);
  if (!nodeDef) return;

  const panel = document.getElementById('right-panel');
  panel.className = 'open';

  document.getElementById('rp-title').textContent = nodeDef.id;
  document.getElementById('rp-node-badge').textContent = nodeDef.id;

  // Find trace data for this node if we have a last trace
  let traceData = null;
  if (lastTrace?.node_trace) {
    const nodeMap = {
      classify: 'classify_intent', agent: 'agent_execute',
      tool: 'tool_node', rag: 'prompt_llm_search_knowledge_base',
      chat: 'prompt_llm_chat'
    };
    const backendName = nodeMap[nodeId] || nodeId;
    traceData = lastTrace.node_trace.filter(n => n.node === backendName);
  }

  const body = document.getElementById('rp-body');
  body.innerHTML = '';

  // Node description
  const descs = {
    START: 'Entry point of the graph. Receives the raw user message.',
    classify: 'Embedding-based semantic router using bge-base-en-v1.5. Classifies intent without LLM calls. <15ms.',
    agent: 'ReAct executor powered by Qwen3.5-2B-AWQ. Calls query_database tool via tool_choice=auto.',
    tool: 'Wrapped ToolNode executing the query_database SQLite tool. Validates SQL before execution.',
    rag: 'Full RAG pipeline: ChromaDB vector search → parent chunk expansion → LLM generation.',
    chat: 'Direct LLM call for conversational queries. No tools, no RAG.',
    END: 'Terminal node. The final AIMessage content is returned to the user.',
  };

  body.innerHTML += rpSection('Description', `<div class="rp-value" style="font-family:var(--font-sans);font-size:12px;line-height:1.6">${descs[nodeId] || nodeId}</div>`);

  if (traceData && traceData.length > 0) {
    const d = traceData[traceData.length - 1]; // most recent
    const ms = d.duration_ms || 0;
    const maxMs = 3000;
    const pct = Math.min(100, (ms / maxMs) * 100).toFixed(1);

    body.innerHTML += rpSection('Last Execution', `
      <div class="rp-timing-bar">
        <div class="rp-timing-track"><div class="rp-timing-fill" style="width:${pct}%"></div></div>
        <span class="rp-timing-ms">${ms}ms</span>
      </div>
      <div class="rp-kv-list" style="margin-top:8px">
        <div class="rp-kv"><span class="rp-key">turn</span><span class="rp-val">${d.turn}</span></div>
        <div class="rp-kv"><span class="rp-key">msgs_in</span><span class="rp-val">${d.incoming_state?.message_count || 0}</span></div>
        <div class="rp-kv"><span class="rp-key">intent</span><span class="rp-val">${d.incoming_state?.message_intent || '—'}</span></div>
      </div>`);

    // Outgoing payload
    const out = d.outgoing_state || {};
    body.innerHTML += rpSection('Outgoing Payload', `<div class="rp-value">${escHtml(JSON.stringify(out, null, 2))}</div>`);

    // Incoming payload
    const inc = d.incoming_state || {};
    body.innerHTML += rpSection('Incoming Payload', `<div class="rp-value">${escHtml(JSON.stringify(inc, null, 2))}</div>`);
  } else if (nodeId === 'START' && lastTrace) {
    const userMsg = lastTrace.user_message || lastTrace.query || (lastTrace.conversation && lastTrace.conversation.find(m => m.role === 'human')?.content) || 'No query available';
    body.innerHTML += rpSection('User Input', `<div class="sql-block" style="color:var(--yellow);font-size:12px;white-space:pre-wrap;font-family:var(--font-mono)">${escHtml(userMsg)}</div>`);
  } else if (nodeId === 'END' && lastTrace) {
    const aiMsg = lastTrace.ai_response || lastTrace.answer || (lastTrace.conversation && [...lastTrace.conversation].reverse().find(m => m.role === 'ai')?.content) || 'No response available';
    body.innerHTML += rpSection('Final AI Response', `<div class="sql-block" style="color:var(--yellow);font-size:12px;white-space:pre-wrap;font-family:var(--font-sans)">${escHtml(aiMsg)}</div>`);
  } else {
    body.innerHTML += `<p style="color:var(--muted);font-size:12px">Send a query to see live execution data for this node.</p>`;
  }
}

function rpSection(label, content) {
  return `<div class="rp-section"><div class="rp-label">${label}</div>${content}</div>`;
}

function closeRightPanel() {
  const panel = document.getElementById('right-panel');
  panel.className = 'right-panel-closed';
}

// ── GRAPH ANIMATION ──────────────────────────────────────
async function animateGraphTrace(nodeTrace) {
  if (!nodesDS || !edgesDS || !nodeTrace.length) return;

  // Reset all nodes and edges
  GRAPH_NODES.forEach(n => nodesDS.update(makeNodeData(n)));
  edgesDS.get().forEach(e => edgesDS.update({ id: e.id, width: 2, color: { color: '#2A3340' } }));

  const nodeMap = {
    classify_intent: 'classify', agent_execute: 'agent',
    tool_node: 'tool', prompt_llm_search_knowledge_base: 'rag',
    prompt_llm_chat: 'chat'
  };

  let prev = 'START';
  const latencies = {};
  nodeTrace.forEach(n => {
    const gid = nodeMap[n.node];
    if (gid) latencies[gid] = n.duration_ms;
  });

  // Highlight START
  const startDef = GRAPH_NODES.find(n => n.id === 'START');
  nodesDS.update(makeNodeData(startDef));
  await delay(200);

  for (const traceNode of nodeTrace) {
    const gid = nodeMap[traceNode.node];
    if (!gid) continue;

    const nodeDef = GRAPH_NODES.find(n => n.id === gid);
    if (!nodeDef) continue;

    // Highlight edge from prev → current
    const edge = edgesDS.get({ filter: e => e.from === prev && e.to === gid });
    if (edge.length) {
      edgesDS.update({ id: edge[0].id, width: 4, color: { color: '#FFD700', highlight: '#FFD700' } });
    }

    // Update node with latency but no highlight
    nodesDS.update(makeNodeData(nodeDef, traceNode.duration_ms));
    await delay(400);

    prev = gid;
  }

  // Highlight to END
  const endDef = GRAPH_NODES.find(n => n.id === 'END');
  const lastEdge = edgesDS.get({ filter: e => e.from === prev && e.to === 'END' });
  if (lastEdge.length) {
    edgesDS.update({ id: lastEdge[0].id, width: 4, color: { color: '#FFD700', highlight: '#FFD700' } });
  }
  nodesDS.update(makeNodeData(endDef));
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

// ── SESSIONS TAB ─────────────────────────────────────────
async function loadSessions() {
  document.getElementById('sessions-loading').style.display = 'flex';
  document.getElementById('sessions-list').innerHTML = '';
  try {
    const r = await fetch(`${API}/api/sessions`);
    const data = await r.json();
    renderSessions(data.sessions || []);
  } catch {
    document.getElementById('sessions-loading').innerHTML = '<p style="color:var(--red)">Failed to load sessions</p>';
  }
}

function renderSessions(sessions) {
  document.getElementById('sessions-loading').style.display = 'none';
  const list = document.getElementById('sessions-list');
  if (!sessions.length) {
    list.innerHTML = '<p style="color:var(--muted);font-size:13px;padding:24px">No sessions found. Run the agent first.</p>';
    return;
  }

  list.innerHTML = sessions.map(s => {
    const intentTags = (s.intents || []).map(i => {
      const cls = i === 'chat' ? 'intent-chat' : i === 'query_user_information' ? 'intent-query' : 'intent-search';
      return `<span class="intent-tag ${cls}">${i.replace(/_/g,' ')}</span>`;
    }).join('');

    return `
      <div class="session-card" onclick="openSessionDetail('${s.id}', this)">
        <div class="session-card-header">
          <span class="session-id">${s.id.slice(0,18)}…</span>
          <span class="session-time">${s.session_start ? new Date(s.session_start).toLocaleString() : s.filename}</span>
        </div>
        <div class="session-stats">
          <div class="session-stat"><div class="session-stat-val">${s.node_count}</div><div class="session-stat-key">nodes</div></div>
          <div class="session-stat"><div class="session-stat-val">${s.message_count}</div><div class="session-stat-key">messages</div></div>
          <div class="session-stat"><div class="session-stat-val">${s.file_size_kb}kb</div><div class="session-stat-key">log size</div></div>
        </div>
        <div class="session-intents">${intentTags || '<span style="font-size:10px;color:var(--muted)">no intents</span>'}</div>
      </div>`;
  }).join('');
}

async function openSessionDetail(sessionId, cardEl) {
  document.querySelectorAll('.session-card').forEach(c => c.classList.remove('selected'));
  cardEl.classList.add('selected');

  const panel = document.getElementById('session-detail');
  panel.classList.add('open');
  document.getElementById('detail-session-title').textContent = 'Loading...';
  document.getElementById('detail-body').innerHTML = '<div class="loading-state"><div class="spinner"></div><p>Loading session...</p></div>';

  try {
    const r = await fetch(`${API}/api/sessions/${sessionId}`);
    const data = await r.json();
    renderSessionDetail(data);
  } catch (e) {
    document.getElementById('detail-body').innerHTML = `<p style="color:var(--red)">Error: ${e.message}</p>`;
  }
}

function renderSessionDetail(data) {
  document.getElementById('detail-session-title').textContent = data.session_id || 'Session';
  document.getElementById('detail-session-sub').textContent = data.session_start || '';

  const nodeColors = {
    classify_intent: '#3B82F6', agent_execute: '#8B5CF6',
    tool_node: '#F59E0B', prompt_llm_search_knowledge_base: '#10B981',
    prompt_llm_chat: '#6366F1'
  };

  const timeline = (data.node_trace || []).map((n, i) => {
    const color = nodeColors[n.node] || '#64748B';
    const isLast = i === (data.node_trace || []).length - 1;
    const out = n.outgoing_state || {};
    let detail = '';
    if (out.message_intent) detail += `Intent: ${out.message_intent} · `;
    if (out.messages_appended) detail += `${out.messages_appended} msg appended · `;
    detail += `${n.duration_ms}ms`;

    return `
      <div class="node-trace-item">
        <div class="node-trace-line">
          <div class="node-trace-dot" style="background:${color}"></div>
          ${!isLast ? '<div class="node-trace-connector"></div>' : ''}
        </div>
        <div class="node-trace-content">
          <div class="node-trace-header">
            <span class="node-trace-name">${n.node}</span>
            <span class="node-trace-ms">${n.duration_ms}ms</span>
          </div>
          <div class="node-trace-detail">${detail}</div>
        </div>
      </div>`;
  }).join('');

  const convo = (data.conversation || []).map(m => {
    const roleColor = m.role === 'human' ? 'var(--yellow)' : m.role === 'tool' ? 'var(--amber)' : 'var(--muted-bright)';
    return `<div style="margin-bottom:10px;padding:8px 10px;background:var(--panel);border:1px solid var(--border);border-radius:6px">
      <div style="font-size:10px;font-weight:700;letter-spacing:0.08em;color:${roleColor};margin-bottom:4px;text-transform:uppercase">${m.role}</div>
      <div style="font-size:12px;color:var(--muted-bright);line-height:1.5;white-space:pre-wrap">${escHtml((m.content || '').slice(0, 400))}${(m.content || '').length > 400 ? '…' : ''}</div>
      ${m.tool_calls ? `<div style="margin-top:6px;font-family:var(--font-mono);font-size:10px;color:var(--amber)">🔧 ${m.tool_calls[0]?.tool}(${JSON.stringify(m.tool_calls[0]?.args || {})})</div>` : ''}
    </div>`;
  }).join('');

  document.getElementById('detail-body').innerHTML = `
    <div style="margin-bottom:20px">
      <div class="node-timeline-title">Execution Timeline (${(data.node_trace||[]).length} nodes)</div>
      <div class="node-timeline">${timeline}</div>
    </div>
    <div>
      <div class="node-timeline-title">Conversation (${(data.conversation||[]).length} messages)</div>
      ${convo}
    </div>`;
}

function closeSessionDetail() {
  document.getElementById('session-detail').classList.remove('open');
  document.querySelectorAll('.session-card').forEach(c => c.classList.remove('selected'));
}

// ── RETRIEVAL TAB ─────────────────────────────────────────
async function loadRetrievalTraces() {
  document.getElementById('retrieval-loading').style.display = 'flex';
  document.getElementById('retrieval-list').innerHTML = '';
  try {
    const r = await fetch(`${API}/api/retrieval-traces?limit=50`);
    const data = await r.json();
    renderRetrievalTraces(data.traces || []);
  } catch {
    document.getElementById('retrieval-loading').innerHTML = '<p style="color:var(--red)">Failed to load traces</p>';
  }
}

function renderRetrievalTraces(traces) {
  document.getElementById('retrieval-loading').style.display = 'none';
  const list = document.getElementById('retrieval-list');
  if (!traces.length) {
    list.innerHTML = '<p style="color:var(--muted);font-size:13px;padding:24px">No retrieval traces yet. Run a knowledge-base query first.</p>';
    return;
  }

  list.innerHTML = traces.map((t, i) => {
    const ts = t.timestamp ? new Date(t.timestamp).toLocaleString() : '';
    const chunks = t.top_k_chunks || [];
    const ms = t.duration_ms || 0;
    const noCtx = t.no_context;

    const chunksHtml = chunks.map(c => {
      const conf = c.confidence || 'low';
      const sim = ((c.similarity || 0) * 100).toFixed(1);
      return `
        <div class="chunk-item conf-${conf}">
          <div class="chunk-header">
            <span class="chunk-id" onclick="event.stopPropagation(); openChunkModal('chunk', '${c.chunk_id}')" style="cursor:pointer;text-decoration:underline" title="Click to view chunk text">${c.chunk_id || '—'}</span>
            <span class="chunk-sim ${conf}">${sim}%</span>
          </div>
          ${c.parent_id ? `<div class="chunk-parent" onclick="event.stopPropagation(); openChunkModal('parent', '${c.parent_id}')" style="cursor:pointer;text-decoration:underline" title="Click to view parent section text">parent: ${c.parent_id}</div>` : ''}
        </div>`;
    }).join('');

    return `
      <div class="retrieval-card">
        <div class="retrieval-card-header" onclick="toggleRetrieval(this)">
          <span class="retrieval-query">${escHtml(t.query || '—')}</span>
          <div class="retrieval-meta">
            ${noCtx ? '<span style="font-size:10px;color:var(--red);font-weight:700">NO_CONTEXT</span>' : ''}
            <span class="retrieval-ms">${ms}ms</span>
            <span class="retrieval-time">${ts}</span>
            <svg class="retrieval-chevron" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"/></svg>
          </div>
        </div>
        <div class="retrieval-card-body">
          <div style="font-size:11px;color:var(--muted);margin-bottom:8px">
            ${chunks.length} chunks · ${t.context_block_count || 0} parents expanded · 
            areas: ${(t.product_areas || []).join(', ') || '—'}
          </div>
          <div class="chunk-list">${chunksHtml}</div>
        </div>
      </div>`;
  }).join('');
}

function toggleRetrieval(header) {
  const body = header.nextElementSibling;
  const chevron = header.querySelector('.retrieval-chevron');
  body.classList.toggle('open');
  header.classList.toggle('open');
  chevron.classList.toggle('open');
}

async function openChunkModal(type, id) {
  document.getElementById('db-modal-overlay').classList.add('open');
  document.getElementById('modal-table-name').textContent = `${type.toUpperCase()}: ${id}`;
  document.getElementById('modal-table-info').textContent = 'Loading...';
  document.getElementById('modal-body').innerHTML = '<div class="loading-state"><div class="spinner"></div></div>';

  try {
    const r = await fetch(`${API}/api/knowledge-base/item?type=${type}&id=${encodeURIComponent(id)}`);
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'Not found');
    
    document.getElementById('modal-table-info').textContent = data.item.source_file ? `Source File: ${data.item.source_file}` : 'Content';
    document.getElementById('modal-body').innerHTML = `
      <div style="padding:20px;color:var(--text);font-size:13px;line-height:1.6;white-space:pre-wrap;font-family:var(--font-mono);max-height:60vh;overflow-y:auto;background:var(--bg);border-radius:8px;border:1px solid var(--border)">${escHtml(data.item.text)}</div>
    `;
  } catch(e) {
    document.getElementById('modal-body').innerHTML = `<p style="color:var(--red);padding:20px">${e.message}</p>`;
  }
}
