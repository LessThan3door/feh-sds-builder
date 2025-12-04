const API_BASE = window.location.origin; // works on Render

async function fetchTopUnits(n) {
  try {
    const res = await fetch(API_BASE + `/top-units?n=${n}`);
    const data = await res.json();

    if (data.error) {
      showError("Failed to get top units: " + data.error);
      return [];
    }

    return data.units || [];
  } catch (e) {
    showError("Network error while getting top units: " + e.toString());
    return [];
  }
}


async function postJSON(url, data) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data)
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error('Server error: ' + res.status + ' ' + txt);
  }
  return res.json();
}

function parseSeedUnits(text) {
  // Split into lines
  const lines = text.trim().split('\n');

  // Convert each line into an array of unit names
  const seedUnits = lines.map(line => {
    if (!line.trim()) return [];
    return line
      .split(',')
      .map(u => u.trim())
      .filter(Boolean);
  });

  // Ensure exactly 4 teams (pad with empty arrays if needed)
  while (seedUnits.length < 4) seedUnits.push([]);
  if (seedUnits.length > 4) seedUnits.length = 4;

  return seedUnits;
}


function showError(msg){ document.getElementById('errors').innerHTML = '<div class="error">'+msg+'</div>'; }
function clearError(){ document.getElementById('errors').innerHTML = ''; }

let current_results = [];
let banned_assignments = [];
let undo_stack = [];
let last_all_available_units = [];
let last_must_use = [];

function snapshot(){ return {results: JSON.parse(JSON.stringify(current_results)), bans: JSON.parse(JSON.stringify(banned_assignments))}; }
function pushUndo(){ undo_stack.push(snapshot()); if(undo_stack.length>30) undo_stack.shift(); }
function doUndo(){ if(!undo_stack.length){ showError('Nothing to undo'); return; } const s=undo_stack.pop(); current_results=s.results; banned_assignments=s.bans; renderTeams(); }

function renderTeams(){
  const container = document.getElementById('teams'); container.innerHTML = '';
  current_results.forEach((obj, idx)=>{
    const card = document.createElement('div'); card.className='team-card'; card.dataset.team=idx;
    card.addEventListener('dragover',e=>{ e.preventDefault(); card.classList.add('drop-target'); });
    card.addEventListener('dragleave',()=>card.classList.remove('drop-target'));
    card.addEventListener('drop',e=>{ e.preventDefault(); card.classList.remove('drop-target'); try{ const payload=JSON.parse(e.dataTransfer.getData('text/plain')); if(payload.unit){ pushUndo(); moveUnit(payload.unit, payload.from, idx); } }catch(err){} });
    const h = document.createElement('h3');
    const captainSkill = obj.captain_skill || (obj.team && obj.team.captain_skill);
    h.textContent = 'Team ' + (idx+1) + (captainSkill ? ' — ' + captainSkill : '');
    card.appendChild(h);

    const ol = document.createElement('ol');
    // normalize the team array whether obj.team is array or {team: [...]} or already array
    const teamArr = Array.isArray(obj.team) ? obj.team : (Array.isArray(obj.team && obj.team.team) ? obj.team.team : []);
    (teamArr || []).forEach(u=>{
      const li=document.createElement('li'); li.className='unit-item'; li.textContent=u; li.setAttribute('draggable','true'); li.addEventListener('dragstart',ev=>{ ev.dataTransfer.setData('text/plain', JSON.stringify({unit:u, from:idx})); li.classList.add('dragging'); }); li.addEventListener('dragend',ev=>{ li.classList.remove('dragging'); }); const rem=document.createElement('button'); rem.className='tiny-btn'; rem.textContent='Remove'; rem.addEventListener('click',()=>{ pushUndo(); removeUnit(u, idx); }); li.appendChild(rem); ol.appendChild(li); }); card.appendChild(ol);
    container.appendChild(card);
  });
  // controls
  const controls = document.createElement('div'); controls.className='edit-controls';
  const rerun = document.createElement('button'); rerun.textContent='Re-Run Builder (fill slots)'; rerun.addEventListener('click',()=>{ pushUndo(); regenerateFromEdits(); });
  const undo = document.createElement('button'); undo.textContent='Undo'; undo.addEventListener('click',()=>doUndo());
  const reset = document.createElement('button'); reset.textContent='Reset Edits'; reset.addEventListener('click',()=>{ pushUndo(); resetEdits(); });
  controls.appendChild(rerun); controls.appendChild(undo); controls.appendChild(reset);
  container.appendChild(controls);
}

function getTeamArray(obj){
  return Array.isArray(obj.team) ? obj.team : obj.team.team;
}

function removeUnit(unit, team){
  const arr = getTeamArray(current_results[team]);
  current_results[team].team = arr.filter(u=>u!==unit);
  banned_assignments.push({unit:unit, team:team});
  renderTeams();
}

function moveUnit(unit, from, to){
  const fromArr = getTeamArray(current_results[from]);
  const toArr = getTeamArray(current_results[to]);

  current_results[from].team = fromArr.filter(u=>u!==unit);
  if(!toArr.includes(unit)) toArr.push(unit);

  current_results[to].team = toArr;
  banned_assignments.push({unit:unit, team:from});
  renderTeams();
}

async function regenerateFromEdits(){
  clearError();
  const edited = current_results.map(x => {
    if (Array.isArray(x.team)) return x.team.slice();
    if (x.team && Array.isArray(x.team.team)) return x.team.team.slice();
    return [];
  });
  const numTeams = current_results.length;
  const payload = { edited_teams: edited, banned_assignments: banned_assignments.slice(), all_available_units: last_all_available_units.slice(), must_use_units: last_must_use.slice(), num_teams: numTeams };
  try{
    const res = await postJSON(API_BASE + '/regenerate', payload);
    current_results = res;
    renderTeams();
  }catch(e){ showError(e.toString()); }
}

function resetEdits(){ banned_assignments = []; document.getElementById('generate').click(); }

document.getElementById('generate').addEventListener('click', async ()=>{
  clearError();
  banned_assignments = [];
  const available = document.getElementById('available_units').value.split('\n').map(s=>s.trim()).filter(Boolean);
  const raw = document.getElementById('seed_units').value.trim();
  const seed = parseSeedUnits(raw);
  const must_use = document.getElementById('must_use_units').value.split('\n').map(s=>s.trim()).filter(Boolean);
  if(available.length===0){ showError('Please provide available units'); return; }
  last_all_available_units = available.slice(); last_must_use = must_use.slice();
  try{
    const numTeams = parseInt(document.getElementById('num_teams').value) || 4;
    const res = await postJSON(API_BASE + '/generate', { available_units: available, seed_units: seed, must_use_units: must_use, num_teams: numTeams });
    current_results = res;
    undo_stack = []; pushUndo();
    renderTeams();
  }catch(e){ showError(e.toString()); }
});

document.getElementById('autofill').addEventListener('click', async () => {
  clearError();

  const raw = document.getElementById('autofill_amount').value;
  const n = parseInt(raw);

  if (isNaN(n) || n <= 0) {
    showError("Please enter a valid positive number (e.g., 50)");
    return;
  }

  const units = await fetchTopUnits(n);
  if (!units.length) {
    showError("No units returned from server.");
    return;
  }

  document.getElementById('available_units').value = units.join('\n');
});


document.getElementById('upload').addEventListener('click', async ()=>{
  const f = document.getElementById('csvfile').files[0]; if(!f){ alert('No file'); return; }
  const fd = new FormData(); fd.append('file', f);
  try{
    const r = await fetch(API_BASE + '/upload-csv', { method:'POST', body: fd }); const j = await r.json(); alert('Uploaded: '+j.filename);
  }catch(e){ showError('Upload failed: '+e.toString()); }
});

document.getElementById('clear').addEventListener('click', ()=>{ document.getElementById('available_units').value=''; document.getElementById('seed_units').value=''; document.getElementById('must_use_units').value=''; document.getElementById('csvfile').value=''; current_results=[]; banned_assignments=[]; undo_stack=[]; renderTeams(); });
