(() => {
  'use strict';
  // Keep routing local: Chrome may still have the previous manifest cached.
  // The bridge tests check parity with page.js and background.js.
  function explorerRoute(path) {
    if (/^\/analysis(?:\/|$)/.test(path)) return 'analysis';
    if (/^\/study\/\w{8}(?:\/\w{8})?\/?$/.test(path)) return 'study';
    // These eight-letter routes are site tools, rather than game IDs.
    if (/^\/(?:training|practice)(?:\/|$)/.test(path)) return;
    // Finished games are now served under /game/ (older links redirect there).
    if (/^\/(?:game\/)?(?:\w{8}(?:\/(?:white|black)(?:\/analysis)?)?|\w{12})\/?$/.test(path)) return 'game';
  }
  const CHANNEL = 'marvin-explorer-v1';
  if (!explorerRoute(location.pathname)) return;
  const DEFAULTS = {enabled: false, rating: 1800, timeControl: '600+0'};
  const REQUEST_DELAY = 30;  // ms between a position change and its request
  // BEGIN DISPLAY MATH — self-contained so cached manifests need no new script.
  const DISPLAY_DEFAULTS = {model: '79m', includeResignation: false, showMoveTime: false, timeStatistic: 'mode'};
  function displaySettings(value = {}) {
    value = value && typeof value === 'object' ? value : {};
    return {model: ['79m', '23m'].includes(value.model) ? value.model : '79m',
      includeResignation: typeof value.includeResignation === 'boolean' ? value.includeResignation : false,
      showMoveTime: typeof value.showMoveTime === 'boolean' ? value.showMoveTime : false,
      timeStatistic: ['mode', 'typical', 'mean'].includes(value.timeStatistic) ? value.timeStatistic : 'mode'};
  }
  function typicalSeconds(probabilities, temperature, topP) {
    if (!Array.isArray(probabilities) || probabilities.length !== 121 ||
        probabilities.some(p => !Number.isFinite(p) || p < 0)) return undefined;
    const peak = Math.max(...probabilities);
    if (!(peak > 0)) return undefined;
    // Relative logs avoid underflow at low temperature. Zeros remain zeros.
    const ranked = probabilities.map((p, second) => ({second,
      p: p > 0 ? Math.exp((Math.log(p) - Math.log(peak)) / temperature) : 0}))
      .sort((a, b) => b.p - a.p || a.second - b.second);
    const total = ranked.reduce((sum, item) => sum + item.p, 0);
    let mass = 0, weighted = 0;
    for (const item of ranked) {
      mass += item.p; weighted += item.p * item.second;
      if (topP < 1 && mass / total >= topP) break;
    }
    return weighted / mass;
  }
  // Seconds as marvin-engine shows them: hundredths below a second, tenths above.
  function seconds(value) { return value.toFixed(value < 1 ? 2 : 1) + 's'; }
  function timeLabel(timing, options) {
    if (options.timeStatistic === 'mode') {
      if (!Number.isFinite(timing.modeSeconds)) return undefined;
      return timing.modeSeconds >= 119.5 ? '120s+' : seconds(timing.modeSeconds);
    }
    // The host computes typical from the full (sub-second) distribution; an older host only
    // sent whole-second probabilities. Never label its full mean as a sharpened estimate.
    const value = options.timeStatistic === 'typical'
      ? (Number.isFinite(timing.typicalSeconds) ? timing.typicalSeconds : typicalSeconds(timing.probabilities, 0.5, 1))
      : timing.meanSeconds;
    return Number.isFinite(value) ? '≈' + seconds(value) : undefined;
  }
  function resignationIndicator(result, options) {
    if (!options.includeResignation || !result.resignation || result.terminal) return undefined;
    const probability = result.resignation.probability;
    return Number.isFinite(probability) && probability >= 0 && probability <= 1 ? probability : undefined;
  }
  // END DISPLAY MATH
  let display = {...DISPLAY_DEFAULTS};
  let settings = {...DEFAULTS};
  let state = {};
  let hostInfo;
  let loadingInfo;
  let result;
  let error = '';
  let requestKey = '';
  let generation = 0;
  let debounce;
  let showAll = false;
  let filtersOpen = false;
  let ui;
  let nativeBox;
  let lastRender = '';
  let ready = false;
  let restoringSource = false;
  let timePopup, timeAnchor, timeShowTimer, timeHideTimer, timePinned = false;

  const command = (type, data = {}) => window.postMessage({channel: CHANNEL, direction: 'command', type, ...data}, location.origin);
  const send = message => chrome.runtime.sendMessage({...message, finishedGame: state.finishedGame === true});
  const active = () => ready && settings.enabled && state.available && state.supported && state.enabled && state.db === 'lichess';
  const el = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const button = (text, fn, className = '') => {
    const node = el('button', className, text);
    node.type = 'button';
    node.addEventListener('click', fn);
    return node;
  };
  function save() { chrome.storage.local.set({preferences: settings}).catch(() => {}); }

  function stop() {
    closeTime();
    generation++;
    clearTimeout(debounce);
    requestKey = '';
    result = undefined;
    send({type: 'cancel'}).catch(() => {});
  }
  function selectDatabase(db) {
    restoringSource = false;
    filtersOpen = false;
    settings.enabled = db === 'marvin';
    save();
    stop();
    error = '';
    // Marvin is an independent visible tab. Keep Lichess's internal database
    // selector within its supported values and preserve all native filters.
    command('database', {db: db === 'marvin' ? 'lichess' : db});
    refresh();
  }
  function tcLabel(tc) {
    const [base, inc] = tc.split('+').map(Number);
    return base < 60 ? base + 's+' + inc : base / 60 + '+' + inc;
  }
  function filterChange(name, value) {
    settings[name] = value;
    save(); stop(); error = ''; refresh();
  }
  function createUI() {
    const root = el('section', 'me-root');
    root.setAttribute('aria-label', 'Marvin opening explorer');
    const toolbar = el('div', 'explorer-title me-tabs');
    toolbar.setAttribute('role', 'tablist');
    toolbar.setAttribute('aria-label', 'Opening databases');
    const tabs = new Map();
    for (const [db, label] of [['masters', 'Masters'], ['lichess', 'Lichess database'], ['player', 'Player'], ['marvin', 'Marvin']]) {
      const tab = button(label, () => selectDatabase(db), 'button-link me-tab');
      tab.replaceChildren(el('strong', '', db === 'lichess' ? 'Lichess' : label));
      if (db === 'lichess') tab.append(document.createTextNode(' database'));
      tab.setAttribute('role', 'tab');
      tab.dataset.database = db;
      tab.addEventListener('keydown', event => {
        const keys = [...tabs.keys()];
        let index = keys.indexOf(db);
        if (event.key === 'ArrowRight') index = (index + 1) % keys.length;
        else if (event.key === 'ArrowLeft') index = (index + keys.length - 1) % keys.length;
        else if (event.key === 'Home') index = 0;
        else if (event.key === 'End') index = keys.length - 1;
        else return;
        event.preventDefault();
        tabs.get(keys[index]).focus();
        selectDatabase(keys[index]);
      });
      tabs.set(db, tab);
      toolbar.append(tab);
    }
    const configure = button('', () => {
      if (settings.enabled) { filtersOpen = !filtersOpen; refresh(); }
      else command('configure');
    }, 'fbt toconf me-configure');
    configure.setAttribute('aria-controls', 'marvin-explorer-filters');
    configure.setAttribute('aria-label', 'Open configuration');
    toolbar.append(configure);
    const panel = el('div', 'me-panel');
    const compatibility = el('p', 'me-error');
    compatibility.hidden = true;
    const filters = el('div', 'explorer__config me-filters');
    filters.id = 'marvin-explorer-filters';
    const speed = el('section', 'speed');
    speed.append(el('label', '', 'Time control'));
    const tc = el('div', 'choices');
    tc.setAttribute('role', 'group');
    tc.setAttribute('aria-label', 'Marvin time control');
    speed.append(tc);
    const ratings = el('section', 'rating');
    ratings.append(el('label', '', 'Average rating'));
    const rating = el('div', 'choices');
    rating.setAttribute('role', 'group');
    rating.setAttribute('aria-label', 'Marvin player rating');
    ratings.append(rating);
    const close = el('section', 'save');
    const done = button('All set!', () => { filtersOpen = false; refresh(); }, 'button button-green text');
    done.dataset.icon = '\ue023';
    close.append(done);
    filters.append(speed, ratings, close);
    const output = el('div', 'me-output');
    panel.append(filters, output);
    root.append(toolbar, compatibility, panel);
    return {root, toolbar, tabs, configure, compatibility, panel, filters, rating, tc, output};
  }
  function updateFilters() {
    if (!hostInfo || !ui) return;

    const filterKey = JSON.stringify(hostInfo);
    if (ui.filters.dataset.key !== filterKey) {
      ui.filters.dataset.key = filterKey;
      ui.rating.replaceChildren();
      for (let rating = hostInfo.ratingMin; rating <= hostInfo.ratingMax; rating += hostInfo.ratingStep) {
        const choice = button(String(rating), () => filterChange('rating', rating));
        choice.dataset.value = String(rating);
        ui.rating.append(choice);
      }
      ui.tc.replaceChildren(...hostInfo.timeControls.map(tc => {
        const choice = button(tcLabel(tc), () => filterChange('timeControl', tc));
        choice.dataset.value = tc;
        return choice;
      }));
      settings.rating = Math.max(hostInfo.ratingMin, Math.min(hostInfo.ratingMax, settings.rating));
      settings.rating = hostInfo.ratingMin + Math.round((settings.rating-hostInfo.ratingMin)/hostInfo.ratingStep)*hostInfo.ratingStep;
      if (!hostInfo.timeControls.includes(settings.timeControl)) settings.timeControl = hostInfo.timeControls[0];
    }
    for (const choice of ui.rating.children)
      choice.setAttribute('aria-pressed', String(Number(choice.dataset.value) === settings.rating));
    for (const choice of ui.tc.children)
      choice.setAttribute('aria-pressed', String(choice.dataset.value === settings.timeControl));
  }
  function wdlBar(wdl) {
    const bar = el('div', 'bar');
    const label = `White ${(wdl.white*100).toFixed(1)}%, Draw ${(wdl.draws*100).toFixed(1)}%, Black ${(wdl.black*100).toFixed(1)}%`;
    bar.title = label;
    bar.setAttribute('role', 'img');
    bar.setAttribute('aria-label', label);
    for (const key of ['white', 'draws', 'black']) {
      const percent = wdl[key]*100;
      const part = el('span', key, percent > 12 ? `${Math.round(percent)}${percent > 20 ? '%' : ''}` : '');
      part.style.width = `${Math.round(percent*10)/10}%`;
      bar.append(part);
    }
    return bar;
  }
  // Policy rows arrive before their WDL; a pending bar holds the row's place.
  function wdlCell(wdl) {
    if (wdl) return wdlBar(wdl);
    const bar = el('div', 'bar me-wdl-pending');
    bar.title = 'Evaluating White / Draw / Black…';
    bar.setAttribute('role', 'img');
    bar.setAttribute('aria-label', 'White / Draw / Black not evaluated yet');
    const part = el('span', 'draws', '…');
    part.style.width = '100%';
    bar.append(part);
    return bar;
  }
  // Think time and WDL arrive after the policy. Each cell is filled in place
  // when its value changes, so hover, focus and an open time popup survive.
  function fillCell(entry, move, fen) {
    const value = entry.kind === 'sum' ? result.wdl : move?.[entry.kind];
    const key = JSON.stringify(value ?? null);
    if (key === entry.key) return;
    entry.key = key;
    if (entry.kind === 'time') entry.cell.replaceChildren(...(value ? [timeButton(move, fen)] : []));
    else entry.cell.replaceChildren(wdlCell(value));
  }
  function patchCells() {
    if (!ui?.cells || !result?.moves) return;
    const byUci = new Map(result.moves.map(move => [move.uci, move]));
    for (const entry of ui.cells) fillCell(entry, byUci.get(entry.uci), result.fen);
    ui.panel.setAttribute('aria-busy', String(!!result.partial));
  }
  const percentage = prob => `${(prob*100).toFixed(0)}%`;
  function closeTime() {
    clearTimeout(timeShowTimer); clearTimeout(timeHideTimer);
    if (timePopup) timePopup.hidden = true;
    if (timeAnchor) {
      timeAnchor.setAttribute('aria-expanded', 'false');
      timeAnchor.removeAttribute('aria-describedby');
    }
    timeAnchor = undefined; timePinned = false;
  }
  function deferTimeClose() {
    clearTimeout(timeShowTimer); clearTimeout(timeHideTimer);
    if (!timePinned) timeHideTimer = setTimeout(closeTime, 180);
  }
  function openTime(anchor, move, fen) {
    clearTimeout(timeShowTimer); clearTimeout(timeHideTimer);
    if (!anchor.isConnected || !active() || filtersOpen || state.position?.fen !== fen) return closeTime();
    if (anchor !== timeAnchor) { closeTime(); timeAnchor = anchor; }
    if (!timePopup) {
      timePopup = el('div', 'me-time-popup');
      timePopup.id = 'marvin-move-time';
      timePopup.setAttribute('role', 'tooltip');
      timePopup.addEventListener('mouseenter', () => clearTimeout(timeHideTimer));
      timePopup.addEventListener('mouseleave', deferTimeClose);
      document.body.append(timePopup);
    }
    const timing = move.time;
    // Every move gets the same 40 bars, equally wide in log(1 + seconds) from 0 to 120 s, as in
    // marvin-engine: fine near 0 for quick moves, coarse for long thinks.
    const chart = el('div', 'me-time-chart');
    chart.setAttribute('aria-hidden', 'true');
    chart.style.gridTemplateColumns = `repeat(${timing.bins.length}, minmax(0, 1fr))`;
    const peak = Math.max(...timing.bins);
    for (const p of timing.bins) {
      const mark = el('i');
      mark.style.height = `${Math.round(p / peak * 42)}px`;
      chart.append(mark);
    }
    const axis = el('div', 'me-time-axis');
    axis.setAttribute('aria-hidden', 'true');
    for (const [s, label] of [[0, '0s'], [1, '1s'], [5, '5s'], [30, '30s'], [120, '120s+']]) {
      const tick = el('span', '', label);
      tick.style.left = `${(100 * Math.log1p(s) / Math.log1p(120)).toFixed(1)}%`;
      axis.append(tick);
    }
    const stats = el('div', 'me-time-stats');
    const mode = timeLabel(timing, {timeStatistic: 'mode'});
    const typical = timeLabel(timing, {timeStatistic: 'typical'}), mean = timeLabel(timing, {timeStatistic: 'mean'});
    stats.append(el('span', '', 'Mode ' + mode));
    if (typical) stats.append(el('span', '', 'Typical ' + typical));
    stats.append(el('span', '', 'Mean ' + mean));
    timePopup.replaceChildren(chart, axis, stats);
    timePopup.setAttribute('aria-label', `${move.san}: think time mode ${mode}${typical ? `, typical ${typical}` : ''}, mean ${mean}. Log time axis from 0 to 120 seconds or longer.`);
    timePopup.hidden = false;
    anchor.setAttribute('aria-expanded', 'true');
    anchor.setAttribute('aria-describedby', timePopup.id);
    // Portal to the body so Lichess's scrollable explorer cannot crop the box.
    const rect = anchor.getBoundingClientRect(), box = timePopup.getBoundingClientRect();
    const width = document.documentElement.clientWidth, height = document.documentElement.clientHeight;
    let left = rect.right + 8;
    if (left + box.width > width - 8) left = rect.left - box.width - 8;
    left = Math.max(8, Math.min(left, width - box.width - 8));
    const top = Math.max(8, Math.min(rect.top - 8, height - box.height - 8));
    timePopup.style.left = left + 'px'; timePopup.style.top = top + 'px';
  }
  function timeButton(move, fen) {
    const anchor = button('', event => {
      event.stopPropagation();
      if (timeAnchor === anchor && timePinned) return closeTime();
      openTime(anchor, move, fen);
      if (timeAnchor === anchor) timePinned = true;
    }, 'me-time-trigger');
    anchor.setAttribute('aria-label', `Thinking time for ${move.san}`);
    anchor.setAttribute('aria-expanded', 'false');
    const label = display.showMoveTime ? timeLabel(move.time, display) : undefined;
    if (label !== undefined) {
      anchor.textContent = label;
      anchor.classList.add('me-time-value');
      anchor.setAttribute('aria-label', `${display.timeStatistic} thinking time for ${move.san}: ${label}. Show distribution`);
    } else {
      const face = el('span', 'me-clock-face'); face.setAttribute('aria-hidden', 'true');
      anchor.append(face);
    }
    anchor.addEventListener('mouseenter', () => {
      clearTimeout(timeShowTimer); clearTimeout(timeHideTimer);
      if (!timePinned) timeShowTimer = setTimeout(() => openTime(anchor, move, fen), 250);
    });
    anchor.addEventListener('mouseleave', deferTimeClose);
    anchor.addEventListener('focus', () => openTime(anchor, move, fen));
    anchor.addEventListener('blur', deferTimeClose);
    return anchor;
  }
  window.addEventListener('keydown', event => { if (event.key === 'Escape') closeTime(); });
  window.addEventListener('pointerdown', event => {
    if (!timeAnchor?.contains(event.target) && !timePopup?.contains(event.target)) closeTime();
  });
  window.addEventListener('scroll', closeTime, true);
  window.addEventListener('resize', closeTime);
  function render() {
    if (!ui) return;
    // Time and WDL are excluded: they fill in progressively without rebuilding the table.
    const shape = result && {...result, wdl: undefined, partial: undefined, evaluated: undefined,
      pending: undefined, moves: result.moves?.map(move => ({...move, wdl: undefined, time: undefined}))};
    const key = JSON.stringify([active(), settings, display, shape, error, showAll, !!hostInfo, filtersOpen]);
    if (key === lastRender) return patchCells();
    closeTime();
    lastRender = key;
    ui.cells = undefined;
    ui.panel.hidden = !active();
    if (!active()) return;
    ui.filters.hidden = !hostInfo || !filtersOpen;
    ui.output.hidden = filtersOpen && !!hostInfo;
    ui.output.replaceChildren();
    ui.panel.setAttribute('aria-busy', String((!result || !!result.partial) && !error));
    if (error) {
      ui.output.append(el('p', 'me-error', error), button('Retry', () => {
        stop(); hostInfo = undefined; loadingInfo = undefined; error = ''; refresh();
      }, 'me-retry'));
      return;
    }
    if (!result) {
      const loader = el('div', 'me-loading');
      loader.setAttribute('role', 'status');
      loader.setAttribute('aria-label', hostInfo ? 'Computing predictions' : 'Connecting to Marvin');
      const dots = el('span', 'ddloader');
      dots.append(el('span'), el('span'), el('span'));
      loader.append(dots);
      ui.output.append(loader);
      return;
    }
    if (result.terminal) {
      ui.output.append(el('p', 'me-terminal', `Game over · ${result.terminal}`), wdlBar(result.wdl));
      return;
    }
    const resignation = resignationIndicator(result, display);
    const table = el('table', 'moves me-moves' + (display.showMoveTime ? '' : ' me-time-disabled'));
    const head = el('thead'); const headRow = el('tr');
    headRow.append(el('th', '', 'Move'));
    const policyHeading = el('th', '', 'Policy');
    if (!display.showMoveTime) policyHeading.colSpan = 2;
    headRow.append(policyHeading);
    if (display.showMoveTime) headRow.append(el('th', 'me-time-cell'));
    headRow.append(el('th', '', 'White / Draw / Black'));
    head.append(headRow); table.append(head);
    const body = el('tbody'); body.dataset.fen = result.fen;
    const displayed = showAll ? result.moves : result.moves.slice(0, 12);
    const cells = [];
    const cell = (kind, uci, node, move) => {
      const entry = {kind, uci, cell: node};
      fillCell(entry, move, renderedFen);
      cells.push(entry);
      return node;
    };
    const renderedFen = result.fen;
    for (const move of displayed) {
      const row = el('tr');
      row.dataset.uci = move.uci;
      const moveCell = el('td');
      const play = () => command('move', {uci: move.uci, fen: renderedFen});
      const moveButton = button(move.san, () => play(), 'me-move');
      moveButton.setAttribute('aria-label', `Play ${move.san}`);
      moveCell.append(moveButton);
      const probCell = el('td', 'me-probability');
      const probability = el('span', '', percentage(move.probability));
      probability.title = `${(move.probability*100).toFixed(2)}%`;
      probCell.append(probability);
      if (display.showMoveTime) {
        row.append(moveCell, probCell, cell('time', move.uci, el('td', 'me-time-cell'), move));
      } else {
        probCell.colSpan = 2;
        row.append(moveCell, probCell);
      }
      row.append(cell('wdl', move.uci, el('td'), move));
      row.addEventListener('click', event => { if (!event.target.closest('button')) play(); });
      row.addEventListener('mouseenter', () => command('hover', {uci: move.uci, fen: renderedFen}));
      row.addEventListener('mouseleave', () => command('hover', {uci: null, fen: renderedFen}));
      body.append(row);
    }
    const summary = el('tr', 'sum');
    const summaryName = el('td', '', 'Σ'); summaryName.title = 'Policy-weighted evaluation of all legal moves';
    const sumProb = el('td', 'me-probability', resignation === undefined ? '100%' : 'R ' + percentage(resignation));
    sumProb.title = resignation === undefined ? 'Total probability across all legal moves' : 'Marvin resignation estimate';
    const sumWdl = cell('sum', undefined, el('td'));
    sumProb.colSpan = 2;
    summary.append(summaryName, sumProb, sumWdl);
    body.append(summary);
    table.append(body); ui.output.append(table);
    ui.cells = cells;
    if (result.moves.length > displayed.length || showAll) {
      const more = button('', () => { showAll = !showAll; render(); }, 'me-more fbt');
      more.textContent = showAll ? '⌃' : '⌄';
      more.setAttribute('aria-label', showAll ? 'Show fewer moves' : 'Show all moves');
      more.title = more.getAttribute('aria-label');
      ui.output.append(more);
    }
  }
  async function infer() {
    if (!active()) return;
    if (!hostInfo) {
      if (loadingInfo) return;
      loadingInfo = send({type: 'info'});
      try {
        const reply = await loadingInfo;
        if (!reply?.ok) throw Error(reply?.error || 'Cannot connect to Marvin. Run install.ps1, then Retry.');
        hostInfo = reply.result;
        updateFilters();
      } catch (exc) { error = exc.message; }
      finally { loadingInfo = undefined; }
      if (error || !active()) return render();
    }
    const position = state.position;
    const filters = {rating: settings.rating, timeControl: settings.timeControl, model: display.model};
    const key = JSON.stringify([position, filters]);
    if (key === requestKey) return;
    requestKey = key;
    const token = ++generation;
    result = undefined; error = ''; showAll = false; render();
    try {
      const reply = await send({type: 'analyse', position, filters, token});
      if (token !== generation || !active()) return;
      if (!reply?.ok) throw Error(reply?.error || 'No reply from Marvin.');
      if (reply.result.fen !== position.fen) throw Error('Received a result for a different position. Retry.');
      result = reply.result;
    } catch (exc) { if (token === generation) error = exc.message; }
    if (token === generation) render();
  }
  // Partial results (policy first, then WDL per batch) for the current request.
  chrome.runtime.onMessage?.addListener(message => {
    if (message?.type !== 'marvin-progress' || message.token !== generation || !active()) return;
    const partial = message.result;
    if (!partial?.partial || !Array.isArray(partial.moves) || partial.fen !== state.position?.fen) return;
    // The final reply travels separately and may arrive first; never regress.
    if (result && (!result.partial || result.evaluated >= partial.evaluated)) return;
    result = partial;
    render();
  });
  function refresh() {
    const found = document.querySelector('main.analyse section.explorer-box:not(.me-root)');
    if (nativeBox && nativeBox !== found) nativeBox.classList.remove('me-native-hidden', 'me-has-tabs');
    nativeBox = found;
    if (!nativeBox) { closeTime(); ui?.root.remove(); return; }
    if (!ui) ui = createUI();
    // Attach INSIDE the existing grid area; Lichess retains ownership of its
    // native children. Our node is reattached if a virtual-DOM redraw removes it.
    if (ui.root.parentNode !== nativeBox) nativeBox.prepend(ui.root);
    const showTabs = ready && state.available && state.supported && state.enabled;
    ui.root.hidden = !(showTabs || (ready && state.error));
    ui.toolbar.hidden = !showTabs;
    const selected = settings.enabled ? 'marvin' : state.db;
    for (const [db, tab] of ui.tabs) {
      const isSelected = db === selected;
      tab.setAttribute('aria-selected', String(isSelected));
      tab.tabIndex = isSelected ? 0 : -1;
      if (tab.classList.contains('active') !== isSelected) tab.classList.toggle('active', isSelected);
    }
    const configOpen = settings.enabled ? filtersOpen : state.configOpen;
    ui.configure.hidden = false;
    ui.configure.setAttribute('aria-label', configOpen ? 'Close configuration' : 'Open configuration');
    ui.configure.setAttribute('aria-expanded', String(!!configOpen));
    ui.configure.dataset.icon = configOpen ? '\ue02a' : '\ue005';
    ui.compatibility.hidden = !state.error;
    if (ui.compatibility.textContent !== (state.error || '')) ui.compatibility.textContent = state.error || '';
    if (nativeBox.classList.contains('me-native-hidden') !== !!active()) nativeBox.classList.toggle('me-native-hidden', !!active());
    if (nativeBox.classList.contains('me-has-tabs') !== !!showTabs) nativeBox.classList.toggle('me-has-tabs', !!showTabs);
    updateFilters();
    render();
    if (active() && !error) {
      const nextKey = JSON.stringify([state.position, {rating: settings.rating, timeControl: settings.timeControl, model: display.model}]);
      if (nextKey !== requestKey) {
        if (requestKey) { stop(); render(); }
        clearTimeout(debounce);
        // Short: the host keeps only each tab's latest position and stops a
        // superseded one between batches, so this only merges same-frame changes.
        debounce = setTimeout(infer, REQUEST_DELAY);
      }
    }
  }
  window.addEventListener('message', event => {
    if (event.source !== window || event.origin !== location.origin || event.data?.channel !== CHANNEL || event.data.direction !== 'state') return;
    const next = event.data.data;
    if (!next || typeof next !== 'object') return;
    const wasActive = active();
    const changedPosition = JSON.stringify(state.position) !== JSON.stringify(next.position);
    if (restoringSource && next.available && next.supported) {
      restoringSource = false;
      if (next.db !== 'lichess') command('database', {db: 'lichess'});
    }
    if (next.available && state.db && state.db !== next.db && next.db !== 'lichess' && settings.enabled) {
      settings.enabled = false;
      save();
    }
    state = next;
    if (wasActive && (!active() || changedPosition)) { stop(); error = ''; }
    refresh();
  });
  // Observe child replacement, not our own attribute changes. Schedule one
  // refresh; never poll inference or rebuild focused filter controls.
  let scheduled = false;
  new MutationObserver(records => {
    if (records.every(record => record.type === 'attributes' && record.target !== nativeBox)) return;
    if (records.every(record => ui?.root.contains(record.target) || timePopup?.contains(record.target))) return;
    if (!scheduled) { scheduled = true; requestAnimationFrame(() => { scheduled = false; refresh(); }); }
  }).observe(document.documentElement, {childList: true, subtree: true, attributes: true, attributeFilter: ['class']});
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== 'local' || !changes.displayPreferences) return;
    display = displaySettings(changes.displayPreferences.newValue);
    refresh();  // re-renders, and re-runs the analysis if the model changed
  });
  chrome.storage.local.get(['preferences', 'displayPreferences']).then(saved => {
    display = displaySettings(saved.displayPreferences);
    settings = {...DEFAULTS, ...saved.preferences}; ready = true;
    restoringSource = settings.enabled;
    command('refresh'); refresh();
  }).catch(() => { ready = true; command('refresh'); refresh(); });
  window.addEventListener('pagehide', stop);
  command('refresh');
})();
