const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const extension = path.join(__dirname, '../extension');

function background() {
  const handlers = {}, messages = [], replies = [], timers = new Map(), forwarded = [];
  let timerId = 0, requestId = 0;
  const port = {postMessage: msg => messages.push(msg), disconnect() {},
    onMessage: {addListener: fn => handlers.reply = fn}, onDisconnect: {addListener: fn => handlers.disconnect = fn}};
  const chrome = {runtime: {connectNative: () => port,
    onMessage: {addListener: fn => handlers.request = fn}}, tabs: {onRemoved: {addListener: fn => handlers.removed = fn},
      sendMessage: async (tabId, message, options) => { forwarded.push({tabId, message, options}); }}};
  const sandbox = vm.createContext({
    chrome, URL, crypto: {randomUUID: () => String(++requestId)},
    setTimeout: fn => { timers.set(++timerId, fn); return timerId; }, clearTimeout: id => timers.delete(id),
  });
  vm.runInContext(fs.readFileSync(path.join(extension, 'background.js'), 'utf8'), sandbox);
  const sender = {url: 'https://lichess.org/analysis', tab: {id: 7}, frameId: 0, documentId: 'doc'};
  const request = (msg, s = sender) => handlers.request(msg, s, reply => replies.push(reply));
  return {handlers, messages, replies, request, sender, chrome, timers, forwarded};
}

test('rejects live games, other origins, and child frames', () => {
  const bg = background();
  for (const sender of [{...bg.sender, url: 'https://lichess.org/abcd1234'},
    {...bg.sender, url: 'https://evil.example/analysis'}, {...bg.sender, frameId: 1}]) {
    assert.equal(bg.request({type: 'analyse'}, sender), false);
  }
  assert.equal(bg.messages.length, 0);
});

test('finished-game and study routes can reach the native host', () => {
  for (const route of ['/abcd1234', '/abcd1234/white', '/abcd1234/black/analysis', '/abcd1234WXYZ',
                       '/game/abcd1234', '/game/abcd1234/black',
                       '/study/STUDY123', '/study/STUDY123/CHAPTER1']) {
    const bg = background();
    const sender = {...bg.sender, url: 'https://lichess.org' + route};
    assert.equal(bg.request({type:'info', finishedGame:!route.startsWith('/study/')}, sender), true, route);
    assert.equal(bg.messages.length, 1, route);
  }
});

test('game requests require completed analysis, while cancellation stays available', () => {
  const bg = background();
  const sender = {...bg.sender, url:'https://lichess.org/abcd1234/black'};
  for (const type of ['info', 'analyse']) {
    assert.equal(bg.request({type}, sender), false);
    assert.equal(bg.request({type, finishedGame:false}, sender), false);
  }
  assert.equal(bg.messages.length, 0);
  bg.request({type:'analyse', finishedGame:true}, sender);
  assert.equal(bg.request({type:'cancel'}, sender), false);
  assert.match(bg.replies[0].error, /Cancelled/);
});

test('study lists, embeds and unrelated routes cannot reach the host', () => {
  const bg = background();
  for (const route of ['/study', '/study/mine/hot', '/study/STUDY123/CHAPTER1/config',
                       '/study/embed/STUDY123/CHAPTER1', '/editor', '/training',
                       '/abcd1234/white/mini', '/abcd1234/forecasts']) {
    assert.equal(bg.request({type:'analyse', finishedGame:true},
                           {...bg.sender, url:'https://lichess.org'+route}), false, route);
  }
  assert.equal(bg.messages.length, 0);
});
test('correlates simultaneous tabs and supersedes one tab only', () => {
  const bg = background();
  bg.request({type: 'analyse'});
  bg.request({type: 'analyse'}, {...bg.sender, tab: {id: 8}});
  bg.request({type: 'analyse'});
  assert.equal(bg.replies.length, 1);
  assert.match(bg.replies[0].error, /Superseded/);
  bg.handlers.reply({id: '1', ok: true, result: 'stale'});
  assert.equal(bg.replies.length, 1);
  bg.handlers.reply({id: '3', ok: true, result: 'current'});
  bg.handlers.reply({id: '2', ok: true, result: 'other tab'});
  assert.deepEqual(bg.replies.slice(1).map(x => x.result), ['current', 'other tab']);
});
test('partial results go to the requesting document; only the final reply answers', () => {
  const bg = background();
  bg.request({type: 'analyse', token: 5});
  const timersBefore = bg.timers.size;
  bg.handlers.reply({id: '1', ok: true, partial: true, result: {fen: 'f', partial: true, evaluated: 0}});
  assert.equal(bg.replies.length, 0);
  assert.equal(bg.forwarded.length, 1);
  assert.equal(bg.forwarded[0].tabId, 7);
  assert.deepEqual(JSON.parse(JSON.stringify(bg.forwarded[0].options)), {documentId: 'doc'});
  assert.equal(bg.forwarded[0].message.type, 'marvin-progress');
  assert.equal(bg.forwarded[0].message.token, 5);
  assert.equal(bg.timers.size, timersBefore, 'progress restarts the request timeout');
  bg.handlers.reply({id: '1', ok: true, result: {fen: 'f'}});
  assert.equal(bg.replies.length, 1);
  assert.equal(bg.timers.size, timersBefore - 1, 'the request timeout is cleared');
  bg.handlers.reply({id: '1', ok: true, partial: true, result: {fen: 'f', partial: true, evaluated: 8}});
  assert.equal(bg.forwarded.length, 1, 'late progress for a finished request is dropped');
});

test('disconnect and cancellation release pending callbacks', () => {
  const bg = background();
  bg.request({type: 'analyse'});
  bg.request({type: 'cancel'});
  assert.match(bg.replies[0].error, /Cancelled/);
  bg.request({type: 'info'});
  bg.chrome.runtime.lastError = {message: 'Specified native messaging host not found.'};
  bg.handlers.disconnect();
  assert.match(bg.replies.at(-1).error, /install.ps1/);
});

function page({pathname = '/analysis', ongoing, mainPresent = true, study, legacy = true} = {}) {
  const messages = [], handlers = {}, moves = [], dbChanges = [], frames = [];
  let database = 'lichess', configOpen = false, playerName = '', selectedPlayer;
  const config = {data: {
    db: value => { database = value; dbChanges.push(value); },
    open: value => value === undefined ? configOpen : (configOpen = value),
    playerName: {value: () => playerName},
  }, selectPlayer: name => { selectedPlayer = name; database = 'player'; },
  toggleOpen: () => { configOpen = !configOpen; }};
  const ctrl = {node: {fen: 'test fen'}, nodeList: [{fen: 'root fen'}, {uci: 'e2e4'}],
    data: {game: {variant: {key: 'standard'}}}, ongoing, study,
    redraw() {},
    playUci: move => moves.push(move),
    explorer: {enabled: () => true, db: () => database, reload() {}, config, setHovering() {}}};
  // Current Lichess publishes only a narrow API without the controller.
  const window = {lichess: {analysis: {navigate: legacy ? {ctrl} : {}}},
    postMessage: msg => messages.push(msg), addEventListener: (name, fn) => handlers[name] = fn};
  const location = {origin: 'https://lichess.org', pathname};
  const sandbox = vm.createContext({
    window, location, document: {querySelector: () => mainPresent ? {} : null}, setInterval() {},
    requestAnimationFrame: fn => frames.push(fn),
  });
  vm.runInContext(fs.readFileSync(path.join(extension, 'page.js'), 'utf8'), sandbox);
  const command = data => handlers.message({source: window, origin: location.origin,
    data: {channel: 'marvin-explorer-v1', direction: 'command', ...data}});
  // Simulates the AnalyseCtrl constructor: an object of the page's realm whose
  // publish step reads the absent `forecast` property.
  const construct = () => vm.runInContext('o => { const c = Object.assign({navigate: {}}, o); c.forecast; return c; }', sandbox)(ctrl);
  const frame = () => frames.splice(0).forEach(fn => fn());
  return {messages, command, moves, ctrl, location, dbChanges, config, sandbox, construct, frame,
    selectedPlayer: () => selectedPlayer, setPlayerName: name => { playerName = name; }};
}
test('publishes the selected branch and accepts moves only for the current FEN', () => {
  const p = page();
  assert.deepEqual(JSON.parse(JSON.stringify(p.messages[0].data.position)),
    {fen: 'test fen', initialFen: 'root fen', moves: ['e2e4']});
  p.command({type: 'move', fen: 'old fen', uci: 'e7e5'});
  p.command({type: 'move', fen: 'test fen', uci: 'e7e5'});
  assert.deepEqual(p.moves, ['e7e5']);
  p.ctrl.explorer.db = () => 'masters';
  p.command({type: 'move', fen: 'test fen', uci: 'e7e5'});
  assert.equal(p.moves.length, 1);
});

test('captures the controller while Lichess constructs it, then removes the hook', () => {
  const p = page({legacy: false});
  assert.equal(p.messages[0].data.available, false);
  const hooked = () => vm.runInContext("Object.prototype.hasOwnProperty.call(Object.prototype, 'forecast')", p.sandbox);
  assert.equal(hooked(), true);
  // Unrelated objects see a normal missing property and normal assignments.
  assert.equal(vm.runInContext("(() => { 'use strict'; const o = {}; const before = o.forecast; o.forecast = 3; return [before, o.forecast, Object.keys(o).join()]; })()", p.sandbox).join(), ',3,forecast');
  assert.equal(hooked(), true);
  const ctrl = p.construct();
  assert.equal(hooked(), false);
  p.command({type: 'refresh'});
  assert.equal(p.messages.at(-1).data.available, true);
  assert.deepEqual(JSON.parse(JSON.stringify(p.messages.at(-1).data.position)),
    {fen: 'test fen', initialFen: 'root fen', moves: ['e2e4']});
  p.command({type: 'move', fen: 'test fen', uci: 'e7e5'});
  assert.deepEqual(p.moves, ['e7e5']);
  ctrl.node = {fen: 'next fen'};
  p.command({type: 'refresh'});
  assert.equal(p.messages.at(-1).data.position.fen, 'next fen');
});

test('a new position is published on the next frame, without waiting for the poll', () => {
  const p = page();
  p.frame();
  const before = p.messages.length;
  p.frame();
  assert.equal(p.messages.length, before, 'an unchanged position publishes nothing');
  p.ctrl.node = {fen: 'next fen'};
  p.ctrl.nodeList = [{fen: 'root fen'}, {uci: 'e2e4'}, {uci: 'e7e5'}];
  p.frame();
  assert.equal(p.messages.length, before + 1);
  assert.equal(p.messages.at(-1).data.position.fen, 'next fen');
});

test('reports an update is needed when the controller is unreachable', () => {
  const p = page({legacy: false});
  assert.match(p.messages[0].data.error || '', /extension needs an update/);
  // Finished games say so too, instead of silently hiding the Marvin tab.
  const game = page({legacy: false, pathname: '/game/abcd1234/white'});
  assert.match(game.messages[0].data.error || '', /extension needs an update/);
});

test('analysis pages, finished games and studies export the same selected branch', () => {
  for (const pathname of ['/analysis', '/analysis/standard', '/abcd1234',
                          '/abcd1234/black', '/abcd1234/white/analysis', '/abcd1234WXYZ',
                          '/game/abcd1234', '/game/abcd1234/black',
                          '/study/STUDY123', '/study/STUDY123/CHAPTER1']) {
    const p = page({pathname, ongoing:false, study:pathname.startsWith('/study/') ? {} : undefined});
    assert.equal(p.messages[0].data.available, true, pathname);
    assert.equal(p.messages[0].data.finishedGame, !pathname.startsWith('/analysis') && !pathname.startsWith('/study/'));
    assert.deepEqual(JSON.parse(JSON.stringify(p.messages[0].data.position)),
                     {fen:'test fen', initialFen:'root fen', moves:['e2e4']}, pathname);
    p.command({type:'move', fen:'test fen', uci:'e7e5'});
    assert.deepEqual(p.moves, ['e7e5'], pathname);
  }
});

test('ongoing games and missing completed-game state never expose predictions or moves', () => {
  for (const pathname of ['/abcd1234', '/abcd1234/white/analysis', '/abcd1234WXYZ']) {
    for (const ongoing of [true, undefined]) {
      const p = page({pathname, ongoing});
      assert.equal(p.messages[0].data.available, false);
      p.command({type:'move', fen:'test fen', uci:'e7e5'});
      assert.equal(p.moves.length, 0);
      p.ctrl.ongoing = false;
      p.command({type:'refresh'});
      assert.equal(p.messages.at(-1).data.available, true);
      p.ctrl.ongoing = true;
      p.command({type:'refresh'});
      assert.equal(p.messages.at(-1).data.available, false);
    }
  }
  const round = page({pathname:'/abcd1234', ongoing:false, mainPresent:false});
  assert.equal(round.messages[0].data.available, false);
});

test('study chapter and branch changes replace the complete history, including custom starting FEN', () => {
  const p = page({pathname:'/study/STUDY123/CHAPTER1', ongoing:false, study:{}});
  p.ctrl.nodeList = [{fen:'custom starting fen'}, {uci:'a7a8q'}, {uci:'h7h6'}];
  p.ctrl.node = {fen:'new chapter fen'};
  p.location.pathname = '/study/STUDY123/CHAPTER2';
  p.command({type:'refresh'});
  assert.deepEqual(JSON.parse(JSON.stringify(p.messages.at(-1).data.position)),
                   {initialFen:'custom starting fen', moves:['a7a8q','h7h6'], fen:'new chapter fen'});
  p.command({type:'move', fen:'test fen', uci:'a1b1'});
  assert.equal(p.moves.length, 0, 'stale chapter cannot play a move');
  p.command({type:'move', fen:'new chapter fen', uci:'a1b1'});
  assert.deepEqual(p.moves, ['a1b1']);
});

test('manifest injects both worlds on the new route types', () => {
  const manifest = JSON.parse(fs.readFileSync(path.join(extension, 'manifest.json'), 'utf8'));
  for (const entry of manifest.content_scripts) {
    assert.deepEqual(entry.matches, ['https://lichess.org/*']);
    assert.notEqual(entry.all_frames, true);
    assert.deepEqual(entry.js, [entry.world === 'MAIN' ? 'page.js' : 'content.js']);
    // page.js must run before Lichess builds the analysis controller.
    if (entry.world === 'MAIN') assert.equal(entry.run_at, 'document_start');
  }
});

// Only the original entry file is executed, as with a pre-update manifest still
// cached by Chrome. MAIN and isolated scripts cannot rely on a helper global.
function content(pathname, {response, enabled = false, deferred = false} = {}) {
  class Element {
    constructor(tag) {
      this.tagName = tag; this.children = []; this.dataset = {}; this.style = {};
      this.attributes = {}; this.events = {}; this.hidden = false; this.className = '';
      this.classList = {
        contains: name => this.className.split(' ').includes(name),
        toggle: (name, on) => {
          const classes = new Set(this.className.split(' ').filter(Boolean));
          if (on) classes.add(name); else classes.delete(name);
          this.className = [...classes].join(' ');
        },
        remove: (...names) => names.forEach(name => this.classList.toggle(name, false)),
        add: (...names) => names.forEach(name => this.classList.toggle(name, true)),
      };
    }
    append(...nodes) { for (const node of nodes) { node.parentNode = this; this.children.push(node); } }
    prepend(node) { node.parentNode = this; this.children.unshift(node); }
    replaceChildren(...nodes) { this.children = []; this.append(...nodes); }
    setAttribute(name, value) { this.attributes[name] = value; }
    removeAttribute(name) { delete this.attributes[name]; }
    getAttribute(name) { return this.attributes[name]; }
    contains(node) { return node === this || this.children.some(child => child.contains?.(node)); }
    closest(tag) { return this.tagName === tag ? this : this.parentNode?.closest(tag); }
    get isConnected() { return true; }
    getBoundingClientRect() { return {left:250,right:270,top:200,width:190,height:100}; }
    addEventListener(name, fn) { this.events[name] = fn; }
    remove() { if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(n => n !== this); }
  }
  const nativeBox = new Element('section'), handlers = {}, commands = [], requests = [], timers = new Map();
  let timerId = 0;
  const window = {addEventListener: (name, fn) => handlers[name] = fn,
    postMessage: message => commands.push(message)};
  const location = {origin: 'https://lichess.org', pathname};
  const document = {body: new Element('body'), documentElement: {clientWidth:800,clientHeight:600}, querySelector: () => nativeBox,
    createElement: tag => new Element(tag), createTextNode: text => ({textContent:text})};
  const sandbox = vm.createContext({window, location, document,
    MutationObserver: class { observe() {} },
    chrome: {storage: {onChanged: {addListener: fn => handlers.storage = fn}, local: {get: async () => ({preferences:{enabled}}), set: async () => {}}},
             runtime: {onMessage: {addListener: fn => handlers.runtime = fn}, sendMessage: async message => {
               requests.push(message);
               if (message.type === 'analyse' && deferred) return new Promise(resolve => { handlers.resolve = resolve; });
               return {ok:true,result:message.type === 'info' ? {ratingMin:1200,ratingMax:2500,ratingStep:100,timeControls:['600+0']} : response};
             }}},
    setTimeout(fn, delay) { timers.set(++timerId, {fn,delay}); return timerId; },
    clearTimeout(id) { timers.delete(id); }, requestAnimationFrame() {},
  });
  vm.runInContext(fs.readFileSync(path.join(extension, 'content.js'), 'utf8'), sandbox);
  return {nativeBox, commands, requests, document, handlers, timers,
    async fire(delay) {
      for (const [id,timer] of [...timers]) if (timer.delay === delay && timers.has(id)) { timers.delete(id); await timer.fn(); }
    },
    state: data => handlers.message({source:window,
    origin:location.origin, data:{channel:'marvin-explorer-v1', direction:'state', data}})};
}

test('entry scripts create the four visible tabs without any shared helper global', async () => {
  for (const pathname of ['/analysis', '/abcd1234/white', '/study/STUDY123/CHAPTER1']) {
    const p = page({pathname, ongoing:false});
    const c = content(pathname);
    await Promise.resolve(); // preferences loaded
    c.state(p.messages[0].data);
    const root = c.nativeBox.children[0], toolbar = root.children[0];
    assert.equal(root.hidden, false, pathname);
    assert.equal(toolbar.hidden, false, pathname);
    assert.deepEqual(toolbar.children.slice(0, 4).map(tab => tab.dataset.database),
                     ['masters', 'lichess', 'player', 'marvin']);
    assert.equal(c.commands.some(command => command.type === 'refresh'), true);
    p.ctrl.ongoing = true;
    p.command({type:'refresh'});
    c.state(p.messages.at(-1).data);
    assert.equal(root.hidden, true, 'ongoing games restore the native explorer');
  }
});

test('clock alone opens timing; native hovers/clicks survive and stale timing closes', async () => {
  const wdl = {white:.5,draws:.1,black:.4};
  const response = {fen:'test fen',wdl,moves:[{uci:'e2e4',san:'e4',probability:.987,wdl,
    time:{bins:[.1,.6,.2,.1],modeSeconds:1,meanSeconds:2.5}}]};
  const c = content('/analysis', {response,enabled:true});
  const p = page();
  await Promise.resolve();
  c.handlers.storage({displayPreferences:{newValue:{showMoveTime:true}}},'local');
  c.state(p.messages[0].data);
  await c.fire(30);
  const all = node => [node,...node.children.flatMap(child => child.children ? all(child) : [])];
  const row = all(c.nativeBox).find(node => node.dataset.uci === 'e2e4');
  assert.ok(row);
  const clock = all(row).find(node => node.classList.contains('me-time-trigger'));
  row.events.mouseenter();
  assert.equal(c.commands.at(-1).type, 'hover');
  assert.equal(c.document.body.children.length, 0, 'row hover does not show time');
  assert.equal(row.children[1].children[0].title, '98.70%');
  assert.match(row.children[3].children[0].title, /White 50.0%/);
  clock.events.mouseenter();
  await c.fire(250);
  const popup = c.document.body.children[0];
  assert.equal(popup.hidden, false);
  assert.deepEqual(popup.children[1].children.map(tick => tick.textContent), ['0s','1s','5s','30s','120s+'], 'fixed log axis');
  assert.equal(popup.children[2].children[0].textContent, 'Mode 1.0s');
  assert.match(popup.children[2].children[1].textContent, /Mean ≈2.5s/);
  assert.ok(popup.children[0].children.every(bar => /^\d+px$/.test(bar.style.height)));
  const movesBefore = c.commands.filter(command => command.type === 'move').length;
  let stopped = false;
  clock.events.click({stopPropagation() { stopped = true; }});
  assert.equal(stopped, true);
  row.events.click({target:clock}); // bubbled clicks are also excluded
  assert.equal(c.commands.filter(command => command.type === 'move').length, movesBefore);
  assert.equal(c.requests.filter(request => request.type === 'analyse').length, 1);
  c.handlers.keydown({key:'Escape'});
  assert.equal(popup.hidden, true);
  clock.events.focus();
  c.state({...p.messages[0].data,position:{...p.messages[0].data.position,fen:'new fen'}});
  assert.equal(popup.hidden, true);
  clock.events.focus();
  assert.equal(popup.hidden, true, 'old move cannot reopen a stale popup');
  row.events.click({target:row.children[0]});
  assert.equal(c.commands.at(-1).type, 'move');
});

test('policy rows render before think time and WDL, which then fill in place', async () => {
  const time = {bins:[1],modeSeconds:1,meanSeconds:1};
  const wdl = {white:.5,draws:.1,black:.4};
  const moves = done => [{uci:'e2e4',san:'e4',probability:.7,wdl:done > 0 ? wdl : null,time:done > 0 ? time : null},
                         {uci:'d2d4',san:'d4',probability:.3,wdl:done > 1 ? wdl : null,time:done > 1 ? time : null}];
  const c = content('/analysis', {enabled:true, deferred:true});
  const p = page();
  await Promise.resolve();
  c.handlers.storage({displayPreferences:{newValue:{showMoveTime:true}}},'local');
  c.state(p.messages[0].data);
  // The request stays open while progress arrives, so do not await it.
  const inflight = c.fire(30);
  for (let i = 0; i < 5; i++) await Promise.resolve();
  const request = c.requests.find(r => r.type === 'analyse');
  assert.ok(request && Number.isInteger(request.token));
  const all = node => [node,...node.children.flatMap(child => child.children ? all(child) : [])];
  const rows = () => all(c.nativeBox).filter(node => node.dataset.uci);
  const pendingBars = () => all(c.nativeBox).filter(node => node.classList.contains('me-wdl-pending')).length;
  const clocks = () => all(c.nativeBox).filter(node => node.classList.contains('me-time-trigger')).length;
  const progress = (done, token = request.token) => c.handlers.runtime({type:'marvin-progress', token,
    result:{fen:'test fen', partial:true, evaluated:done, pending:2, wdl:null, moves:moves(done)}});
  assert.equal(rows().length, 0, 'loading');
  progress(0, request.token + 1);
  assert.equal(rows().length, 0, 'other requests are ignored');
  progress(0);
  const [first] = rows();
  assert.deepEqual(rows().map(row => row.dataset.uci), ['e2e4', 'd2d4']);
  assert.equal(pendingBars(), 3, 'two rows and the summary wait for WDL');
  assert.equal(clocks(), 0, 'think time is not known yet');
  progress(1);
  assert.equal(rows()[0], first, 'time and WDL fill without rebuilding rows');
  assert.equal(pendingBars(), 2);
  assert.equal(clocks(), 1);
  const clock = all(first).find(node => node.classList.contains('me-time-trigger'));
  clock.events.focus();
  const popup = c.document.body.children[0];
  assert.equal(popup.hidden, false);
  assert.match(first.children.at(-1).children[0].title, /White 50.0%/);
  c.handlers.resolve({ok:true, result:{fen:'test fen', wdl, moves:moves(2)}});
  await inflight;
  assert.equal(rows()[0], first);
  assert.equal(pendingBars(), 0);
  assert.equal(clocks(), 2);
  assert.equal(all(first).find(node => node.classList.contains('me-time-trigger')), clock, 'unchanged cells are kept');
  assert.equal(popup.hidden, false, 'an open time popup survives later batches');
  progress(1);
  assert.equal(pendingBars(), 0, 'a late partial never replaces the final result');
});

test('self-contained route classifiers stay identical in every script world', () => {
  const routes = ['page.js', 'content.js', 'background.js'].map(file =>
    fs.readFileSync(path.join(extension, file), 'utf8')
      .match(/function explorerRoute\(path\) \{[\s\S]*?\n\s*\}/)?.[0].replace(/\r/g, '').replace(/^ +/gm, ''));
  assert.ok(routes[0]);
  for (const route of routes) assert.equal(route, routes[0]);
  const classify = vm.runInNewContext('(' + routes[0] + ')');
  for (const path of ['/analysis', '/analysis/standard']) assert.equal(classify(path), 'analysis');
  for (const path of ['/study/STUDY123', '/study/STUDY123/CHAPTER1']) assert.equal(classify(path), 'study');
  for (const path of ['/abcd1234', '/abcd1234/white', '/abcd1234/black/analysis', '/abcd1234WXYZ',
                      '/game/abcd1234', '/game/abcd1234/white', '/game/abcd1234/black/'])
    assert.equal(classify(path), 'game');
  for (const path of ['/training', '/practice', '/study', '/study/mine/hot', '/editor',
                      '/study/embed/STUDY123/CHAPTER1', '/abcd1234/white/mini',
                      '/game/export/abcd1234', '/game/abcd1234/edit', '/game'])
    assert.equal(classify(path), undefined);
});
test('leaves unsupported variants and non-analysis pages alone', () => {
  const p = page();
  p.ctrl.data.game.variant.key = 'chess960';
  p.command({type: 'refresh'});
  assert.equal(p.messages.at(-1).data.supported, false);
  p.command({type: 'move', fen: 'test fen', uci: 'e7e5'});
  p.location.pathname = '/abcd1234';
  p.command({type: 'refresh'});
  assert.equal(p.messages.at(-1).data.available, false);
  assert.equal(p.moves.length, 0);
});

test('database tabs retain Lichess Player selection and configuration behavior', () => {
  const p = page();
  p.command({type: 'database', db: 'masters'});
  assert.equal(p.ctrl.explorer.db(), 'masters');
  p.command({type: 'database', db: 'player'});
  assert.equal(p.selectedPlayer(), 'me');
  assert.equal(p.ctrl.explorer.db(), 'player');
  p.setPlayerName('SavedPlayer');
  p.command({type: 'database', db: 'player'});
  assert.equal(p.selectedPlayer(), 'SavedPlayer');
  p.command({type: 'configure'});
  assert.equal(p.messages.at(-1).data.configOpen, true);
  p.command({type: 'configure'});
  assert.equal(p.messages.at(-1).data.configOpen, false);
  p.command({type: 'database', db: 'marvin'});
  assert.equal(p.ctrl.explorer.db(), 'player', 'never stores an unsupported database in Lichess');
});

test('Player opens configuration when there is no account to select', () => {
  const p = page();
  p.config.selectPlayer = () => {};
  p.command({type: 'database', db: 'player'});
  assert.equal(p.ctrl.explorer.db(), 'player');
  assert.equal(p.config.data.open(), true);
});

test('toolbar toggles a separate resignation indicator without changing policy or WDL', async () => {
  const wdl = {white:.5,draws:.1,black:.4};
  const response = {fen:'test fen',wdl,
    resignation:{probability:.2,wdl:{white:0,draws:0,black:1}},
    moves:[{uci:'e2e4',san:'e4',probability:1,wdl,
      time:{bins:[.25,.75],maxSeconds:5,modeSeconds:1,meanSeconds:1.75,probabilities:[0,.25,.75,...Array(118).fill(0)]}}]};
  const c=content('/analysis',{response,enabled:true}), p=page();
  await Promise.resolve(); c.handlers.storage({displayPreferences:{newValue:{includeResignation:true,showMoveTime:true}}},'local');
  c.state(p.messages[0].data); await c.fire(30);
  const all=node=>[node,...node.children.flatMap(child=>child.children?all(child):[])];
  const indicator=all(c.nativeBox).find(node=>node.textContent==='R 20%');
  assert.ok(indicator,'the Σ row shows the resignation estimate');
  assert.equal(all(c.nativeBox).filter(node=>node.tagName==='tr').length,3,'only the legal move and Σ rows are shown');
  const requests=c.requests.length;
  c.handlers.storage({displayPreferences:{newValue:{includeResignation:false,showMoveTime:true,timeStatistic:'typical'}}},'local');
  assert.equal(all(c.nativeBox).some(node=>node.textContent==='R 20%'),false);
  let trigger=all(c.nativeBox).find(node=>node.classList.contains('me-time-trigger'));
  assert.equal(trigger.textContent,'≈1.9s','typical from an older host\'s whole-second probabilities');
  trigger.events.mouseenter(); await c.fire(250);
  assert.equal(c.document.body.children[0].hidden,false,'the time popup is open');
  c.handlers.storage({displayPreferences:{newValue:{includeResignation:true,showMoveTime:false}}},'local');
  trigger=all(c.nativeBox).find(node=>node.classList.contains('me-time-trigger'));
  assert.equal(trigger,undefined,'disabling move time removes the clock and its time-hover target');
  assert.equal(c.document.body.children[0].hidden,true,'disabling move time closes an open time popup');
  assert.equal(c.requests.length,requests,'display changes perform no inference or cancellation');
  assert.equal(response.moves[0].probability,1,'cached model policy is unchanged');
  assert.deepEqual(response.wdl,wdl,'the original policy evaluation is unchanged');
});

test('the toolbar model choice is sent with requests and re-runs the analysis', async () => {
  const wdl = {white:.5,draws:.1,black:.4};
  const response = {fen:'test fen',wdl,moves:[{uci:'e2e4',san:'e4',probability:1,wdl,
    time:{bins:Array(40).fill(.025),modeSeconds:.5,typicalSeconds:.55,meanSeconds:.69,subsecond:true}}]};
  const c=content('/analysis',{response,enabled:true}), p=page();
  await Promise.resolve();
  c.state(p.messages[0].data); await c.fire(30);
  const analyses=()=>c.requests.filter(request=>request.type==='analyse');
  assert.equal(analyses().length,1);
  assert.equal(analyses()[0].filters.model,'79m','79M is the default');
  c.handlers.storage({displayPreferences:{newValue:{model:'23m',showMoveTime:true,timeStatistic:'typical'}}},'local');
  await c.fire(30);
  assert.equal(analyses().length,2,'switching model re-runs the analysis');
  assert.equal(analyses()[1].filters.model,'23m');
  const all=node=>[node,...node.children.flatMap(child=>child.children?all(child):[])];
  const trigger=all(c.nativeBox).find(node=>node.classList.contains('me-time-trigger'));
  assert.equal(trigger.textContent,'≈0.55s','sub-second values show hundredths');
  c.handlers.storage({displayPreferences:{newValue:{model:'23m',showMoveTime:true,timeStatistic:'mode'}}},'local');
  await c.fire(30);
  assert.equal(analyses().length,2,'display-only changes perform no inference');
});
