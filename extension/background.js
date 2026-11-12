/* One independent native host connection for this extension. */
// Entry scripts are self-contained. The bridge tests check route parity.
function explorerRoute(path) {
  if (/^\/analysis(?:\/|$)/.test(path)) return 'analysis';
  if (/^\/study\/\w{8}(?:\/\w{8})?\/?$/.test(path)) return 'study';
  // These eight-letter routes are site tools, rather than game IDs.
  if (/^\/(?:training|practice)(?:\/|$)/.test(path)) return;
  // Finished games are now served under /game/ (older links redirect there).
  if (/^\/(?:game\/)?(?:\w{8}(?:\/(?:white|black)(?:\/analysis)?)?|\w{12})\/?$/.test(path)) return 'game';
}
const HOST = 'com.marvin.lichess_explorer';
let port;
let idleTimer;
const pending = new Map();

function finish(id, reply) {
  const item = pending.get(id);
  if (!item) return;
  clearTimeout(item.timer);
  pending.delete(id);
  item.respond(reply);
}

const TIMEOUT = 120000;
const timeout = id => setTimeout(() => {
  finish(id, {ok: false, error: 'Marvin took too long to respond. Check the model setup, then Retry.'});
}, TIMEOUT);

// The host reports policy rows before their WDL, then after each WDL batch.
// Partial results go straight to the requesting document; the final reply
// still answers the original sendMessage.
function progress(id, message) {
  const item = pending.get(id);
  if (!item || item.type !== 'analyse') return;
  clearTimeout(item.timer);
  item.timer = timeout(id);
  const update = {type: 'marvin-progress', token: item.token, result: message.result};
  const target = item.documentId ? {documentId: item.documentId} : {frameId: 0};
  try { chrome.tabs.sendMessage?.(item.tabId, update, target)?.catch?.(() => {}); } catch {}
}

function connect() {
  if (port) return port;
  const connected = chrome.runtime.connectNative(HOST);
  port = connected;
  connected.onMessage.addListener(message => message?.partial ? progress(message.id, message) : finish(message.id, message));
  connected.onDisconnect.addListener(() => {
    const error = chrome.runtime.lastError?.message || 'The Marvin host disconnected.';
    if (port !== connected) return;
    port = undefined;
    for (const id of pending.keys()) finish(id, {ok: false, error: `${error} Run install.ps1 in the Marvin Explorer project, then Retry.`});
  });
  return connected;
}

function touchHost() {
  clearTimeout(idleTimer);
  idleTimer = setTimeout(() => {
    if (pending.size) return touchHost();
    port?.disconnect();
    port = undefined;
  }, 300000);
}

function allowedSender(sender, message) {
  try {
    const url = new URL(sender.url || sender.tab?.url);
    if (sender.tab?.id == null || sender.frameId !== 0 || url.origin !== 'https://lichess.org') return false;
    const route = explorerRoute(url.pathname);
    if (route === 'analysis' || route === 'study') return true;
    // MAIN-world state verifies the actual analysis controller is no longer
    // ongoing. Cancellation remains available when that state changes.
    return route === 'game' &&
      (message?.finishedGame === true || message?.type === 'cancel');
  } catch { return false; }
}

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (!allowedSender(sender, message) || !['analyse', 'info', 'cancel'].includes(message?.type)) return false;
  const client = `${sender.tab.id}:${sender.documentId || 0}`;
  if (message.type === 'cancel') {
    port?.postMessage({type: 'cancel', client});
    for (const [id, item] of pending) if (item.client === client) finish(id, {ok: false, error: 'Cancelled.'});
    respond({ok: true});
    return false;
  }
  let id;
  try {
    const native = connect();
    id = crypto.randomUUID();
    if (message.type === 'analyse') {
      for (const [oldId, item] of pending) {
        if (item.client === client && item.type === 'analyse') finish(oldId, {ok: false, error: 'Superseded.'});
      }
    }
    pending.set(id, {respond, client, type: message.type, token: message.token,
      tabId: sender.tab.id, documentId: sender.documentId, timer: timeout(id)});
    native.postMessage({id, client, type: message.type, position: message.position, filters: message.filters});
    touchHost();
  } catch (error) {
    if (id && pending.has(id)) finish(id, {ok: false, error: error.message});
    else respond({ok: false, error: error.message});
  }
  return true;
});

chrome.tabs.onRemoved.addListener(tabId => {
  for (const [id, item] of pending) if (item.client.startsWith(`${tabId}:`)) {
    port?.postMessage({type: 'cancel', client: item.client});
    finish(id, {ok: false, error: 'Tab closed.'});
  }
});
