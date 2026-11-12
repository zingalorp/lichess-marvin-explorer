/* Runs in MAIN world at document_start. Lichess no longer exposes the analysis
 * controller (lichess.analysis is now a narrow API), so it is captured while
 * the controller is constructed; see captureController. Keep this narrow,
 * version-sensitive adapter separate from UI. */
(() => {
  'use strict';
  // Keep routing local: Chrome may still have the previous manifest cached.
  // The bridge tests check parity with content.js and background.js.
  function explorerRoute(path) {
    if (/^\/analysis(?:\/|$)/.test(path)) return 'analysis';
    if (/^\/study\/\w{8}(?:\/\w{8})?\/?$/.test(path)) return 'study';
    // These eight-letter routes are site tools, rather than game IDs.
    if (/^\/(?:training|practice)(?:\/|$)/.test(path)) return;
    // Finished games are now served under /game/ (older links redirect there).
    if (/^\/(?:game\/)?(?:\w{8}(?:\/(?:white|black)(?:\/analysis)?)?|\w{12})\/?$/.test(path)) return 'game';
  }
  const CHANNEL = 'marvin-explorer-v1';
  const UNAVAILABLE = 'Lichess analysis integration is unavailable. Refresh the page; if this persists, the extension needs an update.';
  let lastState = '';
  let captured;
  // Older Lichess releases exposed lichess.analysis.navigate.ctrl directly.
  const controller = () => captured || window.lichess?.analysis?.navigate?.ctrl;
  const route = () => explorerRoute(location.pathname);
  // The manifest covers Lichess's dynamic game URLs. Other pages do no work.
  if (!route()) return;
  captureController();

  /* The AnalyseCtrl constructor ends with a call that publishes lichess.analysis
   * and first reads `ctrl.forecast`, which is only an own property on ongoing
   * correspondence games (never supported here). A temporary Object.prototype
   * accessor sees that read with `this` set to the controller. It returns
   * undefined like the missing property, keeps assignments as own properties,
   * and is removed as soon as the controller is found (or after page load). */
  function captureController() {
    const proto = Object.prototype, key = 'forecast';
    if (Object.prototype.hasOwnProperty.call(proto, key)) return;
    const looksLikeCtrl = value => value !== null && typeof value === 'object' &&
      typeof value.playUci === 'function' && typeof value.navigate === 'object' &&
      value.explorer && Array.isArray(value.nodeList);
    const release = () => {
      if (Object.getOwnPropertyDescriptor(proto, key)?.get === descriptor.get) delete proto[key];
    };
    const descriptor = {
      configurable: true, enumerable: false,
      get() {
        if (!captured && looksLikeCtrl(this)) { captured = this; release(); }
        return undefined;
      },
      set(value) {
        Object.defineProperty(this, key, {value, writable: true, enumerable: true, configurable: true});
      },
    };
    try { Object.defineProperty(proto, key, descriptor); } catch { return; }
    window.addEventListener?.('load', () => setTimeout(release, 60000), {once: true});
  }
  const isAnalysis = () => {
    const context = route();
    const ctrl = controller();
    return context && document.querySelector('main.analyse') && ctrl?.ongoing !== true &&
      (context !== 'game' || ctrl?.ongoing === false);
  };

  // Lichess publishes lichess.analysis only for an analysis controller without
  // forecasts (never an ongoing game's), so a finished-game page that has it
  // but no reachable controller is a compatibility failure, not a live game.
  const missingGameController = () => route() === 'game' && !controller() &&
    !!window.lichess?.analysis && !!document.querySelector('main.analyse');

  function state() {
    if (!isAnalysis()) {
      return missingGameController() ? {available: false, error: UNAVAILABLE} : {available: false};
    }
    const ctrl = controller();
    if (!ctrl?.node?.fen || !ctrl.explorer || !Array.isArray(ctrl.nodeList)) {
      return {available: false, error: UNAVAILABLE};
    }
    const variant = ctrl.data?.game?.variant?.key;
    return {
      available: true, supported: ['standard', 'fromPosition'].includes(variant),
      finishedGame: route() === 'game' && ctrl.ongoing === false,
      enabled: ctrl.explorer.enabled(), db: ctrl.explorer.db(),
      configOpen: ctrl.explorer.config.data.open(),
      position: {fen: ctrl.node.fen, initialFen: ctrl.nodeList[0].fen,
        moves: ctrl.nodeList.slice(1).map(node => node.uci)},
      opening: ctrl.node.opening?.name || '',
    };
  }

  function publish(force = false) {
    try {
      const data = state();
      const key = JSON.stringify(data);
      if (force || key !== lastState) {
        lastState = key;
        window.postMessage({channel: CHANNEL, direction: 'state', data}, location.origin);
      }
    } catch {
      window.postMessage({channel: CHANNEL, direction: 'state', data: {available: false,
        error: 'Cannot read this Lichess analysis position. Refresh the page.'}}, location.origin);
    }
  }

  window.addEventListener('message', event => {
    if (event.source !== window || event.origin !== location.origin ||
        event.data?.channel !== CHANNEL || event.data.direction !== 'command') return;
    const message = event.data;
    if (message.type === 'refresh') return publish(true);
    const ctrl = controller();
    if (!isAnalysis() || !ctrl?.explorer || !['standard', 'fromPosition'].includes(ctrl.data?.game?.variant?.key)) return;
    if (message.type === 'move' && message.fen === ctrl.node.fen && ctrl.explorer.enabled() && ctrl.explorer.db() === 'lichess' && /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(message.uci)) {
      ctrl.playUci(message.uci);
    } else if (message.type === 'hover' && message.fen === ctrl.node.fen) {
      if (message.uci === null || /^[a-h][1-8][a-h][1-8][qrbn]?$/.test(message.uci)) ctrl.explorer.setHovering(message.fen, message.uci);
    } else if (message.type === 'database' && ['lichess', 'masters', 'player'].includes(message.db)) {
      const config = ctrl.explorer.config;
      if (message.db === 'player') {
        // Match Lichess's Player tab: reuse the selected player, or choose the
        // signed-in account; show configuration if no player can be selected.
        config.selectPlayer(config.data.playerName.value() || 'me');
        if (ctrl.explorer.db() !== 'player') {
          config.data.db('player');
          config.data.open(true);
        }
      } else config.data.db(message.db);
      ctrl.explorer.reload();
    } else if (message.type === 'configure') {
      ctrl.explorer.config.toggleOpen();
      ctrl.redraw();
    }
    publish(true);
  });
  setInterval(publish, 200);
  // A new position is published on the next frame, not the next 200 ms poll, so
  // Marvin starts as soon as a move is played or navigated to. Only the node
  // identity is compared per frame; the poll still covers every other change.
  let lastNode;
  function watchPosition() {
    const node = controller()?.node;
    if (node !== lastNode) {
      lastNode = node;
      publish();
    }
    requestAnimationFrame(watchPosition);
  }
  if (typeof requestAnimationFrame === 'function') requestAnimationFrame(watchPosition);
  publish(true);
})();
