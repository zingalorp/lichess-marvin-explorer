(() => {
  'use strict';
  const DEFAULTS = {model: '79m', includeResignation: false, showMoveTime: false,
    timeStatistic: 'mode'};
  const form = document.getElementById('display-settings');
  const error = document.getElementById('save-error');
  const controls = Object.fromEntries(Object.keys(DEFAULTS).map(key => [key, document.getElementById(key)]));
  let preferences = {...DEFAULTS};
  let writes = Promise.resolve();
  let revision = 0;
  function render() {
    for (const [key, input] of Object.entries(controls)) {
      if (input.type === 'checkbox') input.checked = preferences[key];
      else input.value = String(preferences[key]);
    }
    document.getElementById('timing-options').hidden = !preferences.showMoveTime;
  }
  function normalize(saved = {}) {
    const result = {...DEFAULTS};
    for (const key of ['includeResignation', 'showMoveTime']) if (typeof saved?.[key] === 'boolean') result[key] = saved[key];
    if (['mode', 'typical', 'mean'].includes(saved?.timeStatistic)) result.timeStatistic = saved.timeStatistic;
    if (['79m', '23m'].includes(saved?.model)) result.model = saved.model;
    return result;
  }
  form.addEventListener('submit', event => event.preventDefault());
  for (const [key, input] of Object.entries(controls)) input.addEventListener('change', () => {
    if (!input.reportValidity()) return;
    const value = input.type === 'checkbox' ? input.checked : input.type === 'number' ? Number(input.value) : input.value;
    preferences[key] = value; revision++; render();
    writes = writes.then(async () => {
      const saved = await chrome.storage.local.get('displayPreferences');
      await chrome.storage.local.set({displayPreferences: normalize({...saved.displayPreferences, [key]: value})});
      error.hidden = true;
    }).catch(() => { error.textContent = 'Could not save settings. Try again.'; error.hidden = false; });
  });
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area === 'local' && changes.displayPreferences) {
      const version = revision;
      writes.then(async () => {
        const saved = await chrome.storage.local.get('displayPreferences');
        if (version !== revision) return;
        preferences = normalize(saved.displayPreferences); render();
      }).catch(() => {});
    }
  });
  chrome.storage.local.get('displayPreferences').then(saved => {
    if (revision === 0) { preferences = normalize(saved.displayPreferences); render(); }
  }).catch(() => { error.textContent = 'Could not load settings.'; error.hidden = false; });
  render();
})();
