// Development fixture only; shared by the real content and toolbar popup scripts.
(() => {
  const listeners = [];
  window.chrome = {storage: {
    onChanged: {addListener: listener => listeners.push(listener)},
    local: {
      get: async keys => Object.fromEntries((Array.isArray(keys) ? keys : [keys])
        .map(key => [key, JSON.parse(localStorage.getItem('test-' + key) || '{}')])),
      set: async values => {
        const changes = {};
        for (const [key, value] of Object.entries(values)) {
          changes[key] = {oldValue: JSON.parse(localStorage.getItem('test-' + key) || '{}'), newValue: value};
          localStorage.setItem('test-' + key, JSON.stringify(value));
        }
        listeners.forEach(fn => fn(changes, 'local'));
      }
    }
  }};
  window.addEventListener('storage', event => {
    if (!event.key?.startsWith('test-')) return;
    const key = event.key.slice(5);
    listeners.forEach(fn => fn({[key]: {newValue: JSON.parse(event.newValue || '{}')}}, 'local'));
  });
})();
