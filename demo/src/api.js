const API_BASE = resolveApiBase(import.meta.env.VITE_API_BASE);

function isLocalHost(hostname) {
  return hostname === 'localhost' || hostname === '127.0.0.1';
}

function resolveApiBase(explicitBase) {
  if (!explicitBase) {
    return '/api';
  }
  if (explicitBase.startsWith('http://') || explicitBase.startsWith('https://')) {
    try {
      const url = new URL(explicitBase);
      if (isLocalHost(url.hostname) && !isLocalHost(window.location.hostname)) {
        return '/api';
      }
    } catch (error) {
      return explicitBase;
    }
  }
  return explicitBase;
}

async function requestJson(path, options = {}) {
  const { timeoutMs, signal, ...fetchOptions } = options;
  const controller = new AbortController();
  if (signal) {
    if (signal.aborted) {
      controller.abort();
    } else {
      signal.addEventListener('abort', () => controller.abort(), { once: true });
    }
  }
  const timeoutId = timeoutMs ? setTimeout(() => controller.abort(), timeoutMs) : null;

  try {
    const response = await fetch(path, {
      headers: {
        'Content-Type': 'application/json',
        ...(fetchOptions.headers || {})
      },
      ...fetchOptions,
      signal: controller.signal
    });

    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      const message = payload.error || payload.message || `请求失败 (${response.status})`;
      throw new Error(message);
    }
    return payload;
  } catch (error) {
    if (error.name === 'AbortError') {
      throw new Error('请求超时，请检查后端服务是否可用');
    }
    throw error;
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
}

export async function fetchSamples(task = 'funcbound') {
  const params = new URLSearchParams();
  if (task) {
    params.set('task', task);
  }
  const suffix = params.toString() ? `?${params.toString()}` : '';
  const data = await requestJson(`${API_BASE}/samples${suffix}`);
  return data.samples || [];
}

export async function startValidationMetrics({ refresh = false, task = 'funcbound' } = {}) {
  const params = new URLSearchParams();
  if (refresh) {
    params.set('refresh', '1');
  }
  if (task) {
    params.set('task', task);
  }
  const suffix = params.toString() ? `?${params.toString()}` : '';
  return requestJson(`${API_BASE}/validation/start${suffix}`);
}

export async function fetchValidationStatus({ offset = 0, task = 'funcbound' } = {}) {
  const params = new URLSearchParams();
  if (offset) {
    params.set('offset', String(offset));
  }
  if (task) {
    params.set('task', task);
  }
  const suffix = params.toString() ? `?${params.toString()}` : '';
  return requestJson(`${API_BASE}/validation/status${suffix}`);
}

export async function fetchShowcaseRecommendations(options = {}, task = 'funcbound') {
  const params = new URLSearchParams();
  if (task) {
    params.set('task', task);
  }
  Object.entries(options).forEach(([key, value]) => {
    if (value === undefined || value === null || value === '') {
      return;
    }
    params.set(key, String(value));
  });
  const suffix = params.toString() ? `?${params.toString()}` : '';
  return requestJson(`${API_BASE}/showcase/recommend${suffix}`, { timeoutMs: 15000 });
}

export async function fetchShowcaseStatus({ task = 'funcbound' } = {}) {
  const params = new URLSearchParams();
  if (task) {
    params.set('task', task);
  }
  const suffix = params.toString() ? `?${params.toString()}` : '';
  return requestJson(`${API_BASE}/showcase/status${suffix}`, { timeoutMs: 10000 });
}

export async function clearCaches({ task = 'funcbound' } = {}) {
  const params = new URLSearchParams();
  if (task) {
    params.set('task', task);
  }
  const suffix = params.toString() ? `?${params.toString()}` : '';
  return requestJson(`${API_BASE}/cache/clear${suffix}`, { timeoutMs: 10000 });
}

export async function runPrediction({ sampleId, start, end, task = 'funcbound' }) {
  return requestJson(`${API_BASE}/predict`, {
    method: 'POST',
    body: JSON.stringify({ sampleId, start, end, task })
  });
}
