const API_BASE = import.meta.env.VITE_API_BASE ?? '/api';

async function requestJson(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {})
    },
    ...options
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const message = payload.error || payload.message || `请求失败 (${response.status})`;
    throw new Error(message);
  }
  return payload;
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
  return requestJson(`${API_BASE}/showcase/recommend${suffix}`);
}

export async function runPrediction({ sampleId, start, end, task = 'funcbound' }) {
  return requestJson(`${API_BASE}/predict`, {
    method: 'POST',
    body: JSON.stringify({ sampleId, start, end, task })
  });
}
