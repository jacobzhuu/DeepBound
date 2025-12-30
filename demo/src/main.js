import {
  getTaskConfig,
  computeMetrics,
  countBoundaryHits,
  formatPercent
} from './demoDisplay.js';
import {
  clearCaches,
  fetchSamples,
  fetchShowcaseRecommendations,
  fetchShowcaseStatus,
  fetchValidationStatus,
  runPrediction,
  startValidationMetrics
} from './api.js';
import {
  appendLog,
  clearLogs,
  getElements,
  initMetricsChart,
  populateSampleSelect,
  renderSequence,
  renderDisassembly,
  setProgress,
  setMetricsNote,
  setRangeInputs,
  setMetricsPlaceholder,
  setMetricsScope,
  setRunState,
  setStatusBadge,
  setSequenceMeta,
  setLegend,
  setInsightLabels,
  toggleDiffs,
  toggleLabels,
  updateInsights,
  updateMetrics,
  updateMetricsChart
} from './ui.js';

const elements = getElements();
const chart = initMetricsChart(elements.metricsChart);
const DEFAULT_WINDOW = 96;
const RUN_DEBOUNCE_MS = 250;
const METRIC_LABELS = ['精确率', '召回率', 'F1'];
const RECOMMEND_OPTIONS = {
  top: 3,
  maxSamples: 30,
  window: 256,
  stride: 64,       // Denser stride for more thorough search
  maxWindows: 10000, // Effectively unlimited windows to cover full file
  minBetterBoundaries: 0 // Allow recommendations based purely on F1 delta even without count diff
};

const state = {
  task: 'funcbound',
  samples: [],
  sample: null,
  range: { start: 0, end: 0 },
  running: false,
  pendingTimer: null,
  queued: false,
  metricsScope: 'slice',
  sliceMetrics: null,
  sliceCounts: null,
  sliceNote: '',
  validationMetrics: null,
  validationPolling: false,
  validationLogOffset: 0,
  validationTimer: null,
  validationRetryCount: 0,
  recommending: false,
  recommendToken: 0
};

function getActiveTaskConfig() {
  return getTaskConfig(state.task);
}

function applyTaskMode() {
  const taskConfig = getActiveTaskConfig();
  setLegend(elements, taskConfig.legend, 'end');
  setInsightLabels(elements, {
    deepboundStart: 'DeepBound 起始命中',
    deepboundEnd: 'DeepBound 结束命中',
    idaStart: 'IDA 起始命中',
    idaEnd: 'IDA 结束命中'
  });

  if (!taskConfig.supportsValidation && state.metricsScope === 'validation') {
    state.metricsScope = 'slice';
  }
  setMetricsScope(elements, state.metricsScope);
  setMetricsNote(elements, '');
  renderMetricsPlaceholder(METRIC_LABELS, taskConfig.supportsIda);
}

function getDefaultRange(sampleLength) {
  const end = Math.min(sampleLength - 1, DEFAULT_WINDOW);
  return { start: 0, end: Math.max(0, end) };
}

function clampRange(range, maxIndex) {
  let start = Number.isFinite(range.start) ? range.start : 0;
  let end = Number.isFinite(range.end) ? range.end : maxIndex;

  start = Math.max(0, Math.min(maxIndex, start));
  end = Math.max(0, Math.min(maxIndex, end));

  if (start > end) {
    [start, end] = [end, start];
  }

  return { start, end };
}

function setRunning(isRunning) {
  state.running = isRunning;
  setRunState(elements, {
    running: isRunning,
    statusText: isRunning ? '运行中' : '空闲'
  });
  if (elements.recommendButton) {
    elements.recommendButton.disabled = state.recommending || isRunning;
  }
}

function setRecommendState(isRunning, progress = null) {
  if (!elements.recommendButton) {
    return;
  }
  elements.recommendButton.disabled = isRunning || state.running;
  if (isRunning) {
    if (progress !== null && progress >= 0) {
      elements.recommendButton.textContent = `推荐中... ${progress}%`;
      elements.recommendButton.style.setProperty('--progress', `${progress}%`);
      elements.recommendButton.classList.add('progress-bar-button');
    } else {
      elements.recommendButton.textContent = '推荐中...';
      elements.recommendButton.classList.remove('progress-bar-button');
    }
  } else {
    elements.recommendButton.textContent = '自动推荐样例';
    elements.recommendButton.classList.remove('progress-bar-button');
    elements.recommendButton.style.removeProperty('--progress');
  }

  if (!state.running) {
    setStatusBadge(elements, {
      text: isRunning ? '推荐中' : '空闲',
      active: isRunning
    });
  }
}

function toChartValue(value) {
  return Number.isFinite(value) ? value : 0;
}

function countLabels(labels, labelSet) {
  if (!Array.isArray(labels) || !labelSet || !labelSet.length) {
    return 0;
  }
  const target = new Set(labelSet);
  return labels.reduce((total, label) => total + (target.has(label) ? 1 : 0), 0);
}

function updateControls() {
  if (!state.sample) return;
  const maxIndex = Math.max(0, state.sample.length - 1);
  state.range = clampRange(state.range, maxIndex);
  setRangeInputs(elements, state.range, maxIndex);
}

function renderMetricsPlaceholder(labels, showIda = true) {
  setMetricsPlaceholder(elements, '-');
  updateMetricsChart(chart, {
    labels,
    deepbound: labels.map(() => 0),
    ida: labels.map(() => 0),
    showIda
  });
}

function handleValidationStatus(payload) {
  if (typeof payload.progress === 'number') {
    setProgress(elements, payload.progress);
  }

  if (Array.isArray(payload.logs)) {
    payload.logs.forEach((entry) => {
      appendLog(elements, { message: entry.message, level: entry.level });
    });
  }
  if (typeof payload.logCount === 'number') {
    state.validationLogOffset = payload.logCount;
  }

  if (payload.status === 'done' && payload.result) {
    state.validationMetrics = payload.result;
    if (payload.result.totalTokens) {
      appendLog(elements, {
        message: `测试集指标已更新（字节: ${payload.result.totalTokens}）`,
        level: 'info'
      });
    }
    if (state.metricsScope === 'validation') {
      applyMetricsScope();
    }
  }

  if (payload.status === 'error') {
    appendLog(elements, {
      message: `测试集指标计算失败: ${payload.error || '未知错误'}`,
      level: 'alert'
    });
  }
}

async function pollValidationStatus() {
  if (!state.validationPolling) {
    return;
  }
  try {
    const payload = await fetchValidationStatus({
      offset: state.validationLogOffset,
      task: state.task
    });
    state.validationRetryCount = 0;
    handleValidationStatus(payload);
    if (payload.status === 'running') {
      state.validationTimer = setTimeout(pollValidationStatus, 700);
    } else {
      state.validationPolling = false;
      state.validationTimer = null;
    }
  } catch (error) {
    state.validationRetryCount += 1;
    if (state.validationRetryCount <= 3) {
      appendLog(elements, { message: `状态查询重试中 (${state.validationRetryCount}/3): ${error.message}`, level: 'info' });
      state.validationTimer = setTimeout(pollValidationStatus, 2000);
    } else {
      state.validationPolling = false;
      state.validationTimer = null;
      state.validationRetryCount = 0;
      appendLog(elements, { message: `测试集状态查询失败: ${error.message}`, level: 'alert' });
    }
  }
}

async function ensureValidationMetrics({ refresh = false } = {}) {
  const taskConfig = getActiveTaskConfig();
  if (!taskConfig.supportsValidation) {
    return;
  }
  if (!refresh && (state.validationMetrics || state.validationPolling)) {
    return;
  }
  try {
    if (state.validationTimer) {
      clearTimeout(state.validationTimer);
      state.validationTimer = null;
    }
    state.validationPolling = false;
    state.validationMetrics = null;
    state.validationLogOffset = 0;
    setProgress(elements, 0);
    
    const payload = await startValidationMetrics({ task: state.task, refresh });
    handleValidationStatus(payload);
    
    if (payload.status === 'running') {
      state.validationPolling = true;
      pollValidationStatus();
    }
  } catch (error) {
    appendLog(elements, { message: `测试集指标启动失败: ${error.message}`, level: 'alert' });
  }
}

function applyMetricsScope() {
  const taskConfig = getActiveTaskConfig();
  if (!taskConfig.supportsValidation && state.metricsScope === 'validation') {
    state.metricsScope = 'slice';
  }
  setMetricsScope(elements, state.metricsScope);
  
  if (elements.runValidationButton) {
    elements.runValidationButton.classList.toggle('hidden', state.metricsScope !== 'validation');
  }

  if (state.metricsScope === 'slice') {
    if (!state.sliceMetrics) {
      renderMetricsPlaceholder(METRIC_LABELS, taskConfig.supportsIda);
      setMetricsNote(elements, '');
      return;
    }
    setMetricsNote(elements, state.sliceNote);
    updateMetrics(elements, {
      deepbound: state.sliceMetrics.deepbound,
      ida: taskConfig.supportsIda ? state.sliceMetrics.ida : null,
      formatPercent
    });
    updateMetricsChart(chart, {
      labels: METRIC_LABELS,
      deepbound: [
        toChartValue(state.sliceMetrics.deepbound.precision),
        toChartValue(state.sliceMetrics.deepbound.recall),
        toChartValue(state.sliceMetrics.deepbound.f1)
      ],
      ida: taskConfig.supportsIda
        ? [
          toChartValue(state.sliceMetrics.ida.precision),
          toChartValue(state.sliceMetrics.ida.recall),
          toChartValue(state.sliceMetrics.ida.f1)
        ]
        : [],
      showIda: taskConfig.supportsIda
    });
    return;
  }

  setMetricsNote(elements, '测试集整体指标，不对应当前片段');
  if (!state.validationMetrics) {
    renderMetricsPlaceholder(METRIC_LABELS, taskConfig.supportsIda);
    if (!state.validationPolling) {
       ensureValidationMetrics();
    }
    return;
  }
  
  const valResults = state.validationMetrics.metrics;
  updateMetrics(elements, {
    deepbound: valResults.deepbound,
    ida: taskConfig.supportsIda ? valResults.ida : null,
    formatPercent
  });
  updateMetricsChart(chart, {
    labels: METRIC_LABELS,
    deepbound: [
      toChartValue(valResults.deepbound.precision),
      toChartValue(valResults.deepbound.recall),
      toChartValue(valResults.deepbound.f1)
    ],
    ida: taskConfig.supportsIda
      ? [
        toChartValue(valResults.ida.precision),
        toChartValue(valResults.ida.recall),
        toChartValue(valResults.ida.f1)
      ]
      : [],
    showIda: taskConfig.supportsIda
  });
}

function renderFromPayload(payload) {
  const taskConfig = getActiveTaskConfig();
  const sampleLength = payload.sampleLength ?? state.sample?.length ?? 0;
  const start = payload.start ?? state.range.start;
  const end = payload.end ?? state.range.end;

  state.range = { start, end };
  if (state.sample) {
    state.sample.length = sampleLength;
  }

  setRangeInputs(elements, state.range, Math.max(0, sampleLength - 1));
  setSequenceMeta(elements, {
    sampleName: state.sample?.name || payload.sampleId,
    start,
    end,
    total: sampleLength
  });

  renderSequence(elements, {
    tokens: payload.tokens,
    truthLabels: payload.truthLabels,
    idaLabels: payload.idaLabels,
    deepboundLabels: payload.deepboundLabels,
    labelMeta: taskConfig.labelMeta,
    metaLabels: taskConfig.rowMetaLabels,
    startIndex: start
  });

  renderDisassembly(payload.disassembly, payload);

  const metrics = {
    deepbound: computeMetrics(payload.truthLabels, payload.deepboundLabels, taskConfig.boundaryLabels),
    ida: taskConfig.supportsIda && payload.idaLabels
      ? computeMetrics(payload.truthLabels, payload.idaLabels, taskConfig.boundaryLabels)
      : null
  };

  const counts = payload.counts ?? {
    deepbound: countBoundaryHits(payload.truthLabels, payload.deepboundLabels, {
      startLabels: taskConfig.startLabels,
      endLabels: taskConfig.endLabels
    }),
    ida: taskConfig.supportsIda && payload.idaLabels
      ? countBoundaryHits(payload.truthLabels, payload.idaLabels, {
        startLabels: taskConfig.startLabels,
        endLabels: taskConfig.endLabels
      })
      : null
  };

  state.sliceMetrics = metrics;
  state.sliceCounts = counts;
  const truthBoundaryCount = metrics.deepbound.boundaryTrue;
  const truthStartCount = countLabels(payload.truthLabels, taskConfig.startLabels);
  const truthEndCount = countLabels(payload.truthLabels, taskConfig.endLabels);
  if (truthBoundaryCount === 0) {
    state.sliceNote = '当前片段没有真实边界，精确率/召回率/F1 不适用';
  } else {
    const parts = [];
    if (taskConfig.startLabels.length) {
      const label = taskConfig.labelMeta[taskConfig.startLabels[0]]?.short || 'S';
      parts.push(`${label}=${truthStartCount}`);
    }
    if (taskConfig.endLabels.length) {
      const label = taskConfig.labelMeta[taskConfig.endLabels[0]]?.short || 'R';
      parts.push(`${label}=${truthEndCount}`);
    }
    state.sliceNote = `真实边界: ${parts.join(' | ')}`;
  }

  updateInsights(elements, {
    deepboundHits: counts.deepbound,
    idaHits: counts.ida
  });

  if (state.metricsScope === 'slice') {
    applyMetricsScope();
  }
}

async function runPredictionForState() {
  if (state.running || !state.sample) return;
  setRunning(true);
  if (state.metricsScope === 'slice') {
    state.sliceNote = '';
    setMetricsNote(elements, '');
  }
  setProgress(elements, 12);
  appendLog(elements, { message: '正在请求后端推理', level: 'info' });

  const startedAt = performance.now();
  const requested = {
    task: state.task,
    sampleId: state.sample.id,
    start: state.range.start,
    end: state.range.end
  };
  try {
    const payload = await runPrediction({
      task: requested.task,
      sampleId: requested.sampleId,
      start: requested.start,
      end: requested.end
    });

    const elapsed = Math.round(performance.now() - startedAt);
    const isStale = !state.sample
      || state.task !== requested.task
      || state.sample.id !== requested.sampleId
      || state.range.start !== requested.start
      || state.range.end !== requested.end;
    if (isStale) {
      setProgress(elements, 0);
      appendLog(elements, { message: '已忽略过期响应', level: 'alert' });
      return;
    }
    renderFromPayload(payload);
    setProgress(elements, 100);
    appendLog(elements, { message: `推理完成（${elapsed} ms）`, level: 'info' });
    if (payload.device) {
      appendLog(elements, { message: `设备: ${payload.device}`, level: 'info' });
    }
    if (payload.clamped) {
      appendLog(elements, { message: '范围已按最大长度截断', level: 'alert' });
    }
    if (payload.timing) {
      appendLog(elements, {
        message: `I/O ${payload.timing.ioMs} ms | 推理 ${payload.timing.predictMs} ms`,
        level: 'info'
      });
    }
  } catch (error) {
    setProgress(elements, 0);
    appendLog(elements, { message: `错误: ${error.message}`, level: 'alert' });
  } finally {
    setRunning(false);
    if (state.queued) {
      state.queued = false;
      scheduleRun();
    }
  }
}

function scheduleRun() {
  if (state.running) {
    state.queued = true;
    return;
  }
  if (state.pendingTimer) {
    clearTimeout(state.pendingTimer);
  }
  state.pendingTimer = setTimeout(() => {
    state.pendingTimer = null;
    runPredictionForState();
  }, RUN_DEBOUNCE_MS);
}

function handleRunDemo() {
  scheduleRun();
}

async function handleRecommendSample() {
  const taskConfig = getActiveTaskConfig();
  if (!taskConfig.supportsRecommend) {
    appendLog(elements, { message: '当前任务不支持推荐样例', level: 'alert' });
    return;
  }
  if (!elements.recommendButton || state.recommending) {
    return;
  }
  if (!state.samples.length) {
    appendLog(elements, { message: '样本未加载，无法推荐', level: 'alert' });
    return;
  }
  state.recommending = true;
  const recommendToken = ++state.recommendToken;
  const context = {
    task: state.task,
    sampleId: state.sample?.id,
    start: state.range.start,
    end: state.range.end
  };
  setRecommendState(true, 0);
  appendLog(elements, {
    message: '正在当前样本中搜索 DeepBound 更优样例',
    level: 'info'
  });

  try {
    const options = {
      ...RECOMMEND_OPTIONS,
      sampleId: state.sample?.id
    };
    let job = await fetchShowcaseRecommendations(options, state.task);

    while (job.status === 'running' || job.status === 'idle') {
      if (recommendToken !== state.recommendToken) {
        return;
      }
      setRecommendState(true, job.progress);
      await new Promise((resolve) => setTimeout(resolve, 500));
      job = await fetchShowcaseStatus({ task: state.task });
    }

    if (recommendToken !== state.recommendToken) {
      appendLog(elements, { message: '已忽略过期的推荐响应', level: 'alert' });
      return;
    }

    if (job.status === 'error') {
      throw new Error(job.error || 'Unknown error');
    }

    const payload = job.result;

    const isStale = !state.sample
      || state.task !== context.task
      || state.sample.id !== context.sampleId;

    if (isStale) {
      appendLog(elements, { message: '推荐结果已过期（样本已切换），已忽略', level: 'alert' });
      return;
    }
    const items = payload.items || [];
    if (!items.length) {
      appendLog(elements, { message: '当前样本中未找到满足条件的推荐片段', level: 'alert' });
      return;
    }

    const best = items[0];
    const nextSample = state.samples.find((sample) => sample.id === best.sampleId);
    if (!nextSample) {
      appendLog(elements, { message: `推荐样本未在列表中: ${best.sampleId}`, level: 'alert' });
      return;
    }

    state.sample = nextSample;
    state.range = { start: best.start, end: best.end };
    elements.sampleSelect.value = nextSample.id;
    updateControls();

    if (payload.cached) {
      appendLog(elements, { message: '推荐结果来自缓存', level: 'info' });
    }
    appendLog(elements, {
      message: `推荐样例: ${best.sampleId} ${best.start}-${best.end}`,
      level: 'info'
    });
    if (best.deepbound && best.ida) {
      const delta = best.deltaF1 ?? (best.deepbound.f1 - best.ida.f1);
      appendLog(elements, {
        message: `DeepBound F1 ${formatPercent(best.deepbound.f1)} | IDA F1 ${formatPercent(best.ida.f1)} | ΔF1 ${delta.toFixed(3)}`,
        level: 'info'
      });
    } else if (best.deepbound) {
      appendLog(elements, {
        message: `DeepBound Precision ${formatPercent(best.deepbound.precision)} | Recall ${formatPercent(best.deepbound.recall)} | F1 ${formatPercent(best.deepbound.f1)}`,
        level: 'info'
      });
      if (typeof best.boundaryCount === 'number') {
        appendLog(elements, {
          message: `真实边界数量: ${best.boundaryCount}`,
          level: 'info'
        });
      }
      if (best.hits?.deepbound) {
        appendLog(elements, {
          message: `DeepBound 命中: 起始 ${best.hits.deepbound.startHits ?? 0} | 终点 ${best.hits.deepbound.endHits ?? 0}`,
          level: 'info'
        });
      }
    }
    const betterStarts = best.betterStarts || [];
    const betterEnds = best.betterEnds || [];
    if (betterStarts.length) {
      appendLog(elements, {
        message: `更优起点示例: ${betterStarts.join(', ')}`,
        level: 'info'
      });
    }
    if (betterEnds.length) {
      appendLog(elements, {
        message: `更优终点示例: ${betterEnds.join(', ')}`,
        level: 'info'
      });
    }
    if (typeof best.betterStartCount === 'number' || typeof best.betterEndCount === 'number') {
      appendLog(elements, {
        message: `更优边界数量: 起点 ${best.betterStartCount ?? 0} | 终点 ${best.betterEndCount ?? 0}`,
        level: 'info'
      });
    }

    scheduleRun();
  } catch (error) {
    appendLog(elements, { message: `推荐样例失败: ${error.message}`, level: 'alert' });
  } finally {
    state.recommending = false;
    setRecommendState(false);
  }
}

async function handleReset() {
  if (!state.samples.length) return;
  const task = state.task;
  const taskConfig = getActiveTaskConfig();
  state.queued = false;
  if (state.pendingTimer) {
    clearTimeout(state.pendingTimer);
    state.pendingTimer = null;
  }
  if (state.validationTimer) {
    clearTimeout(state.validationTimer);
    state.validationTimer = null;
  }
  state.validationPolling = false;
  state.validationRetryCount = 0;
  state.validationMetrics = null;
  state.validationLogOffset = 0;
  state.recommendToken += 1;
  state.recommending = false;
  setRecommendState(false);
  if (state.metricsScope === 'validation') {
    renderMetricsPlaceholder(METRIC_LABELS, taskConfig.supportsIda);
    setMetricsNote(elements, '测试集整体指标，不对应当前片段');
  }

  try {
    await clearCaches({ task });
    appendLog(elements, { message: '已清除缓存：推荐样例与测试集指标将重新计算', level: 'info' });
  } catch (error) {
    appendLog(elements, { message: `清除缓存失败（将继续使用现有缓存）: ${error.message}`, level: 'alert' });
  }

  state.sample = state.samples[0];
  state.range = getDefaultRange(state.sample.length);
  elements.sampleSelect.value = state.sample.id;
  updateControls();
  scheduleRun();
}

function handleSampleChange(event) {
  const nextSample = state.samples.find((sample) => sample.id === event.target.value);
  if (!nextSample) return;
  state.sample = nextSample;
  state.range = getDefaultRange(nextSample.length);
  updateControls();
  scheduleRun();
}

function handleRangeChange() {
  state.range = {
    start: Number.parseInt(elements.startInput.value, 10),
    end: Number.parseInt(elements.endInput.value, 10)
  };
  updateControls();
  scheduleRun();
}

async function loadSamples() {
  setRunning(true);
  clearLogs(elements);
  setProgress(elements, 5);
  appendLog(elements, { message: '正在加载样本列表', level: 'info' });

  try {
    const samples = await fetchSamples(state.task);
    if (!samples.length) {
      throw new Error('后端未返回可用样本');
    }

    state.samples = samples;
    state.sample = samples[0];
    state.range = getDefaultRange(state.sample.length);

    populateSampleSelect(elements.sampleSelect, samples);
    elements.sampleSelect.value = state.sample.id;
    updateControls();

    appendLog(elements, { message: `已加载 ${samples.length} 个样本`, level: 'info' });
  } catch (error) {
    appendLog(elements, { message: `错误: ${error.message}`, level: 'alert' });
  } finally {
    setRunning(false);
    setProgress(elements, 0);
  }
}

async function init() {
  document.body.classList.add('ready');
  toggleLabels(elements, elements.toggleLabels.checked);
  toggleDiffs(elements, elements.toggleDiffs.checked);
  applyTaskMode();
  elements.sampleSelect.addEventListener('change', handleSampleChange);
  elements.startInput.addEventListener('change', handleRangeChange);
  elements.endInput.addEventListener('change', handleRangeChange);
  elements.toggleLabels.addEventListener('change', (event) => {
    toggleLabels(elements, event.target.checked);
  });
  elements.toggleDiffs.addEventListener('change', (event) => {
    toggleDiffs(elements, event.target.checked);
  });
  elements.runButton.addEventListener('click', handleRunDemo);
  elements.resetButton.addEventListener('click', handleReset);
  if (elements.recommendButton) {
    elements.recommendButton.addEventListener('click', handleRecommendSample);
  }
  if (elements.metricScopeButtons) {
    elements.metricScopeButtons.forEach((button) => {
      button.addEventListener('click', () => {
        const nextScope = button.dataset.metricsScope;
        if (!nextScope || nextScope === state.metricsScope) {
          return;
        }
        state.metricsScope = nextScope;
        applyMetricsScope();
      });
    });
  }

  await loadSamples();
  if (state.samples.length) {
    scheduleRun();
  }

  applyMetricsScope();

  if (elements.runValidationButton) {
    elements.runValidationButton.addEventListener('click', () => {
      ensureValidationMetrics({ refresh: true });
    });
  }

  if (elements.disassemblyHeader && elements.disassemblyCard) {
    elements.disassemblyHeader.addEventListener('click', () => {
      elements.disassemblyCard.classList.toggle('collapsed');
    });
  }
}

init();
