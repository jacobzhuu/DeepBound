import Chart from 'chart.js/auto';
import { getTaskConfig } from './demoDisplay.js';

function getElements() {
  return {
    sampleSelect: document.getElementById('sample-select'),
    startInput: document.getElementById('start-index'),
    endInput: document.getElementById('end-index'),
    toggleLabels: document.getElementById('toggle-labels'),
    toggleDiffs: document.getElementById('toggle-diffs'),
    runButton: document.getElementById('run-demo'),
    resetButton: document.getElementById('reset-demo'),
    recommendButton: document.getElementById('recommend-sample'),
    runStatus: document.getElementById('run-status'),
    progressLabel: document.getElementById('progress-label'),
    progressFill: document.getElementById('progress-fill'),
    logList: document.getElementById('log-list'),
    logTemplate: document.getElementById('log-item-template'),
    sequenceGrid: document.getElementById('sequence-grid'),
    sequenceMeta: document.getElementById('sequence-meta'),
    rowTruth: document.getElementById('row-truth'),
    rowIda: document.getElementById('row-ida'),
    rowXda: document.getElementById('row-deepbound'),
    rowMetaTruth: document.getElementById('row-meta-truth'),
    rowMetaIda: document.getElementById('row-meta-ida'),
    rowMetaXda: document.getElementById('row-meta-deepbound'),
    legendStart: document.getElementById('legend-start'),
    legendMiddle: document.getElementById('legend-middle'),
    legendNone: document.getElementById('legend-none'),
    metricXdaPrec: document.getElementById('metric-deepbound-prec'),
    metricXdaRecall: document.getElementById('metric-deepbound-recall'),
    metricXdaF1: document.getElementById('metric-deepbound-f1'),
    metricIdaPrec: document.getElementById('metric-ida-prec'),
    metricIdaRecall: document.getElementById('metric-ida-recall'),
    metricIdaF1: document.getElementById('metric-ida-f1'),
    metricsCard: document.querySelector('.card.metrics'),
    metricsScopeBadge: document.getElementById('metrics-scope-badge'),
    runValidationButton: document.getElementById('run-validation'),
    metricScopeButtons: document.querySelectorAll('[data-metrics-scope]'),
    metricsNote: document.getElementById('metrics-note'),
    insightXdaStart: document.getElementById('insight-deepbound-start'),
    insightXdaEnd: document.getElementById('insight-deepbound-end'),
    insightIdaStart: document.getElementById('insight-ida-start'),
    insightIdaEnd: document.getElementById('insight-ida-end'),
    insightLabelXdaStart: document.getElementById('insight-label-deepbound-start'),
    insightLabelXdaEnd: document.getElementById('insight-label-deepbound-end'),
    insightLabelIdaStart: document.getElementById('insight-label-ida-start'),
    insightLabelIdaEnd: document.getElementById('insight-label-ida-end'),
    metricsChart: document.getElementById('metrics-chart'),
    disassemblyView: document.getElementById('disassembly-view'),
    disassemblyHeader: document.getElementById('disassembly-header'),
    disassemblyCard: document.querySelector('.card.disassembly')
  };
}

function highlightAsm(text) {
  if (!text) return '';
  // Simple heuristic for AT&T syntax often used by objdump
  
  const spaceIdx = text.indexOf(' ');
  let mnemonic = text;
  let operands = '';
  
  if (spaceIdx > 0) {
    mnemonic = text.substring(0, spaceIdx);
    operands = text.substring(spaceIdx);
  }
  
  const highlightedMnemonic = `<span class="asm-mnemonic">${mnemonic}</span>`;
  
  // Highlight operands
  // Registers: %rax, %r12, etc.
  // Literals: $0x..., 0x..., numbers
  // Punctuation: , ( )
  const highlightedOperands = operands.replace(/([%,][a-zA-Z0-9_]+)|(\$?-?0x[0-9a-fA-F]+)|(\$?-?[0-9]+)|([,()])/g, (match) => {
    if (match.startsWith('%')) return `<span class="asm-register">${match}</span>`;
    if (match.startsWith('$') || match.startsWith('0x') || (match.startsWith('-') && match.length > 1) || /^[0-9]/.test(match)) {
       return `<span class="asm-literal">${match}</span>`;
    }
    if (/^[,()]$/.test(match)) return `<span class="asm-punct">${match}</span>`;
    return match;
  });

  return highlightedMnemonic + highlightedOperands;
}

function renderDisassembly(disassemblyData, payload) {
  const target = document.getElementById('disassembly-view');
  if (!target) return;

  if (!disassemblyData || !Array.isArray(disassemblyData) || disassemblyData.length === 0) {
    target.innerHTML = '<div class="placeholder">无反汇编数据</div>';
    return;
  }

  // Handle error object returned as single item list
  if (disassemblyData.length === 1 && !disassemblyData[0].bytes && disassemblyData[0].text.startsWith('Error')) {
     target.innerHTML = `<div class="placeholder alert">${disassemblyData[0].text}</div>`;
     return;
  }

  const { start, truthLabels, idaLabels, deepboundLabels, task } = payload;
  const taskConfig = getTaskConfig(task);
  
  // Helper to get label at absolute index (relative to slice start)
  const getLabelAt = (labels, idx) => {
      if (!labels || idx < 0 || idx >= labels.length) return 0;
      return labels[idx];
  };

  const isStart = (label) => taskConfig.startLabels.includes(label);
  const isEnd = (label) => taskConfig.endLabels.includes(label);

  const table = document.createElement('table');
  table.className = 'asm-table';
  
  const thead = document.createElement('thead');
  const showIda = !!idaLabels;
  thead.innerHTML = `
    <tr>
      <th style="width: 60px;">Offset</th>
      <th>Bytes</th>
      <th>Instruction</th>
      <th class="center-align" style="width: 48px;">Truth</th>
      ${showIda ? '<th class="center-align" style="width: 48px;">IDA</th>' : ''}
      <th class="center-align" style="width: 48px;">DeepBound</th>
    </tr>
  `;
  table.appendChild(thead);

  const tbody = document.createElement('tbody');
  
  disassemblyData.forEach(instr => {
      const row = document.createElement('tr');
      row.className = 'asm-row';
      
      // Map objdump offset (relative to temp file start 0) to slice index
      const sliceIndex = instr.offset; 
      
      let truthBadge = '', idaBadge = '', deepboundBadge = '';
      let isXdaWin = false;
      let isIdaWin = false;

      if (sliceIndex >= 0 && sliceIndex < truthLabels.length) {
          const tLabel = getLabelAt(truthLabels, sliceIndex);
          const xLabel = getLabelAt(deepboundLabels, sliceIndex);
          const iLabel = showIda ? getLabelAt(idaLabels, sliceIndex) : 0;

          // Status Dots
          if (isStart(tLabel)) truthBadge = `<span class="status-dot truth start" title="Start"></span>`;
          else if (isEnd(tLabel)) truthBadge = `<span class="status-dot truth end" title="End"></span>`;
          
          if (isStart(xLabel)) deepboundBadge = `<span class="status-dot deepbound start" title="Start"></span>`;
          else if (isEnd(xLabel)) deepboundBadge = `<span class="status-dot deepbound end" title="End"></span>`;

          if (showIda) {
              if (isStart(iLabel)) idaBadge = `<span class="status-dot ida start" title="Start"></span>`;
              else if (isEnd(iLabel)) idaBadge = `<span class="status-dot ida end" title="End"></span>`;
          }

          // Winner logic
          if (isStart(tLabel)) {
              const deepboundHit = isStart(xLabel);
              const idaHit = isStart(iLabel);
              if (deepboundHit && !idaHit) isXdaWin = true;
              if (!deepboundHit && idaHit) isIdaWin = true;
          }
      }

      if (isXdaWin) row.classList.add('deepbound-win');
      if (isIdaWin) row.classList.add('ida-win');

      const highlightedText = highlightAsm(instr.text);

      row.innerHTML = `
        <td class="asm-offset">${(start + instr.offset).toString(16).toUpperCase()}</td>
        <td class="asm-bytes">${instr.bytes || ''}</td>
        <td class="asm-text">${highlightedText}</td>
        <td class="center-align">${truthBadge}</td>
        ${showIda ? `<td class="center-align">${idaBadge}</td>` : ''}
        <td class="center-align">${deepboundBadge}</td>
      `;
      tbody.appendChild(row);
  });

  table.appendChild(tbody);
  target.innerHTML = '';
  target.appendChild(table);
}

function populateSampleSelect(select, samples) {
  select.innerHTML = '';
  samples.forEach((sample) => {
    const length = sample.length ?? sample.total ?? 0;
    const option = document.createElement('option');
    option.value = sample.id;
    option.textContent = `${sample.name} (${length} 字节)`;
    select.appendChild(option);
  });
}

function setRangeInputs(elements, range, maxIndex) {
  elements.startInput.min = '0';
  elements.endInput.min = '0';
  elements.startInput.max = String(maxIndex);
  elements.endInput.max = String(maxIndex);
  elements.startInput.value = String(range.start);
  elements.endInput.value = String(range.end);
}

function setStatusBadge(elements, { text, active } = {}) {
  if (!elements.runStatus) {
    return;
  }
  if (typeof text === 'string') {
    elements.runStatus.textContent = text;
  }
  if (typeof active === 'boolean') {
    elements.runStatus.classList.toggle('ghost', !active);
  }
}

function setRunState(elements, { running, statusText }) {
  elements.runButton.disabled = running;
  setStatusBadge(elements, { text: statusText, active: running });
}

function setProgress(elements, progress) {
  const clamped = Math.max(0, Math.min(100, progress));
  elements.progressFill.style.width = `${clamped}%`;
  elements.progressLabel.textContent = `${Math.round(clamped)}%`;
}

function clearLogs(elements) {
  elements.logList.innerHTML = '';
}

function appendLog(elements, { message, level }) {
  const node = elements.logTemplate.content.firstElementChild.cloneNode(true);
  const time = new Date();
  node.querySelector('.log-time').textContent = time.toLocaleTimeString();
  node.querySelector('.log-text').textContent = message;
  if (level === 'alert') {
    node.classList.add('alert');
  } else if (level === 'success') {
    node.classList.add('success');
  }
  elements.logList.appendChild(node);
  elements.logList.scrollTop = elements.logList.scrollHeight;
}

function setSequenceMeta(elements, { sampleName, start, end, total }) {
  elements.sequenceMeta.textContent = `样本: ${sampleName} | 范围: ${start}-${end} | 字节: ${end - start + 1}/${total}`;
}

function updateRowMeta(target, labels, { labelMeta, metaLabels } = {}) {
  if (!target) {
    return;
  }
  if (!Array.isArray(labels) || !labels.length || !Array.isArray(metaLabels) || !metaLabels.length) {
    target.textContent = '';
    return;
  }
  const counts = new Map(metaLabels.map((label) => [label, 0]));
  labels.forEach((label) => {
    if (counts.has(label)) {
      counts.set(label, counts.get(label) + 1);
    }
  });
  const parts = metaLabels.map((label) => {
    const meta = labelMeta[label] || { short: String(label) };
    return `${meta.short}:${counts.get(label) ?? 0}`;
  });
  target.textContent = parts.join(' ');
}

function renderRow({ container, tokens, labels, labelMeta, diffAgainst, startIndex }) {
  container.innerHTML = '';
  const fragment = document.createDocumentFragment();

  const stride = 16;
  const chunkStride = 8;

  for (let i = 0; i < tokens.length; i += stride) {
    const rowDiv = document.createElement('div');
    rowDiv.className = 'grid-row';
    rowDiv.dataset.rowIndex = i;

    // Offset Label
    const offsetDiv = document.createElement('div');
    offsetDiv.className = 'grid-offset';
    // Show offset as 8-digit hex
    const currentOffset = startIndex + i;
    offsetDiv.textContent = currentOffset.toString(16).toUpperCase().padStart(8, '0') + ':';
    offsetDiv.dataset.offset = currentOffset;
    rowDiv.appendChild(offsetDiv);

    // Tokens
    const sliceEnd = Math.min(i + stride, tokens.length);
    for (let j = i; j < sliceEnd; j++) {
      // Add spacer between chunks (but not at the very start of the row)
      if (j > i && (j - i) % chunkStride === 0) {
        const spacer = document.createElement('div');
        spacer.className = 'chunk-spacer';
        rowDiv.appendChild(spacer);
      }

      const token = tokens[j];
      const label = labels[j];
      const meta = labelMeta[label] || labelMeta[0] || { short: '?', className: 'none' };
      
      const span = document.createElement('span');
      span.className = `token ${meta.className}`;
      span.textContent = token;
      span.dataset.label = meta.short;
      span.dataset.index = j; // Relative index in slice
      span.title = `索引 ${startIndex + j} | ${meta.short}`;
      
      if (diffAgainst && diffAgainst[j] !== label) {
        span.classList.add('diff');
      }
      
      rowDiv.appendChild(span);
    }
    fragment.appendChild(rowDiv);
  }
  container.appendChild(fragment);
}

// Shared state for sync interactions to avoid closure issues
const syncState = {
  isScrolling: false,
  activeIdx: null
};

function syncSequenceInteractions(elements) {
  const tracks = [elements.rowTruth, elements.rowIda, elements.rowXda].filter(Boolean);
  
  // Clean up existing listeners if any (by replacing with fresh ones or using a flag)
  // Since we are adding them to the same stable elements, we should be careful.
  
  const handleScroll = (e) => {
    if (syncState.isScrolling) return;
    syncState.isScrolling = true;
    const { scrollTop, scrollLeft } = e.target;
    tracks.forEach(track => {
      if (track !== e.target) {
        track.scrollTop = scrollTop;
        track.scrollLeft = scrollLeft;
      }
    });
    // Use requestAnimationFrame to reset the flag to ensure sync is finished
    requestAnimationFrame(() => {
        syncState.isScrolling = false;
    });
  };

  const clearHighlights = () => {
    document.querySelectorAll('.token.highlight-sync').forEach(el => {
      el.classList.remove('highlight-sync');
    });
  };

  const handleMouseMove = (e) => {
    const token = e.target.closest('.token');
    if (token) {
      const idx = token.dataset.index;
      if (syncState.activeIdx === idx) return;
      syncState.activeIdx = idx;
      
      clearHighlights();
      tracks.forEach(track => {
        const targetToken = track.querySelector(`.token[data-index="${idx}"]`);
        if (targetToken) targetToken.classList.add('highlight-sync');
      });
    } else {
      if (syncState.activeIdx !== null) {
          syncState.activeIdx = null;
          clearHighlights();
      }
    }
  };

  const handleMouseLeave = () => {
    syncState.activeIdx = null;
    clearHighlights();
  };

  const handleClick = (e) => {
    const offsetLabel = e.target.closest('.grid-offset');
    if (offsetLabel) {
      const row = offsetLabel.closest('.grid-row');
      if (row) {
        const top = row.offsetTop - 12;
        tracks.forEach(track => {
          track.scrollTo({ top, behavior: 'smooth' });
        });
      }
    }
  };

  tracks.forEach(track => {
    // Remove old listeners using the fact they are assigned to the element
    // Actually, a better way is to attach them once in getElements or init.
    // But since the elements themselves are stable, let's just use a property.
    if (track._syncAttached) return;
    
    track.addEventListener('scroll', handleScroll);
    track.addEventListener('mousemove', handleMouseMove);
    track.addEventListener('mouseleave', handleMouseLeave);
    track.addEventListener('click', handleClick);
    
    track._syncAttached = true;
  });
}

function renderSequence(elements, payload) {
  const {
    tokens,
    truthLabels,
    idaLabels,
    deepboundLabels,
    labelMeta,
    metaLabels,
    startIndex
  } = payload;

  renderRow({
    container: elements.rowTruth,
    tokens,
    labels: truthLabels,
    labelMeta,
    startIndex
  });

  if (Array.isArray(idaLabels) && idaLabels.length) {
    renderRow({
      container: elements.rowIda,
      tokens,
      labels: idaLabels,
      labelMeta,
      diffAgainst: truthLabels,
      startIndex
    });
  } else if (elements.rowIda) {
    elements.rowIda.innerHTML = '';
  }

  renderRow({
    container: elements.rowXda,
    tokens,
    labels: deepboundLabels,
    labelMeta,
    diffAgainst: truthLabels,
    startIndex
  });

  updateRowMeta(elements.rowMetaTruth, truthLabels, { labelMeta, metaLabels });
  updateRowMeta(elements.rowMetaIda, idaLabels, { labelMeta, metaLabels });
  updateRowMeta(elements.rowMetaXda, deepboundLabels, { labelMeta, metaLabels });

  // Initialize/refresh sync interactions
  syncSequenceInteractions(elements);
}

function toggleLabels(elements, enabled) {
  elements.sequenceGrid.classList.toggle('show-labels', enabled);
}

function toggleDiffs(elements, enabled) {
  elements.sequenceGrid.classList.toggle('show-diffs', enabled);
}

function updateMetrics(elements, { deepbound, ida, formatPercent }) {
  elements.metricXdaPrec.textContent = formatPercent(deepbound.precision);
  elements.metricXdaRecall.textContent = formatPercent(deepbound.recall);
  elements.metricXdaF1.textContent = formatPercent(deepbound.f1);

  if (ida) {
    elements.metricIdaPrec.textContent = formatPercent(ida.precision);
    elements.metricIdaRecall.textContent = formatPercent(ida.recall);
    elements.metricIdaF1.textContent = formatPercent(ida.f1);
  } else {
    elements.metricIdaPrec.textContent = '-';
    elements.metricIdaRecall.textContent = '-';
    elements.metricIdaF1.textContent = '-';
  }
}

function setMetricsPlaceholder(elements, placeholder = '-') {
  elements.metricXdaPrec.textContent = placeholder;
  elements.metricXdaRecall.textContent = placeholder;
  elements.metricXdaF1.textContent = placeholder;
  elements.metricIdaPrec.textContent = placeholder;
  elements.metricIdaRecall.textContent = placeholder;
  elements.metricIdaF1.textContent = placeholder;
}

function setMetricsScope(elements, scope) {
  if (elements.metricsScopeBadge) {
    elements.metricsScopeBadge.textContent = scope === 'validation' ? '测试集整体' : '仅当前片段';
  }
  if (elements.metricScopeButtons) {
    elements.metricScopeButtons.forEach((button) => {
      button.classList.toggle('active', button.dataset.metricsScope === scope);
    });
  }
}

function setMetricsNote(elements, note) {
  if (!elements.metricsNote) {
    return;
  }
  if (!note) {
    elements.metricsNote.textContent = '';
    elements.metricsNote.classList.remove('alert');
    return;
  }
  if (typeof note === 'string') {
    elements.metricsNote.textContent = note;
    elements.metricsNote.classList.remove('alert');
    return;
  }
  elements.metricsNote.textContent = note.text || '';
  elements.metricsNote.classList.toggle('alert', note.tone === 'alert');
}

function setLegend(elements, legend, middleClass = 'end') {
  if (elements.legendStart && legend?.start) {
    elements.legendStart.textContent = legend.start;
  }
  if (elements.legendMiddle && legend?.middle) {
    elements.legendMiddle.textContent = legend.middle;
    elements.legendMiddle.classList.remove('end', 'body');
    elements.legendMiddle.classList.add(middleClass);
  }
  if (elements.legendNone && legend?.none) {
    elements.legendNone.textContent = legend.none;
  }
}

function setInsightLabels(elements, labels = {}) {
  if (elements.insightLabelXdaStart && labels.deepboundStart) {
    elements.insightLabelXdaStart.textContent = labels.deepboundStart;
  }
  if (elements.insightLabelXdaEnd && labels.deepboundEnd) {
    elements.insightLabelXdaEnd.textContent = labels.deepboundEnd;
  }
  if (elements.insightLabelIdaStart && labels.idaStart) {
    elements.insightLabelIdaStart.textContent = labels.idaStart;
  }
  if (elements.insightLabelIdaEnd && labels.idaEnd) {
    elements.insightLabelIdaEnd.textContent = labels.idaEnd;
  }
}

function updateInsights(elements, { deepboundHits, idaHits }) {
  // Helpers to update insight items safely
  const updateItem = (valId, labelId, hits, labelText, barClass) => {
      const valEl = document.getElementById(valId);
      const labelEl = document.getElementById(labelId);
      
      // Find parent insight-item to restructure if needed (simple check)
      if (valEl && valEl.parentElement.classList.contains('insight-item')) {
          const parent = valEl.parentElement;
          // Ensure bar exists
          if (!parent.querySelector('.insight-bar')) {
              const bar = document.createElement('span');
              bar.className = 'insight-bar';
              parent.prepend(bar);
          }
      }

      if (valEl) valEl.textContent = hits ?? '-';
      if (labelEl && labelText) labelEl.textContent = labelText;
  };

  updateItem('insight-deepbound-start', 'insight-label-deepbound-start', deepboundHits ? deepboundHits.startHits : '-', null);
  updateItem('insight-deepbound-end', 'insight-label-deepbound-end', deepboundHits ? deepboundHits.endHits : '-', null);
  updateItem('insight-ida-start', 'insight-label-ida-start', idaHits ? idaHits.startHits : '-', null);
  updateItem('insight-ida-end', 'insight-label-ida-end', idaHits ? idaHits.endHits : '-', null);
}

function initMetricsChart(canvas) {
  const chart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['精确率', '召回率', 'F1'],
      datasets: [
        {
          label: 'DeepBound',
          data: [0, 0, 0],
          backgroundColor: '#F59E0B' // Orange
        },
        {
          label: 'IDA-PRO',
          data: [0, 0, 0],
          backgroundColor: '#3B82F6' // Blue
        }
      ]
    },
    options: {
      responsive: true,
      scales: {
        y: {
          beginAtZero: true,
          max: 1,
          ticks: {
            callback: (value) => `${Math.round(value * 100)}%`
          },
          grid: {
            color: 'rgba(255, 255, 255, 0.06)'
          }
        },
        x: {
          grid: {
            display: false
          }
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: '#cfd7e3'
          }
        }
      }
    }
  });

  return chart;
}

function updateMetricsChart(chart, series) {
  chart.data.labels = series.labels;
  chart.data.datasets[0].data = series.deepbound;
  if (Array.isArray(series.ida)) {
    chart.data.datasets[1].data = series.ida;
  }
  if (typeof series.showIda === 'boolean') {
    chart.data.datasets[1].hidden = !series.showIda;
  }
  chart.update();
}

export {
  appendLog,
  clearLogs,
  getElements,
  initMetricsChart,
  populateSampleSelect,
  renderSequence,
  renderDisassembly,
  setProgress,
  setRangeInputs,
  setMetricsNote,
  setMetricsPlaceholder,
  setMetricsScope,
  setLegend,
  setInsightLabels,
  setRunState,
  setStatusBadge,
  setSequenceMeta,
  toggleDiffs,
  toggleLabels,
  updateInsights,
  updateMetrics,
  updateMetricsChart
};
