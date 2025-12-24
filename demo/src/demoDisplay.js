const TASK_CONFIGS = {
  funcbound: {
    labelMeta: {
      0: { short: '-', className: 'none' },
      1: { short: 'R', className: 'end' },
      2: { short: 'F', className: 'start' }
    },
    boundaryLabels: [1, 2],
    startLabels: [2],
    endLabels: [1],
    rowMetaLabels: [2, 1],
    legend: {
      start: 'F = 函数起始',
      middle: 'R = 函数结束',
      none: '- = 非边界'
    },
    supportsIda: true,
    supportsRecommend: true,
    supportsValidation: true
  }
};

function getTaskConfig(task) {
  return TASK_CONFIGS[task] || TASK_CONFIGS.funcbound;
}

function computeMetrics(truthLabels, predLabels, boundaryLabels = [1, 2]) {
  if (!Array.isArray(truthLabels) || !Array.isArray(predLabels)) {
    return {
      accuracy: 0,
      precision: null,
      recall: null,
      f1: null,
      boundaryTrue: 0,
      boundaryPred: 0
    };
  }
  const total = Math.min(truthLabels.length, predLabels.length);
  let totalCorrect = 0;
  let boundaryCorrect = 0;
  let boundaryTrue = 0;
  let boundaryPred = 0;
  const boundarySet = new Set(boundaryLabels);

  for (let i = 0; i < total; i += 1) {
    const truth = truthLabels[i];
    const pred = predLabels[i];

    if (truth === pred) {
      totalCorrect += 1;
    }

    if (boundarySet.has(truth)) {
      boundaryTrue += 1;
      if (pred === truth) {
        boundaryCorrect += 1;
      }
    }

    if (boundarySet.has(pred)) {
      boundaryPred += 1;
    }
  }

  const precision = boundaryPred === 0 ? null : boundaryCorrect / boundaryPred;
  const recall = boundaryTrue === 0 ? null : boundaryCorrect / boundaryTrue;
  let f1 = null;
  if (precision !== null && recall !== null) {
    const denom = precision + recall;
    f1 = denom === 0 ? 0 : (2 * precision * recall) / denom;
  }
  const accuracy = total === 0 ? 0 : totalCorrect / total;

  return {
    accuracy,
    precision,
    recall,
    f1,
    boundaryTrue,
    boundaryPred
  };
}

function countBoundaryHits(truthLabels, predLabels, { startLabels = [2], endLabels = [1] } = {}) {
  let startHits = 0;
  let endHits = 0;
  if (!Array.isArray(truthLabels) || !Array.isArray(predLabels)) {
    return { startHits, endHits };
  }
  const total = Math.min(truthLabels.length, predLabels.length);
  const startSet = new Set(startLabels);
  const endSet = new Set(endLabels);
  for (let i = 0; i < total; i += 1) {
    if (startSet.has(truthLabels[i]) && predLabels[i] === truthLabels[i]) {
      startHits += 1;
    }
    if (endSet.has(truthLabels[i]) && predLabels[i] === truthLabels[i]) {
      endHits += 1;
    }
  }
  return { startHits, endHits };
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '-';
  }
  return `${(value * 100).toFixed(1)}%`;
}

export {
  TASK_CONFIGS,
  getTaskConfig,
  computeMetrics,
  countBoundaryHits,
  formatPercent
};
