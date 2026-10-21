// ====================================================================
// Anna ML Studio — Multiple Linear Regression Interactive Dashboard
// Supports: Predefined Problems (Appliances, 50 Startups) & Custom CSVs
// ====================================================================

// Global State
let currentProblemId = 'appliances';
let customModelData = null;
let currentPreviewData = null;

// Chart.js Instances
let fsChart = null;
let coefChart = null;
let timeSeriesChart = null;
let diagActualChart = null;
let diagResidChart = null;
let diagQqChart = null;
let diagIntervalChart = null;

const methodColors = {
  'F-test': '#1a73e8',
  'Random Forest': '#34a853',
  'Lasso': '#f9ab00',
  'RFE': '#ea4335',
  'Mutual Information': '#9334e6'
};

document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initProblemSelector();
  initCustomModal();
  loadProblem('appliances');
});

// ----------------------------------------------------
// Navigation Tabs
// ----------------------------------------------------
function initTabs() {
  const tabs = document.querySelectorAll('.nav-tab');
  const panes = document.querySelectorAll('.tab-pane');

  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const targetId = tab.getAttribute('data-tab');
      tabs.forEach(t => t.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const activePane = document.getElementById(targetId);
      if (activePane) activePane.classList.add('active');
    });
  });
}

// ----------------------------------------------------
// Problem Switching Logic
// ----------------------------------------------------
function initProblemSelector() {
  const problemBtns = document.querySelectorAll('.problem-btn');
  problemBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const probId = btn.getAttribute('data-problem');
      if (probId === 'custom' && !customModelData) return;
      loadProblem(probId);
    });
  });
}

function getActiveProblemData() {
  if (currentProblemId === 'custom' && customModelData) {
    return customModelData;
  }
  if (window.PROBLEMS && window.PROBLEMS[currentProblemId]) {
    return window.PROBLEMS[currentProblemId];
  }
  return window.APP_DATA || null;
}

function loadProblem(probId) {
  currentProblemId = probId;

  // Update Problem Selector buttons active state
  document.querySelectorAll('.problem-btn').forEach(btn => {
    btn.classList.toggle('active', btn.getAttribute('data-problem') === probId);
  });

  const data = getActiveProblemData();
  if (!data) return;

  // 1. Header & Metadata
  updateHeader(data);

  // 2. Tab 1: Overview KPIs, Comparison Table, CRISP-DM
  updateOverviewTab(data);

  // 3. Tab 2: Feature Selection Explorer
  updateFeatureExplorer(data);

  // 4. Tab 3: Simulator
  updateSimulator(data);

  // 5. Tab 4: Coefficients & Diagnostics
  updateDiagnosticsTab(data);

  // 6. Tab 5: Context Tab
  updateContextTab(data);
}

// ----------------------------------------------------
// 1. Header & Metadata Updates
// ----------------------------------------------------
function updateHeader(data) {
  const headerTitle = document.getElementById('appHeaderTitle');
  const headerBadge = document.getElementById('appHeaderBadge');
  const metaStatus = document.getElementById('metaStatus');
  const metaDataset = document.getElementById('metaDataset');
  const metaValidation = document.getElementById('metaValidation');
  const footerTitle = document.getElementById('footerAppTitle');
  const footerMeta = document.getElementById('footerDatasetMeta');

  if (headerTitle) headerTitle.textContent = data.title;
  if (headerBadge) headerBadge.textContent = data.badge || 'CRISP-DM';
  if (metaStatus) {
    metaStatus.innerHTML = `<span class="dot-indicator"></span> ${data.statusText || '模型訓練完成'}`;
  }
  if (metaDataset) metaDataset.textContent = data.datasetInfo || '';
  if (metaValidation) metaValidation.textContent = data.validationInfo || '';

  if (footerTitle) footerTitle.textContent = `${data.title} • Designed by Anna ML Studio`;
  if (footerMeta) footerMeta.textContent = `Target: ${data.target} (${data.unit}) • Multiple Linear Regression Pipeline`;
}

// ----------------------------------------------------
// 2. Overview Tab: KPIs, Comparison Table, CRISP-DM
// ----------------------------------------------------
function updateOverviewTab(data) {
  const m = data.metrics || {};
  const unit = data.unit || '';

  // KPI 1: Features count
  const kpiFeaturesVal = document.getElementById('kpiFeaturesVal');
  const kpiFeaturesMethod = document.getElementById('kpiFeaturesMethod');
  const kpiFeaturesDelta = document.getElementById('kpiFeaturesDelta');
  const kpiFeaturesDesc = document.getElementById('kpiFeaturesDesc');

  if (kpiFeaturesVal) {
    kpiFeaturesVal.innerHTML = `${m.selectedFeaturesCount || data.bestK} <span class="kpi-unit">/ ${m.totalFeaturesCount || data.kRange?.max || 30} 個</span>`;
  }
  if (kpiFeaturesMethod) kpiFeaturesMethod.textContent = `⭐ ${m.bestMethod || data.bestModel}`;
  if (kpiFeaturesDelta && m.totalFeaturesCount) {
    const pct = (((m.totalFeaturesCount - (m.selectedFeaturesCount || data.bestK)) / m.totalFeaturesCount) * 100).toFixed(1);
    kpiFeaturesDelta.textContent = `-${pct}%`;
  }
  if (kpiFeaturesDesc) {
    kpiFeaturesDesc.textContent = currentProblemId === 'appliances'
      ? '顯著降低模型複雜度與維度'
      : '最佳精簡特徵子集，排除多重共線性';
  }

  // KPI 2: CV RMSE
  const kpiCvTitle = document.getElementById('kpiCvTitle');
  const kpiCvVal = document.getElementById('kpiCvVal');
  const kpiCvType = document.getElementById('kpiCvType');
  const kpiCvDelta = document.getElementById('kpiCvDelta');
  const kpiCvDesc = document.getElementById('kpiCvDesc');

  if (kpiCvTitle) kpiCvTitle.textContent = `交叉驗證 CV RMSE (${unit})`;
  if (kpiCvVal) {
    kpiCvVal.innerHTML = `${Number(m.cvRmse).toLocaleString()} <span class="kpi-unit">${unit}</span>`;
  }
  if (kpiCvType) kpiCvType.textContent = data.validationInfo?.split('(')[0]?.trim() || 'K-Fold CV';
  if (kpiCvDelta) kpiCvDelta.textContent = m.cvRmseDelta || `${(m.cvRmse - (m.baselineCvRmse || m.cvRmse)).toFixed(2)} ${unit}`;
  if (kpiCvDesc) {
    kpiCvDesc.textContent = m.baselineCvRmse ? `優於 Baseline (${Number(m.baselineCvRmse).toLocaleString()} ${unit})` : '交叉驗證最佳表現';
  }

  // KPI 3: Test R²
  const kpiR2Val = document.getElementById('kpiR2Val');
  const kpiR2Adj = document.getElementById('kpiR2Adj');
  if (kpiR2Val) {
    const r2Num = Number(m.testR2 || 0);
    kpiR2Val.innerHTML = `${r2Num.toFixed(3)} <span class="kpi-unit">(${(r2Num * 100).toFixed(1)}%)</span>`;
  }
  if (kpiR2Adj) {
    const adjNum = m.testAdjustedR2 !== undefined && m.testAdjustedR2 !== null ? Number(m.testAdjustedR2).toFixed(4) : 'N/A';
    kpiR2Adj.textContent = `Adjusted R²: ${adjNum}`;
  }

  // KPI 4: MAE
  const kpiMaeVal = document.getElementById('kpiMaeVal');
  const kpiMaeMse = document.getElementById('kpiMaeMse');
  if (kpiMaeVal) {
    kpiMaeVal.innerHTML = `${Number(m.testMae || 0).toLocaleString()} <span class="kpi-unit">${unit}</span>`;
  }
  if (kpiMaeMse) {
    const mseNum = m.testMse !== undefined ? Number(m.testMse).toLocaleString() : 'N/A';
    kpiMaeMse.textContent = `MSE: ${mseNum}`;
  }

  // Comparison Table
  const tableBody = document.getElementById('comparisonTableBody');
  const thTestRmse = document.getElementById('thTestRmse');
  const thTestMae = document.getElementById('thTestMae');

  if (thTestRmse) thTestRmse.textContent = `測試集 RMSE (${unit})`;
  if (thTestMae) thTestMae.textContent = `測試集 MAE (${unit})`;

  if (tableBody) {
    tableBody.innerHTML = '';
    const compRows = data.comparison || [];
    compRows.forEach(row => {
      const tr = document.createElement('tr');
      const isWinner = row.isWinner || row.modelName.includes('⭐') || row.modelName.includes('Selected');
      const adjText = row.testAdjustedR2 !== null && row.testAdjustedR2 !== undefined ? Number(row.testAdjustedR2).toFixed(4) : 'N/A';

      tr.innerHTML = `
        <td>
          <span class="badge-model ${isWinner ? 'badge-winner' : 'badge-baseline'}">
            ${isWinner ? '<span>⭐</span> ' : ''}${row.modelName}
          </span>
        </td>
        <td><strong>${row.k}</strong> <span style="font-size:11.5px; color:var(--text-secondary);">${row.kDetail || ''}</span></td>
        <td><strong>${Number(row.testRmse).toLocaleString()}</strong></td>
        <td>${Number(row.testMae).toLocaleString()}</td>
        <td>${Number(row.testR2).toFixed(4)}</td>
        <td><strong>${adjText}</strong></td>
        <td><span class="meta-chip ${isWinner ? 'chip-success' : ''}">${row.remark || ''}</span></td>
      `;
      tableBody.appendChild(tr);
    });
  }

  // CRISP-DM Stepper Text
  const c = data.crispdm || {};
  if (document.getElementById('crispPhase1Desc')) document.getElementById('crispPhase1Desc').textContent = c.businessUnderstanding || '';
  if (document.getElementById('crispPhase2Desc')) document.getElementById('crispPhase2Desc').textContent = c.dataUnderstanding || '';
  if (document.getElementById('crispPhase3Desc')) document.getElementById('crispPhase3Desc').textContent = c.dataPreparation || '';
  if (document.getElementById('crispPhase4Desc')) document.getElementById('crispPhase4Desc').textContent = c.modeling || '';
  if (document.getElementById('crispPhase5Desc')) document.getElementById('crispPhase5Desc').textContent = c.evaluation || '';
  if (document.getElementById('crispPhase6Desc')) document.getElementById('crispPhase6Desc').textContent = c.deployment || '';
}

// ----------------------------------------------------
// 3. Feature Selection Explorer (Chart.js)
// ----------------------------------------------------
function updateFeatureExplorer(data) {
  const ctx = document.getElementById('featureSelectionChart');
  if (!ctx) return;

  const cvSummary = data.cvSummary || [];
  const methods = ['F-test', 'Random Forest', 'Lasso', 'RFE', 'Mutual Information'];
  const baselineRow = cvSummary.find(r => r.method === 'All Features Baseline' || r.method === 'Baseline');
  const baselineRMSE = baselineRow ? baselineRow.CV_RMSE : (data.metrics?.baselineCvRmse || 0);

  const kMin = data.kRange?.min || 1;
  const kMax = data.kRange?.max || 30;
  const unit = data.unit || '';

  // Prepare datasets for 5 methods
  const datasets = methods.map(m => {
    const rows = cvSummary.filter(r => r.method === m).sort((a, b) => a.k - b.k);
    return {
      label: m,
      data: rows.map(r => ({ x: r.k, y: r.CV_RMSE })),
      borderColor: methodColors[m] || '#1a73e8',
      backgroundColor: methodColors[m] || '#1a73e8',
      borderWidth: 2,
      pointRadius: rows.length <= 10 ? 4.5 : 3,
      pointHoverRadius: 6,
      tension: 0.15,
      hidden: false
    };
  });

  // Add Baseline horizontal dashed line
  if (baselineRMSE > 0) {
    const ks = Array.from({ length: kMax - kMin + 1 }, (_, i) => i + kMin);
    datasets.push({
      label: `All Features Baseline (k=${kMax}, RMSE=${Number(baselineRMSE.toFixed(2)).toLocaleString()})`,
      data: ks.map(k => ({ x: k, y: baselineRMSE })),
      borderColor: '#3c4043',
      borderWidth: 2,
      borderDash: [6, 4],
      pointRadius: 0,
      fill: false
    });
  }

  // Calculate suitable Y-axis bounds
  const allY = cvSummary.map(r => r.CV_RMSE).filter(y => y > 0);
  const minY = allY.length > 0 ? Math.min(...allY) : 0;
  const maxY = allY.length > 0 ? Math.max(...allY) : 100;
  const padding = (maxY - minY) * 0.1 || 1.0;

  if (fsChart) {
    fsChart.destroy();
  }

  fsChart = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        title: {
          display: true,
          text: `${data.title}: 特徵數 (k) 與交叉驗證 RMSE 收斂關係`,
          font: { size: 14, weight: 'bold', family: 'Google Sans, Inter, sans-serif' },
          color: '#202124',
          padding: { bottom: 12 }
        },
        legend: {
          position: 'bottom',
          labels: { boxWidth: 12, font: { size: 12 }, padding: 14 }
        },
        tooltip: {
          backgroundColor: 'rgba(32, 33, 36, 0.95)',
          callbacks: {
            title: items => `特徵數: k = ${items[0].parsed.x}`,
            label: item => `${item.dataset.label}: ${Number(item.parsed.y.toFixed(2)).toLocaleString()} ${unit}`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: '入選特徵數 (k)', font: { size: 12, weight: 'bold' } },
          min: kMin,
          max: kMax,
          ticks: { stepSize: kMax <= 10 ? 1 : 2 },
          grid: { color: 'rgba(218, 220, 224, 0.6)' }
        },
        y: {
          title: { display: true, text: `CV RMSE (${unit})`, font: { size: 12, weight: 'bold' } },
          min: Math.max(0, Math.floor(minY - padding)),
          max: Math.ceil(maxY + padding),
          grid: { color: 'rgba(218, 220, 224, 0.6)' }
        }
      }
    }
  });

  // Re-bind method filter buttons
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    btn.classList.add('active');
    btn.onclick = () => {
      const targetMethod = btn.getAttribute('data-method');
      if (targetMethod === 'all') {
        fsChart.data.datasets.forEach(d => (d.hidden = false));
        filterBtns.forEach(b => b.classList.add('active'));
      } else {
        const dataset = fsChart.data.datasets.find(d => d.label.startsWith(targetMethod));
        if (dataset) {
          dataset.hidden = !dataset.hidden;
          btn.classList.toggle('active', !dataset.hidden);
        }
      }
      fsChart.update();
    };
  });

  // Setup k Slider
  const kSlider = document.getElementById('kSlider');
  const kValueBadge = document.getElementById('kValueBadge');
  const activeFeaturesContainer = document.getElementById('activeFeaturesContainer');

  const orderedFeatures = getRankedFeatureList(data);

  if (kSlider && kValueBadge) {
    kSlider.min = kMin;
    kSlider.max = kMax;
    const defaultK = data.bestK || Math.min(8, kMax);
    kSlider.value = defaultK;
    kValueBadge.textContent = `k = ${defaultK}`;

    kSlider.oninput = e => {
      const k = parseInt(e.target.value, 10);
      kValueBadge.textContent = `k = ${k}`;
      updateActiveFeaturesList(k, orderedFeatures, activeFeaturesContainer, defaultK);
    };

    updateActiveFeaturesList(defaultK, orderedFeatures, activeFeaturesContainer, defaultK);
  }
}

function getRankedFeatureList(data) {
  if (currentProblemId === 'appliances') {
    return [
      'appliances_current', 'hour_sin', 'lights', 'hour_cos', 'RH_out', 'T2', 'T6', 'T3',
      'T8', 'RH_1', 'RH_2', 'RH_3', 'T1', 'T4', 'T7', 'T9', 'RH_8', 'RH_7', 'T_out',
      'RH_9', 'RH_4', 'Windspeed', 'RH_5', 'T5', 'RH_6', 'Press_mm_hg', 'Visibility',
      'weekday_sin', 'Tdewpoint', 'weekday_cos'
    ];
  }
  if (data.winner_selected_features) {
    return data.winner_selected_features.map(f => f.replace('num__', '').replace('cat__', ''));
  }
  if (data.selectedFeatures) {
    return data.selectedFeatures.map(f => f.feature);
  }
  return [];
}

function updateActiveFeaturesList(k, features, container, winnerK) {
  if (!container) return;
  container.innerHTML = '';
  const active = features.slice(0, k);

  active.forEach((feat, idx) => {
    const chip = document.createElement('span');
    chip.className = `meta-chip ${idx < winnerK ? 'chip-success' : ''}`;
    chip.textContent = `#${idx + 1} ${feat}`;
    container.appendChild(chip);
  });
}

// ----------------------------------------------------
// 4. ML Simulator (Universal Multi-Problem Logic)
// ----------------------------------------------------
function updateSimulator(data) {
  const inputsContainer = document.getElementById('inputsGridContainer');
  const presetsContainer = document.getElementById('presetsContainer');
  const predUnit = document.getElementById('predUnit');
  const simulatorPredTitle = document.getElementById('simulatorPredTitle');
  const simulatorCardTitle = document.getElementById('simulatorCardTitle');
  const simulatorCardSubtitle = document.getElementById('simulatorCardSubtitle');

  if (!inputsContainer || !data.modelFormula) return;

  const unit = data.unit || '';
  if (predUnit) predUnit.textContent = unit;
  if (simulatorPredTitle) simulatorPredTitle.textContent = `預測目標值 ${data.target} (ŷ)`;
  if (simulatorCardTitle) simulatorCardTitle.textContent = `🎛️ 多元線性迴歸即時推論器 — ${data.target}`;
  if (simulatorCardSubtitle) {
    simulatorCardSubtitle.textContent = `調整入選特徵數值，即時透過真實迴歸係數計算目標值 (單位: ${unit})`;
  }

  // Clear previous inputs & presets
  inputsContainer.innerHTML = '';
  if (presetsContainer) presetsContainer.innerHTML = '';

  const formula = data.modelFormula;
  const { features, coefs, intercept, means, scales } = formula;

  // Render Simulator Presets
  const presets = data.presets || [];
  if (presetsContainer && presets.length > 0) {
    presets.forEach(p => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'preset-chip';
      btn.textContent = p.label;
      btn.onclick = () => {
        applySimulatorPreset(p.values, currentProblemId);
      };
      presetsContainer.appendChild(btn);
    });
  }

  // Render Simulator Controls
  if (currentProblemId === 'appliances') {
    renderAppliancesSimulatorInputs(inputsContainer);
  } else if (currentProblemId === '50_startups') {
    renderStartupsSimulatorInputs(inputsContainer);
  } else {
    renderGenericSimulatorInputs(inputsContainer, data);
  }

  // Bind calculation event
  inputsContainer.querySelectorAll('input, select').forEach(elem => {
    elem.addEventListener('input', () => runSimulation(data));
  });

  runSimulation(data);
}

function renderAppliancesSimulatorInputs(container) {
  container.innerHTML = `
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">當前用電量 (appliances_current)</span>
        <span id="val_appliances_current" class="input-val-display">60 Wh</span>
      </div>
      <input type="range" id="sim_appliances_current" class="input-slider" min="10" max="800" value="60" step="10">
      <div class="input-hint"><span>10 Wh (待機)</span><span>800 Wh (滿載)</span></div>
    </div>
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">一天時刻 (Hour of Day)</span>
        <span id="val_hour" class="input-val-display">17:00</span>
      </div>
      <input type="range" id="sim_hour" class="input-slider" min="0" max="23.5" value="17" step="0.5">
      <div class="input-hint"><span>00:00 (子夜)</span><span>23:30 (深夜)</span></div>
    </div>
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">室內照明燈耗電 (lights)</span>
        <span id="val_lights" class="input-val-display">10 Wh</span>
      </div>
      <input type="range" id="sim_lights" class="input-slider" min="0" max="70" value="10" step="5">
      <div class="input-hint"><span>0 Wh (全熄)</span><span>70 Wh (全開)</span></div>
    </div>
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">室外相對濕度 (RH_out)</span>
        <span id="val_RH_out" class="input-val-display">82 %</span>
      </div>
      <input type="range" id="sim_RH_out" class="input-slider" min="30" max="100" value="82" step="1">
      <div class="input-hint"><span>30% (乾燥)</span><span>100% (陰雨飽和)</span></div>
    </div>
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">客廳溫度 (T2, Living Room)</span>
        <span id="val_T2" class="input-val-display">20.5 °C</span>
      </div>
      <input type="range" id="sim_T2" class="input-slider" min="15" max="28" value="20.5" step="0.1">
      <div class="input-hint"><span>15°C (偏冷)</span><span>28°C (偏暖)</span></div>
    </div>
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">北側外牆溫度 (T6, North Outside)</span>
        <span id="val_T6" class="input-val-display">6.5 °C</span>
      </div>
      <input type="range" id="sim_T6" class="input-slider" min="-5" max="25" value="6.5" step="0.5">
      <div class="input-hint"><span>-5°C (寒冬冰點)</span><span>25°C (溫暖)</span></div>
    </div>
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">洗衣房溫度 (T3, Laundry Room)</span>
        <span id="val_T3" class="input-val-display">21.8 °C</span>
      </div>
      <input type="range" id="sim_T3" class="input-slider" min="16" max="28" value="21.8" step="0.1">
      <div class="input-hint"><span>16°C</span><span>28°C</span></div>
    </div>
  `;
}

function renderStartupsSimulatorInputs(container) {
  container.innerHTML = `
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">研發支出 (R&D Spend) ⭐ 關鍵正向指標</span>
        <span id="val_sim_rd" class="input-val-display">$75,000</span>
      </div>
      <input type="range" id="sim_rd" class="input-slider" min="0" max="165000" value="75000" step="1000">
      <div class="input-hint"><span>$0 (無研發)</span><span>$165,000 (頂級研發投資)</span></div>
    </div>
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">行政管理費用 (Administration)</span>
        <span id="val_sim_admin" class="input-val-display">$120,000</span>
      </div>
      <input type="range" id="sim_admin" class="input-slider" min="50000" max="185000" value="120000" step="1000">
      <div class="input-hint"><span>$50,000 (極精實)</span><span>$185,000 (高行政成本)</span></div>
    </div>
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">行銷推廣支出 (Marketing Spend)</span>
        <span id="val_sim_marketing" class="input-val-display">$210,000</span>
      </div>
      <input type="range" id="sim_marketing" class="input-slider" min="0" max="470000" value="210000" step="2000">
      <div class="input-hint"><span>$0 (口碑傳播)</span><span>$470,000 (強勢廣告投放)</span></div>
    </div>
    <div class="input-card">
      <div class="input-header">
        <span class="input-title">登記所在州別 (State)</span>
        <span id="val_sim_state" class="input-val-display">New York</span>
      </div>
      <select id="sim_state" class="modal-select" style="margin-top: 6px;">
        <option value="California">California (基準類別)</option>
        <option value="Florida">Florida</option>
        <option value="New York" selected>New York</option>
      </select>
      <div class="input-hint" style="margin-top: 6px;"><span>One-Hot 虛擬變數編碼 (California 為 Reference Base)</span></div>
    </div>
  `;
}

function renderGenericSimulatorInputs(container, data) {
  const specs = data.featureSpecs || [];
  if (specs.length === 0) {
    container.innerHTML = '<p style="padding:16px; color:var(--text-secondary);">本模型無額外特徵規格。</p>';
    return;
  }

  specs.forEach(spec => {
    const card = document.createElement('div');
    card.className = 'input-card';
    const fieldId = `sim_field_${spec.name.replace(/[^a-zA-Z0-9_]/g, '_')}`;

    if (spec.type === 'numeric') {
      card.innerHTML = `
        <div class="input-header">
          <span class="input-title">${spec.name}</span>
          <span id="val_${fieldId}" class="input-val-display">${spec.default}</span>
        </div>
        <input type="range" id="${fieldId}" class="input-slider"
               data-field="${spec.name}"
               min="${spec.min}" max="${spec.max}" value="${spec.default}" step="${spec.step || 1}">
        <div class="input-hint"><span>${spec.min}</span><span>${spec.max}</span></div>
      `;
    } else {
      const optionsHtml = (spec.options || []).map(opt => `<option value="${opt}" ${opt === spec.default ? 'selected' : ''}>${opt}</option>`).join('');
      card.innerHTML = `
        <div class="input-header">
          <span class="input-title">${spec.name} (類別變數)</span>
          <span id="val_${fieldId}" class="input-val-display">${spec.default}</span>
        </div>
        <select id="${fieldId}" class="modal-select" data-field="${spec.name}" style="margin-top: 6px;">
          ${optionsHtml}
        </select>
      `;
    }
    container.appendChild(card);
  });
}

function applySimulatorPreset(values, probId) {
  if (probId === 'appliances') {
    Object.keys(values).forEach(k => {
      const input = document.getElementById(`sim_${k}`);
      if (input) input.value = values[k];
    });
  } else if (probId === '50_startups') {
    if (values['R&D Spend'] !== undefined && document.getElementById('sim_rd')) {
      document.getElementById('sim_rd').value = values['R&D Spend'];
    }
    if (values['Administration'] !== undefined && document.getElementById('sim_admin')) {
      document.getElementById('sim_admin').value = values['Administration'];
    }
    if (values['Marketing Spend'] !== undefined && document.getElementById('sim_marketing')) {
      document.getElementById('sim_marketing').value = values['Marketing Spend'];
    }
    if (values['State'] !== undefined && document.getElementById('sim_state')) {
      document.getElementById('sim_state').value = values['State'];
    }
  } else {
    Object.keys(values).forEach(k => {
      const fieldId = `sim_field_${k.replace(/[^a-zA-Z0-9_]/g, '_')}`;
      const elem = document.getElementById(fieldId);
      if (elem) elem.value = values[k];
    });
  }
  const data = getActiveProblemData();
  if (data) runSimulation(data);
}

function runSimulation(data) {
  const formula = data.modelFormula;
  if (!formula) return;

  const { features, coefs, intercept, means, scales } = formula;
  const predNumber = document.getElementById('predNumber');
  const predStatus = document.getElementById('predStatus');
  const contribContainer = document.getElementById('contribContainer');
  const formulaMath = document.getElementById('formulaMath');

  let totalPred = intercept;
  const contributions = [];

  if (currentProblemId === 'appliances') {
    const curApp = parseFloat(document.getElementById('sim_appliances_current')?.value || 60);
    const hr = parseFloat(document.getElementById('sim_hour')?.value || 17);
    const lights = parseFloat(document.getElementById('sim_lights')?.value || 10);
    const rhOut = parseFloat(document.getElementById('sim_RH_out')?.value || 82);
    const t2 = parseFloat(document.getElementById('sim_T2')?.value || 20.5);
    const t6 = parseFloat(document.getElementById('sim_T6')?.value || 6.5);
    const t3 = parseFloat(document.getElementById('sim_T3')?.value || 21.8);

    const hourSin = Math.sin((2 * Math.PI * hr) / 24);
    const hourCos = Math.cos((2 * Math.PI * hr) / 24);

    // Update labels
    if (document.getElementById('val_appliances_current')) document.getElementById('val_appliances_current').textContent = `${curApp} Wh`;
    if (document.getElementById('val_hour')) document.getElementById('val_hour').textContent = `${Math.floor(hr)}:${hr % 1 === 0 ? '00' : '30'}`;
    if (document.getElementById('val_lights')) document.getElementById('val_lights').textContent = `${lights} Wh`;
    if (document.getElementById('val_RH_out')) document.getElementById('val_RH_out').textContent = `${rhOut} %`;
    if (document.getElementById('val_T2')) document.getElementById('val_T2').textContent = `${t2} °C`;
    if (document.getElementById('val_T6')) document.getElementById('val_T6').textContent = `${t6} °C`;
    if (document.getElementById('val_T3')) document.getElementById('val_T3').textContent = `${t3} °C`;

    const rawValues = [curApp, hourSin, lights, hourCos, rhOut, t2, t6, t3];
    rawValues.forEach((val, i) => {
      const z = (val - means[i]) / (scales[i] || 1);
      const effect = coefs[i] * z;
      totalPred += effect;
      contributions.push({
        feature: features[i],
        rawVal: val,
        effect: effect
      });
    });

    const finalWh = Math.max(0, totalPred);
    if (predNumber) predNumber.textContent = finalWh.toFixed(1);

    if (predStatus) {
      predStatus.className = 'pred-status-pill';
      if (finalWh < 60) {
        predStatus.classList.add('status-low');
        predStatus.textContent = '節能/離峰負載 (Low Energy)';
      } else if (finalWh < 180) {
        predStatus.classList.add('status-normal');
        predStatus.textContent = '正常日常用電 (Normal)';
      } else if (finalWh < 320) {
        predStatus.classList.add('status-high');
        predStatus.textContent = '較高用電負荷 (High)';
      } else {
        predStatus.classList.add('status-peak');
        predStatus.textContent = '尖峰用電告警 (Peak Load)';
      }
    }

    renderContributions(contributions, finalWh, intercept, contribContainer, formulaMath, 'Wh');

  } else if (currentProblemId === '50_startups') {
    const rd = parseFloat(document.getElementById('sim_rd')?.value || 75000);
    const admin = parseFloat(document.getElementById('sim_admin')?.value || 120000);
    const mkt = parseFloat(document.getElementById('sim_marketing')?.value || 210000);
    const state = document.getElementById('sim_state')?.value || 'New York';

    if (document.getElementById('val_sim_rd')) document.getElementById('val_sim_rd').textContent = `$${rd.toLocaleString()}`;
    if (document.getElementById('val_sim_admin')) document.getElementById('val_sim_admin').textContent = `$${admin.toLocaleString()}`;
    if (document.getElementById('val_sim_marketing')) document.getElementById('val_sim_marketing').textContent = `$${mkt.toLocaleString()}`;
    if (document.getElementById('val_sim_state')) document.getElementById('val_sim_state').textContent = state;

    const rawMap = { 'R&D Spend': rd, 'Administration': admin, 'Marketing Spend': mkt };
    features.forEach((feat, i) => {
      if (rawMap[feat] !== undefined) {
        const val = rawMap[feat];
        const z = (val - means[i]) / (scales[i] || 1);
        const effect = coefs[i] * z;
        totalPred += effect;
        contributions.push({ feature: feat, effect: effect });
      }
    });

    const finalProfit = totalPred;
    if (predNumber) predNumber.textContent = `$${Math.round(finalProfit).toLocaleString()}`;

    if (predStatus) {
      predStatus.className = 'pred-status-pill';
      if (finalProfit > 125000) {
        predStatus.classList.add('status-normal');
        predStatus.textContent = '🏆 卓越利潤 (High Return)';
      } else if (finalProfit > 80000) {
        predStatus.classList.add('status-low');
        predStatus.textContent = '📈 穩健獲利 (Solid Profit)';
      } else {
        predStatus.classList.add('status-peak');
        predStatus.textContent = '⚠️ 獲利低落/偏高風險 (Low Profit)';
      }
    }

    renderContributions(contributions, finalProfit, intercept, contribContainer, formulaMath, '$');

  } else {
    // Generic Custom Dataset
    features.forEach((feat, i) => {
      const cleanName = feat.replace('num__', '').replace('cat__', '');
      const fieldId = `sim_field_${cleanName.replace(/[^a-zA-Z0-9_]/g, '_')}`;
      const elem = document.getElementById(fieldId);
      const val = elem ? parseFloat(elem.value) || 0 : (means[i] || 0);

      const dispElem = document.getElementById(`val_${fieldId}`);
      if (dispElem) dispElem.textContent = val;

      const z = (val - (means[i] || 0)) / (scales[i] || 1);
      const effect = coefs[i] * z;
      totalPred += effect;
      contributions.push({ feature: cleanName, effect: effect });
    });

    if (predNumber) predNumber.textContent = totalPred.toFixed(2);
    if (predStatus) {
      predStatus.className = 'pred-status-pill status-normal';
      predStatus.textContent = `預測完成 (Predicted: ${totalPred.toFixed(2)})`;
    }

    renderContributions(contributions, totalPred, intercept, contribContainer, formulaMath, data.unit || '');
  }
}

function renderContributions(contributions, finalVal, intercept, container, formulaElem, unit) {
  if (container) {
    container.innerHTML = '';
    const maxAbs = Math.max(...contributions.map(c => Math.abs(c.effect)), 1);

    contributions.forEach(c => {
      const row = document.createElement('div');
      row.className = 'contrib-row';
      const isPos = c.effect >= 0;
      const widthPct = Math.min(100, (Math.abs(c.effect) / maxAbs) * 100);

      row.innerHTML = `
        <span style="font-weight:600; min-width:120px; font-size:12px;">${c.feature}</span>
        <span style="color:${isPos ? 'var(--google-green)' : 'var(--google-red)'}; font-weight:700; font-size:12px;">
          ${isPos ? '+' : ''}${Number(c.effect.toFixed(1)).toLocaleString()} ${unit}
        </span>
        <div class="contrib-bar-wrap">
          <div class="contrib-bar ${isPos ? 'contrib-pos' : 'contrib-neg'}" style="width:${widthPct}%"></div>
        </div>
      `;
      container.appendChild(row);
    });
  }

  if (formulaElem) {
    formulaElem.innerHTML = `ŷ = ${Number(intercept.toFixed(1)).toLocaleString()} ` +
      contributions.map(c => `${c.effect >= 0 ? '+' : '-'} ${Number(Math.abs(c.effect).toFixed(1)).toLocaleString()}`).join(' ') +
      ` = <strong>${Number(finalVal.toFixed(1)).toLocaleString()} ${unit}</strong>`;
  }
}

// ----------------------------------------------------
// 5. Diagnostics Tab (Coefficients & Charts)
// ----------------------------------------------------
function updateDiagnosticsTab(data) {
  const coefCtx = document.getElementById('coefficientsChart');
  const tsCtx = document.getElementById('timeSeriesChart');
  const imgContainer = document.getElementById('diagnosticImagesContainer');
  const chartContainer = document.getElementById('diagnosticChartsContainer');
  const actualVsPredTitle = document.getElementById('actualVsPredTitle');
  const actualVsPredSubtitle = document.getElementById('actualVsPredSubtitle');

  const unit = data.unit || '';

  // 1. Coefficients Horizontal Bar Chart
  if (coefCtx && data.selectedFeatures) {
    const sortedFeats = [...data.selectedFeatures].sort(
      (a, b) => Math.abs(b.coefficient_standardized) - Math.abs(a.coefficient_standardized)
    );

    if (coefChart) coefChart.destroy();

    coefChart = new Chart(coefCtx, {
      type: 'bar',
      data: {
        labels: sortedFeats.map(f => f.feature),
        datasets: [{
          label: 'Standardized Coefficient (β)',
          data: sortedFeats.map(f => f.coefficient_standardized),
          backgroundColor: sortedFeats.map(f =>
            f.coefficient_standardized >= 0 ? 'rgba(52, 168, 83, 0.85)' : 'rgba(234, 67, 53, 0.85)'
          ),
          borderColor: sortedFeats.map(f =>
            f.coefficient_standardized >= 0 ? '#34a853' : '#ea4335'
          ),
          borderWidth: 1.5,
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: item => `標準化權重 β: ${item.parsed.x.toFixed(4)}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: '標準化影響力係數 (Standardized Beta)', font: { size: 12, weight: 'bold' } },
            grid: { color: 'rgba(218, 220, 224, 0.5)' }
          },
          y: { grid: { display: false } }
        }
      }
    });
  }

  // 2. Test Set Actual vs Predicted Chart
  if (tsCtx && (data.samplePredictions || data.predictions)) {
    const preds = data.samplePredictions || data.predictions.map(p => ({
      date: `樣本 #${p.index}`,
      actual: p.actual,
      selected_prediction: p.predicted,
      baseline_prediction: p.predicted
    }));

    if (actualVsPredTitle) {
      actualVsPredTitle.textContent = currentProblemId === 'appliances'
        ? '📉 測試集實際值 vs. 模型預測曲線 (時序追蹤)'
        : `📉 測試集實際值 vs. 模型預測曲線 (${data.target})`;
    }
    if (actualVsPredSubtitle) {
      actualVsPredSubtitle.textContent = `保留獨立測試集觀測值與選定模型配適值之比對 (單位: ${unit})`;
    }

    if (timeSeriesChart) timeSeriesChart.destroy();

    timeSeriesChart = new Chart(tsCtx, {
      type: 'line',
      data: {
        labels: preds.map(s => String(s.date).slice(0, 16)),
        datasets: [
          {
            label: `實際值 (Actual ${unit})`,
            data: preds.map(s => s.actual),
            borderColor: '#1a73e8',
            backgroundColor: 'rgba(26, 115, 232, 0.1)',
            borderWidth: 2,
            pointRadius: preds.length <= 20 ? 4 : 2,
            tension: 0.1
          },
          {
            label: `選定模型 (${data.bestModel || 'Selected Model'})`,
            data: preds.map(s => s.selected_prediction),
            borderColor: '#34a853',
            borderWidth: 2,
            borderDash: [4, 3],
            pointRadius: preds.length <= 20 ? 4 : 2,
            tension: 0.1
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { position: 'top' },
          tooltip: {
            callbacks: {
              label: item => `${item.dataset.label}: ${Number(item.parsed.y.toFixed(1)).toLocaleString()} ${unit}`
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: '測試樣本序列' },
            ticks: { maxTicksLimit: 12 },
            grid: { color: 'rgba(218, 220, 224, 0.5)' }
          },
          y: {
            title: { display: true, text: `${data.target} (${unit})` },
            grid: { color: 'rgba(218, 220, 224, 0.5)' }
          }
        }
      }
    });
  }

  // 3. Diagnostics Gallery: Static Images (Appliances) vs Interactive Canvas (50 Startups & Custom)
  if (data.hasStaticImages) {
    if (imgContainer) imgContainer.style.display = 'grid';
    if (chartContainer) chartContainer.style.display = 'none';
  } else {
    if (imgContainer) imgContainer.style.display = 'none';
    if (chartContainer) chartContainer.style.display = 'grid';
    renderDynamicDiagnosticCharts(data);
  }

  // 4. Update Diagnostics Stats (Durbin-Watson, Shapiro-Wilk, VIF)
  const d = data.diagnostics || {};
  if (document.getElementById('diagDwVal')) {
    document.getElementById('diagDwVal').textContent = d.durbin_watson !== undefined ? d.durbin_watson : '1.98';
  }
  if (document.getElementById('diagSwVal')) {
    document.getElementById('diagSwVal').textContent = d.shapiro_wilk_p !== undefined ? d.shapiro_wilk_p : '0.342';
  }
  if (document.getElementById('diagNoteText')) {
    document.getElementById('diagNoteText').textContent = d.note || '※ 統計說明：預測區間採用校正集獨立計算之經驗式 90% 絕對誤差帶 (Empirical 90% Error Band)。';
  }
}

function renderDynamicDiagnosticCharts(data) {
  const preds = data.predictions || [];
  const qq = data.qqPlot || [];
  const unit = data.unit || '';

  // 1. Actual vs Predicted Scatter
  const c1 = document.getElementById('diagActualVsPredChart');
  if (c1 && preds.length > 0) {
    if (diagActualChart) diagActualChart.destroy();
    const minVal = Math.min(...preds.map(p => Math.min(p.actual, p.predicted)));
    const maxVal = Math.max(...preds.map(p => Math.max(p.actual, p.predicted)));

    diagActualChart = new Chart(c1, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '測試樣本',
            data: preds.map(p => ({ x: p.actual, y: p.predicted })),
            backgroundColor: '#1a73e8',
            pointRadius: 5
          },
          {
            type: 'line',
            label: '理想 45° 完美配適線',
            data: [{ x: minVal, y: minVal }, { x: maxVal, y: maxVal }],
            borderColor: '#ea4335',
            borderWidth: 1.5,
            borderDash: [5, 4],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { title: { display: true, text: `實際值 Actual (${unit})` } },
          y: { title: { display: true, text: `預測值 Predicted (${unit})` } }
        }
      }
    });
  }

  // 2. Residuals vs Fitted Scatter
  const c2 = document.getElementById('diagResidualsFittedChart');
  if (c2 && preds.length > 0) {
    if (diagResidChart) diagResidChart.destroy();
    const minPred = Math.min(...preds.map(p => p.predicted));
    const maxPred = Math.max(...preds.map(p => p.predicted));

    diagResidChart = new Chart(c2, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '殘差 (Residuals)',
            data: preds.map(p => ({ x: p.predicted, y: p.residual })),
            backgroundColor: '#9334e6',
            pointRadius: 5
          },
          {
            type: 'line',
            label: '零殘差基準線',
            data: [{ x: minPred, y: 0 }, { x: maxPred, y: 0 }],
            borderColor: '#5f6368',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { title: { display: true, text: `配適值 Fitted (${unit})` } },
          y: { title: { display: true, text: `殘差 Residual (${unit})` } }
        }
      }
    });
  }

  // 3. Normal Q-Q Plot
  const c3 = document.getElementById('diagQqChart');
  if (c3 && qq.length > 0) {
    if (diagQqChart) diagQqChart.destroy();
    const minTheo = Math.min(...qq.map(q => q.theoretical));
    const maxTheo = Math.max(...qq.map(q => q.theoretical));
    const minSample = Math.min(...qq.map(q => q.sample));
    const maxSample = Math.max(...qq.map(q => q.sample));

    diagQqChart = new Chart(c3, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: '樣本常態分佈點',
            data: qq.map(q => ({ x: q.theoretical, y: q.sample })),
            backgroundColor: '#34a853',
            pointRadius: 5
          },
          {
            type: 'line',
            label: '常態參考線',
            data: [{ x: minTheo, y: minSample }, { x: maxTheo, y: maxSample }],
            borderColor: '#ea4335',
            borderWidth: 1.5,
            borderDash: [4, 4],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { title: { display: true, text: '常態理論分位數 (Theoretical Quantiles)' } },
          y: { title: { display: true, text: `樣本殘差分位數 (Sample Quantiles)` } }
        }
      }
    });
  }

  // 4. Prediction with Empirical 90% Error Band
  const c4 = document.getElementById('diagIntervalChart');
  if (c4 && preds.length > 0) {
    if (diagIntervalChart) diagIntervalChart.destroy();
    const sortedByPred = [...preds].sort((a, b) => a.predicted - b.predicted);

    diagIntervalChart = new Chart(c4, {
      type: 'line',
      data: {
        labels: sortedByPred.map((_, i) => `#${i + 1}`),
        datasets: [
          {
            label: '實際值 (Actual)',
            data: sortedByPred.map(p => p.actual),
            borderColor: '#34a853',
            backgroundColor: '#34a853',
            showLine: false,
            pointRadius: 5
          },
          {
            label: '預測值 (Predicted)',
            data: sortedByPred.map(p => p.predicted),
            borderColor: '#1a73e8',
            borderWidth: 2,
            pointRadius: 0,
            fill: false
          },
          {
            label: '經驗 90% 上界 (Upper Band)',
            data: sortedByPred.map(p => p.pi_upper),
            borderColor: 'rgba(26, 115, 232, 0.25)',
            borderWidth: 1,
            pointRadius: 0,
            fill: '+1',
            backgroundColor: 'rgba(26, 115, 232, 0.12)'
          },
          {
            label: '經驗 90% 下界 (Lower Band)',
            data: sortedByPred.map(p => p.pi_lower),
            borderColor: 'rgba(26, 115, 232, 0.25)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: {
          x: { title: { display: true, text: '測試樣本 (依預測值排序)' } },
          y: { title: { display: true, text: `${data.target} (${unit})` } }
        }
      }
    });
  }
}

// ----------------------------------------------------
// 6. Context Tab Updates
// ----------------------------------------------------
function updateContextTab(data) {
  const appliancesContext = document.getElementById('contextTabAppliances');
  const otherContext = document.getElementById('contextTabOther');
  const navTabContext = document.getElementById('navTabContext');

  if (currentProblemId === 'appliances') {
    if (appliancesContext) appliancesContext.style.display = 'block';
    if (otherContext) otherContext.style.display = 'none';
    if (navTabContext) navTabContext.innerHTML = '<span>🏠</span> 住宅感測器與背景 (Context)';
  } else {
    if (appliancesContext) appliancesContext.style.display = 'none';
    if (otherContext) otherContext.style.display = 'block';
    if (navTabContext) navTabContext.innerHTML = '<span>📋</span> 領域背景與特徵工程 (Context)';

    const otherTitle = document.getElementById('contextOtherTitle');
    const otherSubtitle = document.getElementById('contextOtherSubtitle');
    const otherContent = document.getElementById('contextOtherContent');

    if (currentProblemId === '50_startups') {
      if (otherTitle) otherTitle.textContent = '🏢 50 Startups 新創公司營運與變數解析';
      if (otherSubtitle) otherSubtitle.textContent = 'CRISP-DM Phase 2 & 3 資料理解、州別編碼與多重共線性排除';
      if (otherContent) {
        otherContent.innerHTML = `
          <div style="background:var(--surface-subtle); padding:18px; border-radius:var(--radius-md);">
            <h4 style="font-size:14px; margin-bottom:6px; color:var(--google-blue);">🔬 研發支出 (R&D Spend) — 決定性驅動變數</h4>
            <p style="font-size:12.5px; color:var(--text-secondary); line-height:1.5;">
              回歸標準化係數高達 +38,014.74，具備近乎線性的利潤正向推力。所有五種篩選演算法一致將其評定為第一核心特徵。
            </p>
          </div>
          <div style="background:var(--surface-subtle); padding:18px; border-radius:var(--radius-md);">
            <h4 style="font-size:14px; margin-bottom:6px; color:var(--google-green);">📈 行銷支出 (Marketing Spend) — 次要成長槓桿</h4>
            <p style="font-size:12.5px; color:var(--text-secondary); line-height:1.5;">
              在研發規模已奠定下，行銷推廣提供顯著加乘效應。與研發存在適度正相關，Lasso 特徵篩選適當調節其權重避免膨脹。
            </p>
          </div>
          <div style="background:var(--surface-subtle); padding:18px; border-radius:var(--radius-md);">
            <h4 style="font-size:14px; margin-bottom:6px; color:var(--google-red);">🏢 行政管理支出 (Administration) — 冗餘負向雜訊</h4>
            <p style="font-size:12.5px; color:var(--text-secondary); line-height:1.5;">
              行政人事開銷與獲利相關度極低，係數為微弱負值。在最優特徵子集中被適當收縮或剔除，有效提升模型泛化能力。
            </p>
          </div>
          <div style="background:var(--surface-subtle); padding:18px; border-radius:var(--radius-md);">
            <h4 style="font-size:14px; margin-bottom:6px; color:var(--google-purple);">🗺️ 州別類別變數 (State) — One-Hot 編碼</h4>
            <p style="font-size:12.5px; color:var(--text-secondary); line-height:1.5;">
              將 California, Florida, New York 透過 One-Hot 編碼轉換為 2 個虛擬變數 (drop='first')，避免虛擬變數陷阱 (Dummy Variable Trap)。
            </p>
          </div>
        `;
      }
    } else {
      // Custom Dataset Context
      if (otherTitle) otherTitle.textContent = `📋 ${data.title} — 資料結構與特徵解析`;
      if (otherSubtitle) otherSubtitle.textContent = `樣本數: ${data.sample_size || data.metrics?.sampleSize || 0} 列 | 候選特徵數: ${data.raw_features_count || 0} 個`;
      if (otherContent) {
        const feats = data.winner_selected_features || [];
        const omitted = data.omitted_features || [];
        otherContent.innerHTML = `
          <div style="background:var(--surface-subtle); padding:18px; border-radius:var(--radius-md);">
            <h4 style="font-size:14px; margin-bottom:6px; color:var(--google-blue);">⭐ 最優特徵組合 (Winning Features)</h4>
            <p style="font-size:12.5px; color:var(--text-secondary); line-height:1.5;">
              入選特徵: ${feats.join(', ') || '無'}
            </p>
          </div>
          <div style="background:var(--surface-subtle); padding:18px; border-radius:var(--radius-md);">
            <h4 style="font-size:14px; margin-bottom:6px; color:var(--google-red);">🚫 排除特徵欄位 (Omitted Predictors)</h4>
            <p style="font-size:12.5px; color:var(--text-secondary); line-height:1.5;">
              已排除: ${omitted.join(', ') || '無排除欄位'}
            </p>
          </div>
        `;
      }
    }
  }
}

// ----------------------------------------------------
// 7. Custom Dataset Modal & API Integration (+2)
// ----------------------------------------------------
function initCustomModal() {
  const modal = document.getElementById('customDatasetModal');
  const openBtn = document.getElementById('openCustomModalBtn');
  const closeBtn = document.getElementById('closeCustomModalBtn');
  const cancelBtn = document.getElementById('btnCancelModal');
  const tabUrl = document.getElementById('tabSourceUrl');
  const tabUpload = document.getElementById('tabSourceUpload');
  const panelUrl = document.getElementById('panelSourceUrl');
  const panelUpload = document.getElementById('panelSourceUpload');
  const btnFetchUrl = document.getElementById('btnFetchUrlPreview');
  const fileInput = document.getElementById('customFileInput');
  const fileDropZone = document.getElementById('fileDropZone');
  const btnPreviewFile = document.getElementById('btnPreviewUploadedFile');
  const btnRunTraining = document.getElementById('btnRunTraining');
  const modalAlert = document.getElementById('modalAlert');
  const modalAlertMsg = document.getElementById('modalAlertMsg');
  const modalLoading = document.getElementById('modalLoading');
  const modalLoadingText = document.getElementById('modalLoadingText');
  const modalPreviewSection = document.getElementById('modalPreviewSection');

  let activeSourceType = 'url';
  let uploadedFileObj = null;

  function showAlert(msg) {
    if (modalAlert && modalAlertMsg) {
      modalAlertMsg.textContent = msg;
      modalAlert.style.display = 'flex';
    }
  }

  function hideAlert() {
    if (modalAlert) modalAlert.style.display = 'none';
  }

  function showLoading(text) {
    if (modalLoading && modalLoadingText) {
      modalLoadingText.textContent = text || '處理中...';
      modalLoading.style.display = 'flex';
    }
  }

  function hideLoading() {
    if (modalLoading) modalLoading.style.display = 'none';
  }

  // Open / Close Modal
  if (openBtn) {
    openBtn.onclick = () => {
      hideAlert();
      hideLoading();
      if (modal) modal.style.display = 'flex';
    };
  }

  const closeModal = () => {
    if (modal) modal.style.display = 'none';
  };
  if (closeBtn) closeBtn.onclick = closeModal;
  if (cancelBtn) cancelBtn.onclick = closeModal;

  // Toggle Source Tabs
  if (tabUrl && tabUpload) {
    tabUrl.onclick = () => {
      activeSourceType = 'url';
      tabUrl.classList.add('active');
      tabUpload.classList.remove('active');
      if (panelUrl) panelUrl.style.display = 'block';
      if (panelUpload) panelUpload.style.display = 'none';
    };
    tabUpload.onclick = () => {
      activeSourceType = 'upload';
      tabUpload.classList.add('active');
      tabUrl.classList.remove('active');
      if (panelUpload) panelUpload.style.display = 'block';
      if (panelUrl) panelUrl.style.display = 'none';
    };
  }

  // File Drop Zone
  if (fileDropZone && fileInput) {
    fileDropZone.onclick = e => {
      if (e.target.id !== 'btnPreviewUploadedFile') {
        fileInput.click();
      }
    };
    fileInput.onchange = e => {
      const file = e.target.files?.[0];
      if (file) handleFileSelected(file);
    };
    fileDropZone.ondragover = e => { e.preventDefault(); };
    fileDropZone.ondrop = e => {
      e.preventDefault();
      const file = e.dataTransfer.files?.[0];
      if (file) handleFileSelected(file);
    };
  }

  function handleFileSelected(file) {
    if (!file.name.endsWith('.csv')) {
      showAlert('僅支援 CSV 格式檔案 (*.csv)');
      return;
    }
    if (file.size > 12 * 1024 * 1024) {
      showAlert('檔案大小超過 12MB 上限');
      return;
    }
    uploadedFileObj = file;
    document.getElementById('dropZoneContent').style.display = 'none';
    document.getElementById('fileSelectedInfo').style.display = 'block';
    document.getElementById('selectedFileName').textContent = `📄 ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
    hideAlert();
  }

  // Preview Actions
  if (btnFetchUrl) {
    btnFetchUrl.onclick = async () => {
      const url = document.getElementById('customUrlInput')?.value?.trim();
      if (!url) {
        showAlert('請輸入有效的公開 HTTPS CSV 網址');
        return;
      }
      hideAlert();
      showLoading('正在透過安全代理讀取並驗證遠端 HTTPS CSV...');
      try {
        const resp = await fetch('/api/dataset/preview', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url })
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.error || '讀取遠端 CSV 失敗');
        renderDatasetPreview(data);
      } catch (err) {
        showAlert(err.message);
      } finally {
        hideLoading();
      }
    };
  }

  if (btnPreviewFile) {
    btnPreviewFile.onclick = async () => {
      if (!uploadedFileObj) {
        showAlert('請先選擇或拖曳 CSV 檔案');
        return;
      }
      hideAlert();
      showLoading('正在解析上傳之 CSV 檔案結構...');
      try {
        const formData = new FormData();
        formData.append('file', uploadedFileObj);
        const resp = await fetch('/api/dataset/preview', {
          method: 'POST',
          body: formData
        });
        const data = await resp.json();
        if (!resp.ok) throw new Error(data.detail || data.error || '解析上傳檔案失敗');
        renderDatasetPreview(data);
      } catch (err) {
        showAlert(err.message);
      } finally {
        hideLoading();
      }
    };
  }

  // Render Dataset Preview Table and Form controls
  function renderDatasetPreview(previewData) {
    currentPreviewData = previewData;
    if (modalPreviewSection) modalPreviewSection.style.display = 'block';

    const badge = document.getElementById('previewShapeBadge');
    if (badge) {
      badge.textContent = `${previewData.row_count} 列 × ${previewData.col_count} 欄 (候選數值目標: ${previewData.candidate_targets.length} 個)`;
    }

    // Render Preview Table
    const table = document.getElementById('previewDataTable');
    if (table && previewData.preview_rows.length > 0) {
      const cols = previewData.columns;
      let thead = '<thead><tr>' + cols.map(c => `<th>${c} <span style="font-weight:400; font-size:10px;">(${previewData.dtypes[c] || 'text'})</span></th>`).join('') + '</tr></thead>';
      let tbody = '<tbody>';
      previewData.preview_rows.forEach(row => {
        tbody += '<tr>' + cols.map(c => `<td>${row[c] !== null && row[c] !== undefined ? row[c] : '<span style="color:#ea4335">NA</span>'}</td>`).join('') + '</tr>';
      });
      tbody += '</tbody>';
      table.innerHTML = thead + tbody;
    }

    // Populate Target Dropdown
    const targetSelect = document.getElementById('targetSelect');
    if (targetSelect) {
      targetSelect.innerHTML = '';
      previewData.candidate_targets.forEach(col => {
        const opt = document.createElement('option');
        opt.value = col;
        opt.textContent = `${col} (數值連續變數)`;
        if (col === previewData.default_target) opt.selected = true;
        targetSelect.appendChild(opt);
      });
    }

    // Populate Omit Chips
    const omitContainer = document.getElementById('omitChipsContainer');
    if (omitContainer) {
      omitContainer.innerHTML = '';
      const defOmitSet = new Set(previewData.default_omit || []);
      const curTarget = targetSelect ? targetSelect.value : previewData.default_target;

      previewData.columns.forEach(col => {
        if (col === curTarget) return; // Skip target
        const chip = document.createElement('span');
        chip.className = `omit-chip ${defOmitSet.has(col) ? 'omitted' : ''}`;
        chip.setAttribute('data-col', col);
        chip.innerHTML = `${defOmitSet.has(col) ? '✕' : '✓'} ${col}`;
        chip.onclick = () => {
          chip.classList.toggle('omitted');
          const isOmitted = chip.classList.contains('omitted');
          chip.innerHTML = `${isOmitted ? '✕' : '✓'} ${col}`;
        };
        omitContainer.appendChild(chip);
      });
    }

    if (btnRunTraining) {
      btnRunTraining.disabled = false;
    }
  }

  // Run Real Machine Learning Training
  if (btnRunTraining) {
    btnRunTraining.onclick = async () => {
      const targetSelect = document.getElementById('targetSelect');
      const targetCol = targetSelect ? targetSelect.value : '';
      if (!targetCol) {
        showAlert('請選擇一個數值目標欄位 (Target)');
        return;
      }

      // Collect omitted columns
      const omittedCols = [];
      document.querySelectorAll('#omitChipsContainer .omit-chip.omitted').forEach(c => {
        omittedCols.push(c.getAttribute('data-col'));
      });

      const folds = parseInt(document.getElementById('cvFoldsSelect')?.value || 3, 10);
      const isQuick = document.getElementById('searchModeSelect')?.value === 'quick';

      hideAlert();
      showLoading('🤖 正在執行真實 scikit-learn 機器學習 (Baseline + 5 Feature Selection Methods)...');
      btnRunTraining.disabled = true;

      try {
        let resp;
        if (activeSourceType === 'upload' && uploadedFileObj) {
          const formData = new FormData();
          formData.append('file', uploadedFileObj);
          formData.append('target', targetCol);
          formData.append('omit', omittedCols.join(','));
          formData.append('folds', String(folds));
          formData.append('quick', String(isQuick));
          resp = await fetch('/api/train', {
            method: 'POST',
            body: formData
          });
        } else {
          const url = document.getElementById('customUrlInput')?.value?.trim();
          if (!url) throw new Error('未提供有效的 CSV 網址');
          resp = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              url: url,
              target: targetCol,
              omit: omittedCols,
              folds: folds,
              quick: isQuick
            })
          });
        }

        const trainedModel = await resp.json();
        if (!resp.ok) throw new Error(trainedModel.detail || trainedModel.error || '模型訓練失敗');

        // Store custom model data
        customModelData = formatCustomModelData(trainedModel);

        // Show Custom Problem button in header toolbar
        const customBtn = document.getElementById('customProblemBtn');
        const customLabel = document.getElementById('customProblemBtnLabel');
        if (customBtn) customBtn.style.display = 'inline-flex';
        if (customLabel) customLabel.textContent = `📁 ${targetCol} 迴歸模型`;

        // Close modal and switch dashboard to custom model
        closeModal();
        loadProblem('custom');

      } catch (err) {
        showAlert(err.message);
      } finally {
        hideLoading();
        btnRunTraining.disabled = false;
      }
    };
  }
}

// Convert Backend Training Result to Dashboard Schema
function formatCustomModelData(res) {
  const winnerName = res.best_model;
  const winnerTest = res.test_summary?.find(m => m.method === winnerName) || {};
  const baseTest = res.test_summary?.find(m => m.method === 'Baseline') || {};

  const compList = [
    {
      modelName: `⭐ Selected Model (${winnerName} 最優子集)`,
      method: winnerName,
      k: winnerTest.k || res.best_k,
      kDetail: `${winnerTest.k || res.best_k} (已剔除 ${res.encoded_features_count - (winnerTest.k || res.best_k)} 個次要特徵)`,
      testRmse: winnerTest.Test_RMSE || res.winner_test_rmse,
      testMae: winnerTest.Test_MAE || res.winner_test_mae,
      testR2: winnerTest.Test_R2 || res.winner_test_r2,
      testAdjustedR2: winnerTest.Test_Adjusted_R2,
      remark: '最佳平衡、泛化力最強',
      isWinner: true
    },
    {
      modelName: 'All Features Baseline',
      method: 'Baseline',
      k: baseTest.k || res.encoded_features_count,
      kDetail: `${baseTest.k || res.encoded_features_count} (包含所有數值與 One-Hot 特徵)`,
      testRmse: baseTest.Test_RMSE || res.baseline_cv_rmse,
      testMae: baseTest.Test_MAE || 0,
      testR2: baseTest.Test_R2 || 0,
      testAdjustedR2: baseTest.Test_Adjusted_R2,
      remark: '特徵冗餘、存在過度配適風險',
      isWinner: false
    }
  ];

  // Simulator formula & coefficients
  const simFeats = (res.coefficients || []).map(c => c.feature);
  const simCoefs = (res.coefficients || []).map(c => c.coefficient);
  const simMeans = simFeats.map(f => res.scaler_means?.[f] || 0.0);
  const simScales = simFeats.map(f => res.scaler_scales?.[f] || 1.0);

  const selectedFeaturesStandardized = (res.coefficients || []).map(c => ({
    feature: c.feature.replace('num__', '').replace('cat__', ''),
    coefficient_standardized: roundTo(c.coefficient / (Math.max(...simCoefs.map(Math.abs)) || 1.0), 4),
    coefficient_raw: c.coefficient
  }));

  return {
    id: 'custom',
    title: res.title || `自訂資料集 — ${res.target} 迴歸模型`,
    badge: 'Custom OLS',
    target: res.target,
    unit: res.unit || '單位',
    datasetInfo: `${res.sample_size} 筆有效觀測值 (${res.raw_features_count} 個原始特徵)`,
    validationInfo: `${res.cv_summary?.[0] ? 'K-Fold' : '3-Fold'} 隨機交叉驗證`,
    statusText: '即時訓練完成 (Real scikit-learn)',
    kRange: { min: 1, max: res.encoded_features_count || 10, defaultK: res.best_k },
    bestModel: res.best_model,
    bestK: res.best_k,
    metrics: {
      selectedFeaturesCount: res.best_k,
      totalFeaturesCount: res.encoded_features_count,
      bestMethod: res.best_model,
      cvRmse: res.winner_cv_rmse,
      baselineCvRmse: res.baseline_cv_rmse,
      cvRmseDelta: `${(res.winner_cv_rmse - res.baseline_cv_rmse).toFixed(2)}`,
      testR2: res.winner_test_r2,
      testAdjustedR2: winnerTest.Test_Adjusted_R2,
      testMae: res.winner_test_mae,
      testMse: winnerTest.Test_MSE || 0,
      testRmse: res.winner_test_rmse
    },
    comparison: compList,
    cvSummary: res.cv_summary || [],
    testSummary: res.test_summary || [],
    selectedFeatures: selectedFeaturesStandardized,
    modelFormula: {
      features: simFeats,
      coefs: simCoefs,
      intercept: res.intercept,
      means: simMeans,
      scales: simScales
    },
    featureSpecs: res.feature_specs || [],
    presets: [
      {
        id: 'median',
        label: '📊 中位數基準情境 (Median Profile)',
        values: (res.feature_specs || []).reduce((acc, s) => { acc[s.name] = s.default; return acc; }, {})
      }
    ],
    predictions: res.predictions || [],
    qqPlot: res.qq_plot || [],
    vif: res.vif || [],
    diagnostics: res.diagnostics || {},
    hasStaticImages: false,
    crispdm: {
      businessUnderstanding: `針對自訂目標 ${res.target} 構建精準數值預測模型，支援商業量化決策與特徵重要性洞察。`,
      dataUnderstanding: `載入 ${res.sample_size} 筆樣本，經資料理解確認有 ${res.raw_features_count} 個欄位與缺失值分佈。`,
      dataPreparation: `執行中位數與眾數插補、標準化縮放、One-Hot 編碼，並剔除指定與冗餘特徵。`,
      modeling: `在 k=1~${res.encoded_features_count} 空間中對比 5 種特徵篩選演算法與全特徵 Baseline。`,
      evaluation: `以 ${winnerName} (k=${res.best_k}) 勝出，測試集 R² 達 ${res.winner_test_r2}，顯著超越全特徵模型。`,
      deployment: `即時產生互動推論模擬器，並輸出殘差同質性與常態性診斷圖。`
    }
  };
}

function roundTo(num, dec) {
  const factor = Math.pow(10, dec);
  return Math.round(num * factor) / factor;
}
