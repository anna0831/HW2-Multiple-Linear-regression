// Interactive Logic for Multiple Linear Regression Dashboard
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initFeatureExplorer();
  initSimulator();
  initCoefficientsChart();
  initTimeSeriesChart();
});

// Tab Navigation
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
// 1. Feature Selection Explorer (Chart.js)
// ----------------------------------------------------
let fsChart = null;
const methodColors = {
  'F-test': '#1a73e8',
  'Random Forest': '#34a853',
  'Lasso': '#f9ab00',
  'RFE': '#ea4335',
  'Mutual Information': '#9334e6'
};

function initFeatureExplorer() {
  const ctx = document.getElementById('featureSelectionChart');
  if (!ctx || !window.APP_DATA) return;

  const cvSummary = window.APP_DATA.cvSummary;
  const methods = ['F-test', 'Random Forest', 'Lasso', 'RFE', 'Mutual Information'];
  const baselineRow = cvSummary.find(r => r.method === 'All Features Baseline');
  const baselineRMSE = baselineRow ? baselineRow.CV_RMSE : 67.84;

  const ks = Array.from({ length: 30 }, (_, i) => i + 1);

  const datasets = methods.map(m => {
    const rows = cvSummary.filter(r => r.method === m).sort((a, b) => a.k - b.k);
    return {
      label: m,
      data: rows.map(r => ({ x: r.k, y: r.CV_RMSE })),
      borderColor: methodColors[m],
      backgroundColor: methodColors[m],
      borderWidth: 2,
      pointRadius: 3.5,
      pointHoverRadius: 6,
      tension: 0.15,
      hidden: false
    };
  });

  // Add Baseline Horizontal Line
  datasets.push({
    label: 'All Features Baseline (k=30, RMSE=67.84)',
    data: ks.map(k => ({ x: k, y: baselineRMSE })),
    borderColor: '#3c4043',
    borderWidth: 2,
    borderDash: [6, 4],
    pointRadius: 0,
    fill: false
  });

  fsChart = new Chart(ctx, {
    type: 'line',
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        title: {
          display: true,
          text: 'Appliances Energy Prediction: Number of Features vs. Cross-Validation RMSE',
          font: { size: 15, weight: 'bold', family: 'Google Sans, Inter, sans-serif' },
          color: '#202124',
          padding: { bottom: 15 }
        },
        legend: {
          position: 'bottom',
          labels: {
            boxWidth: 14,
            font: { size: 12, family: 'Google Sans, Inter, sans-serif' },
            padding: 16
          }
        },
        tooltip: {
          backgroundColor: 'rgba(32, 33, 36, 0.95)',
          titleFont: { size: 13, weight: 'bold' },
          bodyFont: { size: 12 },
          padding: 12,
          cornerRadius: 8,
          callbacks: {
            title: items => `Number of Features: k = ${items[0].parsed.x}`,
            label: item => `${item.dataset.label}: ${item.parsed.y.toFixed(3)} Wh`
          }
        }
      },
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: 'Number of Selected Features (k)',
            font: { size: 12, weight: 'bold' },
            color: '#5f6368'
          },
          ticks: { stepSize: 2 },
          min: 1,
          max: 30,
          grid: { color: 'rgba(218, 220, 224, 0.6)' }
        },
        y: {
          title: {
            display: true,
            text: 'Mean Time-Series CV RMSE (Wh)',
            font: { size: 12, weight: 'bold' },
            color: '#5f6368'
          },
          min: 66.5,
          max: 69.5,
          grid: { color: 'rgba(218, 220, 224, 0.6)' }
        }
      }
    }
  });

  // Method filter toggles
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetMethod = btn.getAttribute('data-method');
      if (targetMethod === 'all') {
        fsChart.data.datasets.forEach(d => (d.hidden = false));
        filterBtns.forEach(b => b.classList.add('active'));
      } else {
        const dataset = fsChart.data.datasets.find(d => d.label === targetMethod);
        if (dataset) {
          dataset.hidden = !dataset.hidden;
          btn.classList.toggle('active', !dataset.hidden);
        }
      }
      fsChart.update();
    });
  });

  // Slider for k
  const kSlider = document.getElementById('kSlider');
  const kValueBadge = document.getElementById('kValueBadge');
  const activeFeaturesContainer = document.getElementById('activeFeaturesContainer');

  const orderedFeaturesByFTest = [
    'appliances_current', 'hour_sin', 'lights', 'hour_cos', 'RH_out', 'T2', 'T6', 'T3',
    'T8', 'RH_1', 'RH_2', 'RH_3', 'T1', 'T4', 'T7', 'T9', 'RH_8', 'RH_7', 'T_out',
    'RH_9', 'RH_4', 'Windspeed', 'RH_5', 'T5', 'RH_6', 'Press_mm_hg', 'Visibility',
    'weekday_sin', 'Tdewpoint', 'weekday_cos'
  ];

  if (kSlider && kValueBadge) {
    kSlider.addEventListener('input', e => {
      const k = parseInt(e.target.value, 10);
      kValueBadge.textContent = `k = ${k}`;
      updateActiveFeaturesList(k);
    });
  }

  function updateActiveFeaturesList(k) {
    if (!activeFeaturesContainer) return;
    activeFeaturesContainer.innerHTML = '';
    const active = orderedFeaturesByFTest.slice(0, k);

    active.forEach((feat, idx) => {
      const chip = document.createElement('span');
      chip.className = `meta-chip ${idx < 8 ? 'chip-success' : ''}`;
      chip.textContent = `#${idx + 1} ${feat}`;
      activeFeaturesContainer.appendChild(chip);
    });
  }

  updateActiveFeaturesList(8);
}

// ----------------------------------------------------
// 2. Interactive Multiple Linear Regression Simulator
// ----------------------------------------------------
function initSimulator() {
  if (!window.APP_DATA || !window.APP_DATA.modelFormula) return;

  const formula = window.APP_DATA.modelFormula;
  const { features, coefs, intercept, means, scales } = formula;

  const inputs = {
    appliances_current: document.getElementById('sim_appliances_current'),
    hour: document.getElementById('sim_hour'),
    lights: document.getElementById('sim_lights'),
    RH_out: document.getElementById('sim_RH_out'),
    T2: document.getElementById('sim_T2'),
    T6: document.getElementById('sim_T6'),
    T3: document.getElementById('sim_T3')
  };

  const displays = {
    appliances_current: document.getElementById('val_appliances_current'),
    hour: document.getElementById('val_hour'),
    lights: document.getElementById('val_lights'),
    RH_out: document.getElementById('val_RH_out'),
    T2: document.getElementById('val_T2'),
    T6: document.getElementById('val_T6'),
    T3: document.getElementById('val_T3')
  };

  const predNumber = document.getElementById('predNumber');
  const predStatus = document.getElementById('predStatus');
  const formulaMath = document.getElementById('formulaMath');
  const contribContainer = document.getElementById('contribContainer');

  function calculatePrediction() {
    const curApp = parseFloat(inputs.appliances_current.value);
    const hr = parseFloat(inputs.hour.value);
    const lights = parseFloat(inputs.lights.value);
    const rhOut = parseFloat(inputs.RH_out.value);
    const t2 = parseFloat(inputs.T2.value);
    const t6 = parseFloat(inputs.T6.value);
    const t3 = parseFloat(inputs.T3.value);

    // Compute cyclical hour components
    const hourSin = Math.sin((2 * Math.PI * hr) / 24);
    const hourCos = Math.cos((2 * Math.PI * hr) / 24);

    // Update display labels
    displays.appliances_current.textContent = `${curApp} Wh`;
    displays.hour.textContent = `${Math.floor(hr)}:${hr % 1 === 0 ? '00' : '30'} (sin: ${hourSin.toFixed(2)}, cos: ${hourCos.toFixed(2)})`;
    displays.lights.textContent = `${lights} Wh`;
    displays.RH_out.textContent = `${rhOut} %`;
    displays.T2.textContent = `${t2} °C`;
    displays.T6.textContent = `${t6} °C`;
    displays.T3.textContent = `${t3} °C`;

    // Map input values to the 8 selected feature order:
    // ['appliances_current', 'hour_sin', 'lights', 'hour_cos', 'RH_out', 'T2', 'T6', 'T3']
    const rawValues = [curApp, hourSin, lights, hourCos, rhOut, t2, t6, t3];

    let totalPred = intercept;
    const contributions = [];

    rawValues.forEach((val, i) => {
      const z = (val - means[i]) / scales[i];
      const effect = coefs[i] * z;
      totalPred += effect;
      contributions.push({
        feature: features[i],
        rawVal: val,
        zVal: z,
        coef: coefs[i],
        effect: effect
      });
    });

    // Ensure physical floor of 0 Wh
    const finalWh = Math.max(0, totalPred);
    if (predNumber) predNumber.textContent = finalWh.toFixed(1);

    // Update status badge
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

    // Render contributions breakdown
    if (contribContainer) {
      contribContainer.innerHTML = '';
      const maxAbs = Math.max(...contributions.map(c => Math.abs(c.effect)), 1);

      contributions.forEach(c => {
        const row = document.createElement('div');
        row.className = 'contrib-row';
        const isPos = c.effect >= 0;
        const widthPct = Math.min(100, (Math.abs(c.effect) / maxAbs) * 100);

        row.innerHTML = `
          <span style="font-weight:600; min-width:110px;">${c.feature}</span>
          <span style="color:${isPos ? 'var(--google-green)' : 'var(--google-red)'}; font-weight:700;">
            ${isPos ? '+' : ''}${c.effect.toFixed(2)} Wh
          </span>
          <div class="contrib-bar-wrap">
            <div class="contrib-bar ${isPos ? 'contrib-pos' : 'contrib-neg'}" style="width:${widthPct}%"></div>
          </div>
        `;
        contribContainer.appendChild(row);
      });
    }

    // Mathematical formula view
    if (formulaMath) {
      formulaMath.innerHTML = `ŷ = ${intercept.toFixed(2)} ` +
        contributions.map(c => `${c.effect >= 0 ? '+' : '-'} ${Math.abs(c.effect).toFixed(2)}`).join(' ') +
        ` = <strong>${finalWh.toFixed(2)} Wh</strong>`;
    }
  }

  // Bind input events
  Object.values(inputs).forEach(input => {
    if (input) input.addEventListener('input', calculatePrediction);
  });

  // Presets
  const presets = {
    night: { appliances_current: 30, hour: 3, lights: 0, RH_out: 90, T2: 18.5, T6: 2.0, T3: 20.0 },
    morning: { appliances_current: 120, hour: 7.5, lights: 20, RH_out: 85, T2: 19.8, T6: 5.5, T3: 21.0 },
    cooking: { appliances_current: 450, hour: 19, lights: 30, RH_out: 78, T2: 21.5, T6: 8.0, T3: 22.5 },
    spring: { appliances_current: 60, hour: 14, lights: 0, RH_out: 55, T2: 23.0, T6: 18.0, T3: 22.0 }
  };

  const presetBtns = document.querySelectorAll('.preset-chip');
  presetBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const type = btn.getAttribute('data-preset');
      const p = presets[type];
      if (!p) return;

      inputs.appliances_current.value = p.appliances_current;
      inputs.hour.value = p.hour;
      inputs.lights.value = p.lights;
      inputs.RH_out.value = p.RH_out;
      inputs.T2.value = p.T2;
      inputs.T6.value = p.T6;
      inputs.T3.value = p.T3;
      calculatePrediction();
    });
  });

  calculatePrediction();
}

// ----------------------------------------------------
// 3. Coefficients Horizontal Bar Chart
// ----------------------------------------------------
function initCoefficientsChart() {
  const ctx = document.getElementById('coefficientsChart');
  if (!ctx || !window.APP_DATA || !window.APP_DATA.selectedFeatures) return;

  const featuresData = window.APP_DATA.selectedFeatures.sort(
    (a, b) => Math.abs(b.coefficient_standardized) - Math.abs(a.coefficient_standardized)
  );

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: featuresData.map(f => f.feature),
      datasets: [{
        label: 'Standardized Coefficient (β)',
        data: featuresData.map(f => f.coefficient_standardized),
        backgroundColor: featuresData.map(f =>
          f.coefficient_standardized >= 0 ? 'rgba(52, 168, 83, 0.85)' : 'rgba(234, 67, 53, 0.85)'
        ),
        borderColor: featuresData.map(f =>
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
            label: item => `標準化係數 β: ${item.parsed.x.toFixed(4)}`
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Standardized Coefficient (Effect Size on Standard Deviations)',
            font: { size: 12, weight: 'bold' }
          },
          grid: { color: 'rgba(218, 220, 224, 0.5)' }
        },
        y: {
          grid: { display: false }
        }
      }
    }
  });
}

// ----------------------------------------------------
// 4. Time Series Actual vs. Predicted Chart
// ----------------------------------------------------
function initTimeSeriesChart() {
  const ctx = document.getElementById('timeSeriesChart');
  if (!ctx || !window.APP_DATA || !window.APP_DATA.samplePredictions) return;

  const sample = window.APP_DATA.samplePredictions;

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: sample.map(s => s.date.slice(5, 16)),
      datasets: [
        {
          label: '實際值 (Actual Wh)',
          data: sample.map(s => s.actual),
          borderColor: '#1a73e8',
          backgroundColor: 'rgba(26, 115, 232, 0.1)',
          borderWidth: 2,
          pointRadius: 2,
          tension: 0.1
        },
        {
          label: '選定模型 (Selected Model, k=8)',
          data: sample.map(s => s.selected_prediction),
          borderColor: '#34a853',
          borderWidth: 2,
          borderDash: [4, 3],
          pointRadius: 2,
          tension: 0.1
        },
        {
          label: '全特徵 Baseline (k=30)',
          data: sample.map(s => s.baseline_prediction),
          borderColor: '#ea4335',
          borderWidth: 1.5,
          borderDash: [2, 2],
          pointRadius: 0,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: { position: 'top' },
        tooltip: {
          callbacks: {
            label: item => `${item.dataset.label}: ${item.parsed.y.toFixed(1)} Wh`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: '測試集時間戳記 (2016-04 至 2016-05)' },
          ticks: { maxTicksLimit: 12 },
          grid: { color: 'rgba(218, 220, 224, 0.5)' }
        },
        y: {
          title: { display: true, text: '耗電量 (Wh)' },
          grid: { color: 'rgba(218, 220, 224, 0.5)' }
        }
      }
    }
  });
}
