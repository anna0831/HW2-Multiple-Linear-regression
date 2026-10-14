# Appliances Energy Prediction (CRISP-DM Multiple Linear Regression)

> 國立中興大學 / 機器學習作業 HW2: 多元線性迴歸與特徵選取（CRISP-DM 標準流程）
> 互動式視覺化展示頁面：[GitHub Pages 網址](https://anna0831.github.io/HW2-Multiple-Linear-regression/)

---

## 專案簡介 (Project Overview)

本專案遵循 **CRISP-DM（Cross-Industry Standard Process for Data Mining）** 資料探勘流程，利用比利時低耗能住宅 4.5 個月（約 19,735 筆 10 分鐘級距）的溫濕度感測與氣象數據，建立多元線性迴歸模型預測未來 10 分鐘之家電耗電量（`Appliances` Wh）。

專案包含：
1. **完整的特徵工程**：日/週週期正弦餘弦編碼、當前耗電量滞後特徵（Lag-1 自迴歸項）、排除隨機噪聲（`rv1`, `rv2`）。
2. **5 大特徵選擇演算法比較**：`F-test`、`Lasso`、`RFE`、`Random Forest`、`Mutual Information`。
3. **時間序列外層交叉驗證 (TimeSeriesSplit 3-Fold)**：防止未來資料外洩。
4. **與 Baseline 模型對比**：相較於使用全 30 個特徵的基準模型，選定之 F-test ($k=8$) 在減少 73.3% 特徵維度的同時，達成最優的交叉驗證 RMSE (66.97 Wh) 與最高泛化能力的 Adjusted $R^2$ (0.5454)。
5. **現代化互動式 Web 儀表板 (Anna ML Studio)**：支援特徵篩選動態滑桿、多元線性迴歸即時推論器、情境預設按鈕、邊際貢獻瀑布圖與殘差統計診斷。

---

## 成果摘要 (Results Summary)

| 指標 | 最佳選定模型 (Selected Model, k=8) | 全特徵基準模型 (Baseline, k=30) | 差異 / 優化 |
| :--- | :---: | :---: | :---: |
| **特徵數量 (k)** | **8** | 30 | **-73.3% (維度大幅精簡)** |
| **CV RMSE (Wh)** | **66.97** | 67.84 | **-0.87 Wh (誤差更低)** |
| **Test RMSE (Wh)** | 61.32 | 61.25 | 表現持平，泛化力更穩定 |
| **Test Adjusted $R^2$** | **0.5454** | 0.5439 | **+0.15% (避免過度配適)** |
| **Test MAE (Wh)** | 27.53 | 26.82 | 平均絕對誤差約 27 Wh |

### 入選之 8 個核心特徵（依重要度排序）：
1. `appliances_current`：當前用電量自迴歸項（最強預測因子）
2. `hour_sin`：日循環週期正弦項
3. `lights`：全屋照明耗電量
4. `hour_cos`：日循環週期餘弦項
5. `RH_out`：室外相對濕度
6. `T2`：客廳溫度
7. `T6`：北側外牆溫度（常年背陽無日光直射干擾，反映真實外圍冷空氣微氣候）
8. `T3`：洗衣房溫度

---

## 檔案結構 (Repository Structure)

```text
├── index.html                  # 互動式儀表板主頁面 (Google / Modern Material 3 風格)
├── style.css                   # 儀表板 CSS 樣式系統
├── app.js                      # 儀表板前端互動邏輯、Chart.js 與即時推論核心
├── results_data.js             # 預提取之模型參數與評估數據 (供前端即時推論)
├── appliances_crispdm.py       # CRISP-DM 核心機器學習訓練與評估管線
├── plot_rmse_vs_features.py   # 特徵數量 vs CV RMSE 獨立繪圖腳本
├── energydata_complete.csv     # UCI 家電能耗原始資料集
├── requirements.txt            # Python 相依套件清單
└── results/                    # 模型產出數據與視覺化診斷圖檔
    ├── feature_count_vs_rmse.png   # 特徵數 vs RMSE 收斂曲線 (標題: Appliances Energy Prediction)
    ├── actual_vs_predicted.png     # 測試集實際值 vs 配適值散佈圖
    ├── residuals_vs_fitted.png     # 殘差 vs 配適值圖 (檢查異質變異)
    ├── residual_qq.png             # 殘差常態 Q-Q 圖
    ├── residuals_over_time.png     # 殘差時間序列圖
    ├── test_metrics.csv            # 測試集評估指標對照表
    ├── selected_features.csv       # 8 個核心特徵之標準化迴歸係數
    ├── cv_summary.csv              # 5 種演算法於各 k 之交叉驗證詳細數據
    └── selected_model.joblib       # 最終選定模型與前處理 Pipeline 檔
```

---

## 快速開始 (Quick Start)

### 1. 安裝環境與套件
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 執行模型訓練與特徵選取
```bash
python appliances_crispdm.py
```
*(可選參數: `--csv <路徑>` `--out <輸出資料夾>` `--folds <折數>`)*

### 3. 本地啟動互動式 Web 儀表板
```bash
python3 -m http.server 8080
```
於瀏覽器開啟 `http://localhost:8080` 即可體驗即時推論與動態圖表。
