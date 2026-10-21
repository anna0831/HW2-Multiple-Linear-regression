# Appliances & Multi-Problem Linear Regression Studio (CRISP-DM Lab)

> 國立中興大學 / 智慧物聯網作業 HW2: 多元線性迴歸與特徵選取（CRISP-DM 標準流程 + 兩大加分功能）
> 🌐 線上部署網址 (Vercel Live URL)：[https://hw-2-multiple-linear-regression-c65.vercel.app/](https://hw-2-multiple-linear-regression-c65.vercel.app/)

---

## 專案亮點與加分項目 (Key Achievements & Extra Credits)

本專案升級原 HW2 家電能耗預測儀表板，在完全保留原版 Google Material Design 3 視覺風格、住宅微氣候感測器脈絡與 CRISP-DM 完整流程的前提下，完整實現兩大加分項：

1. **加分項 +1（兩大預設迴歸題目切換）**：
   - 頂部導覽列提供直覺的 Segmented Button 題目切換器：
     - 🏠 **Appliances 能耗預測**：UCI 比利時被動式低能耗住宅 19,735 筆 10 分鐘級距時序數據，預測目標 `Appliances` (Wh)，採用 3-Fold TimeSeriesSplit 嚴格避免資料外洩，F-test $k=8$ 勝出。
     - 🚀 **50 Startups 獲利預測**：美國加州/紐約/佛州 50 家新創公司支出與利潤資料，預測目標 `Profit` ($)，採用 3-Fold 隨機交叉驗證 (K-Fold CV)，對州別 (State) 進行 One-Hot 編碼 (排除基準類別避免虛擬變數陷阱)，Lasso $k=3$ 勝出 (Test $R^2 = 0.9001$)。
   - **零延遲即時切換**：兩大預設題目均預先完成真實 scikit-learn 訓練並打包於 `results_data.js`，切換時無需等待後端運算，標題、單位、KPI 指標、係數、推論模擬器滑桿、殘差診斷與情境預設均自動同步重置。

2. **加分項 +2（任意相容 CSV 資料集與即時機器學習）**：
   - 提供現代化彈出式自訂資料集控制台：
     - **公開 HTTPS CSV 下載網址**：支援任意公開網址（包含強制測試目標 `50_Startups.csv`）。
     - **本機 CSV 檔案上傳**：支援本機檔案拖曳與選擇，大小上限 12MB。
     - **資料理解與預覽 (Data Understanding)**：即時解析前 5 列、列數欄數、缺失值分佈與欄位型態。
     - **智慧目標 (Target Y) 篩選**：自動偵測並過濾僅列出適合迴歸之連續數值欄位。
     - **特徵排除 (Omitted Predictors)**：動態排除 ID、日期時間、高基數非數值或資料外洩變數。
     - **驗證折數與搜尋深度**：支援 3/4/5-Fold 交叉驗證，提供快速模式 (Quick Mode) 與完整搜尋模式 (Full Mode)。
   - **真實機器學習執行 (Real Machine Learning)**：由 Python FastAPI 後端透過 scikit-learn 執行真實 Pipeline（SimpleImputer 中位數/眾數插補、StandardScaler 標準化、OneHotEncoder 編碼、TopK 特徵篩選），同時執行 Baseline 與五種特徵篩選方法（`F-test`、`Lasso`、`RFE`、`Random Forest`、`Mutual Information`），即時計算 Test RMSE、MAE、MSE、$R^2$、Adjusted $R^2$、殘差診斷與經驗 90% 誤差帶。

---

## 成果摘要對照表 (Results Summary)

### 題目 A：Appliances Energy Prediction (時序能耗預測)
| 指標 | 最佳選定模型 (Selected Model, k=8) | 全特徵基準模型 (Baseline, k=30) | 差異 / 優化 |
| :--- | :---: | :---: | :---: |
| **特徵數量 (k)** | **8** | 30 | **-73.3% (大幅精簡維度)** |
| **CV RMSE (Wh)** | **66.97** | 67.84 | **-0.87 Wh (誤差更低)** |
| **Test RMSE (Wh)** | 61.32 | 61.25 | 表現持平，泛化力更穩定 |
| **Test Adjusted $R^2$** | **0.5454** | 0.5439 | **+0.15% (避免過度配適)** |
| **Test MAE (Wh)** | 27.53 | 26.82 | 平均絕對誤差約 27 Wh |

### 題目 B：50 Startups — Profit Prediction (新創獲利預測)
| 指標 | 最佳選定模型 (Lasso, k=3) | 全特徵基準模型 (Baseline, k=5) | 差異 / 優化 |
| :--- | :---: | :---: | :---: |
| **特徵數量 (k)** | **3** (`R&D`, `Admin`, `Marketing`) | 5 (含 State 虛擬變數) | **-40.0% (排除州別雜訊)** |
| **CV RMSE ($)** | **$9,104.68** | $9,550.68 | **-$446.00 (交叉驗證誤差最低)** |
| **Test RMSE ($)** | **$8,995.91** | $9,055.96 | **-$60.05 (測試集泛化最優)** |
| **Test $R^2$** | **0.9001 (90.0%)** | 0.8987 | **預測獲利能力高達 90%** |
| **Test Adjusted $R^2$** | **0.8501** | 0.7721 | **+7.8% (顯著排除多重共線性)** |
| **Test MAE ($)** | $6,979.15 | $6,961.48 | 預測平均絕對誤差約 $6.9k |

---

## 系統架構與 Vercel 部署架構 (Architecture & Deployment)

### 為什麼從 GitHub Pages 改為部署至 Vercel？
- **原 GitHub Pages 限制**：GitHub Pages 僅為純靜態託管（Static Hosting），無法執行 Python 直譯器、FastAPI 伺服器或 scikit-learn 機器學習套件。若要實現加分項 +2「使用者貼上任意 CSV 網址即時執行真實機器學習」，必須具備伺服器端運算能力。
- **Vercel Serverless Functions 優勢**：
  1. **靜態前端邊緣分發**：原生 HTML/CSS/JS 前端資源由全球 CDN 邊緣快取，達到極速載入。
  2. **Python Serverless 運算**：在 `/api` 目錄下透過 Python 3.11 執行輕量化的 FastAPI Serverless Function。
  3. **自動路由整合**：無需分開維護前後端網域，透過 `vercel.json` 統一管理 `/api/(.*)` 路由。
  4. **嚴格控制打包體積**：API 僅載入必要之 `scikit-learn`、`pandas`、`numpy`、`scipy` 與 `fastapi`，並透過 `.vercelignore` 嚴格排除 11.9MB 之原始能耗 CSV 與暫存檔案，確保解壓縮體積 < 60MB（遠低於 Vercel 250MB 上限）。

### 安全性與防護設計 (Security & SSRF Hardening)
- **SSRF 嚴格阻擋**：對外部 CSV 網址進行嚴格 DNS 預解析，封鎖 127.0.0.1、localhost、10.0.0.0/8、192.168.0.0/16、172.16.0.0/12、169.254.169.254 等私有、回環或保留 IP 位址。
- **協定限制**：僅允許標準 HTTPS 協定，禁止明文 HTTP。
- **停用重新導向 (Redirects Disabled)**：全面拒絕 3xx 重新導向，防止利用跳轉繞過 IP 安全過濾。
- **資料流控管**：採用串流下載並限制檔案大小上限為 12MB，超量立即中斷，防範拒絕服務 (DoS) 與記憶體溢位。

---

## 檔案結構 (Repository Structure)

```text
├── index.html                  # 現代化儀表板主頁面 (Google / Modern Material 3 風格)
├── style.css                   # 儀表板 CSS 樣式系統 (含自訂彈窗與題目切換器)
├── app.js                      # 前端狀態管理、Chart.js 動態繪圖、即時推論與 API 串接
├── results_data.js             # 預先封裝之雙問題 (Appliances + 50 Startups) 模型數據
├── vercel.json                 # Vercel Serverless Function 路由與執行時長配置
├── requirements.txt            # Python 相依套件清單 (FastAPI + scikit-learn 輕量化)
├── .vercelignore               # Vercel 部署排除清單 (防止大型 CSV 與暫存檔打包)
├── api/
│   ├── index.py                # FastAPI 伺服器端端點 (/api/health, /preview, /train)
│   └── ml_engine.py            # 機器學習核心引擎 (SSRF 防護、Pipeline、診斷指標)
├── tests/
│   └── test_api_and_ml.py      # 自動化單元與整合測試腳本 (覆蓋 SSRF、預覽、訓練)
├── appliances_crispdm.py       # 原 HW2 家電能耗機器學習訓練腳本
├── energydata_complete.csv     # UCI 家電能耗原始資料集
└── results/                    # 原 Appliances 評估診斷圖檔與數據
```

---

## 本機快速啟動 (Local Quick Start)

本系統支援**單一指令啟動全端服務**，FastAPI 會自動掛載靜態前端：

### 1. 安裝環境與套件
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 執行自動化測試 (Verification Tests)
```bash
python3 -m unittest tests/test_api_and_ml.py
```
*(預期輸出：Ran 7 tests ... OK，驗證 SSRF 防護、健康檢查、預覽與真實模型訓練)*

### 3. 本機啟動儀表板與後端 API
```bash
python3 -m uvicorn api.index:app --host 127.0.0.1 --port 8000
```
開啟瀏覽器訪問 `http://127.0.0.1:8000` 即可使用完整功能：
- 點擊「Appliances」與「50 Startups」進行零延遲預設題目切換。
- 點擊「載入自訂公開 CSV / 上傳檔案」，貼上 `https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv` 或上傳本機 CSV，體驗即時機器學習訓練。

---

## 部署至 Vercel 說明 (Vercel Deployment Guide)

1. **安裝 Vercel CLI 或連接 GitHub Repository**：
   ```bash
   npm i -g vercel
   vercel
   ```
2. **自動建置**：Vercel 會自動讀取 `vercel.json` 與 `requirements.txt`，並將 `api/index.py` 建置為 Python 3.11 Serverless Function。
3. **無伺服器限制檢查**：
   - 記憶體配置：1024 MB
   - 最大執行時限：60 秒
   - 打包體積：排除大型 CSV 後 < 60 MB（遠低於 250 MB 上限）
