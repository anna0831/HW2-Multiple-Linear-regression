"""
國立中興大學 智慧物聯網作業 (AIoT HW2)
作業題目：家電耗電量多元線性迴歸預測與特徵選取（CRISP-DM 標準流程）
學生學號：7115056153（請同學替換為個人學號）
資料集來源：UCI Machine Learning Repository / Kaggle: Appliances energy prediction Data Set
Kaggle 連結：https://www.kaggle.com/datasets/loveall/appliances-energy-prediction

專案遵循 CRISP-DM 六大流程：
1. 商業理解 (Business Understanding)
2. 資料理解 (Data Understanding)
3. 資料準備 (Data Preparation)
4. 模型建立 (Modeling)
5. 模型評估 (Evaluation)
6. 成果部署 (Deployment)
"""

import argparse
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.feature_selection import f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 中文與負號字型設定（相容 macOS 與 Linux）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang TC', 'Heiti TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 固定隨機種子以確保實驗結果可重現
SEED = 42

# 評估的 5 種特徵選擇演算法
METHODS = ['F-test', 'Lasso', 'RFE', 'Random Forest', 'Mutual Information']


# ==============================================================================
# Phase 2 & 3: 資料理解 (Data Understanding) 與 資料準備 (Data Preparation)
# ==============================================================================
def load_and_prepare_data(csv_path: str):
    """
    載入家電能耗資料集，進行時序排序、週期特徵工程、構建自迴歸項 (Lag-1) 與資料清理。
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_cols = {'date', 'Appliances'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"缺少必要欄位: {required_cols - set(df.columns)}")

    # 1. 時間格式轉換、遞增排序並移除重複
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    df = df.sort_values('date').drop_duplicates('date').reset_index(drop=True)

    # 2. 定義預測目標：在 t 時間點預測 t+1（未來 10 分鐘）的耗電量 (shift -1)
    df['target_next_10min'] = df['Appliances'].shift(-1)

    # 3. 週期性特徵工程：日循環與週循環正弦/餘弦編碼
    hour = df['date'].dt.hour + df['date'].dt.minute / 60.0
    dow = df['date'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    df['weekday_sin'] = np.sin(2 * np.pi * dow / 7.0)
    df['weekday_cos'] = np.cos(2 * np.pi * dow / 7.0)

    # 4. 自迴歸項 (Auto-Regressive Feature)：以當前時間點耗電量作為 Lag-1 特徵
    df['appliances_current'] = df['Appliances']

    # 5. 移除最後一筆（因 shift 產生的 NaN 目標）
    df = df.iloc[:-1].copy()

    # 6. 排除隨機噪聲變數 (rv1, rv2) 與非特徵欄位，嚴防資訊洩漏
    excluded = ['date', 'Appliances', 'target_next_10min', 'rv1', 'rv2']
    X = df.drop(columns=[col for col in excluded if col in df.columns])
    X = X.apply(pd.to_numeric, errors='coerce')
    y = df['target_next_10min'].astype(float)

    return df, X, y


def build_preprocessor():
    """前處理 Pipeline：中位數補值 + Z-score 標準化"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


# ==============================================================================
# Phase 4: 模型建立 (Modeling) — 5 種特徵選擇演算法
# ==============================================================================
def rank_features(method: str, X_scaled: np.ndarray, y_train: np.ndarray):
    """
    依據 5 種不同特徵選取演算法計算重要性並排序欄位索引。
    """
    if method == 'F-test':
        scores, _ = f_regression(X_scaled, y_train)
        return np.argsort(-np.nan_to_num(scores, nan=-np.inf))
    elif method == 'Lasso':
        lasso = LassoCV(cv=3, random_state=SEED, max_iter=25000, n_jobs=-1)
        lasso.fit(X_scaled, y_train)
        return np.argsort(-np.abs(lasso.coef_))
    elif method == 'RFE':
        estimator = LinearRegression()
        rfe = RFE(estimator, n_features_to_select=1, step=1)
        rfe.fit(X_scaled, y_train)
        return np.argsort(rfe.ranking_)
    elif method == 'Random Forest':
        rf = RandomForestRegressor(n_estimators=80, random_state=SEED, n_jobs=-1)
        rf.fit(X_scaled, y_train)
        return np.argsort(-rf.feature_importances_)
    elif method == 'Mutual Information':
        mi = mutual_info_regression(X_scaled, y_train, random_state=SEED)
        return np.argsort(-mi)
    else:
        raise ValueError(f"未知的特徵選擇方法: {method}")


def evaluate_metrics(y_true, y_pred):
    """計算常用評估指標：MAE, MSE, RMSE, R²"""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


# ==============================================================================
# Phase 5: 評估與視覺化 (Evaluation & Visualization)
# ==============================================================================
def plot_prediction_with_intervals(df, y_test, preds, cutoff, out_dir, X_train_sel, y_train, X_test_sel, best_model):
    """
    繪製預測圖，嚴格依據經典多元線性迴歸 (OLS) 理論推導 95% 信賴區間 (CI for Mean) 與 95% 預測區間 (PI)。
    【統計嚴謹性保證】：
      1. 殘差標準差 s_e 嚴格自【訓練集】殘差與無偏自由度 (N_train - p - 1) 計算，絕不使用測試集殘差反向校準。
      2. 參數協方差矩陣 Cov(beta) = s_e^2 * (X_train^T X_train)^-1 依據訓練集設計矩陣。
      3. 每個測試點之槓桿值 h_00 = x_0 * (X_train^T X_train)^-1 * x_0^T 反映遠離訓練中心的變異放大效應。
      4. 95% CI = y_hat +/- t_crit * s_e * sqrt(h_00)
      5. 95% PI = y_hat +/- t_crit * s_e * sqrt(1 + h_00)
    """
    pred_df = pd.DataFrame({
        'date': df['date'].iloc[cutoff:].values,
        'actual': y_test.values,
        'selected_prediction': preds['Selected Model'],
        'baseline_prediction': preds['All Features Baseline']
    })

    # 1. 訓練集統計量推導 (嚴格無測試洩漏)
    n_train = len(y_train)
    p = X_train_sel.shape[1]
    dof = n_train - p - 1
    y_train_pred = best_model.predict(X_train_sel)
    rss_train = np.sum((y_train.values - y_train_pred) ** 2)
    residual_se_train = float(np.sqrt(rss_train / dof))

    # 2. 設計矩陣協方差 (X^T X)^-1
    X_tr_des = np.column_stack([np.ones(n_train), X_train_sel])
    XtX_inv = np.linalg.pinv(X_tr_des.T @ X_tr_des)

    # 3. 測試點槓桿值與標準誤推導
    X_te_des = np.column_stack([np.ones(len(y_test)), X_test_sel])
    h_all = np.sum((X_te_des @ XtX_inv) * X_te_des, axis=1)
    tcrit = float(stats.t.ppf(0.975, df=dof))

    ci_margin_all = tcrit * residual_se_train * np.sqrt(np.maximum(0, h_all))
    pi_margin_all = tcrit * residual_se_train * np.sqrt(np.maximum(1.0, 1.0 + h_all))

    # 儲存全測試集包含槓桿值與區間的預測明細
    full_pred_export = pred_df.copy()
    full_pred_export['leverage_h'] = h_all
    full_pred_export['ci_margin'] = ci_margin_all
    full_pred_export['pi_margin'] = pi_margin_all
    full_pred_export['ci_lower'] = np.maximum(0, full_pred_export['selected_prediction'] - ci_margin_all)
    full_pred_export['ci_upper'] = full_pred_export['selected_prediction'] + ci_margin_all
    full_pred_export['pi_lower'] = np.maximum(0, full_pred_export['selected_prediction'] - pi_margin_all)
    full_pred_export['pi_upper'] = full_pred_export['selected_prediction'] + pi_margin_all
    full_pred_export.to_csv(out_dir / 'test_predictions_with_intervals.csv', index=False)

    # 取樣 120 點進行清晰視覺呈現 (每 30 筆取 1 點)
    sample = full_pred_export.iloc[::30].copy().reset_index(drop=True)
    mean_ci = float(np.mean(sample['ci_margin']))
    mean_pi = float(np.mean(sample['pi_margin']))

    plt.figure(figsize=(13, 6), dpi=300)
    x_idx = np.arange(len(sample))

    # 填繪區間
    plt.fill_between(x_idx, sample['pi_lower'], sample['pi_upper'], color='#34a853', alpha=0.18,
                     label=f'95% 預測區間 (Prediction Interval, 訓練集統計推導平均 ±{mean_pi:.1f} Wh)')
    plt.fill_between(x_idx, sample['ci_lower'], sample['ci_upper'], color='#1a73e8', alpha=0.35,
                     label=f'95% 平均值信賴區間 (Confidence Interval for Mean, 平均 ±{mean_ci:.1f} Wh)')

    # 繪製曲線
    plt.plot(x_idx, sample['actual'], color='#1a73e8', marker='o', markersize=4, linewidth=1.5,
             label='實際耗電量 (Actual Wh)')
    plt.plot(x_idx, sample['selected_prediction'], color='#1e8e3e', linestyle='--', linewidth=2,
             label=f'選定模型預測 (Selected Model k={p}, F-test)')
    plt.plot(x_idx, sample['baseline_prediction'], color='#ea4335', linestyle=':', linewidth=1.5,
             alpha=0.8, label=f'全特徵基準 (Baseline k={X_test_sel.shape[1] if hasattr(X_test_sel, "shape") else 30})')

    step = max(1, len(sample) // 8)
    plt.xticks(x_idx[::step], sample['date'].astype(str).str.slice(5, 16).iloc[::step], rotation=20, ha='right', fontsize=9)
    plt.xlabel('測試集時間戳記 (Test Set Timestamp)', fontsize=11, fontweight='bold', labelpad=8)
    plt.ylabel('家電耗電量 (Appliances Wh)', fontsize=11, fontweight='bold', labelpad=8)
    plt.title('家電耗電量多元線性迴歸預測 — 95% 信賴區間與預測區間分析 (嚴格 OLS 統計推導)', fontsize=14, fontweight='bold', pad=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right', fontsize=10)
    plt.tight_layout()

    out_file = out_dir / 'prediction_with_interval.png'
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"[視覺化] 成功依據訓練集嚴格統計推導儲存預測圖（含信賴與預測區間）至: {out_file.resolve()}")
    print(f"       訓練集殘差標準誤 s_e = {residual_se_train:.4f} Wh | 平均 95% CI 邊際 = ±{mean_ci:.2f} Wh | 平均 95% PI 邊際 = ±{mean_pi:.2f} Wh")


# ==============================================================================
# 主程式執行入口
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='CRISP-DM 家電耗電量多元線性迴歸與特徵選取')
    parser.add_argument('--csv', default='energydata_complete.csv', help='資料集 CSV 路徑')
    parser.add_argument('--out', default='results', help='輸出成果資料夾')
    parser.add_argument('--folds', type=int, default=3, help='時間序列交叉驗證折數')
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print(" 國立中興大學 智慧物聯網作業 (AIoT HW2) - 多元線性迴歸分析")
    print("==================================================================")

    # 1. 載入資料
    print("\n[Step 1] 正在載入與預處理資料集...")
    df, X, y = load_and_prepare_data(args.csv)
    n_samples, n_features = X.shape
    feature_names = np.array(X.columns)
    print(f"資料筆數: {n_samples:,} 筆 | 候選特徵數: {n_features} 個")

    # 2. 時間序列 80/20 劃分（嚴格保留最後 20% 為無資料外洩之測試集）
    cutoff = int(n_samples * 0.8)
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    print(f"訓練集筆數: {len(X_train):,} 筆 | 獨立測試集筆數: {len(X_test):,} 筆")

    # 3. 全特徵 Baseline 模型訓練
    print("\n[Step 2] 訓練全特徵 Baseline 基準模型 (k=30)...")
    preprocessor = build_preprocessor()
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    baseline_model = LinearRegression().fit(X_train_scaled, y_train)
    y_pred_baseline = baseline_model.predict(X_test_scaled)

    # 4. TimeSeriesSplit 外層交叉驗證比較 5 種特徵選擇方法
    print(f"\n[Step 3] 執行 {args.folds}-Fold 時間序列交叉驗證與特徵篩選 (k=1~{n_features})...")
    tscv = TimeSeriesSplit(n_splits=args.folds)
    splits = list(tscv.split(X_train))
    cv_records = []

    for fold_i, (train_idx, val_idx) in enumerate(splits, 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        prep = build_preprocessor().fit(X_tr)
        X_tr_s = prep.transform(X_tr)
        X_val_s = prep.transform(X_val)

        # 記錄 Baseline
        b_pred = LinearRegression().fit(X_tr_s, y_tr).predict(X_val_s)
        cv_records.append({'method': 'All Features Baseline', 'k': n_features, 'fold': fold_i, **evaluate_metrics(y_val, b_pred)})

        # 評估 5 種方法在各 k 之表現
        for method in METHODS:
            ranked_idx = rank_features(method, X_tr_s, y_tr)
            for k in range(1, n_features + 1):
                subset_idx = ranked_idx[:k]
                k_pred = LinearRegression().fit(X_tr_s[:, subset_idx], y_tr).predict(X_val_s[:, subset_idx])
                cv_records.append({'method': method, 'k': k, 'fold': fold_i, **evaluate_metrics(y_val, k_pred)})

    cv_df = pd.DataFrame(cv_records)
    cv_df.to_csv(out_dir / 'cv_fold_results.csv', index=False)

    cv_summary = cv_df.groupby(['method', 'k'], as_index=False).agg(
        CV_RMSE=('RMSE', 'mean'),
        CV_RMSE_sd=('RMSE', 'std'),
        CV_MAE=('MAE', 'mean'),
        CV_MSE=('MSE', 'mean'),
        CV_R2=('R2', 'mean')
    )
    cv_summary.to_csv(out_dir / 'cv_summary.csv', index=False)

    # 5. 挑選最佳特徵選擇演算法與最佳特徵數 k (最低 CV RMSE)
    candidates = cv_summary[cv_summary['method'].isin(METHODS)]
    best_candidate = candidates.sort_values(['CV_RMSE', 'k', 'method']).iloc[0]
    best_method = str(best_candidate['method'])
    best_k = int(best_candidate['k'])
    print(f"\n[Step 4] 最佳特徵選擇演算法: {best_method} | 最佳特徵數 k: {best_k} (CV RMSE: {best_candidate['CV_RMSE']:.4f} Wh)")

    # 6. 使用全訓練集與選定特徵重新配適模型
    prep_final = build_preprocessor().fit(X_train)
    X_train_full = prep_final.transform(X_train)
    X_test_full = prep_final.transform(X_test)

    final_ranking = rank_features(best_method, X_train_full, y_train)
    selected_indices = final_ranking[:best_k]
    selected_features = feature_names[selected_indices]

    best_model = LinearRegression().fit(X_train_full[:, selected_indices], y_train)
    y_pred_selected = best_model.predict(X_test_full[:, selected_indices])

    preds = {
        'Selected Model': y_pred_selected,
        'All Features Baseline': y_pred_baseline
    }

    # 7. 計算測試集指標 (包含 Adjusted R²)
    metrics = []
    n_test = len(y_test)
    for name, pred in preds.items():
        m = evaluate_metrics(y_test, pred)
        p = best_k if name == 'Selected Model' else n_features
        m['Adjusted_R2'] = (1 - (1 - m['R2']) * (n_test - 1) / (n_test - p - 1)) if n_test > p + 1 else np.nan
        metrics.append({'model': name, 'features': p, **m})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out_dir / 'test_metrics.csv', index=False)
    print("\n--- 獨立測試集評估結果 ---")
    print(metrics_df.to_string(index=False))

    # 8. 儲存入選特徵標準化係數與預測結果
    pd.DataFrame({'feature': selected_features, 'coefficient_standardized': best_model.coef_}).to_csv(
        out_dir / 'selected_features.csv', index=False
    )
    pd.DataFrame({
        'date': df['date'].iloc[cutoff:].values,
        'actual': y_test.values,
        'selected_prediction': y_pred_selected,
        'baseline_prediction': y_pred_baseline
    }).to_csv(out_dir / 'test_predictions.csv', index=False)

    # 9. 產出預測圖（包含嚴格統計推導之 95% 信賴區間與預測區間）
    plot_prediction_with_intervals(
        df=df,
        y_test=y_test,
        preds=preds,
        cutoff=cutoff,
        out_dir=out_dir,
        X_train_sel=X_train_full[:, selected_indices],
        y_train=y_train,
        X_test_sel=X_test_full[:, selected_indices],
        best_model=best_model
    )

    print("\n[完成] 所有模型訓練、特徵篩選、評估指標與視覺化圖表已產出完成！")


if __name__ == '__main__':
    main()
