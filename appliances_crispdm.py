"""CRISP-DM 專案：家電耗電量預測（預測未來 10 分鐘區間之 Appliances 耗電量）

本專案遵循 CRISP-DM 資料探勘標準流程：
1. 商業理解 (Business Understanding)：透過歷史用電與環境感測數據，提前預測家電耗電，支援節能與需量調度。
2. 資料理解 (Data Understanding)：檢查特徵欄位、缺失值分佈、目標變數分佈與特徵相關性。
3. 資料準備 (Data Preparation)：時間序列排序、週期特徵工程（時/星期正弦餘弦）、建構滯後項 (lag-1)、排除資料外洩欄位。
4. 模型建立 (Modeling)：使用 5 種特徵選擇演算法（F-test, Lasso, RFE, Random Forest, Mutual Information）
   結合 TimeSeriesSplit 交叉驗證，選出最佳特徵子集與線性迴歸模型。
5. 評估 (Evaluation)：在獨立保留的 20% 時間序列測試集上比較最佳模型與 Baseline（全特徵模型）。
6. 部署 (Deployment)：儲存模型 Pipeline、特徵名冊及殘差分析診斷視覺化圖表。

執行方式：
    python appliances_crispdm.py --csv energydata_complete.csv --out results
"""

import argparse
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 設定無 GUI 後端，適用於伺服器或批次繪圖
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.base import clone
from sklearn.feature_selection import f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 固定隨機種子以確保實驗可重現性
SEED = 42

# 評估的 5 種特徵選取方法
METHODS = ['F-test', 'Lasso', 'RFE', 'Random Forest', 'Mutual Information']


def load_data(csv):
    """載入資料並進行時間特徵工程與目標變數定義。

    參數:
        csv (str): energydata_complete.csv 檔案路徑

    回傳:
        data (DataFrame): 包含工程後所有欄位的完整資料框
        X (DataFrame): 供模型訓練的特徵矩陣
        y (Series): 目標變數（未來 10 分鐘之家電耗電量）
    """
    data = pd.read_csv(csv)
    # 去除欄位名稱前後的空白字元
    data.columns = data.columns.str.strip()

    # 確認必要欄位是否存在
    needed = {'date', 'Appliances'}
    if not needed.issubset(data.columns):
        raise ValueError(f'缺少必要欄位: {needed - set(data.columns)}')

    # 轉換時間格式，依時間遞增排序並移除重複時間戳記
    data['date'] = pd.to_datetime(data['date'], errors='raise')
    data = data.sort_values('date').drop_duplicates('date').reset_index(drop=True)

    # 定義目標變數：在 t 時間點預測 t+1（下一個 10 分鐘）的耗電量 (shift -1)
    data['target_next_10min'] = data['Appliances'].shift(-1)

    # 時間週期特徵工程（以正弦/餘弦編碼擷取日循環與週循環）
    hour = data['date'].dt.hour + data['date'].dt.minute / 60
    dow = data['date'].dt.dayofweek
    data['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    data['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    data['weekday_sin'] = np.sin(2 * np.pi * dow / 7)
    data['weekday_cos'] = np.cos(2 * np.pi * dow / 7)

    # 將當前時間點的家電耗電量作為 lag-1 自迴歸特徵（此數值在 t 時已知，非未來洩漏）
    data['appliances_current'] = data['Appliances']

    # 移除最後一筆（因 shift(-1) 產生之 NaN 目標）
    data = data.iloc[:-1].copy()

    # 排除非預測特徵與隨機噪聲變數 (rv1, rv2)
    excluded = ['date', 'Appliances', 'target_next_10min', 'rv1', 'rv2']
    X = data.drop(columns=[c for c in excluded if c in data.columns])
    X = X.apply(pd.to_numeric, errors='coerce')
    y = data['target_next_10min'].astype(float)
    return data, X, y


def ranking(method, scaled, y, k=None):
    """在訓練集上依指定方法計算特徵重要性，回傳重要性由高至低的欄位索引。

    參數:
        method (str): 特徵評估方法名稱
        scaled (ndarray): 已標準化之特徵矩陣
        y (Series/ndarray): 訓練集目標變數
        k (int, optional): 指定特徵數（部分方法適用）

    回傳:
        ndarray: 由重要到不重要的特徵索引陣列
    """
    if method == 'F-test':
        # 單變量線性迴歸 F 檢定（統計顯著性）
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            scores, _ = f_regression(scaled, y)
        scores = np.nan_to_num(scores, nan=-np.inf)
        return np.argsort(-scores, kind='stable')

    if method == 'Mutual Information':
        # 互資訊法：衡量非線性關聯性
        scores = mutual_info_regression(scaled, y, random_state=SEED)
        return np.argsort(-scores, kind='stable')

    if method == 'Random Forest':
        # 隨機森林特徵重要性（基於雜質度減少量）
        model = RandomForestRegressor(n_estimators=80, min_samples_leaf=3,
                                      random_state=SEED, n_jobs=-1)
        model.fit(scaled, y)
        return np.argsort(-model.feature_importances_, kind='stable')

    if method == 'Lasso':
        # LassoCV L1 正則化：以時間序列交叉驗證挑選正則係數 alpha
        model = LassoCV(cv=TimeSeriesSplit(n_splits=3), max_iter=15000,
                        alphas=60, random_state=SEED)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(scaled, y)
        return np.argsort(-np.abs(model.coef_), kind='stable')

    if method == 'RFE':
        # 遞迴特徵消除法 (Recursive Feature Elimination)
        model = RFE(LinearRegression(), n_features_to_select=1, step=1)
        model.fit(scaled, y)
        return np.argsort(model.ranking_, kind='stable')

    raise ValueError(f'不支援的特徵選取方法: {method}')


def predictor():
    """建立基礎資料前處理 Pipeline（中位數填補與標準化縮放）。"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def scoring(y, pred):
    """計算常用迴歸評估指標：MAE, MSE, RMSE, R²。"""
    return dict(
        MAE=mean_absolute_error(y, pred),
        MSE=mean_squared_error(y, pred),
        RMSE=np.sqrt(mean_squared_error(y, pred)),
        R2=r2_score(y, pred)
    )


def main(args):
    # 建立輸出結果目錄
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. 載入資料與時間序列切分（前 80% 訓練集，後 20% 測試集，保持時序順序防洩漏）
    data, X, y = load_data(args.csv)
    cutoff = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    names = np.array(X.columns)
    n_features = len(names)

    print(f'總筆數: {len(X)}; 模型特徵數: {n_features}; 訓練/測試劃分: {len(X_train)}/{len(X_test)}')
    if n_features <= 13 or n_features >= 40:
        print('警告: 特徵數量超出 14~39 規範範圍')
    print('目標: 利用 t 時刻觀測特徵，預測 t+1 (未來 10 分鐘) 之家電耗電量')
    print('測試集時間區間:', data['date'].iloc[cutoff], '至', data['date'].iloc[-1])
    print('當前耗電量 Appliances(t) 作為合法滯後特徵保留。\n')

    # 匯出資料理解摘要（缺失值分佈、相關矩陣、目標變數統計量）
    pd.DataFrame({'feature': names, 'missing_train': X_train.isna().sum().values}).to_csv(
        out / 'feature_inventory.csv', index=False)
    X_train.corr(numeric_only=True).to_csv(out / 'feature_correlations.csv')
    y_train.describe().to_csv(out / 'target_summary.csv')

    # 2. Baseline 模型（無特徵選取，全特徵線性迴歸 Pipeline）
    cv = TimeSeriesSplit(n_splits=args.folds)
    baseline = Pipeline([
        ("preprocessing", predictor()),
        ("regression", LinearRegression())
    ])

    # 計算 Baseline 的時間序列交叉驗證 RMSE
    cv_mse = -cross_val_score(
        baseline,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_squared_error"
    )
    baseline_cv_rmse = np.sqrt(cv_mse).mean()

    # 訓練 Baseline 並在獨立測試集上進行推論與評估
    baseline.fit(X_train, y_train)
    y_pred_baseline = baseline.predict(X_test)

    baseline_mse = mean_squared_error(y_test, y_pred_baseline)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2 = r2_score(y_test, y_pred_baseline)

    print("--- Baseline 模型評估（無特徵選取，使用全部特徵）---")
    print("Baseline CV RMSE:", baseline_cv_rmse)
    print("Baseline Test MSE:", baseline_mse)
    print("Baseline Test RMSE:", baseline_rmse)
    print("Baseline Test R²:", baseline_r2)
    print("---------------------------------------------------\n")

    # 3. 階段四：時間序列外層交叉驗證（依前進窗口劃分，所有轉換與特徵排序僅在訓練折內進行）
    splits = list(cv.split(X_train))
    ks = list(range(1, n_features + 1))
    rows = []
    for fi, (tr, val) in enumerate(splits, 1):
        Xt, Xv = X_train.iloc[tr], X_train.iloc[val]
        yt, yv = y_train.iloc[tr], y_train.iloc[val]

        # 僅用折內訓練資料擬合前處理器，防止資料外洩
        prep = predictor().fit(Xt)
        a = prep.transform(Xt)
        b = prep.transform(Xv)

        # 記錄全特徵 Baseline 在各 fold 的評估表現
        baseline_pred = LinearRegression().fit(a, yt).predict(b)
        rows.append({'method': 'All Features Baseline', 'k': n_features,
                     'fold': fi, **scoring(yv, baseline_pred)})

        # 依 5 種不同特徵選取方法進行特徵重要度排序與子集評估
        for method in METHODS:
            print(f'Fold {fi}/{args.folds}: 執行 {method} 特徵排序中...', flush=True)
            order = ranking(method, a, yt)
            # 測試前 1 至前 k 個特徵的預測表現
            for k in ks:
                idx = order[:k]
                pred = LinearRegression().fit(a[:, idx], yt).predict(b[:, idx])
                rows.append({'method': method, 'k': k, 'fold': fi,
                             **scoring(yv, pred)})

    # 儲存每折評估指標與整體交叉驗證摘要
    fold_results = pd.DataFrame(rows)
    fold_results.to_csv(out / 'cv_fold_results.csv', index=False)
    summary = (fold_results.groupby(['method', 'k'], as_index=False)
               .agg(CV_RMSE=('RMSE', 'mean'), CV_RMSE_sd=('RMSE', 'std'),
                    CV_MAE=('MAE', 'mean'), CV_MSE=('MSE', 'mean'),
                    CV_R2=('R2', 'mean')))
    summary.to_csv(out / 'cv_summary.csv', index=False)

    # 依 CV RMSE 挑選最佳特徵選取方法與最佳特徵數 k
    eligible = summary[summary['method'].isin(METHODS)]
    winner = eligible.sort_values(['CV_RMSE', 'k', 'method']).iloc[0]
    method, k = str(winner['method']), int(winner['k'])

    # 4. 階段五：使用全訓練集與選定特徵重新配適模型，於獨立測試集評估
    prep = predictor().fit(X_train)
    a, b = prep.transform(X_train), prep.transform(X_test)
    order = ranking(method, a, y_train)
    chosen = order[:k]
    best = LinearRegression().fit(a[:, chosen], y_train)

    preds = {
        'Selected Model': best.predict(b[:, chosen]),
        'All Features Baseline': y_pred_baseline
    }

    metrics = []
    for key, pred in preds.items():
        measure = scoring(y_test, pred)
        p = k if key == 'Selected Model' else n_features
        # 計算校正後判定係數 (Adjusted R²)
        measure['Adjusted_R2'] = (1 - (1 - measure['R2']) * (len(y_test) - 1) /
                                  (len(y_test) - p - 1)) if len(y_test) > p + 1 else np.nan
        metrics.append({'model': key, 'features': p, **measure})

    # 匯出評估指標、入選特徵權重係數、測試集預測明細與最終模型
    pd.DataFrame(metrics).to_csv(out / 'test_metrics.csv', index=False)
    pd.DataFrame({'feature': names[chosen], 'coefficient_standardized': best.coef_}).to_csv(
        out / 'selected_features.csv', index=False)
    pd.DataFrame({'date': data['date'].iloc[cutoff:].values,
                  'actual': y_test.values,
                  'selected_prediction': preds['Selected Model'],
                  'baseline_prediction': preds['All Features Baseline']}).to_csv(
        out / 'test_predictions.csv', index=False)
    joblib.dump({'preprocessor': prep, 'model': best,
                 'selected_indices': chosen, 'input_columns': list(names),
                 'forecast_horizon': 'next 10-minute interval'},
                out / 'selected_model.joblib')

    # 5. 繪製診斷與結果視覺化圖表
    # (1) 特徵數量 vs 交叉驗證 RMSE 曲線
    plt.figure(figsize=(10, 6))
    for m in METHODS:
        part = summary[summary.method == m]
        plt.plot(part.k, part.CV_RMSE, marker='.', label=m)
    baseline_rmse_cv = float(summary[summary.method == 'All Features Baseline']['CV_RMSE'].iloc[0])
    plt.axhline(baseline_rmse_cv, ls='--', color='black', label='All features baseline')
    plt.xlabel('Number of selected features (k)')
    plt.ylabel('Mean time-series CV RMSE (Wh)')
    plt.title('Appliances Energy Prediction')
    plt.legend(loc='lower right'); plt.grid(alpha=.2); plt.tight_layout()
    plt.savefig(out / 'feature_count_vs_rmse.png', dpi=180); plt.close()

    # (2) 實際值 vs 預測值散佈圖
    pred = preds['Selected Model']
    residual = y_test.to_numpy() - pred
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_test, pred, alpha=.18, s=9)
    lower, upper = min(y_test.min(), pred.min()), max(y_test.max(), pred.max())
    ax.plot([lower, upper], [lower, upper], 'r--')
    ax.set(xlabel='Actual (Wh)', ylabel='Predicted (Wh)', title='Actual vs predicted')
    fig.tight_layout(); fig.savefig(out / 'actual_vs_predicted.png', dpi=180); plt.close(fig)

    # (3) 殘差 vs 配適值散佈圖（檢查異質變異與非線性）
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(pred, residual, alpha=.18, s=9)
    ax.axhline(0, ls='--', color='red')
    ax.set(xlabel='Predicted (Wh)', ylabel='Residual (Wh)', title='Residuals vs fitted')
    fig.tight_layout(); fig.savefig(out / 'residuals_vs_fitted.png', dpi=180); plt.close(fig)

    # (4) 殘差常態 Q-Q 圖（檢驗常態分佈假設）
    fig, ax = plt.subplots(figsize=(6, 5))
    stats.probplot(residual, dist='norm', plot=ax)
    ax.set_title('Residual Q-Q plot')
    fig.tight_layout(); fig.savefig(out / 'residual_qq.png', dpi=180); plt.close(fig)

    # (5) 殘差時間序列圖（檢查隨時間的序列自相關性或漂移）
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(data['date'].iloc[cutoff:].values, residual, linewidth=.5)
    ax.axhline(0, ls='--', color='red')
    ax.set(xlabel='Date', ylabel='Residual (Wh)', title='Residuals over time')
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out / 'residuals_over_time.png', dpi=180); plt.close(fig)

    # 印出最終摘要報告
    print('最佳特徵選取方法:', method, '| 最佳特徵數 k:', k, '| CV RMSE:', round(winner['CV_RMSE'], 4))
    print('選出之特徵清單:', list(names[chosen]))
    print('\n', pd.DataFrame(metrics).to_string(index=False))
    print('所有產出檔案已儲存至:', out.resolve())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRISP-DM 家電耗電量預測模型')
    parser.add_argument('--csv', default='energydata_complete.csv', help='資料集路徑 (預設: energydata_complete.csv)')
    parser.add_argument('--out', default='results', help='輸出成果資料夾 (預設: results)')
    parser.add_argument('--folds', type=int, default=3, help='時間序列交叉驗證折數 (預設: 3)')
    main(parser.parse_args())
