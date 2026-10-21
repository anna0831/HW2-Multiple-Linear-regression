"""
Machine Learning Engine for Multiple Linear Regression Dashboard.
Provides:
- SSRF-safe URL fetching
- CSV parsing & validation
- Real scikit-learn Pipeline with ColumnTransformer, StandardScaler, OneHotEncoder, SimpleImputer
- Five feature selection methods: F-test, Lasso, RFE, Random Forest, Mutual Information
- K-Fold Cross-Validation, held-out test evaluation
- Model diagnostics: Actual vs Predicted, Residuals vs Fitted, Q-Q data, Durbin-Watson, Shapiro-Wilk, VIF
- Feature specifications for interactive simulator
"""
import io
import ipaddress
import socket
import time
from urllib.parse import urlparse
import numpy as np
import pandas as pd
import requests
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, f_regression, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

METHODS = ['F-test', 'Lasso', 'RFE', 'Random Forest', 'Mutual Information']

def validate_and_fetch_url(url: str, max_bytes: int = 12 * 1024 * 1024) -> str:
    """Safely fetch CSV from public HTTPS endpoint with strict SSRF protection."""
    parsed = urlparse(url)
    if parsed.scheme.lower() != 'https':
        raise ValueError("安全限制：僅允許公開 HTTPS 網址 (Only public HTTPS URLs allowed)")
    if not parsed.hostname:
        raise ValueError("無效的主機名稱 (Invalid hostname)")
    if parsed.username or parsed.password:
        raise ValueError("不允許在網址中包含使用者認證資訊 (Credentials in URL not permitted)")

    # DNS check to block internal/private/reserved IP addresses
    try:
        addr_info = socket.getaddrinfo(parsed.hostname, 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror:
        raise ValueError(f"無法解析主機名稱 (DNS resolution failed): {parsed.hostname}")

    for item in addr_info:
        ip_str = item[4][0]
        ip = ipaddress.ip_address(ip_str)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified or not ip.is_global:
            raise ValueError(f"安全阻擋：不允許存取非公開或私有 IP 位址 ({ip_str})")

    headers = {'User-Agent': 'Anna-ML-Studio/2.0 (+https://github.com/anna0831)'}
    with requests.get(url, headers=headers, stream=True, timeout=15, allow_redirects=False) as resp:
        if 300 <= resp.status_code < 400:
            raise ValueError("為防止安全性漏洞，不允許自動重新導向 (Redirects disabled)。請提供最終直接指向 CSV 的 HTTPS 網址。")
        resp.raise_for_status()

        chunks = []
        total_size = 0
        for chunk in resp.iter_content(chunk_size=65536):
            if chunk:
                total_size += len(chunk)
                if total_size > max_bytes:
                    raise ValueError(f"CSV 檔案大小超過 {max_bytes // (1024 * 1024)} MB 上限")
                chunks.append(chunk)

    raw_bytes = b"".join(chunks)
    try:
        return raw_bytes.decode('utf-8')
    except UnicodeDecodeError:
        return raw_bytes.decode('latin1')

def parse_csv_dataframe(csv_text_or_bytes) -> pd.DataFrame:
    """Parse CSV text or bytes into a pandas DataFrame."""
    if isinstance(csv_text_or_bytes, str):
        buf = io.StringIO(csv_text_or_bytes)
    else:
        buf = io.BytesIO(csv_text_or_bytes)
    df = pd.read_csv(buf)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def preview_dataset(df: pd.DataFrame) -> dict:
    """Generate preview overview for arbitrary dataset."""
    if df.empty:
        raise ValueError("CSV 檔案為空或無有效資料列")
    if len(df) < 10:
        raise ValueError("資料集列數過少，至少需要 10 列以供預覽與分析")

    columns = list(df.columns)
    missing_counts = {str(k): int(v) for k, v in df.isna().sum().to_dict().items()}
    dtypes = {}
    candidate_targets = []
    default_omit = []

    for col in columns:
        col_lower = col.lower()
        # Clean numeric candidate check
        numeric_series = pd.to_numeric(df[col], errors='coerce')
        valid_num_count = numeric_series.notna().sum()
        is_num = valid_num_count >= 0.80 * len(df) and numeric_series.nunique(dropna=True) > 1

        if is_num:
            dtypes[col] = "numeric"
            if valid_num_count >= 15:
                candidate_targets.append(col)
        else:
            dtypes[col] = "categorical"

        # Check for ID / index / uuid / datetime / high cardinality
        if col_lower in ('id', 'index', 'unnamed: 0', 'uid', 'uuid', 'row_id'):
            default_omit.append(col)
        elif any(t in col_lower for t in ('date', 'time', 'timestamp', 'datetime')):
            default_omit.append(col)
        elif not is_num:
            n_unique = df[col].nunique(dropna=True)
            if n_unique > 15 or (len(df) > 30 and n_unique / len(df) > 0.25):
                default_omit.append(col)

    if not candidate_targets:
        raise ValueError("未在資料集中偵測到適合作為迴歸 Target 的連續數值欄位")

    # Pick default target
    preferred = [c for c in candidate_targets if c.lower() in ('profit', 'medv', 'target', 'price', 'y', 'appliances', 'sales')]
    default_target = preferred[0] if preferred else candidate_targets[-1]

    preview_rows = df.head(5).replace({np.nan: None}).to_dict(orient='records')
    # Clean preview rows for JSON serialization
    clean_preview = []
    for r in preview_rows:
        clean_row = {}
        for k, v in r.items():
            if isinstance(v, (np.integer, int)):
                clean_row[k] = int(v)
            elif isinstance(v, (np.floating, float)):
                clean_row[k] = None if np.isnan(v) else round(float(v), 4)
            else:
                clean_row[k] = str(v) if v is not None else ""
        clean_preview.append(clean_row)

    return {
        "row_count": len(df),
        "col_count": len(columns),
        "columns": columns,
        "dtypes": dtypes,
        "missing_counts": missing_counts,
        "candidate_targets": candidate_targets,
        "default_target": default_target,
        "default_omit": default_omit,
        "preview_rows": clean_preview
    }

class TopKSelector(BaseEstimator, TransformerMixin):
    """Select top k features according to ranking method."""
    def __init__(self, method='F-test', k=1):
        self.method = method
        self.k = k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        k_eff = max(1, min(self.k, n_features))

        if self.method == 'RFE':
            self.mask_ = RFE(LinearRegression(), n_features_to_select=k_eff).fit(X, y).support_
            return self

        if self.method == 'F-test':
            vals = f_regression(X, y)[0]
            vals = np.nan_to_num(vals, nan=-np.inf)
        elif self.method == 'Lasso':
            # Adaptive CV folds for small sample size
            inner_cv = min(3, max(2, len(y) // 4))
            lasso = LassoCV(cv=inner_cv, random_state=42, max_iter=10000)
            lasso.fit(X, y)
            vals = np.abs(lasso.coef_)
        elif self.method == 'Random Forest':
            rf = RandomForestRegressor(n_estimators=40, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            vals = rf.feature_importances_
        elif self.method == 'Mutual Information':
            vals = mutual_info_regression(X, y, random_state=42)
        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        order = np.argsort(-vals, kind='stable')
        self.mask_ = np.zeros(n_features, dtype=bool)
        self.mask_[order[:k_eff]] = True
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.mask_]

    def get_support(self):
        return self.mask_

def build_regression_pipeline(X: pd.DataFrame, method: str = None, k: int = None):
    num_cols = X.select_dtypes(include='number').columns.tolist()
    cat_cols = X.select_dtypes(exclude='number').columns.tolist()

    transformers = []
    if num_cols:
        transformers.append(('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ]), num_cols))

    if cat_cols:
        transformers.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
        ]), cat_cols))

    preprocessor = ColumnTransformer(transformers)
    steps = [('prep', preprocessor)]
    if method and k is not None:
        steps.append(('select', TopKSelector(method=method, k=k)))
    steps.append(('ols', LinearRegression()))
    return Pipeline(steps)

def get_pipeline_feature_names(pipeline: Pipeline):
    prep = pipeline.named_steps['prep']
    names = prep.get_feature_names_out()
    if 'select' in pipeline.named_steps:
        return names[pipeline.named_steps['select'].get_support()]
    return names

def calc_durbin_watson(residuals: np.ndarray) -> float:
    res = np.asarray(residuals, dtype=float)
    if len(res) < 2:
        return 2.0
    diff = np.diff(res)
    denom = np.sum(res ** 2)
    return float(np.sum(diff ** 2) / denom) if denom != 0 else 2.0

def calc_vifs(X_arr: np.ndarray, feature_names: list) -> list:
    if X_arr.shape[1] <= 1:
        return [{'feature': feature_names[0], 'vif': 1.0}]
    vifs = []
    for i, name in enumerate(feature_names):
        y_col = X_arr[:, i]
        x_cols = np.delete(X_arr, i, axis=1)
        try:
            ols = LinearRegression().fit(x_cols, y_col)
            r2 = max(0.0, min(0.99999, float(ols.score(x_cols, y_col))))
            vif_val = round(1.0 / (1.0 - r2), 2) if r2 < 0.9999 else 999.0
        except Exception:
            vif_val = None
        vifs.append({'feature': name, 'vif': vif_val})
    return vifs

def train_regression_model(
    df: pd.DataFrame,
    target_col: str,
    omit_list: list = None,
    folds: int = 3,
    quick: bool = True
) -> dict:
    """Train Multiple Linear Regression models and return complete comparison and diagnostics."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    if target_col not in df.columns:
        raise ValueError(f"未找到指定的目標欄位: '{target_col}'")

    # Clean target
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=[target_col])
    if len(df) < 15:
        raise ValueError(f"資料集有效列數僅 {len(df)} 列，至少需要 15 列有效數值進行交叉驗證")

    omit = set(omit_list or [])
    omit.add(target_col)

    # Clean numeric-like strings (e.g. "$1,200", "50%")
    for c in df.columns:
        if c != target_col and not pd.api.types.is_numeric_dtype(df[c]):
            cleaned = pd.to_numeric(df[c].astype(str).str.replace(r'[$,% ]', '', regex=True), errors='coerce')
            if cleaned.notna().sum() >= 0.85 * len(df):
                df[c] = cleaned

    # Auto drop single-value columns, IDs, and high-cardinality non-numeric columns
    for c in df.columns:
        c_low = c.lower()
        if c == target_col:
            continue
        if df[c].nunique(dropna=True) <= 1:
            omit.add(c)
            continue
        if c_low in ('id', 'index', 'unnamed: 0', 'uid', 'uuid', 'row_id'):
            omit.add(c)
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            n_unq = df[c].nunique(dropna=True)
            if any(term in c_low for term in ('date', 'time', 'timestamp', 'datetime')):
                omit.add(c)
            elif n_unq > 12 or (len(df) > 20 and n_unq / len(df) > 0.20):
                omit.add(c)

    feature_cols = [c for c in df.columns if c not in omit]
    if not feature_cols:
        raise ValueError("排除無效與冗餘欄位後，已無可用預測特徵。請檢查排除設定。")

    X = df[feature_cols].copy()
    y = df[target_col].astype(float)

    # 80/20 train/test split with fixed random seed
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, random_state=42)

    # Dev/Calibration split (for empirical 90% error interval calibration)
    Xdev, Xcal, ydev, ycal = train_test_split(Xtr, ytr, test_size=0.20, random_state=123)

    # Determine encoded dimension
    probe_pipe = build_regression_pipeline(Xtr)
    probe_pipe.named_steps['prep'].fit(Xtr)
    all_encoded_names = list(probe_pipe.named_steps['prep'].get_feature_names_out())
    n_encoded = len(all_encoded_names)

    # Bound folds by sample size
    actual_folds = min(folds, max(2, len(Xdev) // 3), 5)
    cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42)

    def cv_score(model):
        scores = -cross_val_score(model, Xdev, ydev, cv=cv, scoring='neg_mean_squared_error')
        rmse_arr = np.sqrt(np.maximum(0, scores))
        return float(np.mean(rmse_arr)), float(np.std(rmse_arr))

    # Baseline evaluation
    base_rmse, base_std = cv_score(build_regression_pipeline(Xtr))
    cv_records = [{
        'method': 'Baseline',
        'k': n_encoded,
        'CV_RMSE': base_rmse,
        'CV_STD': base_std
    }]

    if quick:
        k_candidates = sorted(set([1, 2, 3, 4, 5, 7, 10, 15, 20, n_encoded]))
        k_grid = [k for k in k_candidates if 1 <= k <= n_encoded]
    else:
        k_grid = list(range(1, n_encoded + 1))

    for method in METHODS:
        for k in k_grid:
            rmse, std = cv_score(build_regression_pipeline(Xtr, method, k))
            cv_records.append({
                'method': method,
                'k': k,
                'CV_RMSE': rmse,
                'CV_STD': std
            })

    cv_df = pd.DataFrame(cv_records)

    # Find winning k for each method
    leaders = []
    for method in METHODS:
        subset = cv_df[cv_df['method'] == method].sort_values(['CV_RMSE', 'k'])
        if not subset.empty:
            leaders.append(subset.iloc[0])

    baseline_row = cv_df[cv_df['method'] == 'Baseline'].iloc[0]
    comparison_candidates = [baseline_row] + leaders

    test_summary = []
    fitted_models = {}
    intervals = {}

    for row in comparison_candidates:
        m_name = row['method']
        k_val = int(row['k'])
        pipe = build_regression_pipeline(Xtr, None if m_name == 'Baseline' else m_name, k_val)
        pipe.fit(Xdev, ydev)
        fitted_models[m_name] = pipe

        preds = pipe.predict(Xte)
        sel_features = list(get_pipeline_feature_names(pipe))

        # Empirical 90% absolute residual interval from calibration set
        cal_preds = pipe.predict(Xcal)
        cal_err = np.abs(ycal.to_numpy() - cal_preds)
        half_width = float(np.quantile(cal_err, 0.90)) if len(cal_err) > 0 else float(np.std(cal_err))
        intervals[m_name] = half_width

        n_test = len(yte)
        p_features = len(sel_features)
        mse = float(mean_squared_error(yte, preds))
        r2 = float(r2_score(yte, preds))
        adj_r2 = float(1 - (1 - r2) * (n_test - 1) / (n_test - p_features - 1)) if n_test > p_features + 1 else None

        test_summary.append({
            'method': m_name,
            'k': p_features,
            'CV_RMSE': float(row['CV_RMSE']),
            'CV_STD': float(row['CV_STD']),
            'Test_MAE': float(mean_absolute_error(yte, preds)),
            'Test_MSE': mse,
            'Test_RMSE': float(np.sqrt(mse)),
            'Test_R2': r2,
            'Test_Adjusted_R2': adj_r2,
            'selected_features': sel_features
        })

    # Pick overall winner based on lowest CV_RMSE
    sorted_leaders = sorted(leaders, key=lambda x: (x['CV_RMSE'], x['k']))
    winner_name = sorted_leaders[0]['method']
    winner_k = int(sorted_leaders[0]['k'])
    winner_model = fitted_models[winner_name]

    # Held-out predictions and diagnostics
    y_pred_test = winner_model.predict(Xte)
    residuals = yte.to_numpy() - y_pred_test
    winner_width = intervals[winner_name]

    # Predictions table (sorted by predicted value)
    sort_order = np.argsort(y_pred_test)
    predictions_table = []
    for rank, idx in enumerate(sort_order):
        p_val = float(y_pred_test[idx])
        act_val = float(yte.iloc[idx])
        predictions_table.append({
            'index': rank + 1,
            'actual': round(act_val, 2),
            'predicted': round(p_val, 2),
            'residual': round(float(residuals[idx]), 2),
            'ci_lower': round(p_val - winner_width * 0.5, 2),
            'ci_upper': round(p_val + winner_width * 0.5, 2),
            'pi_lower': round(p_val - winner_width, 2),
            'pi_upper': round(p_val + winner_width, 2)
        })

    # Statistical diagnostics
    dw_val = round(calc_durbin_watson(residuals), 3)
    shapiro_val = None
    if 3 <= len(residuals) <= 5000:
        try:
            shapiro_val = round(float(stats.shapiro(residuals).pvalue), 4)
        except Exception:
            shapiro_val = None

    # Normal Q-Q Plot
    qq_data = []
    if len(residuals) >= 3:
        osm, osr = stats.probplot(residuals, dist="norm")
        qq_data = [{'theoretical': round(float(qt), 3), 'sample': round(float(qs), 3)} for qt, qs in zip(osm[0], osm[1])]

    # Extract coefficients and scaler for simulator
    ols_step = winner_model.named_steps['ols']
    prep_step = winner_model.named_steps['prep']
    winner_sel_names = list(get_pipeline_feature_names(winner_model))

    num_cols = Xtr.select_dtypes(include='number').columns.tolist()
    cat_cols = Xtr.select_dtypes(exclude='number').columns.tolist()

    scaler_means = {}
    scaler_scales = {}
    if num_cols and 'num' in prep_step.named_transformers_:
        scale_step = prep_step.named_transformers_['num'].named_steps['scale']
        for col, m, s in zip(num_cols, scale_step.mean_, scale_step.scale_):
            scaler_means[col] = float(m)
            scaler_scales[col] = float(s)

    cat_categories = {}
    if cat_cols and 'cat' in prep_step.named_transformers_:
        enc_step = prep_step.named_transformers_['cat'].named_steps['encode']
        for col, cats in zip(cat_cols, enc_step.categories_):
            cat_categories[col] = [str(c) for c in cats[1:]]

    coefficients_list = [
        {'feature': name, 'coefficient': round(float(coef), 4)}
        for name, coef in zip(winner_sel_names, ols_step.coef_)
    ]

    # Feature specs for dynamic simulator sliders
    feature_specs = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            s = df[col].dropna()
            min_v = float(s.min())
            max_v = float(s.max())
            med_v = float(s.median())
            step_v = round((max_v - min_v) / 100, 2) if max_v > min_v else 1.0
            feature_specs.append({
                'name': col,
                'type': 'numeric',
                'min': min_v,
                'max': max_v,
                'mean': round(float(s.mean()), 2),
                'default': round(med_v, 2),
                'step': step_v if step_v > 0 else 0.01
            })
        else:
            cats = sorted(df[col].dropna().unique().tolist())
            feature_specs.append({
                'name': col,
                'type': 'categorical',
                'options': [str(c) for c in cats],
                'default': str(cats[0]) if cats else ""
            })

    # VIF multicollinearity calculation on transformed training data
    Xdev_trans = prep_step.transform(Xdev)
    if 'select' in winner_model.named_steps:
        Xdev_trans = winner_model.named_steps['select'].transform(Xdev_trans)
    vif_records = calc_vifs(Xdev_trans, winner_sel_names)

    winner_test_metric = next(item for item in test_summary if item['method'] == winner_name)

    return {
        'dataset_id': f"custom_{int(time.time() * 1000)}",
        'title': f"自訂資料集 — {target_col} 迴歸模型",
        'unit': "單位",
        'target': target_col,
        'sample_size': len(df),
        'train_size': len(Xtr),
        'test_size': len(Xte),
        'raw_features_count': len(X.columns),
        'encoded_features_count': n_encoded,
        'raw_features': list(X.columns),
        'encoded_features': all_encoded_names,
        'omitted_features': list(omit),
        'best_model': winner_name,
        'best_k': winner_k,
        'baseline_cv_rmse': round(base_rmse, 3),
        'winner_cv_rmse': round(float(sorted_leaders[0]['CV_RMSE']), 3),
        'winner_test_rmse': round(winner_test_metric['Test_RMSE'], 3),
        'winner_test_r2': round(winner_test_metric['Test_R2'], 4),
        'winner_test_mae': round(winner_test_metric['Test_MAE'], 3),
        'cv_summary': cv_records,
        'test_summary': test_summary,
        'winner_selected_features': winner_sel_names,
        'coefficients': coefficients_list,
        'intercept': round(float(ols_step.intercept_), 4),
        'scaler_means': scaler_means,
        'scaler_scales': scaler_scales,
        'cat_categories': cat_categories,
        'feature_specs': feature_specs,
        'predictions': predictions_table,
        'qq_plot': qq_data,
        'vif': vif_records,
        'diagnostics': {
            'durbin_watson': dw_val,
            'shapiro_wilk_p': shapiro_val,
            'note': '診斷數值於獨立 20% 測試集計算；區間為校正集經驗 90% 殘差誤差帶 (Empirical Error Band)。'
        },
        'interval_half_width': round(winner_width, 2)
    }
