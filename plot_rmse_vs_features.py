"""繪製特徵數量 (Number of Features) 與交叉驗證 RMSE 的關係圖

讀取 results/cv_summary.csv 並繪製：
- 5 種特徵篩選法在不同特徵數 k (1~30) 下的 CV RMSE 變化趨勢
- 全特徵 Baseline 基準虛線
- 標註全域最佳點 (Best Model: k=8, F-test)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_rmse_vs_features(csv_path='results/cv_summary.csv', output_path='results/feature_count_vs_rmse.png'):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"找不到檔案: {csv_path}，請先執行 appliances_crispdm.py 生成成果！")

    df = pd.read_csv(csv_file)
    
    # 分離 Baseline 與各特徵選擇方法
    baseline_df = df[df['method'] == 'All Features Baseline']
    baseline_rmse = float(baseline_df['CV_RMSE'].iloc[0])
    methods_df = df[df['method'] != 'All Features Baseline']

    # 找出最佳點（最低 CV RMSE）
    best_row = methods_df.sort_values('CV_RMSE').iloc[0]
    best_method = best_row['method']
    best_k = int(best_row['k'])
    best_rmse = float(best_row['CV_RMSE'])

    # 設定配色與標記樣式
    color_map = {
        'F-test': '#1f77b4',              # 藍色
        'Random Forest': '#2ca02c',       # 綠色
        'Lasso': '#ff7f0e',               # 橘色
        'RFE': '#d62728',                 # 紅色
        'Mutual Information': '#9467bd'   # 紫色
    }
    marker_map = {
        'F-test': 'o',
        'Random Forest': 's',
        'Lasso': '^',
        'RFE': 'D',
        'Mutual Information': 'v'
    }

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 繪製各方法折線
    for method in ['F-test', 'Random Forest', 'Lasso', 'RFE', 'Mutual Information']:
        sub = methods_df[methods_df['method'] == method].sort_values('k')
        if sub.empty:
            continue
        ax.plot(
            sub['k'], sub['CV_RMSE'],
            label=method,
            color=color_map.get(method, None),
            marker=marker_map.get(method, '.'),
            markersize=5,
            linewidth=1.8,
            alpha=0.9
        )

    # 繪製 Baseline 基準線
    ax.axhline(
        y=baseline_rmse,
        color='#333333',
        linestyle='--',
        linewidth=1.5,
        label=f'All Features Baseline (k=30, RMSE={baseline_rmse:.2f})'
    )

    # 標註最佳點 (k=8, F-test)
    ax.scatter([best_k], [best_rmse], color='crimson', s=120, zorder=5, edgecolors='black', linewidth=1.5)
    ax.annotate(
        f'Best: {best_method}\n$k={best_k}$, RMSE={best_rmse:.2f} Wh',
        xy=(best_k, best_rmse),
        xytext=(best_k + 2.5, best_rmse + 0.35),
        arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff2cc', edgecolor='#d6b656', alpha=0.95)
    )

    # 設定圖表標籤與格線
    ax.set_title('Appliances Energy Prediction', fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Number of Selected Features ($k$)', fontsize=11, labelpad=8)
    ax.set_ylabel('Mean Time-Series CV RMSE (Wh)', fontsize=11, labelpad=8)
    ax.set_xticks(range(1, 31, 2))
    ax.set_xlim(0.5, 30.5)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9.5, loc='lower right')

    plt.tight_layout()
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f'成功繪製並儲存圖表至: {output_file.resolve()}')

if __name__ == '__main__':
    plot_rmse_vs_features()
