import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os
import logging
import sys
import config
from prediction import predict_vad_trajectory
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_detailed_timeseries_visualizations(df, output_dir):
    logger.info("Creating detailed time series visualizations")

    # MAE Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['mae_overall'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['mae_overall'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {df["mae_overall"].median():.4f}')
    plt.axvline(df['mae_overall'].mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {df["mae_overall"].mean():.4f}')
    plt.xlabel('Overall MAE')
    plt.ylabel('Number of Words')
    plt.title('Distribution of MAE Across Words')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mae_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved MAE distribution to {output_path}")

    # RMSE Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['rmse_overall'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.axvline(df['rmse_overall'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {df["rmse_overall"].median():.4f}')
    plt.axvline(df['rmse_overall'].mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {df["rmse_overall"].mean():.4f}')
    plt.xlabel('Overall RMSE')
    plt.ylabel('Number of Words')
    plt.title('Distribution of RMSE Across Words')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rmse_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved RMSE distribution to {output_path}")

    # MAE by VAD Dimension
    plt.figure(figsize=(10, 6))
    mae_data = [df['mae_v'], df['mae_a'], df['mae_d']]
    bp1 = plt.boxplot(mae_data, patch_artist=True, tick_labels=['Valence', 'Arousal', 'Dominance'])
    colors = ['lightblue', 'lightgreen', 'lightpink']
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.ylabel('MAE')
    plt.title('MAE Distribution by VAD Dimension')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mae_by_dimension.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved MAE by dimension to {output_path}")

    # RMSE by VAD Dimension
    plt.figure(figsize=(10, 6))
    rmse_data = [df['rmse_v'], df['rmse_a'], df['rmse_d']]
    bp2 = plt.boxplot(rmse_data, patch_artist=True, tick_labels=['Valence', 'Arousal', 'Dominance'])
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.ylabel('RMSE')
    plt.title('RMSE Distribution by VAD Dimension')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rmse_by_dimension.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved RMSE by dimension to {output_path}")

    # MAE Performance Ranking
    plt.figure(figsize=(10, 6))
    sorted_mae = df.sort_values('mae_overall').reset_index(drop=True)
    sorted_mae['rank_percentile'] = range(1, len(sorted_mae) + 1)
    sorted_mae['rank_percentile'] = sorted_mae['rank_percentile'] / len(sorted_mae) * 100
    plt.plot(sorted_mae['rank_percentile'], sorted_mae['mae_overall'], linewidth=2, color='navy')
    plt.fill_between(sorted_mae['rank_percentile'], sorted_mae['mae_overall'], alpha=0.3, color='lightblue')
    plt.xlabel('Word Rank Percentile')
    plt.ylabel('MAE')
    plt.title('MAE Performance Ranking')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mae_ranking.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved MAE ranking to {output_path}")

    # RMSE Performance Ranking
    plt.figure(figsize=(10, 6))
    sorted_rmse = df.sort_values('rmse_overall').reset_index(drop=True)
    sorted_rmse['rank_percentile'] = range(1, len(sorted_rmse) + 1)
    sorted_rmse['rank_percentile'] = sorted_rmse['rank_percentile'] / len(sorted_rmse) * 100
    plt.plot(sorted_rmse['rank_percentile'], sorted_rmse['rmse_overall'], linewidth=2, color='darkred')
    plt.fill_between(sorted_rmse['rank_percentile'], sorted_rmse['rmse_overall'], alpha=0.3, color='lightcoral')
    plt.xlabel('Word Rank Percentile')
    plt.ylabel('RMSE')
    plt.title('RMSE Performance Ranking')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rmse_ranking.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved RMSE ranking to {output_path}")


def _plot_word_case_study_full_vad(ax, row, word, config, title_suffix=""):
    if word in config.full_vad_data and 'temporal_vad' in config.full_vad_data[word]:
        word_data = config.full_vad_data[word]['temporal_vad']
        years = word_data['x']
        v_vals = word_data['v']
        a_vals = word_data['a']
        d_vals = word_data['d']

        valid_data = []
        for i, year in enumerate(years):
            if (year is not None and
                    v_vals[i] is not None and
                    a_vals[i] is not None and
                    d_vals[i] is not None):
                valid_data.append((year, v_vals[i], a_vals[i], d_vals[i]))

        if valid_data:
            valid_years, valid_v, valid_a, valid_d = zip(*valid_data)

            ax.plot(valid_years, valid_v, 'b-', linewidth=2, label='Valence', alpha=0.8)
            ax.plot(valid_years, valid_a, 'g-', linewidth=2, label='Arousal', alpha=0.8)
            ax.plot(valid_years, valid_d, 'r-', linewidth=2, label='Dominance', alpha=0.8)

            ax.axvline(x=1980, color='black', linestyle='--', alpha=0.7, label='Prediction Point')

            ax.scatter([2010], [row['actual_v']], color='blue', s=100, label='Actual V',
                       marker='o', edgecolor='black', linewidth=1)
            ax.scatter([2010], [row['pred_v']], color='blue', s=100, label='Pred V',
                       marker='s', edgecolor='black', linewidth=1)

            ax.scatter([2010], [row['actual_a']], color='green', s=100, label='Actual A',
                       marker='o', edgecolor='black', linewidth=1)
            ax.scatter([2010], [row['pred_a']], color='green', s=100, label='Pred A',
                       marker='s', edgecolor='black', linewidth=1)

            ax.scatter([2010], [row['actual_d']], color='red', s=100, label='Actual D',
                       marker='o', edgecolor='black', linewidth=1)
            ax.scatter([2010], [row['pred_d']], color='red', s=100, label='Pred D',
                       marker='s', edgecolor='black', linewidth=1)

            ax.set_xlim(min(valid_years) - 5, max(valid_years) + 5)
            ax.set_ylim(-1.1, 1.1)
        else:
            ax.text(0.5, 0.5, f'No valid time series data for "{word}"',
                    ha='center', va='center', transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, f'Data not found for "{word}"',
                ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Year')
    ax.set_ylabel('VAD Values')
    ax.grid(True, alpha=0.3)


def create_best_word_case_studies(df, output_dir):
    logger.info("Creating best word case studies")
    for metric, title_prefix in [('mae_overall', 'MAE'), ('rmse_overall', 'RMSE')]:
        best_words = df.nsmallest(6, metric)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Case Studies: 6 Best Performing Words (Lowest {title_prefix})',
                     fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for i, (_, row) in enumerate(best_words.iterrows()):
            ax = axes[i]
            _plot_word_case_study_full_vad(ax, row, row['word'], config)

            mae_val = row['mae_overall']
            rmse_val = row['rmse_overall']
            ax.set_title(f"'{row['word']}'\nMAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}",
                         fontsize=10)

            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        for i in range(len(best_words), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = os.path.join(output_dir, f'best_word_case_studies_{metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved best case studies ({title_prefix}) to {output_path}")


def create_worst_word_case_studies(df, output_dir):
    logger.info("Creating worst word case studies")
    for metric, title_prefix in [('mae_overall', 'MAE'), ('rmse_overall', 'RMSE')]:
        worst_words = df.nlargest(6, metric)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Case Studies: 6 Worst Performing Words (Highest {title_prefix})',
                     fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for i, (_, row) in enumerate(worst_words.iterrows()):
            ax = axes[i]
            _plot_word_case_study_full_vad(ax, row, row['word'], config)

            mae_val = row['mae_overall']
            rmse_val = row['rmse_overall']
            ax.set_title(f"'{row['word']}'\nMAE: {mae_val:.4f}, RMSE: {rmse_val:.4f}",
                         fontsize=10)

            if i == 0:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

        for i in range(len(worst_words), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        output_path = os.path.join(output_dir, f'worst_word_case_studies_{metric}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved worst case studies ({title_prefix}) to {output_path}")


def create_performance_heatmap(df, output_dir):
    logger.info("Creating performance heatmaps")

    def get_word_characteristics(word, row):
        return {
            'length_category': 'Short' if len(word) <= 5 else 'Medium' if len(word) <= 8 else 'Long',
            'mae_category': 'Excellent' if row['mae_overall'] <= 0.01 else 'Good' if row['mae_overall'] <= 0.05 else 'Poor',
            'rmse_category': 'Excellent' if row['rmse_overall'] <= 0.015 else 'Good' if row['rmse_overall'] <= 0.075 else 'Poor',
            'variance_category': 'Low' if row['vad_variance'] <= 0.01 else 'Medium' if row['vad_variance'] <= 0.05 else 'High'
        }

    char_data = [get_word_characteristics(row['word'], row) for _, row in df.iterrows()]
    char_df = pd.DataFrame(char_data)
    char_df['mae'] = df['mae_overall']
    char_df['rmse'] = df['rmse_overall']

    # MAE heatmap
    plt.figure(figsize=(10, 6))
    heatmap_mae = char_df.groupby(['length_category', 'variance_category'])['mae'].mean().unstack()
    sns.heatmap(heatmap_mae, annot=True, fmt='.4f', cmap='RdYlBu_r', center=0.02)
    plt.title('Average MAE by Word Length and VAD Variance')
    plt.xlabel('VAD Variance Category')
    plt.ylabel('Word Length Category')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mae_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved MAE heatmap to {output_path}")

    # RMSE heatmap
    plt.figure(figsize=(10, 6))
    heatmap_rmse = char_df.groupby(['length_category', 'variance_category'])['rmse'].mean().unstack()
    sns.heatmap(heatmap_rmse, annot=True, fmt='.4f', cmap='RdYlBu_r', center=0.03)
    plt.title('Average RMSE by Word Length and VAD Variance')
    plt.xlabel('VAD Variance Category')
    plt.ylabel('Word Length Category')
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rmse_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved RMSE heatmap to {output_path}")


def create_mae_rmse_comparison(df, output_dir):
    logger.info("Creating MAE vs RMSE comparison visualizations")

    # Scatter plot: MAE vs RMSE
    plt.figure(figsize=(10, 8))
    plt.scatter(df['mae_overall'], df['rmse_overall'], alpha=0.6, color='blue')
    plt.plot([0, df['mae_overall'].max()], [0, df['mae_overall'].max()], 'r--', alpha=0.8, label='MAE=RMSE line')
    plt.xlabel('MAE')
    plt.ylabel('RMSE')
    plt.title('MAE vs RMSE Correlation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    corr = df['mae_overall'].corr(df['rmse_overall'])
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mae_rmse_correlation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved MAE vs RMSE correlation to {output_path}")

    # Histogram of MAE/RMSE ratio
    plt.figure(figsize=(10, 6))
    ratio = df['rmse_overall'] / df['mae_overall']
    plt.hist(ratio, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(ratio.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ratio.mean():.2f}')
    plt.xlabel('RMSE/MAE Ratio')
    plt.ylabel('Number of Words')
    plt.title('Distribution of RMSE/MAE Ratio')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'rmse_mae_ratio.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved RMSE/MAE ratio to {output_path}")

    # Dimension comparison
    plt.figure(figsize=(10, 6))
    dimensions = ['v', 'a', 'd']
    mae_means = [df[f'mae_{dim}'].mean() for dim in dimensions]
    rmse_means = [df[f'rmse_{dim}'].mean() for dim in dimensions]
    x = np.arange(len(dimensions))
    width = 0.35
    plt.bar(x - width / 2, mae_means, width, label='MAE', alpha=0.8, color='skyblue')
    plt.bar(x + width / 2, rmse_means, width, label='RMSE', alpha=0.8, color='lightcoral')
    plt.xlabel('VAD Dimension')
    plt.ylabel('Error Value')
    plt.title('Mean MAE vs RMSE by Dimension')
    plt.xticks(x, ['Valence', 'Arousal', 'Dominance'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'mae_rmse_by_dimension.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved MAE vs RMSE by dimension to {output_path}")

    # Performance categories comparison
    plt.figure(figsize=(10, 6))
    mae_excellent = (df['mae_overall'] <= 0.01).sum()
    mae_good = ((df['mae_overall'] > 0.01) & (df['mae_overall'] <= 0.05)).sum()
    mae_poor = (df['mae_overall'] > 0.05).sum()
    rmse_excellent = (df['rmse_overall'] <= 0.015).sum()
    rmse_good = ((df['rmse_overall'] > 0.015) & (df['rmse_overall'] <= 0.075)).sum()
    rmse_poor = (df['rmse_overall'] > 0.075).sum()
    categories = ['Excellent', 'Good', 'Poor']
    mae_counts = [mae_excellent, mae_good, mae_poor]
    rmse_counts = [rmse_excellent, rmse_good, rmse_poor]
    x = np.arange(len(categories))
    width = 0.35
    plt.bar(x - width / 2, mae_counts, width, label='MAE', alpha=0.8, color='skyblue')
    plt.bar(x + width / 2, rmse_counts, width, label='RMSE', alpha=0.8, color='lightcoral')
    plt.xlabel('Performance Category')
    plt.ylabel('Number of Words')
    plt.title('Performance Categories: MAE vs RMSE')
    plt.xticks(x, categories)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'performance_categories.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved performance categories to {output_path}")


def create_summary_table(df, output_dir):
    logger.info("Creating summary table")
    summary_path = os.path.join(output_dir, 'performance_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("EMOTRACKER TIME SERIES PREDICTION PERFORMANCE SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total words analyzed: {len(df):,}\n")
        f.write("Prediction task: 1980 -> 2010 (30-year forecasting)\n\n")

        f.write("OVERALL PERFORMANCE DISTRIBUTION:\n")
        f.write("-" * 40 + "\n")

        f.write("MAE STATISTICS:\n")
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        for q in quantiles:
            f.write(f"{q * 100:4.0f}th percentile: {df['mae_overall'].quantile(q):.6f}\n")
        f.write(f"Mean: {df['mae_overall'].mean():.6f}\n")
        f.write(f"Std:  {df['mae_overall'].std():.6f}\n\n")

        f.write("RMSE STATISTICS:\n")
        for q in quantiles:
            f.write(f"{q * 100:4.0f}th percentile: {df['rmse_overall'].quantile(q):.6f}\n")
        f.write(f"Mean: {df['rmse_overall'].mean():.6f}\n")
        f.write(f"Std:  {df['rmse_overall'].std():.6f}\n\n")

        f.write("PERFORMANCE CATEGORIES:\n")
        f.write("-" * 40 + "\n")

        mae_excellent = (df['mae_overall'] <= 0.01).sum()
        mae_good = ((df['mae_overall'] > 0.01) & (df['mae_overall'] <= 0.05)).sum()
        mae_fair = ((df['mae_overall'] > 0.05) & (df['mae_overall'] <= 0.1)).sum()
        mae_poor = (df['mae_overall'] > 0.1).sum()
        total = len(df)

        f.write("MAE Categories:\n")
        f.write(f"Excellent (MAE <= 0.01):     {mae_excellent:5d} words ({mae_excellent / total:.1%})\n")
        f.write(f"Good (0.01 < MAE <= 0.05):   {mae_good:5d} words ({mae_good / total:.1%})\n")
        f.write(f"Fair (0.05 < MAE <= 0.10):   {mae_fair:5d} words ({mae_fair / total:.1%})\n")
        f.write(f"Poor (MAE > 0.10):           {mae_poor:5d} words ({mae_poor / total:.1%})\n\n")

        rmse_excellent = (df['rmse_overall'] <= 0.015).sum()
        rmse_good = ((df['rmse_overall'] > 0.015) & (df['rmse_overall'] <= 0.075)).sum()
        rmse_fair = ((df['rmse_overall'] > 0.075) & (df['rmse_overall'] <= 0.15)).sum()
        rmse_poor = (df['rmse_overall'] > 0.15).sum()

        f.write("RMSE Categories:\n")
        f.write(f"Excellent (RMSE <= 0.015):   {rmse_excellent:5d} words ({rmse_excellent / total:.1%})\n")
        f.write(f"Good (0.015 < RMSE <= 0.075): {rmse_good:5d} words ({rmse_good / total:.1%})\n")
        f.write(f"Fair (0.075 < RMSE <= 0.15):  {rmse_fair:5d} words ({rmse_fair / total:.1%})\n")
        f.write(f"Poor (RMSE > 0.15):          {rmse_poor:5d} words ({rmse_poor / total:.1%})\n\n")

        f.write("DIMENSION-SPECIFIC PERFORMANCE:\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"{'Dimension':<10} {'Correlation':<12} {'R-squared':<10} {'Mean MAE':<10} {'Mean RMSE':<10} "
            f"{'Med MAE':<10} {'Med RMSE':<10}\n")
        f.write("-" * 70 + "\n")

        for dim_code, dim_name in [('v', 'Valence'), ('a', 'Arousal'), ('d', 'Dominance')]:
            corr = df[f'actual_{dim_code}'].corr(df[f'pred_{dim_code}'])
            mean_mae = df[f'mae_{dim_code}'].mean()
            mean_rmse = df[f'rmse_{dim_code}'].mean()
            median_mae = df[f'mae_{dim_code}'].median()
            median_rmse = df[f'rmse_{dim_code}'].median()
            f.write(
                f"{dim_name:<10} {corr:<12.4f} {corr ** 2:<10.4f} {mean_mae:<10.6f} {mean_rmse:<10.6f} "
                f"{median_mae:<10.6f} {median_rmse:<10.6f}\n")

        f.write(f"\nTOP 10 BEST PERFORMERS (MAE):\n")
        f.write("-" * 40 + "\n")
        best_mae = df.nsmallest(10, 'mae_overall')
        for i, (_, row) in enumerate(best_mae.iterrows(), 1):
            f.write(f"{i:2d}. {row['word']:<15} MAE: {row['mae_overall']:.6f} RMSE: {row['rmse_overall']:.6f}\n")

        f.write(f"\nTOP 10 BEST PERFORMERS (RMSE):\n")
        f.write("-" * 40 + "\n")
        best_rmse = df.nsmallest(10, 'rmse_overall')
        for i, (_, row) in enumerate(best_rmse.iterrows(), 1):
            f.write(f"{i:2d}. {row['word']:<15} MAE: {row['mae_overall']:.6f} RMSE: {row['rmse_overall']:.6f}\n")

        f.write(f"\nTOP 10 WORST PERFORMERS (MAE):\n")
        f.write("-" * 40 + "\n")
        worst_mae = df.nlargest(10, 'mae_overall')
        for i, (_, row) in enumerate(worst_mae.iterrows(), 1):
            f.write(f"{i:2d}. {row['word']:<15} MAE: {row['mae_overall']:.6f} RMSE: {row['rmse_overall']:.6f}\n")

        f.write(f"\nTOP 10 WORST PERFORMERS (RMSE):\n")
        f.write("-" * 40 + "\n")
        worst_rmse = df.nlargest(10, 'rmse_overall')
        for i, (_, row) in enumerate(worst_rmse.iterrows(), 1):
            f.write(f"{i:2d}. {row['word']:<15} MAE: {row['mae_overall']:.6f} RMSE: {row['rmse_overall']:.6f}\n")

    logger.info(f"Saved summary to {summary_path}")


def analyze_word_performance(output_dir="forecasting_evaluation_results"):
    if not config.resources_loaded_pytorch:
        logger.error("PyTorch resources not loaded - cannot proceed with analysis")
        return False

    logger.info(f"Starting word performance analysis - output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    error_log_path = os.path.join(output_dir, 'error_log.txt')
    test_words = [word for word, data in config.full_vad_data.items() if 'temporal_vad' in data]
    predict_from_year, predict_to_year = 1980, 2010
    results, failed_log = [], []

    logger.info(f"Found {len(test_words)} words with temporal VAD data")
    logger.info(f"Prediction task: {predict_from_year} -> {predict_to_year}")

    batch_size = 100
    total_batches = (len(test_words) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(test_words), batch_size):
        current_batch = batch_idx // batch_size + 1
        batch_words = test_words[batch_idx:batch_idx + batch_size]
        logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch_words)} words)")

        for word in batch_words:
            try:
                word_data = config.full_vad_data[word]['temporal_vad']
                years, v, a, d = word_data['x'], word_data['v'], word_data['a'], word_data['d']

                actuals = next(((v[j], a[j], d[j]) for j, yr in enumerate(years) if
                                yr == predict_to_year and all(val is not None for val in [v[j], a[j], d[j]])), None)
                if actuals is None:
                    failed_log.append(f"WORD: {word} - No actual values for {predict_to_year}")
                    continue

                predictions = predict_vad_trajectory(word, predict_from_year, predict_to_year)
                if not predictions:
                    failed_log.append(f"WORD: {word} - No predictions returned")
                    continue

                pred = predictions[0]
                preds = [pred['v'], pred['a'], pred['d']]
                valid_vad = [val for sub in zip(v, a, d) for val in sub if val is not None]

                mae_overall = mean_absolute_error(actuals, preds)
                rmse_overall = math.sqrt(mean_squared_error(actuals, preds))

                mae_v = abs(actuals[0] - preds[0])
                mae_a = abs(actuals[1] - preds[1])
                mae_d = abs(actuals[2] - preds[2])

                rmse_v = math.sqrt((actuals[0] - preds[0]) ** 2)
                rmse_a = math.sqrt((actuals[1] - preds[1]) ** 2)
                rmse_d = math.sqrt((actuals[2] - preds[2]) ** 2)

                results.append({
                    'word': word,
                    'actual_v': actuals[0], 'actual_a': actuals[1], 'actual_d': actuals[2],
                    'pred_v': preds[0], 'pred_a': preds[1], 'pred_d': preds[2],
                    'mae_overall': mae_overall,
                    'rmse_overall': rmse_overall,
                    'mae_v': mae_v, 'mae_a': mae_a, 'mae_d': mae_d,
                    'rmse_v': rmse_v, 'rmse_a': rmse_a, 'rmse_d': rmse_d,
                    'vad_variance': np.var(valid_vad) if len(valid_vad) > 1 else 0.0,
                })

            except Exception as e:
                failed_log.append(f"WORD: {word} - EXCEPTION: {e}")

    with open(error_log_path, 'w') as f:
        f.write("Words that failed prediction:\n" + "=" * 50 + "\n\n" + "\n".join(failed_log))

    successful = len(results)
    failed_count = len(failed_log)
    total_processed = successful + failed_count

    if total_processed > 0:
        success_rate = successful / total_processed * 100
        logger.info(f"Analysis complete: {successful} successful, {failed_count} failed ({success_rate:.1f}% success)")
        logger.info(f"Error log saved to {error_log_path}")

    if not results:
        logger.error("No successful predictions to analyze")
        return False

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'word_performance_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Raw results saved to {csv_path}")

    create_analysis_report(df, output_dir)
    create_individual_visualizations(df, output_dir)
    return True


def create_analysis_report(df, output_dir):
    logger.info("Creating detailed analysis report")
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("WORD PERFORMANCE ANALYSIS REPORT\n" + "=" * 40 + "\n\n")
        f.write(f"Total words analyzed: {len(df)}\n\n")

        f.write("OVERALL MAE PERFORMANCE:\n")
        f.write(f"{df['mae_overall'].describe().to_string()}\n\n")

        f.write("OVERALL RMSE PERFORMANCE:\n")
        f.write(f"{df['rmse_overall'].describe().to_string()}\n\n")

        f.write("TOP 10 BEST PERFORMING WORDS (MAE):\n")
        f.write(
            df.nsmallest(10, 'mae_overall')[['word', 'mae_overall', 'rmse_overall']].to_string(index=False) + "\n\n")

        f.write("TOP 10 BEST PERFORMING WORDS (RMSE):\n")
        f.write(
            df.nsmallest(10, 'rmse_overall')[['word', 'mae_overall', 'rmse_overall']].to_string(index=False) + "\n\n")

        f.write("TOP 10 WORST PERFORMING WORDS (MAE):\n")
        f.write(df.nlargest(10, 'mae_overall')[['word', 'mae_overall', 'rmse_overall']].to_string(index=False) + "\n\n")

        f.write("TOP 10 WORST PERFORMING WORDS (RMSE):\n")
        f.write(
            df.nlargest(10, 'rmse_overall')[['word', 'mae_overall', 'rmse_overall']].to_string(index=False) + "\n\n")

    logger.info(f"Analysis report saved to {report_path}")


def create_individual_visualizations(df, output_dir):
    logger.info("Creating individual visualization files")
    plt.style.use('default')

    # MAE Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['mae_overall'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Overall MAE Across All Words')
    plt.xlabel('Overall MAE')
    plt.tight_layout()
    output_path = os.path.join(output_dir, '1_mae_distribution_simple.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug(f"Saved MAE distribution to {output_path}")

    # RMSE Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['rmse_overall'], bins=30, kde=True, color='lightcoral')
    plt.title('Distribution of Overall RMSE Across All Words')
    plt.xlabel('Overall RMSE')
    plt.tight_layout()
    output_path = os.path.join(output_dir, '1_rmse_distribution_simple.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug(f"Saved RMSE distribution to {output_path}")

    # MAE by Dimension
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['mae_v', 'mae_a', 'mae_d']], notch=True, patch_artist=True)
    plt.title('MAE Distribution by VAD Dimension')
    plt.ylabel('MAE')
    plt.gca().set_xticklabels(['Valence', 'Arousal', 'Dominance'])
    plt.tight_layout()
    output_path = os.path.join(output_dir, '2_mae_dimension_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug(f"Saved MAE dimension comparison to {output_path}")

    # RMSE by Dimension
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['rmse_v', 'rmse_a', 'rmse_d']], notch=True, patch_artist=True)
    plt.title('RMSE Distribution by VAD Dimension')
    plt.ylabel('RMSE')
    plt.gca().set_xticklabels(['Valence', 'Arousal', 'Dominance'])
    plt.tight_layout()
    output_path = os.path.join(output_dir, '2_rmse_dimension_comparison.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug(f"Saved RMSE dimension comparison to {output_path}")

    # Best Performers - MAE
    plt.figure(figsize=(10, 8))
    best_mae_df = df.nsmallest(15, 'mae_overall')
    sns.barplot(x='mae_overall', y='word', data=best_mae_df, color='g', alpha=0.8)
    plt.title('Top 15 Best Performing Words (Lowest MAE)')
    plt.xlabel('Overall MAE')
    plt.ylabel('Word')
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3a_best_performers_mae.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug(f"Saved best MAE performers to {output_path}")

    # Best Performers - RMSE
    plt.figure(figsize=(10, 8))
    best_rmse_df = df.nsmallest(15, 'rmse_overall')
    sns.barplot(x='rmse_overall', y='word', data=best_rmse_df, color='darkgreen', alpha=0.8)
    plt.title('Top 15 Best Performing Words (Lowest RMSE)')
    plt.xlabel('Overall RMSE')
    plt.ylabel('Word')
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3b_best_performers_rmse.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug(f"Saved best RMSE performers to {output_path}")

    # Worst Performers - MAE
    plt.figure(figsize=(10, 8))
    worst_mae_df = df.nlargest(15, 'mae_overall')
    sns.barplot(x='mae_overall', y='word', data=worst_mae_df, color='r', alpha=0.8)
    plt.title('Top 15 Worst Performing Words (Highest MAE)')
    plt.xlabel('Overall MAE')
    plt.ylabel('Word')
    plt.tight_layout()
    output_path = os.path.join(output_dir, '4a_worst_performers_mae.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug(f"Saved worst MAE performers to {output_path}")

    # Worst Performers - RMSE
    plt.figure(figsize=(10, 8))
    worst_rmse_df = df.nlargest(15, 'rmse_overall')
    sns.barplot(x='rmse_overall', y='word', data=worst_rmse_df, color='darkred', alpha=0.8)
    plt.title('Top 15 Worst Performing Words (Highest RMSE)')
    plt.xlabel('Overall RMSE')
    plt.ylabel('Word')
    plt.tight_layout()
    output_path = os.path.join(output_dir, '4b_worst_performers_rmse.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.debug(f"Saved worst RMSE performers to {output_path}")


def enhanced_analyze_word_performance(output_dir="forecasting_evaluation_results"):
    logger.info("Starting enhanced word performance analysis")

    if analyze_word_performance(output_dir):
        results_path = os.path.join(output_dir, 'word_performance_results.csv')
        if os.path.exists(results_path):
            logger.info("Loading results and creating enhanced visualizations")
            df = pd.read_csv(results_path)

            create_detailed_timeseries_visualizations(df, output_dir)
            create_best_word_case_studies(df, output_dir)
            create_worst_word_case_studies(df, output_dir)
            create_performance_heatmap(df, output_dir)
            create_mae_rmse_comparison(df, output_dir)
            create_summary_table(df, output_dir)

            logger.info("analysis completed successfully")
            return True
        else:
            logger.error(f"Results file not found: {results_path}")
    else:
        logger.error("Initial analysis failed")
    return False


def main():
    if not os.path.exists('config.py'):
        logger.error("config.py not found - run script from api/ directory")
        return

    logger.info("Starting EmoTracker word performance analysis")

    if enhanced_analyze_word_performance():
        logger.info("Analysis completed successfully - check 'forecasting_evaluation_results' directory")
    else:
        logger.error("Analysis failed - check error messages and log file")


if __name__ == "__main__":
    main()