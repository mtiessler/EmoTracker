import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os
import logging
import config
from prediction import predict_vad_trajectory
import seaborn as sns
from scipy import stats

# todo change printing by logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_detailed_timeseries_visualizations(df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Distribution of errors
    axes[0, 0].hist(df['mae_overall'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].axvline(df['mae_overall'].median(), color='red', linestyle='--', linewidth=2,
                       label=f'Median: {df["mae_overall"].median():.4f}')
    axes[0, 0].axvline(df['mae_overall'].mean(), color='orange', linestyle='--', linewidth=2,
                       label=f'Mean: {df["mae_overall"].mean():.4f}')
    axes[0, 0].set_xlabel('Overall MAE')
    axes[0, 0].set_ylabel('Number of Words')
    axes[0, 0].set_title('Distribution of Time Series Prediction Errors')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Log scale distribution
    axes[0, 1].hist(df['mae_overall'], bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    axes[0, 1].set_yscale('log')
    axes[0, 1].set_xlabel('Overall MAE')
    axes[0, 1].set_ylabel('Number of Words (log scale)')
    axes[0, 1].set_title('Prediction Error Distribution (Log Scale)')
    axes[0, 1].grid(True, alpha=0.3)

    # Box plot for summary statistics
    box_data = [df['mae_overall']]
    bp = axes[1, 0].boxplot(box_data, patch_artist=True, tick_labels=['All Words'])
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][0].set_alpha(0.7)
    axes[1, 0].set_ylabel('Overall MAE')
    axes[1, 0].set_title('Error Distribution Summary Statistics')
    axes[1, 0].grid(True, alpha=0.3)

    q1, median, q3 = df['mae_overall'].quantile([0.25, 0.5, 0.75])
    axes[1, 0].text(1.1, q1, f'Q1: {q1:.4f}', ha='left')
    axes[1, 0].text(1.1, median, f'Median: {median:.4f}', ha='left')
    axes[1, 0].text(1.1, q3, f'Q3: {q3:.4f}', ha='left')

    # Q-Q plot
    stats.probplot(df['mae_overall'], dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Are Errors Normally Distributed?')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'detailed_error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Performance ranking analysis
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    sorted_df = df.sort_values('mae_overall').reset_index(drop=True)
    sorted_df['rank'] = range(1, len(sorted_df) + 1)
    sorted_df['rank_percentile'] = sorted_df['rank'] / len(sorted_df) * 100

    axes[0].plot(sorted_df['rank_percentile'], sorted_df['mae_overall'], linewidth=2, color='navy')
    axes[0].fill_between(sorted_df['rank_percentile'], sorted_df['mae_overall'], alpha=0.3, color='lightblue')
    axes[0].set_xlabel('Word Rank Percentile')
    axes[0].set_ylabel('MAE')
    axes[0].set_title('Time Series Prediction Performance Ranking')
    axes[0].grid(True, alpha=0.3)

    excellent_threshold, good_threshold = 0.01, 0.05
    axes[0].axhline(y=excellent_threshold, color='green', linestyle='--', alpha=0.7, label='Excellent (<0.01)')
    axes[0].axhline(y=good_threshold, color='orange', linestyle='--', alpha=0.7, label='Good (<0.05)')
    axes[0].legend()

    axes[1].plot(sorted_df['mae_overall'], sorted_df['rank_percentile'], linewidth=3, color='darkred')
    axes[1].set_xlabel('MAE Threshold')
    axes[1].set_ylabel('Percentage of Words Below Threshold')
    axes[1].set_title('Cumulative Performance: % of Words Achieving Each Error Level')
    axes[1].grid(True, alpha=0.3)

    for threshold in [0.01, 0.02, 0.05, 0.1]:
        pct_below = (sorted_df['mae_overall'] <= threshold).mean() * 100
        axes[1].axvline(x=threshold, color='gray', linestyle=':', alpha=0.7)
        axes[1].text(threshold, pct_below + 5, f'{pct_below:.1f}%', ha='center', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_ranking_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_word_case_study(ax, row, word, config):
    if word in config.full_vad_data and 'temporal_vad' in config.full_vad_data[word]:
        word_data = config.full_vad_data[word]['temporal_vad']
        years, v_vals, a_vals, d_vals = word_data['x'], word_data['v'], word_data['a'], word_data['d']
        valid_data = [(y, v) for y, v in zip(years, v_vals) if y is not None and v is not None]

        if valid_data:
            valid_years, valid_v = zip(*valid_data)
            ax.plot(valid_years, valid_v, 'b-', linewidth=2, label='Valence', alpha=0.8)
            ax.axvline(x=1980, color='red', linestyle='--', alpha=0.7, label='Prediction Point')
            ax.scatter([2010], [row['actual_v']], color='blue', s=80, label='Actual 2010', marker='o',
                       edgecolor='black', linewidth=1)
            ax.scatter([2010], [row['pred_v']], color='red', s=80, label='Predicted 2010', marker='s',
                       edgecolor='black', linewidth=1)
            ax.set_xlim(min(valid_years) - 5, max(valid_years) + 5)
        else:
            ax.text(0.5, 0.5, f'No valid time series data for "{word}"', ha='center', va='center',
                    transform=ax.transAxes)
    else:
        ax.text(0.5, 0.5, f'Data not found for "{word}"', ha='center', va='center', transform=ax.transAxes)

    ax.set_xlabel('Year')
    ax.set_ylabel('Valence')
    ax.grid(True, alpha=0.3)


def create_best_word_case_studies(df, output_dir):
    best_words = df.nsmallest(6, 'mae_overall')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Case Studies: 6 Best Performing Words (Lowest MAE)', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, (_, row) in enumerate(best_words.iterrows()):
        ax = axes[i]
        _plot_word_case_study(ax, row, row['word'], config)
        ax.set_title(f"'{row['word']}'\nMAE: {row['mae_overall']:.4f}", fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)

    for i in range(len(best_words), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'best_word_case_studies.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_worst_word_case_studies(df, output_dir):
    worst_words = df.nlargest(6, 'mae_overall')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Case Studies: 6 Worst Performing Words (Highest MAE)', fontsize=16, fontweight='bold')
    axes = axes.flatten()

    for i, (_, row) in enumerate(worst_words.iterrows()):
        ax = axes[i]
        _plot_word_case_study(ax, row, row['word'], config)
        ax.set_title(f"'{row['word']}'\nMAE: {row['mae_overall']:.4f}", fontsize=10)
        if i == 0:
            ax.legend(fontsize=8)

    for i in range(len(worst_words), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(output_dir, 'worst_word_case_studies.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_heatmap(df, output_dir):

    def get_word_characteristics(word, row):
        return {
            'length_category': 'Short' if len(word) <= 5 else 'Medium' if len(word) <= 8 else 'Long',
            'mae_category': 'Excellent' if row['mae_overall'] <= 0.01 else 'Good' if row[
                                                                                         'mae_overall'] <= 0.05 else 'Poor',
            'variance_category': 'Low' if row['vad_variance'] <= 0.01 else 'Medium' if row[
                                                                                           'vad_variance'] <= 0.05 else 'High'
        }

    char_data = [get_word_characteristics(row['word'], row) for _, row in df.iterrows()]
    char_df = pd.DataFrame(char_data)
    char_df['mae'] = df['mae_overall']  # Add MAE for aggregation
    heatmap_data = char_df.groupby(['length_category', 'variance_category'])['mae'].mean().unstack()

    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlBu_r', center=0.02)
    plt.title('Average MAE by Word Length and VAD Variance')
    plt.xlabel('VAD Variance Category')
    plt.ylabel('Word Length Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_patterns_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()


def create_reviewer_summary_table(df, output_dir):
    summary_path = os.path.join(output_dir, 'reviewer_performance_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("EMOTRACKER TIME SERIES PREDICTION PERFORMANCE SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total words analyzed: {len(df):,}\n")
        f.write("Prediction task: 1980 -> 2010 (30-year forecasting)\n\n")

        f.write("OVERALL PERFORMANCE DISTRIBUTION (MAE):\n")
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
        for q in quantiles:
            f.write(f"{q * 100:4.0f}th percentile: {df['mae_overall'].quantile(q):.6f}\n")
        f.write(f"\nMean MAE: {df['mae_overall'].mean():.6f}\n")
        f.write(f"Standard deviation: {df['mae_overall'].std():.6f}\n\n")

        f.write("PERFORMANCE CATEGORIES:\n")
        excellent = (df['mae_overall'] <= 0.01).sum()
        good = ((df['mae_overall'] > 0.01) & (df['mae_overall'] <= 0.05)).sum()
        fair = ((df['mae_overall'] > 0.05) & (df['mae_overall'] <= 0.1)).sum()
        poor = (df['mae_overall'] > 0.1).sum()
        total = len(df)
        f.write(f"Excellent (MAE <= 0.01): {excellent:5d} words ({excellent / total:.1%})\n")
        f.write(f"Good (0.01 < MAE <= 0.05): {good:5d} words ({good / total:.1%})\n")
        f.write(f"Fair (0.05 < MAE <= 0.10): {fair:5d} words ({fair / total:.1%})\n")
        f.write(f"Poor (MAE > 0.10):      {poor:5d} words ({poor / total:.1%})\n\n")

        f.write("DIMENSION-SPECIFIC PERFORMANCE:\n")
        f.write(f"{'Dimension':<10} {'Correlation':<12} {'R-squared':<10} {'Mean MAE':<10} {'Median MAE':<12}\n")
        f.write("-" * 57 + "\n")
        for dim_code, dim_name in [('v', 'Valence'), ('a', 'Arousal'), ('d', 'Dominance')]:
            corr = df[f'actual_{dim_code}'].corr(df[f'pred_{dim_code}'])
            mean_mae = df[f'mae_{dim_code}'].mean()
            median_mae = df[f'mae_{dim_code}'].median()
            f.write(f"{dim_name:<10} {corr:<12.4f} {corr ** 2:<10.4f} {mean_mae:<10.6f} {median_mae:<12.6f}\n")

    print(f"Comprehensive reviewer summary saved to {summary_path}")


def analyze_word_performance(output_dir="word_analysis_results"):
    if not config.resources_loaded_pytorch:
        return False

    os.makedirs(output_dir, exist_ok=True)
    error_log_path = os.path.join(output_dir, 'error_log.txt')
    test_words = [word for word, data in config.full_vad_data.items() if 'temporal_vad' in data]
    predict_from_year, predict_to_year = 1980, 2010
    results, failed_log = [], []

    batch_size = 100  # Increased batch size for less frequent (internal) logging if any
    for i in range(0, len(test_words), batch_size):
        batch_words = test_words[i:i + batch_size]
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

                results.append({
                    'word': word,
                    'actual_v': actuals[0], 'actual_a': actuals[1], 'actual_d': actuals[2],
                    'pred_v': preds[0], 'pred_a': preds[1], 'pred_d': preds[2],
                    'mae_overall': mean_absolute_error(actuals, preds),
                    'mae_v': abs(actuals[0] - preds[0]),
                    'mae_a': abs(actuals[1] - preds[1]),
                    'mae_d': abs(actuals[2] - preds[2]),
                    'vad_variance': np.var(valid_vad) if len(valid_vad) > 1 else 0.0,
                })

            except Exception as e:
                failed_log.append(f"WORD: {word} - EXCEPTION: {e}")

    # Write errors to a log file
    with open(error_log_path, 'w') as f:
        f.write("Words that failed prediction:\n" + "=" * 50 + "\n\n" + "\n".join(failed_log))

    successful = len(results)
    failed_count = len(failed_log)
    total_processed = successful + failed_count
    if total_processed > 0:
        success_rate = successful / total_processed * 100
        print(f"Analysis complete: {successful} successful, {failed_count} failed ({success_rate:.1f}% success).")
        print(f"Error log saved to {error_log_path}")

    if not results:
        print("No successful predictions to analyze.")
        return False

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, 'word_performance_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Raw results saved to {csv_path}")

    create_analysis_report(df, output_dir)
    create_individual_visualizations(df, output_dir)
    return True


def create_analysis_report(df, output_dir):
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write("WORD PERFORMANCE ANALYSIS REPORT\n" + "=" * 40 + "\n\n")
        f.write(f"Total words analyzed: {len(df)}\n\n")
        f.write("OVERALL PERFORMANCE (MAE):\n")
        f.write(f"{df['mae_overall'].describe().to_string()}\n\n")

        f.write("TOP 10 BEST PERFORMING WORDS:\n")
        f.write(df.nsmallest(10, 'mae_overall')[['word', 'mae_overall']].to_string(index=False) + "\n\n")

        f.write("TOP 10 WORST PERFORMING WORDS:\n")
        f.write(df.nlargest(10, 'mae_overall')[['word', 'mae_overall']].to_string(index=False) + "\n\n")

    print(f"Analysis report saved to {report_path}")


def create_individual_visualizations(df, output_dir):
    plt.style.use('default')

    # MAE Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['mae_overall'], bins=30, kde=True)
    plt.title('Distribution of Overall MAE Across All Words')
    plt.xlabel('Overall MAE')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_mae_distribution.png'), dpi=300)
    plt.close()

    # MAE by Dimension
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df[['mae_v', 'mae_a', 'mae_d']], notch=True, patch_artist=True)
    plt.title('MAE Distribution by VAD Dimension')
    plt.ylabel('MAE')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_dimension_comparison.png'), dpi=300)
    plt.close()

    # Best Performers
    plt.figure(figsize=(8, 6))
    best_df = df.nsmallest(10, 'mae_overall')
    sns.barplot(x='mae_overall', y='word', data=best_df, color='g', alpha=0.8)
    plt.title('Top 10 Best Performing Words (Lowest MAE)')
    plt.xlabel('Overall MAE')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3a_best_performers.png'), dpi=300)
    plt.close()

    # Worst Performers
    plt.figure(figsize=(8, 6))
    worst_df = df.nlargest(10, 'mae_overall')
    sns.barplot(x='mae_overall', y='word', data=worst_df, color='r', alpha=0.8)
    plt.title('Top 10 Worst Performing Words (Highest MAE)')
    plt.xlabel('Overall MAE')
    plt.ylabel('Word')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4a_worst_performers.png'), dpi=300)
    plt.close()


def enhanced_analyze_word_performance(output_dir="word_analysis_results"):
    if analyze_word_performance(output_dir):
        results_path = os.path.join(output_dir, 'word_performance_results.csv')
        if os.path.exists(results_path):
            df = pd.read_csv(results_path)
            create_detailed_timeseries_visualizations(df, output_dir)
            create_best_word_case_studies(df, output_dir)
            create_worst_word_case_studies(df, output_dir)
            create_performance_heatmap(df, output_dir)
            create_reviewer_summary_table(df, output_dir)
            print("\nEnhanced analysis and visualizations complete!")
            return True
    return False


def main():
    if not os.path.exists('config.py'):
        print("ERROR: Run this script from the api/ directory!")
        return

    if enhanced_analyze_word_performance():
        print("\nCOMPLETE ANALYSIS FINISHED!")
        print("Check the 'word_analysis_results' directory for all outputs.")
    else:
        print("\nAnalysis failed. Check the error messages and log file.")


if __name__ == "__main__":
    main()