import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os
import logging
import config
from prediction import predict_vad_trajectory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def analyze_word_performance(output_dir="word_analysis_results"):
    print("=== COMPREHENSIVE WORD PERFORMANCE ANALYSIS ===")

    if not config.resources_loaded_pytorch:
        return False

    os.makedirs(output_dir, exist_ok=True)
    error_log_path = os.path.join(output_dir, 'error_log.txt')

    test_words = []
    for word, data in config.full_vad_data.items():
        if 'temporal_vad' in data:
            test_words.append(word)

    print(f"Selected {len(test_words)} words for testing")

    predict_from_year = 1980
    predict_to_year = 2010

    results = []
    successful = 0
    failed = 0
    error_log = []

    print(f"Testing predictions from {predict_from_year} to {predict_to_year}...")

    batch_size = 50
    total_batches = (len(test_words) + batch_size - 1) // batch_size

    for batch_idx in range(0, len(test_words), batch_size):
        batch_words = test_words[batch_idx:batch_idx + batch_size]
        current_batch = batch_idx // batch_size + 1

        print(f"Processing batch {current_batch}/{total_batches} ({len(batch_words)} words)")

        for word in batch_words:
            try:
                word_data = config.full_vad_data[word]['temporal_vad']
                years = word_data['x']
                v_vals = word_data['v']
                a_vals = word_data['a']
                d_vals = word_data['d']

                actual_values = None
                for j, year in enumerate(years):
                    if year == predict_to_year:
                        if all(val is not None for val in [v_vals[j], a_vals[j], d_vals[j]]):
                            actual_values = [v_vals[j], a_vals[j], d_vals[j]]
                        break

                if actual_values is None:
                    failed += 1
                    error_log.append(f"WORD: {word} - ERROR: No actual values found for year {predict_to_year}")
                    continue

                predictions = predict_vad_trajectory(word, predict_from_year, predict_to_year)

                if predictions and len(predictions) > 0:
                    pred = predictions[0]
                    predicted_values = [pred['v'], pred['a'], pred['d']]

                    # Calculate errors
                    mae_overall = mean_absolute_error(actual_values, predicted_values)
                    rmse_overall = math.sqrt(mean_squared_error(actual_values, predicted_values))

                    mae_v = abs(actual_values[0] - predicted_values[0])
                    mae_a = abs(actual_values[1] - predicted_values[1])
                    mae_d = abs(actual_values[2] - predicted_values[2])

                    trajectory_length = len([y for y in years if y is not None])
                    valid_vad_values = []
                    for j in range(len(years)):
                        if v_vals[j] is not None:
                            valid_vad_values.append(v_vals[j])
                        if a_vals[j] is not None:
                            valid_vad_values.append(a_vals[j])
                        if d_vals[j] is not None:
                            valid_vad_values.append(d_vals[j])

                    vad_variance = np.var(valid_vad_values) if len(valid_vad_values) > 1 else 0.0

                    results.append({
                        'word': word,
                        'actual_v': actual_values[0],
                        'actual_a': actual_values[1],
                        'actual_d': actual_values[2],
                        'pred_v': predicted_values[0],
                        'pred_a': predicted_values[1],
                        'pred_d': predicted_values[2],
                        'mae_overall': mae_overall,
                        'rmse_overall': rmse_overall,
                        'mae_v': mae_v,
                        'mae_a': mae_a,
                        'mae_d': mae_d,
                        'word_length': len(word),
                        'trajectory_length': trajectory_length,
                        'vad_variance': vad_variance,
                        'error_v': predicted_values[0] - actual_values[0],
                        'error_a': predicted_values[1] - actual_values[1],
                        'error_d': predicted_values[2] - actual_values[2]
                    })

                    successful += 1
                else:
                    failed += 1
                    error_log.append(f"WORD: {word} - ERROR: No predictions returned from model")

            except Exception as e:
                failed += 1
                error_reason = str(e)
                if "not found" in error_reason.lower():
                    error_log.append(f"WORD: {word} - ERROR: Word not found in model vocabulary")
                else:
                    error_log.append(f"WORD: {word} - ERROR: {error_reason}")
                continue

        print(f"  Batch {current_batch} complete - Running totals: {successful} successful, {failed} failed")

    with open(error_log_path, 'w') as f:
        f.write("Words that failed prediction\n")
        f.write("=" * 50 + "\n\n")
        for error in error_log:
            f.write(error + "\n")

    print(f"Analysis complete: {successful} successful, {failed} failed")
    print(f"Success rate: {successful / (successful + failed) * 100:.1f}%")
    print(f"Error log saved to {error_log_path}")

    if len(results) == 0:
        print("No successful predictions to analyze!")
        return False

    df = pd.DataFrame(results)

    csv_path = os.path.join(output_dir, 'word_performance_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"✓ Raw results saved to {csv_path}")

    create_analysis_report(df, output_dir)
    create_individual_visualizations(df, output_dir)

    return True


def create_analysis_report(df, output_dir):
    report_path = os.path.join(output_dir, 'analysis_report.txt')

    with open(report_path, 'w') as f:
        f.write("WORD PERFORMANCE ANALYSIS REPORT\n")
        f.write("=" * 40 + "\n\n")

        f.write(f"Total words analyzed: {len(df)}\n\n")

        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"Mean MAE: {df['mae_overall'].mean():.4f}\n")
        f.write(f"Median MAE: {df['mae_overall'].median():.4f}\n")
        f.write(f"Std MAE: {df['mae_overall'].std():.4f}\n")
        f.write(f"Range: {df['mae_overall'].min():.4f} - {df['mae_overall'].max():.4f}\n\n")

        f.write("DIMENSION-SPECIFIC PERFORMANCE:\n")
        for dim, col in [('Valence', 'mae_v'), ('Arousal', 'mae_a'), ('Dominance', 'mae_d')]:
            f.write(f"{dim}:\n")
            f.write(f"  Mean MAE: {df[col].mean():.4f}\n")
            f.write(f"  Median MAE: {df[col].median():.4f}\n")
            f.write(f"  Std MAE: {df[col].std():.4f}\n\n")

        f.write("TOP 10 BEST PERFORMING WORDS:\n")
        best = df.nsmallest(10, 'mae_overall')
        for i, (_, row) in enumerate(best.iterrows(), 1):
            f.write(f"{i:2d}. {row['word']:<15} (MAE: {row['mae_overall']:.4f})\n")
        f.write("\n")

        f.write("TOP 10 WORST PERFORMING WORDS:\n")
        worst = df.nlargest(10, 'mae_overall')
        for i, (_, row) in enumerate(worst.iterrows(), 1):
            f.write(f"{i:2d}. {row['word']:<15} (MAE: {row['mae_overall']:.4f})\n")
        f.write("\n")

        corr = df['word_length'].corr(df['mae_overall'])
        f.write(f"Word length correlation with MAE: {corr:.3f}\n")

        traj_corr = df['trajectory_length'].corr(df['mae_overall'])
        var_corr = df['vad_variance'].corr(df['mae_overall'])
        f.write(f"Trajectory length correlation with MAE: {traj_corr:.3f}\n")
        f.write(f"VAD variance correlation with MAE: {var_corr:.3f}\n")

    print(f"✓ Analysis report saved to {report_path}")


def create_individual_visualizations(df, output_dir):
    plt.style.use('default')
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd']

    print("Creating individual visualization files...")

    # 1. MAE Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['mae_overall'], bins=30, alpha=0.7, edgecolor='black', color=colors[0])
    plt.axvline(df['mae_overall'].median(), color='red', linestyle='--', linewidth=2,
                label=f'Median: {df["mae_overall"].median():.4f}')
    plt.axvline(df['mae_overall'].mean(), color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {df["mae_overall"].mean():.4f}')
    plt.xlabel('Overall MAE', fontsize=12)
    plt.ylabel('Number of Words', fontsize=12)
    plt.title('Distribution of Overall MAE Across All Words', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = (f'n = {len(df)} words\nStd = {df["mae_overall"].std():.4f}\nMin '
                  f'= {df["mae_overall"].min():.4f}\nMax = {df["mae_overall"].max():.4f}')
    plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '1_mae_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Dimension Comparison (Box Plot)
    plt.figure(figsize=(10, 6))
    dim_data = [df['mae_v'], df['mae_a'], df['mae_d']]
    box_plot = plt.boxplot(dim_data, tick_labels=['Valence', 'Arousal', 'Dominance'],
                           patch_artist=True, notch=True)

    for patch, color in zip(box_plot['boxes'], colors[:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel('MAE', fontsize=12)
    plt.title('MAE Distribution by VAD Dimension', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add mean values as text
    means = [df['mae_v'].mean(), df['mae_a'].mean(), df['mae_d'].mean()]
    for i, mean_val in enumerate(means):
        plt.text(i + 1, mean_val, f'μ={mean_val:.4f}', ha='center', va='bottom',
                 fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '2_dimension_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Best Performing Words - Split into multiple plots
    # Top 10 Best
    plt.figure(figsize=(8, 6))
    best_10 = df.nsmallest(10, 'mae_overall')
    y_pos = range(len(best_10))

    bars = plt.barh(y_pos, best_10['mae_overall'], color=colors[1], alpha=0.8)

    word_labels = []
    for word in best_10['word']:
        if len(word) > 15:
            word_labels.append(word[:15] + '...')
        else:
            word_labels.append(word)

    plt.yticks(y_pos, word_labels, fontsize=10)
    plt.xlabel('Overall MAE', fontsize=12)
    plt.title('Top 10 Best Performing Words (Lowest MAE)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, best_10['mae_overall'])):
        plt.text(val + max(best_10['mae_overall']) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3a_best_performers_top10.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Next 10 Best (11-20)
    plt.figure(figsize=(8, 6))
    best_11_20 = df.nsmallest(20, 'mae_overall').tail(10)
    y_pos = range(len(best_11_20))

    bars = plt.barh(y_pos, best_11_20['mae_overall'], color=colors[1], alpha=0.6)

    # Truncate long words for display
    word_labels = []
    for word in best_11_20['word']:
        if len(word) > 15:
            word_labels.append(word[:15] + '...')
        else:
            word_labels.append(word)

    plt.yticks(y_pos, word_labels, fontsize=10)
    plt.xlabel('Overall MAE', fontsize=12)
    plt.title('Next 10 Best Performing Words (11th-20th Lowest MAE)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, best_11_20['mae_overall'])):
        plt.text(val + max(best_11_20['mae_overall']) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3b_best_performers_next10.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Worst Performing Words - Split into multiple plots
    # Top 10 Worst
    plt.figure(figsize=(8, 6))
    worst_10 = df.nlargest(10, 'mae_overall')
    y_pos = range(len(worst_10))

    bars = plt.barh(y_pos, worst_10['mae_overall'], color=colors[0], alpha=0.8)

    # Truncate long words for display
    word_labels = []
    for word in worst_10['word']:
        if len(word) > 15:
            word_labels.append(word[:15] + '...')
        else:
            word_labels.append(word)

    plt.yticks(y_pos, word_labels, fontsize=10)
    plt.xlabel('Overall MAE', fontsize=12)
    plt.title('Top 10 Worst Performing Words (Highest MAE)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, worst_10['mae_overall'])):
        plt.text(val + max(worst_10['mae_overall']) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4a_worst_performers_top10.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Next 10 Worst (11-20)
    plt.figure(figsize=(8, 6))
    worst_11_20 = df.nlargest(20, 'mae_overall').tail(10)
    y_pos = range(len(worst_11_20))

    bars = plt.barh(y_pos, worst_11_20['mae_overall'], color=colors[0], alpha=0.6)

    # Truncate long words for display
    word_labels = []
    for word in worst_11_20['word']:
        if len(word) > 15:
            word_labels.append(word[:15] + '...')
        else:
            word_labels.append(word)

    plt.yticks(y_pos, word_labels, fontsize=10)
    plt.xlabel('Overall MAE', fontsize=12)
    plt.title('Next 10 Worst Performing Words (11th-20th Highest MAE)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, worst_11_20['mae_overall'])):
        plt.text(val + max(worst_11_20['mae_overall']) * 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.4f}', va='center', ha='left', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '4b_worst_performers_next10.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Word Length vs Performance
    plt.figure(figsize=(10, 6))
    plt.scatter(df['word_length'], df['mae_overall'], alpha=0.6, color=colors[2], s=30)
    plt.xlabel('Word Length (characters)', fontsize=12)
    plt.ylabel('Overall MAE', fontsize=12)
    plt.title('Word Length vs Prediction Performance', fontsize=14, fontweight='bold')

    # Add correlation and trend line
    corr = df['word_length'].corr(df['mae_overall'])
    z = np.polyfit(df['word_length'], df['mae_overall'], 1)
    p = np.poly1d(z)
    plt.plot(df['word_length'], p(df['word_length']), "r--", alpha=0.8, linewidth=2)

    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
             fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '5_word_length_vs_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Actual vs Predicted Scatter Plots (Individual plots for each dimension)
    # Valence
    plt.figure(figsize=(8, 6))
    plt.scatter(df['actual_v'], df['pred_v'], alpha=0.6, color=colors[0], s=20)

    # Perfect prediction line
    min_val = min(df['actual_v'].min(), df['pred_v'].min())
    max_val = max(df['actual_v'].max(), df['pred_v'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Valence', fontsize=12)
    plt.ylabel('Predicted Valence', fontsize=12)
    plt.title('Actual vs Predicted Valence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add correlation and R²
    corr = df['actual_v'].corr(df['pred_v'])
    r_squared = corr ** 2
    plt.text(0.05, 0.95, f'r = {corr:.3f}\nR² = {r_squared:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6a_actual_vs_predicted_valence.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Arousal
    plt.figure(figsize=(8, 6))
    plt.scatter(df['actual_a'], df['pred_a'], alpha=0.6, color=colors[1], s=20)

    # Perfect prediction line
    min_val = min(df['actual_a'].min(), df['pred_a'].min())
    max_val = max(df['actual_a'].max(), df['pred_a'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Arousal', fontsize=12)
    plt.ylabel('Predicted Arousal', fontsize=12)
    plt.title('Actual vs Predicted Arousal', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add correlation and R²
    corr = df['actual_a'].corr(df['pred_a'])
    r_squared = corr ** 2
    plt.text(0.05, 0.95, f'r = {corr:.3f}\nR² = {r_squared:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6b_actual_vs_predicted_arousal.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Dominance
    plt.figure(figsize=(8, 6))
    plt.scatter(df['actual_d'], df['pred_d'], alpha=0.6, color=colors[2], s=20)

    # Perfect prediction line
    min_val = min(df['actual_d'].min(), df['pred_d'].min())
    max_val = max(df['actual_d'].max(), df['pred_d'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=2, label='Perfect Prediction')

    plt.xlabel('Actual Dominance', fontsize=12)
    plt.ylabel('Predicted Dominance', fontsize=12)
    plt.title('Actual vs Predicted Dominance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Add correlation and R²
    corr = df['actual_d'].corr(df['pred_d'])
    r_squared = corr ** 2
    plt.text(0.05, 0.95, f'r = {corr:.3f}\nR² = {r_squared:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '6c_actual_vs_predicted_dominance.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Cumulative Distribution of MAE
    plt.figure(figsize=(10, 6))
    sorted_mae = np.sort(df['mae_overall'])
    cumulative_prob = np.arange(1, len(sorted_mae) + 1) / len(sorted_mae)

    plt.plot(sorted_mae, cumulative_prob, linewidth=3, color=colors[3])
    plt.xlabel('Overall MAE', fontsize=12)
    plt.ylabel('Cumulative Probability', fontsize=12)
    plt.title('Cumulative Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Add percentile lines
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        val = np.percentile(sorted_mae, p)
        plt.axvline(val, color='red', linestyle=':', alpha=0.7)
        plt.text(val, p / 100, f'{p}th: {val:.3f}', rotation=90,
                 verticalalignment='bottom', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '7_cumulative_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Error Distribution by Dimension (Violin Plot)
    plt.figure(figsize=(10, 6))
    error_data = [df['error_v'], df['error_a'], df['error_d']]

    parts = plt.violinplot(error_data, positions=[1, 2, 3], showmeans=True, showextrema=True)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)

    plt.xticks([1, 2, 3], ['Valence', 'Arousal', 'Dominance'])
    plt.ylabel('Prediction Error (Predicted - Actual)', fontsize=12)
    plt.title('Distribution of Prediction Errors by Dimension', fontsize=14, fontweight='bold')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3)

    # Add mean error values
    means = [df['error_v'].mean(), df['error_a'].mean(), df['error_d'].mean()]
    for i, mean_val in enumerate(means, 1):
        plt.text(i, mean_val, f'μ={mean_val:.4f}', ha='center', va='bottom',
                 fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '8_error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Performance vs Trajectory Characteristics
    plt.figure(figsize=(12, 5))

    # Subplot 1: Trajectory Length vs Performance
    plt.subplot(1, 2, 1)
    plt.scatter(df['trajectory_length'], df['mae_overall'], alpha=0.6, color=colors[4], s=30)
    plt.xlabel('Trajectory Length (time points)', fontsize=11)
    plt.ylabel('Overall MAE', fontsize=11)
    plt.title('Trajectory Length vs Performance', fontsize=12, fontweight='bold')

    corr = df['trajectory_length'].corr(df['mae_overall'])
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.grid(True, alpha=0.3)

    # Subplot 2: VAD Variance vs Performance
    plt.subplot(1, 2, 2)
    plt.scatter(df['vad_variance'], df['mae_overall'], alpha=0.6, color=colors[5], s=30)
    plt.xlabel('VAD Variance', fontsize=11)
    plt.ylabel('Overall MAE', fontsize=11)
    plt.title('VAD Variance vs Performance', fontsize=12, fontweight='bold')

    corr = df['vad_variance'].corr(df['mae_overall'])
    plt.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '9_trajectory_characteristics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plot_files = [
        "1_mae_distribution.png - Overall MAE distribution",
        "2_dimension_comparison.png - Performance by VAD dimension",
        "3a_best_performers_top10.png - Top 10 best performing words",
        "3b_best_performers_next10.png - Next 10 best performing words (11-20)",
        "4a_worst_performers_top10.png - Top 10 worst performing words",
        "4b_worst_performers_next10.png - Next 10 worst performing words (11-20)",
        "5_word_length_vs_performance.png - Word length correlation",
        "6a_actual_vs_predicted_valence.png - Valence prediction accuracy",
        "6b_actual_vs_predicted_arousal.png - Arousal prediction accuracy",
        "6c_actual_vs_predicted_dominance.png - Dominance prediction accuracy",
        "7_cumulative_distribution.png - Cumulative error distribution",
        "8_error_distribution.png - Error distributions (violin plots)",
        "9_trajectory_characteristics.png - Trajectory length and variance effects"
    ]

    return plot_files


def main():
    if not os.path.exists('config.py'):
        print("ERROR: Run this script from the api/ directory!")
        return

    success = analyze_word_performance()

    if success:
        print("Analysis worked. Check the 'word_analysis_results' directory")
    else:
        print("Analysis failed. Check the error messages above.")


if __name__ == "__main__":
    main()