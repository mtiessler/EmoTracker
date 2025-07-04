import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def rescale_to_gold_standard(values, from_range=(0, 1), to_range=(1, 9)):
    values = np.array(values)

    if np.max(values) == np.min(values):
        middle = (to_range[0] + to_range[1]) / 2
        return np.full_like(values, middle)

    from_min, from_max = from_range
    to_min, to_max = to_range

    normalized = (values - from_min) / (from_max - from_min)
    rescaled = normalized * (to_max - to_min) + to_min

    return rescaled


def adaptive_rescale_to_gold_standard(estimates_df, gold_standard_df, method='minmax'):
    rescaled_df = estimates_df.copy()
    dimensions = ['valence', 'arousal', 'dominance']

    print(f"Rescaling method: {method}")
    print("Scale transformation:")

    for dim in dimensions:
        est_vals = estimates_df[dim].values
        gold_vals = gold_standard_df[dim].values

        if method == 'minmax':
            rescaled_vals = rescale_to_gold_standard(
                est_vals,
                from_range=(np.min(est_vals), np.max(est_vals)),
                to_range=(np.min(gold_vals), np.max(gold_vals))
            )
        else:
            rescaled_vals = est_vals

        rescaled_df[dim] = rescaled_vals

        print(f"  {dim}: [{np.min(est_vals):.3f}, {np.max(est_vals):.3f}] -> "
              f"[{np.min(rescaled_vals):.3f}, {np.max(rescaled_vals):.3f}] "
              f"(Gold: [{np.min(gold_vals):.1f}, {np.max(gold_vals):.1f}])")

    return rescaled_df


def load_gold_standard(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 4:
                    word = parts[0].lower()
                    v, a, d = float(parts[1]), float(parts[2]), float(parts[3])
                    data.append({'word': word, 'valence': v, 'arousal': a, 'dominance': d})

    df = pd.DataFrame(data)
    print(f"Loaded gold standard: {len(df)} words")

    print("Gold standard statistics:")
    for dim in ['valence', 'arousal', 'dominance']:
        values = df[dim]
        print(f"  {dim}: mean={values.mean():.1f}, std={values.std():.1f}, "
              f"range=[{values.min():.1f}, {values.max():.1f}]")

    return df


def load_temporal_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded temporal data: {len(data)} words")
    return data


def extract_historical_estimates(temporal_data, target_year=1835):
    estimates = []

    for word, word_data in temporal_data.items():
        if 'temporal_vad' not in word_data:
            continue

        temporal_vad = word_data['temporal_vad']
        x_values = temporal_vad.get('x', [])
        v_values = temporal_vad.get('v', [])
        a_values = temporal_vad.get('a', [])
        d_values = temporal_vad.get('d', [])

        if not x_values or len(x_values) != len(v_values):
            continue

        closest_idx = None
        min_diff = float('inf')

        for i, year in enumerate(x_values):
            if year is None:
                continue
            diff = abs(year - target_year)
            if diff < min_diff:
                min_diff = diff
                closest_idx = i

        if closest_idx is None:
            continue

        v_val = v_values[closest_idx]
        a_val = a_values[closest_idx]
        d_val = d_values[closest_idx]

        if all(val is not None for val in [v_val, a_val, d_val]):
            estimates.append({
                'word': word.lower(),
                'valence': float(v_val),
                'arousal': float(a_val),
                'dominance': float(d_val)
            })

    df = pd.DataFrame(estimates)
    print(f"Extracted historical estimates: {len(df)} words for year ~{target_year}")

    print("Original estimate statistics:")
    for dim in ['valence', 'arousal', 'dominance']:
        values = df[dim]
        print(f"  {dim}: mean={values.mean():.3f}, std={values.std():.3f}, "
              f"range=[{values.min():.3f}, {values.max():.3f}]")

    return df


def evaluate_correlation(gold_standard, estimates, rescale_method='minmax'):
    merged = pd.merge(gold_standard, estimates, on='word', suffixes=('_gold', '_est'))
    merged = merged.dropna()

    print(f"\nEvaluation Dataset:")
    print(f"  Overlapping words: {len(merged)}")
    print(f"  Sample words: {', '.join(merged['word'].head(10).tolist())}")

    if len(merged) < 5:
        print("ERROR: Too few overlapping words for reliable evaluation")
        return None

    original_estimates = merged[['word', 'valence_est', 'arousal_est', 'dominance_est']].copy()
    original_estimates.columns = ['word', 'valence', 'arousal', 'dominance']

    gold_subset = merged[['word', 'valence_gold', 'arousal_gold', 'dominance_gold']].copy()
    gold_subset.columns = ['word', 'valence', 'arousal', 'dominance']

    rescaled_estimates = adaptive_rescale_to_gold_standard(
        original_estimates, gold_subset, method=rescale_method
    )

    rescaled_merged = pd.merge(
        gold_subset, rescaled_estimates,
        on='word', suffixes=('_gold', '_rescaled')
    )

    dimensions = ['valence', 'arousal', 'dominance']
    results = {
        'n_words': len(merged),
        'words': merged['word'].tolist(),
        'original_data': merged,
        'rescaled_data': rescaled_merged
    }

    print(f"\nCorrelation Analysis:")
    print(
        f"{'Dimension':<12} {'Original r':<12} {'Rescaled r':<12} "
        f"{'P-value':<10} {'Significance':<12} {'MAE':<8} {'RMSE':<8}")
    print("-" * 90)

    all_gold, all_rescaled = [], []

    for dim in dimensions:
        gold_vals = merged[f'{dim}_gold'].values
        orig_vals = merged[f'{dim}_est'].values
        orig_r, _ = pearsonr(gold_vals, orig_vals)

        rescaled_vals = rescaled_merged[f'{dim}_rescaled'].values
        rescaled_r, p = pearsonr(gold_vals, rescaled_vals)

        # for me visually seeing it fast :PP
        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"

        mae = mean_absolute_error(gold_vals, rescaled_vals)
        rmse = np.sqrt(mean_squared_error(gold_vals, rescaled_vals))

        print(f"{dim.title():<12} {orig_r:<12.3f} {rescaled_r:<12.3f} {p:<10.4f} {sig:<12} {mae:<8.3f} {rmse:<8.3f}")

        all_gold.extend(gold_vals)
        all_rescaled.extend(rescaled_vals)

        results[f'{dim}_orig_r'] = orig_r
        results[f'{dim}_rescaled_r'] = rescaled_r
        results[f'{dim}_p'] = p
        results[f'{dim}_mae'] = mae
        results[f'{dim}_rmse'] = rmse

    overall_orig_r, _ = pearsonr(
        np.concatenate([merged[f'{dim}_gold'].values for dim in dimensions]),
        np.concatenate([merged[f'{dim}_est'].values for dim in dimensions])
    )
    overall_rescaled_r, overall_p = pearsonr(all_gold, all_rescaled)

    if overall_p < 0.001:
        overall_sig = "***"
    elif overall_p < 0.01:
        overall_sig = "**"
    elif overall_p < 0.05:
        overall_sig = "*"
    else:
        overall_sig = "ns"

    print(f"{'Overall':<12} {overall_orig_r:<12.3f} {overall_rescaled_r:<12.3f} {overall_p:<10.4f} {overall_sig:<12}")

    results['overall_orig_r'] = overall_orig_r
    results['overall_rescaled_r'] = overall_rescaled_r
    results['overall_p'] = overall_p

    return results


def print_summary_statistics(results):
    if not results:
        return

    print(f"\nSummary Statistics:")
    print(f"  Number of words evaluated: {results['n_words']}")
    print(f"  Original overall correlation: r = {results['overall_orig_r']:.3f}")
    print(f"  Rescaled overall correlation: r = {results['overall_rescaled_r']:.3f}")
    print(f"  Improvement from rescaling: Δr = {results['overall_rescaled_r'] - results['overall_orig_r']:+.3f}")

    print(f"\nDimension-wise Results (after rescaling):")
    dimensions = ['valence', 'arousal', 'dominance']
    for dim in dimensions:
        r = results[f'{dim}_rescaled_r']
        p = results[f'{dim}_p']
        mae = results[f'{dim}_mae']

        sig_level = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        print(f"  {dim.title()}: r = {r:.3f} ({sig_level}), MAE = {mae:.3f}")


def save_detailed_results(results, filename='evaluation_results.json'):
    if not results:
        return

    summary = {
        'evaluation_summary': {
            'n_words': results['n_words'],
            'words_evaluated': results['words']
        },
        'correlations': {
            'original': {
                'overall': results['overall_orig_r'],
                'valence': results['valence_orig_r'],
                'arousal': results['arousal_orig_r'],
                'dominance': results['dominance_orig_r']
            },
            'rescaled': {
                'overall': results['overall_rescaled_r'],
                'valence': results['valence_rescaled_r'],
                'arousal': results['arousal_rescaled_r'],
                'dominance': results['dominance_rescaled_r']
            }
        },
        'statistical_significance': {
            'overall_p_value': results['overall_p'],
            'valence_p_value': results['valence_p'],
            'arousal_p_value': results['arousal_p'],
            'dominance_p_value': results['dominance_p']
        },
        'error_metrics': {
            'valence_mae': results['valence_mae'],
            'arousal_mae': results['arousal_mae'],
            'dominance_mae': results['dominance_mae'],
            'valence_rmse': results['valence_rmse'],
            'arousal_rmse': results['arousal_rmse'],
            'dominance_rmse': results['dominance_rmse']
        },
        'improvement': {
            'overall_improvement': results['overall_rescaled_r'] - results['overall_orig_r']
        }
    }

    with open(filename, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDetailed results saved to: {filename}")


def find_file(filename, search_dirs):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for search_dir in search_dirs:
        full_path = os.path.join(script_dir, search_dir, filename)
        if os.path.exists(full_path):
            return full_path
    return None


def main():
    print("EmoTracker Gold Standard Evaluation")

    # Find files
    gold_file = find_file("goldEN.vad", ["../../data/VAD_Lexicons/Golden_VAD", "../data"])
    ml_file = find_file("ml_ready_temporal_vad_data.json", ["../../data/Generated_VAD_Dataset"])

    if not gold_file:
        print("ERROR: Cannot find goldEN.vad file")
        return

    if not ml_file:
        print("ERROR: Cannot find ml_ready_temporal_vad_data.json file")
        return

    print(f"Data sources:")
    print(f"  Gold standard: {gold_file}")
    print(f"  ML data: {ml_file}")

    try:
        gold_standard = load_gold_standard(gold_file)
        temporal_data = load_temporal_data(ml_file)

        historical_estimates = extract_historical_estimates(temporal_data, target_year=1835)

        results = evaluate_correlation(gold_standard, historical_estimates, rescale_method='minmax')

        if results:
            print_summary_statistics(results)
            save_detailed_results(results, os.path.join("..",
                                                        "..",
                                                        "data",
                                                        "model_assets_pytorch",
                                                        'gold_standard_evaluation_detailed.json'))

            print(f"\nEvaluation completed successfully.")

        else:
            print("Check data files and try again")

    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()