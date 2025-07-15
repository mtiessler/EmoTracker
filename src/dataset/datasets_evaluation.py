import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import logging
from pathlib import Path

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
    logging.info(f"Loaded gold standard: {len(df)} words")

    return df


def load_temporal_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logging.info(f"Loaded temporal data: {len(data)} words")
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
    logging.info(f"Extracted historical estimates: {len(df)} words for year ~{target_year}")

    return df


def evaluate_single_dataset(gold_standard, estimates, dataset_name, rescale_method='minmax'):
    merged = pd.merge(gold_standard, estimates, on='word', suffixes=('_gold', '_est'))
    merged = merged.dropna()

    if len(merged) < 5:
        logging.warning(f"Too few overlapping words for {dataset_name}: {len(merged)}")
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
        'dataset_name': dataset_name,
        'n_words': len(merged),
        'words': merged['word'].tolist(),
        'original_data': merged,
        'rescaled_data': rescaled_merged
    }

    all_gold, all_rescaled = [], []

    for dim in dimensions:
        gold_vals = merged[f'{dim}_gold'].values
        orig_vals = merged[f'{dim}_est'].values
        orig_r, _ = pearsonr(gold_vals, orig_vals)

        rescaled_vals = rescaled_merged[f'{dim}_rescaled'].values
        rescaled_r, p = pearsonr(gold_vals, rescaled_vals)

        mae = mean_absolute_error(gold_vals, rescaled_vals)
        rmse = np.sqrt(mean_squared_error(gold_vals, rescaled_vals))

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

    results['overall_orig_r'] = overall_orig_r
    results['overall_rescaled_r'] = overall_rescaled_r
    results['overall_p'] = overall_p

    return results


def print_detailed_results(results, dataset_name):
    if not results:
        return

    print(f"\n{dataset_name.upper()} DATASET EVALUATION")
    print("=" * 50)

    print(f"Overlapping words: {results['n_words']}")
    print(f"Sample words: {', '.join(results['words'][:10])}")

    print(f"\nScale transformation:")
    dimensions = ['valence', 'arousal', 'dominance']

    merged = results['original_data']
    rescaled = results['rescaled_data']

    for dim in dimensions:
        est_vals = merged[f'{dim}_est'].values
        rescaled_vals = rescaled[f'{dim}_rescaled'].values
        gold_vals = merged[f'{dim}_gold'].values

        print(f"  {dim}: [{np.min(est_vals):.3f}, {np.max(est_vals):.3f}] -> "
              f"[{np.min(rescaled_vals):.3f}, {np.max(rescaled_vals):.3f}] "
              f"(Gold: [{np.min(gold_vals):.1f}, {np.max(gold_vals):.1f}])")

    print(f"\nCorrelation Analysis:")
    print(f"{'Dimension':<12} {'Original r':<12} {'Rescaled r':<12} {'P-value':<10} {'Sig':<5} {'MAE':<8} {'RMSE':<8}")
    print("-" * 75)

    for dim in dimensions:
        orig_r = results[f'{dim}_orig_r']
        rescaled_r = results[f'{dim}_rescaled_r']
        p = results[f'{dim}_p']
        mae = results[f'{dim}_mae']
        rmse = results[f'{dim}_rmse']

        if p < 0.001:
            sig = "***"
        elif p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"

        print(f"{dim.title():<12} {orig_r:<12.3f} {rescaled_r:<12.3f} {p:<10.4f} {sig:<5} {mae:<8.3f} {rmse:<8.3f}")

    overall_orig_r = results['overall_orig_r']
    overall_rescaled_r = results['overall_rescaled_r']
    overall_p = results['overall_p']

    if overall_p < 0.001:
        overall_sig = "***"
    elif overall_p < 0.01:
        overall_sig = "**"
    elif overall_p < 0.05:
        overall_sig = "*"
    else:
        overall_sig = "ns"

    print(f"{'Overall':<12} {overall_orig_r:<12.3f} {overall_rescaled_r:<12.3f} {overall_p:<10.4f} {overall_sig:<5}")


def print_comparison_table(all_results):
    print("\n" + "=" * 80)
    print("DATASET COMPARISON SUMMARY")
    print("=" * 80)

    print(f"{'Dataset':<12} {'N Words':<8} {'Overall r':<10} {'Val r':<8} {'Aro r':<8} {'Dom r':<8} {'Overall p':<10}")
    print("-" * 80)

    for dataset_name, results in all_results.items():
        if results:
            n_words = results['n_words']
            overall_r = results['overall_rescaled_r']
            val_r = results['valence_rescaled_r']
            aro_r = results['arousal_rescaled_r']
            dom_r = results['dominance_rescaled_r']
            overall_p = results['overall_p']

            print(
                f"{dataset_name.upper():<12} {n_words:<8} {overall_r:<10.3f} {val_r:<8.3f} {aro_r:<8.3f} "
                f"{dom_r:<8.3f} {overall_p:<10.4f}")
        else:
            print(f"{dataset_name.upper():<12} {'FAILED':<8} {'N/A':<10} {'N/A':<8} {'N/A':<8} "
                  f"{'N/A':<8} {'N/A':<10}")

    valid_results = {k: v for k, v in all_results.items() if v is not None}

    if valid_results:
        print(f"\nBEST PERFORMERS:")

        # Overall best
        best_overall = max(valid_results.items(), key=lambda x: x[1]['overall_rescaled_r'])
        print(f"  Overall: {best_overall[0].upper()} (r = {best_overall[1]['overall_rescaled_r']:.3f})")

        # Dimension-wise best
        for dim, dim_name in [('valence', 'Valence'), ('arousal', 'Arousal'), ('dominance', 'Dominance')]:
            best_dim = max(valid_results.items(), key=lambda x: x[1][f'{dim}_rescaled_r'])
            print(f"  {dim_name}: {best_dim[0].upper()} (r = {best_dim[1][f'{dim}_rescaled_r']:.3f})")


def save_all_results(all_results, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save individual results
    for dataset_name, results in all_results.items():
        if results:
            # Create simplified results for JSON serialization
            simplified_results = {
                'dataset_name': dataset_name,
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
                }
            }

            filename = output_path / f"evaluation_{dataset_name}.json"
            with open(filename, 'w') as f:
                json.dump(simplified_results, f, indent=2)

            print(f"Saved {dataset_name} results to: {filename}")

    comparison_summary = {}
    for dataset_name, results in all_results.items():
        if results:
            comparison_summary[dataset_name] = {
                'n_words': results['n_words'],
                'overall_correlation': results['overall_rescaled_r'],
                'valence_correlation': results['valence_rescaled_r'],
                'arousal_correlation': results['arousal_rescaled_r'],
                'dominance_correlation': results['dominance_rescaled_r'],
                'overall_p_value': results['overall_p']
            }
        else:
            comparison_summary[dataset_name] = {'status': 'failed'}

    summary_filename = output_path / "comparison_summary.json"
    with open(summary_filename, 'w') as f:
        json.dump(comparison_summary, f, indent=2)

    print(f"Saved comparison summary to: {summary_filename}")


def find_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidate = os.path.join(script_dir, "..", "..", "data", "VAD_Lexicons", "Golden_VAD", "goldEN.vad")

    gold_file = None
    if os.path.exists(candidate):
        gold_file = candidate

    dataset_base_dir = os.path.join(script_dir, "..", "..", "data", "Generated_VAD_Dataset")
    dataset_files = {}

    for dataset_name in ['nrc', 'warriner', 'memolon']:
        dataset_path = os.path.join(dataset_base_dir, f"dataset_{dataset_name}",
                                    f"emotracker_{dataset_name}.json")
        if os.path.exists(dataset_path):
            dataset_files[dataset_name] = dataset_path

    return gold_file, dataset_files


def main():
    print("Multi-Dataset Gold Standard Evaluation")
    print("=" * 50)

    gold_file, dataset_files = find_files()

    if not gold_file:
        print("ERROR: Cannot find goldEN.vad file")
        return

    if not dataset_files:
        print("ERROR: Cannot find any dataset files")
        return

    print(f"Gold standard file: {gold_file}")
    print(f"Found {len(dataset_files)} datasets:")
    for name, path in dataset_files.items():
        print(f"  {name.upper()}: {path}")

    try:
        gold_standard = load_gold_standard(gold_file)
        print(f"\nGold standard statistics:")
        for dim in ['valence', 'arousal', 'dominance']:
            values = gold_standard[dim]
            print(f"  {dim}: mean={values.mean():.1f}, std={values.std():.1f}, "
                  f"range=[{values.min():.1f}, {values.max():.1f}]")
    except Exception as e:
        print(f"ERROR loading gold standard: {e}")
        return

    all_results = {}

    for dataset_name, dataset_path in dataset_files.items():
        print(f"\n{'=' * 20} EVALUATING {dataset_name.upper()} {'=' * 20}")

        try:
            # Load dataset
            temporal_data = load_temporal_data(dataset_path)

            # Extract estimates
            historical_estimates = extract_historical_estimates(temporal_data, target_year=1835)

            # Evaluate
            results = evaluate_single_dataset(gold_standard, historical_estimates, dataset_name)

            if results:
                all_results[dataset_name] = results
                print_detailed_results(results, dataset_name)
            else:
                all_results[dataset_name] = None
                print(f"Failed to evaluate {dataset_name}")

        except Exception as e:
            print(f"ERROR evaluating {dataset_name}: {e}")
            all_results[dataset_name] = None

    print_comparison_table(all_results)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "..", "data", "evaluation_results")
    save_all_results(all_results, output_dir)

    print(f"\nEvaluation completed!")


if __name__ == "__main__":
    main()
