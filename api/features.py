import numpy as np
from scipy.stats import linregress
from scipy.ndimage import gaussian_filter1d


def _safe_nan_check(value, default=0.0):
    return default if np.isnan(value) else value


def _calculate_velocity_and_trend(series):
    if len(series) < 3:
        return 0.0, 0.0, 0.0

    x_axis = np.arange(len(series))

    try:
        slope, _, r_value, _, _ = linregress(x_axis, series)
        velocity = slope
        trend_strength = abs(r_value)
        trend_direction = np.sign(slope)
    except ValueError:
        velocity = 0.0
        trend_strength = 0.0
        trend_direction = 0.0

    velocity = _safe_nan_check(velocity)
    trend_strength = _safe_nan_check(trend_strength)
    trend_direction = _safe_nan_check(trend_direction)

    return velocity, trend_strength, trend_direction


def _calculate_acceleration(series, sigma=1.0):
    if len(series) < 3:
        return 0.0

    smoothed_series = gaussian_filter1d(series, sigma=sigma, mode='nearest')

    if len(smoothed_series) < 3:
        return 0.0

    # Calculate velocities (first derivatives)
    velocities = np.diff(smoothed_series)

    if len(velocities) < 2:
        return 0.0

    # Calculate accelerations (second derivatives)
    accelerations = np.diff(velocities)
    acceleration = np.mean(accelerations)

    return _safe_nan_check(acceleration)


def _calculate_volatility(series):
    if len(series) <= 1:
        return 0.0

    volatility = np.std(series)
    return _safe_nan_check(volatility)


def _calculate_momentum_oscillator(series):
    if len(series) < 2:
        return 0.0

    recent_change = series[-1] - series[0]
    std_dev = np.std(series)

    momentum_oscillator = recent_change / (std_dev + 1e-8)
    return _safe_nan_check(momentum_oscillator)


def _calculate_relative_strength(series):
    if len(series) < 3:
        return 0.0

    mid_point = len(series) // 2
    if mid_point == 0:
        return 0.0

    first_half_mean = np.mean(series[:mid_point])
    second_half_mean = np.mean(series[mid_point:])

    relative_strength = (second_half_mean - first_half_mean) / (abs(first_half_mean) + 1e-8)
    return _safe_nan_check(relative_strength)


def _calculate_range_position(series):
    if len(series) <= 1:
        return 0.0

    series_min = np.min(series)
    series_max = np.max(series)

    if series_max == series_min:
        # All values are the same, position is in the middle
        return 0.5

    range_position = (series[-1] - series_min) / (series_max - series_min)
    return _safe_nan_check(range_position)


def _calculate_ema_ratio(series):
    if len(series) < 3:
        return 0.0

    # EMA
    alpha = 2.0 / (len(series) + 1)
    ema = series[0]
    for i in range(1, len(series)):
        ema = alpha * series[i] + (1 - alpha) * ema

    # SMA
    sma = np.mean(series)

    # ratio
    ema_ratio = (ema - sma) / (abs(sma) + 1e-8)
    return _safe_nan_check(ema_ratio)


def _calculate_single_dimension_features(series, sigma=1.0):
    if len(series) < 3:
        return [0.0] * 8

    velocity, trend_strength, trend_direction = _calculate_velocity_and_trend(series)
    acceleration = _calculate_acceleration(series, sigma)
    volatility = _calculate_volatility(series)
    momentum_oscillator = _calculate_momentum_oscillator(series)
    relative_strength = _calculate_relative_strength(series)
    range_position = _calculate_range_position(series)
    ema_ratio = _calculate_ema_ratio(series)

    return [
        velocity,
        acceleration,
        trend_strength * trend_direction,
        volatility,
        momentum_oscillator,
        relative_strength,
        range_position,
        ema_ratio
    ]


def _validate_vad_series(vad_series):
    if not isinstance(vad_series, np.ndarray):
        raise ValueError("vad_series must be a numpy array")

    if len(vad_series.shape) != 2:
        raise ValueError("vad_series must be a 2D array")

    n_points, n_dims = vad_series.shape

    if n_points == 0 or n_dims == 0:
        raise ValueError("vad_series cannot have zero points or dimensions")

    return n_points, n_dims


def calculate_momentum_features(vad_series, window_size=5, sigma=1.0):
    n_points, n_dims = _validate_vad_series(vad_series)

    features_per_dim = 8
    momentum_features = []

    for point_idx in range(n_points):
        if point_idx < window_size:
            # Insufficient history - return zeros
            momentum_features.append(np.zeros(n_dims * features_per_dim))
            continue

        window_data = vad_series[point_idx - window_size:point_idx]
        point_features = []

        for dim_idx in range(n_dims):
            dim_series = window_data[:, dim_idx]
            dim_features = _calculate_single_dimension_features(dim_series, sigma)
            point_features.extend(dim_features)

        momentum_features.append(np.array(point_features, dtype=float))

    return np.array(momentum_features, dtype=float)


def get_feature_names(n_dims=3):
    base_feature_names = [
        'velocity',
        'acceleration',
        'trend_strength_direction',
        'volatility',
        'momentum_oscillator',
        'relative_strength',
        'range_position',
        'ema_ratio'
    ]

    dim_labels = ['V', 'A', 'D'] if n_dims == 3 else [f'dim_{i}' for i in range(n_dims)]

    feature_names = []
    for dim_label in dim_labels:
        for base_name in base_feature_names:
            feature_names.append(f"{dim_label}_{base_name}")

    return feature_names


def calculate_feature_statistics(vad_series, window_size=5, sigma=1.0):
    features = calculate_momentum_features(vad_series, window_size, sigma)

    valid_features = features[window_size:]

    if len(valid_features) == 0:
        return {
            'features': features,
            'statistics': {
                'mean': np.array([]),
                'std': np.array([]),
                'min': np.array([]),
                'max': np.array([])
            }
        }

    stats = {
        'features': features,
        'statistics': {
            'mean': np.mean(valid_features, axis=0),
            'std': np.std(valid_features, axis=0),
            'min': np.min(valid_features, axis=0),
            'max': np.max(valid_features, axis=0)
        },
        'feature_names': get_feature_names(vad_series.shape[1])
    }

    return stats