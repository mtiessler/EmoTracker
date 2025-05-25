import numpy as np
import logging
from typing import Optional, Tuple, List


def validate_array_shapes(*arrays):
    shapes = [arr.shape for arr in arrays if arr.size > 0]
    if not shapes:
        return False
    first_shape = shapes[0]
    return all(shape[0] == first_shape[0] for shape in shapes)


def safe_divide(numerator: float, denominator: float, epsilon: float = 1e-8) -> float:
    return numerator / (abs(denominator) + epsilon)


def handle_nan_values(value: float, default: float = 0.0) -> float:
    return default if np.isnan(value) else value


def interpolate_missing_values(series: np.ndarray) -> Optional[np.ndarray]:
    is_nan = np.isnan(series)
    if not is_nan.any():
        return series

    valid_indices = np.where(~is_nan)[0]
    nan_indices = np.where(is_nan)[0]

    if len(valid_indices) < 2:
        return None

    series[nan_indices] = np.interp(nan_indices, valid_indices, series[valid_indices])
    return series


def calculate_statistics(series: np.ndarray) -> Tuple[float, float, float, float]:
    if len(series) == 0:
        return 0.0, 0.0, 0.0, 0.0

    mean_val = np.mean(series)
    std_val = np.std(series) if len(series) > 1 else 0.0
    min_val = np.min(series)
    max_val = np.max(series)

    return mean_val, std_val, min_val, max_val


def split_data_by_year(data_arrays: List[np.ndarray], years: np.ndarray,
                       split_year: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    train_indices = np.where(years <= split_year)[0]
    test_indices = np.where(years > split_year)[0]

    train_data = [arr[train_indices] if arr.size > 0 else np.array([]) for arr in data_arrays]
    test_data = [arr[test_indices] if arr.size > 0 else np.array([]) for arr in data_arrays]

    return train_data, test_data


def log_data_info(data_dict: dict, name: str = "Dataset") -> None:
    logging.info(f"{name} Information:")
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            logging.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            logging.info(f"  {key}: {type(value).__name__}")


def ensure_directory_exists(directory_path: str) -> None:
    import os
    os.makedirs(directory_path, exist_ok=True)