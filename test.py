import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def calculate_metrics(y_true, y_pred):
    """
    Calculate MAE, RMSE, and DTW for the given true and predicted values.

    Parameters:
    y_true (list or array): True values.
    y_pred (list or array): Predicted values.

    Returns:
    dict: Dictionary containing MAE, RMSE, and DTW.
    """

    # Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)

    # Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Dynamic Time Warping
    dtw_distance, _ = fastdtw(y_true, y_pred, dist=euclidean)

    return {"MAE": mae, "RMSE": rmse, "DTW": dtw_distance}

# Example usage
y_true = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y_pred = [1.1, 1.9, 3.2, 3.9, 5.1, 5.8, 7.2, 7.9, 9.5, 9.9]

metrics = calculate_metrics(y_true, y_pred)
metrics

