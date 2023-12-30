import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    dtw_distance, _ = fastdtw(y_true, y_pred, dist=euclidean)

    return {"MAE": mae, "RMSE": rmse, "DTW": dtw_distance}