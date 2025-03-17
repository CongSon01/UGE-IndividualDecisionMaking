# common_utils.py
import numpy as np
import pandas as pd

def normalize_min_max(df, criteria_columns, criteria_type):
    """
    Normalize the data using Min-Max normalization.
    For benefit criteria: (x - min) / (max - min)
    For cost criteria: (max - x) / (max - min)
    
    Args:
        df (DataFrame): The input data.
        criteria_columns (list): List of criteria column names.
        criteria_type (dict): Mapping of criteria names to "benefit" or "cost".
        
    Returns:
        DataFrame: Normalized values in the range [0, 1].
    """
    df_norm = df.copy()
    for col in criteria_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val == min_val:
            df_norm[col] = 0.0
        else:
            if criteria_type.get(col, "benefit") == "benefit":
                df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            else:  # cost criterion
                df_norm[col] = (max_val - df[col]) / (max_val - min_val)
    return df_norm

def normalize_vector(df, criteria_columns):
    """
    Normalize the data using vector normalization (used in TOPSIS).
    Formula: r_ij = x_ij / sqrt(sum_i(x_ij^2))
    
    Args:
        df (DataFrame): The input data.
        criteria_columns (list): List of criteria column names.
    
    Returns:
        DataFrame: The normalized data.
    """
    df_norm = df.copy()
    for col in criteria_columns:
        norm = np.sqrt((df[col]**2).sum())
        if norm != 0:
            df_norm[col] = df[col] / norm
        else:
            df_norm[col] = 0.0
    return df_norm
