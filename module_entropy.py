# module_entropy.py
import numpy as np
import pandas as pd

def compute_entropy_weights(df, criteria_columns, criteria_type):
    """
    Compute criteria weights using the Entropy method:
    1. For cost criteria, transform the data to benefit type.
    2. Normalize the data column-wise: p_ij = x_ij / sum_i(x_ij).
    3. Compute the entropy: E_j = -k * sum(p_ij * ln(p_ij)), where k = 1 / ln(n).
    4. Determine the diversity degree: d_j = 1 - E_j and then compute the weights.
    
    Args:
        df (DataFrame): The input data.
        criteria_columns (list): List of criteria column names.
        criteria_type (dict): Mapping of criteria names to "benefit" or "cost".
        
    Returns:
        dict: A dictionary mapping criteria names to their computed weights.
    """
    df_proc = df.copy()
    # Transform cost criteria into benefit criteria
    for col in criteria_columns:
        if criteria_type.get(col, "benefit") == "cost":
            min_val = df[col].min()
            max_val = df[col].max()
            df_proc[col] = max_val - df[col] + min_val

    # Normalize the data column-wise: p_ij = x_ij / sum_i(x_ij)
    p = df_proc[criteria_columns].copy()
    p = p.apply(lambda x: x / (x.sum() + 1e-6), axis=0)
    
    n = df.shape[0]
    k = 1.0 / np.log(n + 1e-6)
    
    # Compute entropy for each criterion
    E = {}
    for col in criteria_columns:
        p_col = p[col].replace(0, 1e-6)
        E[col] = -k * (p_col * np.log(p_col)).sum()
    
    d = {col: 1 - E[col] for col in criteria_columns}
    d_sum = sum(d.values())
    weights = {col: d[col] / (d_sum + 1e-6) for col in criteria_columns}
    return weights
