# module_topsis.py
import numpy as np
import pandas as pd
from common_utils import normalize_vector

def rank_topsis(df, criteria_columns, weights, criteria_type):
    """
    Rank alternatives using the TOPSIS method:
    1. Normalize the data using vector normalization.
    2. Multiply by criteria weights.
    3. Determine the ideal and anti-ideal solutions.
    4. Compute the Euclidean distances to the ideal and anti-ideal solutions.
    5. Compute the closeness coefficient: CC = d_minus / (d_plus + d_minus).
    
    Args:
        df (DataFrame): Original dataset.
        criteria_columns (list): List of criteria column names.
        weights (dict): Criteria weights.
        criteria_type (dict): Mapping of criteria names to "benefit" or "cost".
    
    Returns:
        DataFrame: Ranking result with columns 'model_name' and 'score' (closeness coefficient), sorted in descending order.
    """
    df_norm = normalize_vector(df, criteria_columns)
    # Apply weights to normalized data
    for col in criteria_columns:
        df_norm[col] = df_norm[col] * weights[col]
        
    # Determine ideal and anti-ideal solutions
    ideal = {}
    anti_ideal = {}
    for col in criteria_columns:
        if criteria_type.get(col, "benefit") == "benefit":
            ideal[col] = df_norm[col].max()
            anti_ideal[col] = df_norm[col].min()
        else:  # For cost criteria, reverse the order
            ideal[col] = df_norm[col].min()
            anti_ideal[col] = df_norm[col].max()
    
    # Compute Euclidean distances
    d_plus = np.sqrt(((df_norm[criteria_columns] - pd.Series(ideal))**2).sum(axis=1))
    d_minus = np.sqrt(((df_norm[criteria_columns] - pd.Series(anti_ideal))**2).sum(axis=1))
    
    # Calculate closeness coefficient
    cc = d_minus / (d_plus + d_minus + 1e-6)
    
    result = df[['model_name']].copy()
    result['score'] = cc
    result = result.sort_values(by='score', ascending=False).reset_index(drop=True)
    return result
