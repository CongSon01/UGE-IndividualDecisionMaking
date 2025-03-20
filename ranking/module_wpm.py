import numpy as np
import pandas as pd
from common_utils import normalize_min_max

def rank_wpm(df, criteria_columns, weights, criteria_type):
    """
    Rank alternatives using the Weighted Product Model (WPM):
    1. Normalize the data using min-max normalization.
    2. Compute the score: score = product_j( normalized_value_j ^ weight_j ).
    
    Args:
        df (DataFrame): Original dataset.
        criteria_columns (list): List of criteria column names.
        weights (dict): Criteria weights.
        criteria_type (dict): Mapping of criteria names to "benefit" or "cost".
    
    Returns:
        DataFrame: Ranking result with columns 'model_name' and 'score', sorted in descending order.
    """
    df_norm = normalize_min_max(df, criteria_columns, criteria_type)
    scores = np.ones(df.shape[0])
    for col in criteria_columns:
        scores *= df_norm[col] ** weights[col]
    result = df[['model_name']].copy()
    result['score'] = scores
    result = result.sort_values(by='score', ascending=False).reset_index(drop=True)
    return result
