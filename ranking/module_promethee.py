import numpy as np
import pandas as pd
from common_utils import normalize_min_max

def rank_promethee(df, criteria_columns, weights, criteria_type):
    """
    Rank alternatives using PROMETHEE:
    1. Normalize the data using min-max normalization.
    2. Use a simple preference function: if the difference (i - k) > 0 then preference = difference; otherwise 0.
    3. Compute the positive flow (phi+) and negative flow (phi-) by pairwise comparison.
    4. The net flow (phi+ - phi-) is used as the ranking score.
    
    Args:
        df (DataFrame): Original dataset.
        criteria_columns (list): List of criteria column names.
        weights (dict): Criteria weights.
        criteria_type (dict): Mapping of criteria names to "benefit" or "cost".
        
    Returns:
        DataFrame: Ranking result with columns 'model_name' and 'score', sorted in descending order.
    """
    df_norm = normalize_min_max(df, criteria_columns, criteria_type)
    n = df.shape[0]
    phi_plus = np.zeros(n)
    phi_minus = np.zeros(n)
    
    # Perform pairwise comparisons for each alternative pair
    for i in range(n):
        for k in range(n):
            if i == k:
                continue
            pref = 0
            for col in criteria_columns:
                diff = df_norm.iloc[i][col] - df_norm.iloc[k][col]
                # Preference function: positive difference only
                pref += weights[col] * (diff if diff > 0 else 0)
            phi_plus[i] += pref
            phi_minus[k] += pref
    net_flow = phi_plus - phi_minus
    result = df[['model_name']].copy()
    result['score'] = net_flow
    result = result.sort_values(by='score', ascending=False).reset_index(drop=True)
    return result
