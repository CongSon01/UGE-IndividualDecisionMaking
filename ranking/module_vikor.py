import numpy as np
import pandas as pd
from common_utils import normalize_min_max

def rank_vikor(df, criteria_columns, weights, criteria_type, v=0.5):
    """
    Rank alternatives using the VIKOR method:
    1. Normalize the data using min-max normalization.
    2. Determine the best (f*) and worst (f-) values for each criterion.
    3. Compute S (the weighted sum of normalized differences) and R (the maximum weighted difference) for each alternative.
    4. Compute the VIKOR index Q for each alternative:
       Q = v*(S - S_min)/(S_max - S_min) + (1 - v)*(R - R_min)/(R_max - R_min)
    5. Alternatives with lower Q are preferred.
    
    Args:
        df (DataFrame): Original dataset.
        criteria_columns (list): List of criteria column names.
        weights (dict): Criteria weights.
        criteria_type (dict): Mapping of criteria names to "benefit" or "cost".
        v (float): The weight of the strategy of the majority of criteria (default is 0.5).
        
    Returns:
        DataFrame: Ranking result with columns 'model_name' and 'score' (Q), sorted in ascending order.
    """
    df_norm = normalize_min_max(df, criteria_columns, criteria_type)
    
    # Determine best and worst values for each criterion
    f_star = {}
    f_worst = {}
    for col in criteria_columns:
        # For normalized data, higher is always better
        f_star[col] = df_norm[col].max()
        f_worst[col] = df_norm[col].min()
    
    n = df.shape[0]
    S = np.zeros(n)
    R = np.zeros(n)
    for i in range(n):
        s_i = 0
        r_i = -np.inf
        for col in criteria_columns:
            diff = f_star[col] - df_norm.iloc[i][col]
            range_val = f_star[col] - f_worst[col] + 1e-6
            val = weights[col] * diff / range_val
            s_i += val
            if val > r_i:
                r_i = val
        S[i] = s_i
        R[i] = r_i
        
    S_min, S_max = S.min(), S.max()
    R_min, R_max = R.min(), R.max()
    
    Q = v * (S - S_min) / (S_max - S_min + 1e-6) + (1 - v) * (R - R_min) / (R_max - R_min + 1e-6)
    
    result = df[['model_name']].copy()
    result['score'] = Q
    result = result.sort_values(by='score', ascending=True).reset_index(drop=True)
    return result
