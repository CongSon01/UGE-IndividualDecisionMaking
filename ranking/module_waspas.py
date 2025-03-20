import pandas as pd
from ranking.module_wsm import rank_wsm
from ranking.module_wpm import rank_wpm

def rank_waspas(df, criteria_columns, weights, criteria_type, lambda_val=0.5):
    """
    Rank alternatives using WASPAS:
    1. Compute scores using both WSM and WPM.
    2. Combine the scores: score = lambda * score_WSM + (1 - lambda) * score_WPM.
    
    Args:
        df (DataFrame): Original dataset.
        criteria_columns (list): List of criteria column names.
        weights (dict): Criteria weights.
        criteria_type (dict): Mapping of criteria names to "benefit" or "cost".
        lambda_val (float): Weighting parameter to combine WSM and WPM (default 0.5).
    
    Returns:
        DataFrame: Ranking result with columns 'model_name' and 'score', sorted in descending order.
    """
    res_wsm = rank_wsm(df, criteria_columns, weights, criteria_type)
    res_wpm = rank_wpm(df, criteria_columns, weights, criteria_type)
    
    # Merge the results on 'model_name' to ensure proper alignment
    merged = pd.merge(res_wsm[['model_name', 'score']], 
                      res_wpm[['model_name', 'score']], 
                      on='model_name', 
                      suffixes=('_wsm', '_wpm'))
    
    # Combine scores from WSM and WPM using the balancing parameter lambda_val
    merged['score'] = lambda_val * merged['score_wsm'] + (1 - lambda_val) * merged['score_wpm']
    
    # Sort alternatives by the final score in descending order
    result = merged[['model_name', 'score']].sort_values(by='score', ascending=False).reset_index(drop=True)
    return result
