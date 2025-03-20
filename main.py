# main.py
import argparse
import time
import pandas as pd

# Import algorithm modules
from weights import *
from ranking import *

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Criteria Decision Making (MCDM) for Samsung Phones Dataset"
    )
    parser.add_argument(
        "--file", type=str, default="cleaned_samsung_phones.csv",
        help="Path to the CSV data file"
    )
    parser.add_argument(
        "--weight_method", type=str, choices=["ahp", "entropy"], default="entropy",
        help="Method to compute criteria weights (ahp or entropy)"
    )
    parser.add_argument(
        "--pairwise_file", type=str, default="pairwise_matrix.xlsx",
        help="Path to the Excel file containing the pairwise comparison matrix (for AHP)"
    )
    parser.add_argument(
        "--rank_method", type=str,
        choices=["wsm", "wpm", "waspas", "topsis", "promethee", "vikor"],
        default="topsis", help="Method to rank alternatives"
    )
    args = parser.parse_args()
    
    # Load dataset
    df = pd.read_csv(args.file)
    
    # Identify criteria columns (all columns except 'model_name')
    criteria_columns = [col for col in df.columns if col != "model_name"]
    
    # Define criteria types: if column name contains 'price' or 'date', treat as cost; otherwise, benefit.
    criteria_type = {}
    for col in criteria_columns:
        if "price" in col.lower() or "date" in col.lower():
            criteria_type[col] = "cost"
        else:
            criteria_type[col] = "benefit"
    
    print("Selected Weight Computation Method:", args.weight_method)
    # Compute weights and measure execution time
    start = time.perf_counter()
    try:
        if args.weight_method == "ahp":
            weights = compute_ahp_weights_from_excel(args.pairwise_file)
        else:  # entropy
            weights = compute_entropy_weights(df, criteria_columns, criteria_type)
    except ValueError as e:
        print("Error in weight computation:", e)
        return
    end = time.perf_counter()
    print("Computed Weights:", weights)
    print("Weight computation time: {:.6f} seconds".format(end - start))
    
    # Rank alternatives based on the selected ranking method and measure execution time
    print("\nSelected Ranking Method:", args.rank_method)
    start = time.perf_counter()
    if args.rank_method == "wsm":
        ranking = rank_wsm(df, criteria_columns, weights, criteria_type)
    elif args.rank_method == "wpm":
        ranking = rank_wpm(df, criteria_columns, weights, criteria_type)
    elif args.rank_method == "waspas":
        ranking = rank_waspas(df, criteria_columns, weights, criteria_type, lambda_val=0.5)
    elif args.rank_method == "topsis":
        ranking = rank_topsis(df, criteria_columns, weights, criteria_type)
    elif args.rank_method == "promethee":
        ranking = rank_promethee(df, criteria_columns, weights, criteria_type)
    elif args.rank_method == "vikor":
        ranking = rank_vikor(df, criteria_columns, weights, criteria_type, v=0.5)
    end = time.perf_counter()
    
    print("Ranking Results (based on 'model_name'):")
    print(ranking)
    print("Ranking computation time: {:.6f} seconds".format(end - start))
    
if __name__ == "__main__":
    main()
