# module_ahp.py
import numpy as np
import pandas as pd

def compute_consistency_ratio(M):
    """
    Compute the Consistency Ratio (CR) for the pairwise comparison matrix.
    
    Args:
        M (ndarray): Pairwise comparison matrix (shape: m x m).
        
    Returns:
        float: The Consistency Ratio.
    """
    m = M.shape[0]
    # Eigen decomposition of the matrix
    eigenvalues, _ = np.linalg.eig(M)
    lambda_max = max(eigenvalues.real)
    CI = (lambda_max - m) / (m - 1) if m > 1 else 0.0
    
    # Random Consistency Index (RI) for matrices of order m
    RI_dict = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    RI = RI_dict.get(m, 1.49)  # use 1.49 if m > 10
    
    CR = CI / RI if RI != 0 else 0.0
    return CR

def compute_ahp_weights_from_excel(excel_file):
    """
    Read a pairwise comparison matrix from an Excel file and compute the AHP weights.
    The Excel file is expected to contain a square matrix (7 x 7) with criteria names as row and column labels.
    The function also computes the Consistency Ratio (CR) and raises an error if CR >= 0.1.
    
    Args:
        excel_file (str): Path to the Excel file containing the pairwise comparison matrix.
    
    Returns:
        dict: A dictionary mapping criteria names to their computed weights.
        
    Raises:
        ValueError: If the matrix is not square or does not have 7 criteria,
                    or if the Consistency Ratio (CR) is not acceptable (>= 0.1).
    """
    # Read the pairwise comparison matrix from Excel
    df_matrix = pd.read_excel(excel_file, index_col=0)
    M = df_matrix.to_numpy(dtype=float)
    
    # Validate the matrix dimensions (should be square and have 7 criteria)
    m = M.shape[0]
    if M.shape[0] != M.shape[1]:
        raise ValueError("The pairwise comparison matrix must be square.")
    if m != 7:
        raise ValueError("The pairwise comparison matrix must have 7 criteria.")
    
    # Compute the Consistency Ratio
    CR = compute_consistency_ratio(M)
    if CR >= 0.1:
        raise ValueError(f"Consistency Ratio is too high (CR = {CR:.4f}). Please adjust the matrix to achieve CR < 0.1.")
    else:
        print(f"CR = {CR:.4f}")
    
    # Compute the principal eigenvector corresponding to the largest eigenvalue
    eigenvalues, eigenvectors = np.linalg.eig(M)
    max_index = np.argmax(eigenvalues.real)
    weight_vector = eigenvectors[:, max_index].real
    weight_vector = np.abs(weight_vector)
    weight_vector = weight_vector / weight_vector.sum()
    
    # Map criteria names (from the Excel file's index) to weights
    criteria_names = df_matrix.index.tolist()
    weights = {criteria_names[i]: weight_vector[i] for i in range(m)}
    return weights