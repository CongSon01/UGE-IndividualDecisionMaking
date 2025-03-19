import pandas as pd
import numpy as np

# Define criteria and their expected weights
criteria = [
    "popularity", 
    "best_price", 
    "highest_price", 
    "screen_size", 
    "memory_size", 
    "battery_size", 
    "release_date"
]

# Expected weights (must sum to 1)
weights = {
    "popularity": 0.10,
    "best_price": 0.30,
    "highest_price": 0.05,
    "screen_size": 0.15,
    "memory_size": 0.15,
    "battery_size": 0.20,
    "release_date": 0.05
}
# weights = {
#     "popularity": 0.12,
#     "best_price": 0.28,
#     "highest_price": 0.05,
#     "screen_size": 0.14,
#     "memory_size": 0.16,
#     "battery_size": 0.20,
#     "release_date": 0.05
# }

# Set maximum perturbation factor (delta) for randomness
delta = 0.05  # 5% variation

n = len(criteria)
# Initialize an n x n matrix with ones on the diagonal
matrix = np.ones((n, n))

# Build the pairwise comparison matrix with random perturbation
for i in range(n):
    for j in range(i + 1, n):
        # Generate a small random perturbation epsilon in [-delta, delta]
        epsilon = np.random.uniform(-delta, delta)
        # Calculate the perturbed value: ideal ratio multiplied by (1 + epsilon)
        value = (weights[criteria[i]] / weights[criteria[j]]) * (1 + epsilon)
        matrix[i, j] = value
        matrix[j, i] = 1.0 / value  # ensure reciprocal property

# Create a DataFrame with criteria names as both index and columns
df_matrix = pd.DataFrame(matrix, index=criteria, columns=criteria)

# Save the DataFrame to an Excel file
df_matrix.to_excel("pairwise_matrix.xlsx", index=True)

print("Excel file 'pairwise_matrix.xlsx' has been created successfully!")
