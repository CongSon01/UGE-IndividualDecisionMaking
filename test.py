import pandas as pd

# Define criteria and their corresponding sample weights
criteria = [
    "popularity", 
    "best_price", 
    "highest_price", 
    "screen_size", 
    "memory_size", 
    "battery_size", 
    "release_date"
]

# Sample weight vector (must sum to 1, here chỉ dùng để tạo ma trận so sánh)
weights = {
    "popularity": 0.3,
    "best_price": 0.1,
    "highest_price": 0.05,
    "screen_size": 0.15,
    "memory_size": 0.1,
    "battery_size": 0.2,
    "release_date": 0.1
}

# Build the pairwise comparison matrix
matrix = []
for i in criteria:
    row = []
    for j in criteria:
        row.append(weights[i] / weights[j])
    matrix.append(row)

# Create a DataFrame with criteria names as both index and columns
df_matrix = pd.DataFrame(matrix, index=criteria, columns=criteria)

# Save the DataFrame to an Excel file
df_matrix.to_excel("pairwise_matrix.xlsx", index=True)

print("Excel file 'pairwise_matrix.xlsx' has been created successfully!")
