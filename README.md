# Multi-Criteria Decision Making (MCDM) for Samsung Phones

[Demo](http://ec2-51-20-54-194.eu-north-1.compute.amazonaws.com:8888/)

This project implements several Multi-Criteria Decision Making (MCDM) methods on a Samsung Phones dataset. The goal is to compute criteria weights and rank phone models (alternatives) based on multiple criteria using various algorithms.

## Installation & Running the Code

### Prerequisites

- **Python 3.6+** is required.
- Install necessary Python packages:
  - `numpy`
  - `pandas`
  - `openpyxl` (for reading Excel files)

### Step-by-Step Instructions

1. **Clone or Download the Project:**

   - Place all the provided `.py` files and the sample Excel/CSV files (`cleaned_samsung_phones.csv` and `pairwise_matrix.xlsx`) into the same directory.
2. **Create a Virtual Environment (Optional but Recommended):**

```bash
   python -m venv mcdm_env
   source mcdm_env/bin/activate   # On Windows: mcdm_env\Scripts\activate
```

3. **Install Required Packages:**

```bash
    pip install -r requirements.txt
```

4. **Prepare the Data Files:**
   Ensure that cleaned_samsung_phones.csv (the dataset) and pairwise_matrix.xlsx (the pairwise comparison matrix) are in the same directory as the code, or note their file paths.
5. **Run the Code:**
   The main script (`main.py`) accepts command-line arguments to choose the weight and ranking methods.
   Example Commands:
   Using AHP for Weights and TOPSIS for Ranking:

```bash
    python main.py --weight_method ahp --pairwise_file pairwise_matrix.xlsx --rank_method topsis
```

Using Entropy for Weights and WASPAS for Ranking:

```bash
    python main.py --weight_method entropy --rank_method waspas
```

### Running the Flask Web Application
1. **Start the Flask app:**
```bash
    python app.py
```
2. **Open your browser and go to:**
```bash
    http://127.0.0.1:8888
```
3. **Open your browser and go to:**
```bash
    http://127.0.0.1:8888
```
4. Upload cleaned_samsung_phones.csv and, if using AHP, provide pairwise_matrix.xlsx.
5. Click "Execute" to run the ranking methods and view results.
## 1. Problem Overview

In this project, we address a typical MCDM problem:

- **Weight Computation:** Determine the importance of each criterion using methods such as:

  - **AHP (Analytic Hierarchy Process):** Uses a pairwise comparison matrix (with consistency ratio CR < 0.1) provided in an Excel file.
  - **Entropy:** Computes weights automatically based on the dispersion (entropy) of the data.
- **Ranking Alternatives:** Once the criteria weights are computed, rank the alternatives (Samsung phone models) using different methods:

  - **WSM (Weighted Sum Model)**
  - **WPM (Weighted Product Model)**
  - **WASPAS (Weighted Aggregated Sum Product Assessment)**
  - **TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)**
  - **PROMETHEE (Preference Ranking Organization Method for Enrichment Evaluations)**
  - **VIKOR (VIseKriterijumska Optimizacija I Kompromisno Rangiranje)**

## 2. Dataset Description

- **cleaned_samsung_phones.csv:**Contains the alternatives (phone models) and various criteria (e.g., price, battery, camera, etc.). The column `model_name` represents the phone model. The remaining columns represent performance, cost, and quality measures.
- **pairwise_matrix.xlsx:**An Excel file with a 7x7 pairwise comparison matrix for the following criteria:

  - `popularity`
  - `best_price`
  - `highest_price`
  - `screen_size`
  - `memory_size`
  - `battery_size`
  - `release_date`

  The matrix is built using the formula:
  \[ a_{ij} = \frac{w_i}{w_j} \]
  (for a sample weight vector), which guarantees a Consistency Ratio (CR) of 0.

## 3. Implemented Algorithms

### Weight Computation Methods

- **AHP:**Reads a pairwise comparison matrix from `pairwise_matrix.xlsx`, computes the principal eigenvector, checks the Consistency Ratio (CR < 0.1), and returns the normalized criteria weights.
- **Entropy:**
  Uses the dispersion of data for each criterion to compute weights in an entirely data-driven and objective manner.

### Ranking Methods

- **WSM (Weighted Sum Model):**Normalizes the data using Min-Max normalization and computes the weighted sum.
- **WPM (Weighted Product Model):**Uses the product of normalized values raised to the power of their weights.
- **WASPAS:**Combines the scores from WSM and WPM using a balancing parameter (λ).
- **TOPSIS:**Uses vector normalization, identifies the ideal and anti-ideal solutions, computes Euclidean distances, and calculates a closeness coefficient.
- **PROMETHEE:**Performs pairwise comparisons with a simple preference function to calculate net flow (phi) scores.
- **VIKOR:**
  Calculates a compromise ranking index based on the aggregated weighted distances from the ideal solution.

## 4. Code Structure

The project is organized into multiple Python modules:

- **common_utils.py:**Contains utility functions (e.g., normalization using Min-Max and vector normalization).
- **module_ahp.py:**Implements AHP weight computation by reading a pairwise comparison matrix from an Excel file.
- **module_entropy.py:**Implements the Entropy method to compute criteria weights based on dataset dispersion.
- **module_wsm.py, module_wpm.py, module_waspas.py, module_topsis.py, module_promethee.py, module_vikor.py:**Each module implements one of the ranking algorithms.
- **main.py:**
  The main driver script that reads the dataset, selects the weight and ranking method based on command-line parameters, computes weights and rankings, and measures the execution time.
