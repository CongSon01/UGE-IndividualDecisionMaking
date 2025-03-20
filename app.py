# app.py
import argparse
import os
import io
import base64
import time
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash, send_file

# Import our MCDM modules
from weights import *
from ranking import *

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your own secret key

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Retrieve weight method (ahp or entropy) from form
        weight_method = request.form.get("weight_method")
        
        # Retrieve the main dataset file: cleaned_samsung_phones.csv
        csv_file = request.files.get("csv_file")
        if csv_file and csv_file.filename != "":
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                flash("Error reading CSV file: " + str(e))
                return redirect(request.url)
        else:
            flash("Please upload the 'cleaned_samsung_phones.csv' file.")
            return redirect(request.url)
        
        # If AHP is selected, retrieve the pairwise matrix file
        if weight_method == "ahp":
            pairwise_file = request.files.get("pairwise_file")
            if not pairwise_file or pairwise_file.filename == "":
                flash("Please upload the 'pairwise_matrix.xlsx' file for AHP weight computation.")
                return redirect(request.url)
        
        # Define criteria columns (all columns except 'model_name')
        criteria_columns = [col for col in df.columns if col != "model_name"]
        
        # Define criteria type: if column name contains 'price', treat it as cost; otherwise, benefit.
        criteria_type = {}
        for col in criteria_columns:
            if "price" in col.lower():
                criteria_type[col] = "cost"
            else:
                criteria_type[col] = "benefit"
        
        # Compute weights and measure execution time for weight computation
        start_weight = time.perf_counter()
        try:
            if weight_method == "ahp":
                # Read the uploaded pairwise file from memory using BytesIO
                pairwise_bytes = pairwise_file.read()
                pairwise_io = io.BytesIO(pairwise_bytes)
                weights = compute_ahp_weights_from_excel(pairwise_io)
            else:
                weights = compute_entropy_weights(df, criteria_columns, criteria_type)
        except Exception as e:
            flash("Error computing weights: " + str(e))
            return redirect(request.url)
        end_weight = time.perf_counter()
        weight_time = end_weight - start_weight
        
        # Define ranking methods to execute (all methods)
        ranking_methods = {
            "WSM": rank_wsm,
            "WPM": rank_wpm,
            "WASPAS": rank_waspas,
            "TOPSIS": rank_topsis,
            "PROMETHEE": rank_promethee,
            "VIKOR": rank_vikor
        }
        ranking_results = {}
        ranking_times = {}

        # Directory to save full ranking files
        output_folder = os.path.join("static", "results")
        os.makedirs(output_folder, exist_ok=True)
        
        # Execute each ranking algorithm and record the top 10 results & execution time.
        for method_name, func in ranking_methods.items():
            start_rank = time.perf_counter()
            # For WASPAS and VIKOR, we pass a default parameter.
            if method_name == "WASPAS":
                result_df = func(df, criteria_columns, weights, criteria_type, lambda_val=0.5)
            elif method_name == "VIKOR":
                result_df = func(df, criteria_columns, weights, criteria_type, v=0.5)
            else:
                result_df = func(df, criteria_columns, weights, criteria_type)
            end_rank = time.perf_counter()
            ranking_time = end_rank - start_rank

            top_10 = result_df.head(10).copy()
            scores = top_10['score']
            mean_score = scores.mean()
            std_score = scores.std()
            cv_value = (std_score / mean_score) if mean_score != 0 else 0.0

            # Convert to HTML
            table_html = top_10.to_html(classes="table table-striped", index=False)

            # Save full ranking to excel file for download
            filename = f"{weight_method}_{method_name}_full_ranking.xlsx"
            filepath = os.path.join(output_folder, filename)
            result_df.to_excel(filepath, index=False)

            # Store results along with CV
            ranking_results[method_name] = {
                'table': table_html,
                'CV': cv_value,
                'download': filename
            }
            ranking_times[method_name] = ranking_time
        
        # Create a pie chart of criteria weights using matplotlib
        fig1, ax1 = plt.subplots()
        labels = list(weights.keys())
        sizes = list(weights.values())
        print(weights)
        ax1.pie(sizes, labels=labels, autopct='%1.2f%%', startangle=90)
        ax1.axis('equal')
        pie_chart = io.BytesIO()
        plt.savefig(pie_chart, format='png')
        pie_chart.seek(0)
        pie_chart_base64 = base64.b64encode(pie_chart.read()).decode('utf-8')
        plt.close(fig1)
        
        # Create a bar chart comparing ranking algorithm execution times
        fig2, ax2 = plt.subplots()
        methods = list(ranking_times.keys())
        times_list = [ranking_times[m] for m in methods]
        bars = ax2.bar(methods, times_list, color='skyblue')
        ax2.set_ylabel("Execution Time (seconds)")
        ax2.set_title("Ranking Algorithms Execution Time Comparison")

        # Annotate each bar with the execution time
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, height + 0.001, f"{height:.6f}",
                    ha='center', va='bottom')

        # Save the bar chart to a bytes buffer
        bar_chart = io.BytesIO()
        plt.savefig(bar_chart, format='png')
        bar_chart.seek(0)
        bar_chart_base64 = base64.b64encode(bar_chart.read()).decode('utf-8')
        plt.close(fig2)

        
        # Render the results page with weights pie chart, top 10 ranking tables, and ranking time bar chart.
        return render_template("results.html",
                               weights=weights,
                               weight_time=weight_time,
                               ranking_results=ranking_results,
                               ranking_times=ranking_times,
                               pie_chart=pie_chart_base64,
                               bar_chart=bar_chart_base64)
    return render_template("index.html")

# Route for downloading full ranking file
@app.route('/download/<method>')
def download_file(method):
    filename = f"{method}_full_ranking.xlsx"
    filepath = os.path.join("static", "results", filename)
    return send_file(filepath, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8888)
