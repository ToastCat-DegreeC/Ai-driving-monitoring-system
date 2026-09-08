import pandas as pd
import numpy as np
import os

def generate_stats():
    csv_path = 'benchmark_universal_masked.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Calculate Summary Statistics
    stats = {
        'Metric': ['HR MAE (BPM)', 'SpO2 MAE (%)'],
        'Mean': [df['mae_hr'].mean(), df['mae_spo2'].mean()],
        'Median': [df['mae_hr'].median(), df['mae_spo2'].median()],
        'Std Dev': [df['mae_hr'].std(), df['mae_spo2'].std()],
        'Min': [df['mae_hr'].min(), df['mae_spo2'].min()],
        'Max': [df['mae_hr'].max(), df['mae_spo2'].max()],
        '68th Percentile': [df['mae_hr'].quantile(0.68), df['mae_spo2'].quantile(0.68)]
    }
    
    stats_df = pd.DataFrame(stats)
    
    print("### Summary Statistics (N=50)")
    print(stats_df.to_markdown(index=False))
    print("\n")
    
    # Identify Outliers (Top 5 highest HR MAE)
    outliers = df.nlargest(5, 'mae_hr')[['subject', 'version', 'mae_hr', 'mae_spo2']]
    print("### Top 5 Boundary Cases (Highest HR MAE)")
    print(outliers.to_markdown(index=False))
    print("\n")
    
    # Full Table (Formatted for Markdown)
    print("### Full Subject-by-Subject Results")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    generate_stats()
