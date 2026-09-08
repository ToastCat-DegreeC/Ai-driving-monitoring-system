import pandas as pd

df = pd.read_csv('benchmark_universal_masked.csv')
df['mae_hr'] = df['mae_hr'].round(2)
df['mae_spo2'] = df['mae_spo2'].round(2)

table = "**Table 4.6.2: Full 50-Subject Universal Benchmark Results**\n"
table += "| Subject ID | Dataset Version | HR MAE (BPM) | SpO2 MAE (%) |\n"
table += "| :--- | :--- | :--- | :--- |\n"
for index, row in df.iterrows():
    table += f"| {row.subject} | {row.version} | {row.mae_hr:.2f} | {row.mae_spo2:.2f} |\n"

with open("temp_table.md", "w") as f:
    f.write(table)
