import pandas as pd

df = pd.read_csv('benchmark_universal_masked.csv')
n = len(df)
half = (n + 1) // 2
df1 = df.iloc[:half].reset_index(drop=True)
df2 = df.iloc[half:].reset_index(drop=True)

table = "**Table 4.6.2: Full 50-Subject Universal Benchmark Results (Compact View)**\n"
table += "| Subject (ID) | HR MAE | SpO2 | | Subject (ID) | HR MAE | SpO2 |\n"
table += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"

for i in range(half):
    row1 = df1.iloc[i]
    val1 = f"| {row1.subject} ({row1.version}) | {row1.mae_hr:.2f} | {row1.mae_spo2:.2f} |"
    if i < len(df2):
        row2 = df2.iloc[i]
        val2 = f" {row2.subject} ({row2.version}) | {row2.mae_hr:.2f} | {row2.mae_spo2:.2f} |"
    else:
        val2 = " | | |"
    table += val1 + val2 + "\n"

with open("temp_compact_table.md", "w") as f:
    f.write(table)
