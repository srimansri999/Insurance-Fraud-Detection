import pandas as pd

# Change this path to your latest train.csv
df = pd.read_csv(
    r"E:\Udemy\Project 4\Insurance Fraud Detection\Artifact\07_14_2026_23_56_49\data_ingestion\ingested\train.csv"
)

print("Total Columns:", len(df.columns))
print()

print("Input Features:")

for col in df.columns:
    if col != "fraud_reported":
        print(col)