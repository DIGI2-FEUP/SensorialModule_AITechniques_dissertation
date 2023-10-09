import pandas as pd

# Column names for the first CSV file
columns1 = ["id", "pH", "temp", "press", "hum", "sol_temp"]

# Column names for the second CSV file
columns2 = ["id", "red", "green", "blue"]

# Load the second CSV file with specified column names
df2 = pd.read_csv("../Images Processor/rgb_dataset.csv", header=None, names=columns2)

# Load the first CSV file with specified column names
df1 = pd.read_csv("../Logs Processor/dataset.csv", header=None, names=columns1)

# Merge the two dataframes on the "id" column
merged_df = pd.merge(df1, df2, on="id")

# Save the merged dataframe to a new CSV file
merged_df.to_csv("dataset.csv", index=False)