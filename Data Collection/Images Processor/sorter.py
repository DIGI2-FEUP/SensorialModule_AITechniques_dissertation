import csv

# Define the CSV file path
csv_file = 'data.csv'

# Read the CSV file into a list of rows
with open(csv_file, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    rows = list(csv_reader)

# Sort the rows based on the first value (assuming it's an integer)
sorted_rows = sorted(rows, key=lambda x: int(x[0]))

# Define the path for the sorted CSV file
sorted_csv_file = 'sorted_' + csv_file

# Write the sorted rows to a new CSV file
with open(sorted_csv_file, mode='w', newline='') as file:
    csv_writer = csv.writer(file)
    csv_writer.writerows(sorted_rows)

print(f"CSV file '{csv_file}' has been sorted and saved as '{sorted_csv_file}'.")