import csv
import os


def rename_most_recent_file(directory_path, new_name):
    files = os.listdir(directory_path)

    # Filter out directories and keep only files
    files = [file for file in files if os.path.isfile(os.path.join(directory_path, file))]

    if not files:
        print("No files found in the directory.")
        return

    # Get the most recent file based on its modification time
    most_recent_file = max(files, key=lambda f: os.path.getmtime(os.path.join(directory_path, f)))

    # Generate the new path with the new name
    new_path = os.path.join(directory_path, new_name)

    # Rename the most recent file
    os.rename(os.path.join(directory_path, most_recent_file), new_path)
    print(f"Renamed '{most_recent_file}' to '{new_name}'.")


def remove_empty_lines(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Remove empty lines
    lines = [line for line in lines if line.strip()]

    with open(file_path, 'w') as file:
        file.writelines(lines)


def process_lines_and_convert_to_csv(input_file_path, output_csv_path, keyword):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    # Process lines starting with the specified keyword and remove the keyword
    processed_lines = [line.replace(keyword, '', 1).strip() for line in lines if line.strip().startswith(keyword)]

    with open(output_csv_path, 'w') as csv_file:
        for line in processed_lines:
            csv_file.write(line + '\n')


def append_csv(file_to_append, target_file):
    with open(file_to_append, 'r') as file1, open(target_file, 'a', newline='') as file2:
        csv_reader = csv.reader(file1)
        csv_writer = csv.writer(file2)

        for row in csv_reader:
            csv_writer.writerow(row)


# directory_path = '..\Data Collection\Timed Collection\logs'
directory_path = 'log'
new_name = 'input.log'
rename_most_recent_file(directory_path, new_name)
# Replace with the path of the input file
# input_file_path = '..\Data Collection\Timed Collection\logs\input.log'
input_file_path = 'log\input.log'
# Replace with the path of the output file
output_csv_path = 'output.csv'
keyword = 'sample:'
remove_empty_lines(input_file_path)
process_lines_and_convert_to_csv(input_file_path, output_csv_path, keyword)
print(f"Lines starting with '{keyword}' processed and saved in CSV file.")
# Replace 'file_to_append.csv' with the source CSV file you want to append
file_to_append = 'output.csv'
# Replace 'target_file.csv' with the target CSV file you want to append to
target_file = 'dataset.csv'
append_csv(file_to_append, target_file)
print(f"Content from '{file_to_append}' appended to '{target_file}'.")
