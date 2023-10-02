import os
import csv


def write_folder_names_to_csv(folder_path, output_file):
    # Get the list of file names in the folder
    file_names = os.listdir(folder_path)

    # Create or overwrite the output CSV file
    with open(output_file, 'w', newline='') as f_csv:
        # Create a CSV writer
        writer_csv = csv.writer(f_csv)

        # Create or overwrite the output text file
        output_text_file = os.path.splitext(output_file)[0] + ".txt"
        with open(output_text_file, 'w') as f_txt:
            # Write each file name as a row in the CSV file and text file with a counter
            for i, file_name in enumerate(file_names, start=1):
                if file_name.endswith('.jpg'):
                    writer_csv.writerow([file_name, i])
                    f_txt.write(file_name + '\n')

    print(f"File names written to {output_file} and {output_text_file} successfully.")

# change the file name here
folder_path = "./Dataset/test/"
output_file = "./Dataset/test/file_names.csv"

write_folder_names_to_csv(folder_path, output_file)
