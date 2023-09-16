import os
import csv

output_filename = 'combined_data.csv'
base_path = "C:\\Users\\Edwin\\Desktop\\Career\\Research\\Induction Motor Inter-turn\\Data"
status_range = ['NO FAULT','A-G','B-G','C-G','A-B','B-C','A-C','ABC','ABCG']
load_range = [round(0.1 * i, 1) for i in range(1, 10)]
time_range = [round(0.1 * i, 1) for i in range(3, 9)]
combined_data = []
i = 0
for status in status_range:
    for load in load_range:
        for time in time_range:
            foldername = os.path.join(base_path, str(status), str(load), str(time))
            filename = os.path.join(foldername, 'matrix.csv')
            
            if os.path.exists(filename):
                with open(filename, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    data = [row for row in csv_reader]
                    
                    for row in data:
                        row.append(status)
                        row.append(i)  # Append status to the end of each row
                        combined_data.append(row)
    i+=1

with open(output_filename, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(combined_data)

print('Combined CSV file has been created:', output_filename)

