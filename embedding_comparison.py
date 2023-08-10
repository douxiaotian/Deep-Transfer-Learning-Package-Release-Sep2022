import csv

# Imports Harvard Embeddings into a dictionary with CUIs as keys and dimension values as dictionary values
embedding_dict = {}
with open("cui2vec_pretrained.csv", 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)   # First row specified as header

    for row in csv_reader:
        key = row[0]
        values = row[1:]
        embedding_dict[key] = values

# Imports Lab CUIs into iterable list
dummy_list = []
with open("all_feature_col.csv", 'r') as file:
    csv_reader = csv.reader(file)
    header = next(csv_reader)

    for row in csv_reader:
        dummy_list.append(row[1])

# Store matching values in new file
output_file = 'matching_symptoms.csv'

with open(output_file, 'w') as file:

    # Defines header row and puts V1 - V500
    header_row = ['Symptom Codes'] + [f'V{i}' for i in range(1, 501)] 
    file.write(','.join(header_row) + '\n')

    # Iterate through "dummy list" and check for matches after removing last two characters
    for item in dummy_list:
        
        # Check if item ends with _A or _P to perform calculations
        is_A = item.endswith('_A')
        is_P = item.endswith('_P')

        # Remove last two characters from item
        modified_item = item[:-2]

        # Check if "modified item" is in "embedding dict"
        if modified_item in embedding_dict:

            # Write "item" (includes _A or _P) as row header in file
            file.write(item + ",")

            # Write corresponding values to file
            values = embedding_dict[modified_item]
            
            # If item ends with "_A", multiply corresponding values by -1
            if is_A:
                values = [str(float(value) * -1) for value in values]
            file.write(",".join(values) + "\n")


