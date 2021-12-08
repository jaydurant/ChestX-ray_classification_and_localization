import csv
import re

parent_separator_1 = "├──"
parent_separator_2 = "└──"


pattern = '([a-z]+\s)+'

def generate_label_map():
    label_dict = {}
    current_parent = None
    label_dict["normal"] = ""
    label_dict[""]

    with open('tree_term_CUI_counts_160K.csv', newline='') as csvfile:
        reader  = csv.reader(csvfile)

        for row in reader:
            label_str = row[0].lower()
            match  = re.search(pattern, label_str)
            label_str_clean = match.group(0).strip()
            
            if label_str.startswith(parent_separator_1) or label_str.startswith(parent_separator_2):
                label_dict[label_str_clean] = ""
                current_parent = label_str_clean
            else:
                label_dict[label_str_clean] = current_parent
        
    return (label_dict, len(label_dict))

