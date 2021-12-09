import csv
import pandas as pd
import re
import os
import ast
from models.selected_labels import selected_labels

parent_separator_1 = "├──"
parent_separator_2 = "└──"


pattern = '([a-z]+\s)+'

def generate_label_map(csv_file):
    label_dict = {}
    current_parent = None
    label_dict["normal"] = ""
    label_dict["exclude"] = ""

    with open(csv_file, newline='') as csvfile:
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


def edit_labels_csv_file_padchest(annontated_image_csv_file, label_csv_file ):
    df = pd.read_csv(annontated_image_csv_file, index_col=0)
    df['Labels'] = df['Labels'].str.strip('[]').str.replace("'", "").str.split(',')
    label_dict, _ = generate_label_map(label_csv_file)

    for row_index in df.index:
        #print(row_index)
        new_labels = []
        row_labels = df.at[row_index, "Labels"]
        #print(row_labels)

        if not isinstance(row_labels, list):
            continue
        for label in row_labels:
            label = label.strip()
            #Check if label is in label map and if there is a parent
            if label in label_dict and label_dict[label]:
                label = label_dict[label]

            if label in selected_labels:
                new_labels.append(label)

        if len(new_labels) == 0:
            new_labels = ["other findings"]
        
        df.at[row_index, "Labels"] = new_labels
    curr_dir = os.getcwd()
    df.drop(df.columns.difference(['ImageID','Labels']), 1, inplace=True)
    df.to_csv(os.path.join(curr_dir, "padchest_img_labels.csv"), index=False)
    print("finished editing ")


edit_labels_csv_file_padchest("/home/jason/projects/ChestX-ray_classification_and_localization/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv", "/home/jason/projects/ChestX-ray_classification_and_localization/tree_term_CUI_counts_160K.csv")

