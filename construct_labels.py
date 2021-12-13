import csv
import pandas as pd
import re
import os
import ast
import math
import numpy as np
from os.path import exists
#from models.selected_labels import selected_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split


parent_separator_1 = "├──"
parent_separator_2 = "└──"

selected_labels = ['aortic elongation', 'atelectasis', 'cardiomegaly', 'catheter' ,'copd signs'
 ,'effusion', 'electrical device', 'emphysema' ,'heart insufficiency',
 'infiltration', 'mass', 'nodule', 'normal', 'nsg tube', 'other findings'
 ,'pneumonia', 'pneumothorax', 'pulmonary edema', 'pulmonary fibrosis',
 'surgery', 'thoracic cage deformation', 'tuberculosis']

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
    drop_rows = []

    for row_index in df.index:
        #print(row_index)
        new_labels = []
        row_labels = df.at[row_index, "Labels"]
        #print(row_labels)

        image_path = df.at[row_index, "ImageID"]
        print(os.path.join("./data", image_path))
        print(exists(os.path.join("./data", image_path)))
        if not exists(os.path.join("./data", image_path)):
            print("hello")
            drop_rows.append(row_index)

        if not isinstance(row_labels, list):
            continue
        for label in row_labels:
            label = label.strip()
            #Check if label is in label map and ifSSS there is a parent
            if label in label_dict and label_dict[label]:
                label = label_dict[label]

            if label in selected_labels:
                new_labels.append(label)

        if len(new_labels) == 0:
            new_labels = ["other findings"]
        
        df.at[row_index, "Labels"] = new_labels
    print(drop_rows,"drop rows")
    df = df.drop(labels=drop_rows, axis=0)
    
    df.drop(df.columns.difference(['ImageID','Labels']), 1, inplace=True)
    curr_dir = os.getcwd()
    df.to_csv(os.path.join(curr_dir, "padchest_img_labels.csv"), index=False)
    print("finished editing ")

def generate_test_val_train_datasets(file):
    df = pd.read_csv(file)
    df['Labels'] = df['Labels'].str.strip('[]').str.replace("'", "").str.split(',')
    mlb = MultiLabelBinarizer()
    mlb.fit([selected_labels])

    for row_index in df.index:
        new_labels = []
        labels = df.at[row_index, "Labels"]
        #print(row_index, type(labels))
        if type(labels) != list and math.isnan(labels):
            labels = ['normal']

        for label in labels:
            label = label.strip()
            new_labels.append(label)


        binary_labels = mlb.transform([new_labels])[0]
        df.at[row_index, "Labels"] = binary_labels
    
    trainval, test = train_test_split(df, test_size=0.1, random_state=1, shuffle=True)
    train, val = train_test_split(trainval, test_size=0.1, random_state=1, shuffle=True)
    
    print("Finished  building train, val, and test sets")
    
    return (train, val, test)

def generate_stratified_test_val_train_datasets(file):
    df = pd.read_csv(file)
    df['Labels'] = df['Labels'].str.strip('[]').str.replace("'", "").str.split(',')
    mlb = MultiLabelBinarizer()
    mlb.fit([selected_labels])

    for row_index in df.index:
        new_labels = []
        labels = df.at[row_index, "Labels"]
        #print(row_index, type(labels))
        if type(labels) != list and math.isnan(labels):
            labels = ['normal']

        for label in labels:
            label = label.strip()
            new_labels.append(label)


        binary_labels = mlb.transform([new_labels])[0]
        df.at[row_index, "Labels"] = binary_labels
    Xlen = len(df["ImageID"])
    y = np.array(df["Labels"].tolist())
    X = np.array(df["ImageId"].tolist()).reshape((Xlen,1))
    X_trainval, y_trainval, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1, random_staet=1)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_trainval, y_trainval, test_size=0.1, random_state=1)
    
    print("Finished  building train, val, and test sets")
    
    return (X_train, y_train, X_test, y_test, X_val, y_val)
