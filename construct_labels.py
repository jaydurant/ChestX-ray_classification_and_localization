import csv
import pandas as pd
import re
import os
import ast
import math
import xml.etree.ElementTree as ET
import numpy as np
from os.path import exists
#from models.selected_labels import selected_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split


parent_separator_1 = "├──"
parent_separator_2 = "└──"

selected_labels = ['aortic elongation', 'atelectasis', 'cardiomegaly', 'catheter' ,'copd signs'
 ,'pleural effusion', 'electrical device', 'emphysema' ,'heart insufficiency',
 'infiltrates', 'mass', 'nodule', 'normal', 'nsg tube', 'other findings'
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

def edit_labels_openi(dir):
    data = {"Labels": [], "ImageID":[]}
    for filename in os.listdir(dir):
        f = os.path.join(dir, filename)
        tree = ET.parse(f)
        images = tree.findall(".//parentImage")
        impression_sentence = tree.find(".//*[@Label='IMPRESSION']")
        

        for image in images:
            image_file = image.attrib['id']
            data['Labels'].append(image_file)
            data['ImageID'].append(impression_sentence)
            
    df = pd.DataFrame(data)
    df.to_csv('openi_image_labels.csv')


def combine_ds_labels(file_label):
    padchest_labels = pd.read_csv("./padchest_img_labels.csv")
    openi_labels = ["No Finding", "Enlarged Cardiomediastinum",
     "Cardiomegaly", "Lung Lesion", "Lung Opacity", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Plerual Other", "Fracture", "Support Devices"]
    df = pd.read_csv(file_label)
    rows_to_add = []
    for row_index in df.index:
        new_label_arrays = []
        for i, label in enumerate(openi_labels):
            row_value = df.at[row_index, label]

            if row_value and i == 0:
                new_label_arrays.append("normal")
            elif row_value and i == 1:
                new_label_arrays.append("other findings")
            elif row_value and i == 2:
                new_label_arrays.append("cardiomegaly")
            elif row_value and i == 3:
                new_label_arrays.append("mass")
            elif row_value and i==4:
                new_label_arrays.append("other findings")
            elif row_value and i==5:
                new_label_arrays.append("pulmonary edema")
            elif row_value and i==6:
                new_label_arrays.append("other findings")
            elif row_value and i==7:
                new_label_arrays.append("pneumonia")
            elif row_value and i==8:
                new_label_arrays.append("atelectasis")
            elif row_value and i==9:
                new_label_arrays.append("pneumothorax")
            elif row_value and i==10:
                new_label_arrays.append("pleural effusion")
            elif row_value and i==11:
                new_label_arrays.append("other findings")
            elif row_value and i==12:
                new_label_arrays.append("thoracic cage deformation")
            elif row_value and i==13:
                new_label_arrays.append("electrical devices")
        new_label_arrays = list(set(new_label_arrays))
        rows_to_add.append([df.at[row_index,"ImageId"], new_label_arrays])
    padchest_labels.append(rows_to_add)

    padchest_labels.to_csv("padchest_openi_labels.csv")
    print("finished combining labels")





def edit_labels_csv_file_padchest(annontated_image_csv_file, label_csv_file ):
    df = pd.read_csv(annontated_image_csv_file, index_col=0)
    df['Labels'] = df['Labels'].str.strip('[]').str.replace("'", "").str.split(',')
    label_dict, _ = generate_label_map(label_csv_file)
    drop_rows = []
    #print(label_dict)
    for row_index in df.index:
        #print(row_index)
        new_labels = []
        row_labels = df.at[row_index, "Labels"]
        #print(row_labels)
        
        
        image_path = df.at[row_index, "ImageID"]
        #print(os.path.join("./data", image_path))
        #print(exists(os.path.join("./data", image_path)))
        
        if not exists(os.path.join("./data", image_path)):
            #print("hello")
            drop_rows.append(row_index)
        
        if not isinstance(row_labels, list):
            continue
        for label in row_labels:
            label = label.strip().lower()

            #Check if label is in label map and if there is a parent
            if label in label_dict and label_dict[label]:
                label = label_dict[label]

            if label in selected_labels:
                new_labels.append(label)

        if len(new_labels) == 0:
            new_labels = ["other findings"]
        
        df.at[row_index, "Labels"] = list(set(new_labels))
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
    X = np.array(df["ImageID"].tolist()).reshape((Xlen,1))
    X_trainval, y_trainval, X_test, y_test = iterative_train_test_split(X, y, test_size=0.1)
    X_train, y_train, X_val, y_val = iterative_train_test_split(X_trainval, y_trainval, test_size=0.1)
    
    print("Finished  building train, val, and test sets")
    
    return (X_train, y_train, X_test, y_test, X_val, y_val)
