from utils.gs_utils import download_blob
import zipfile
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_image
from pathlib import Path
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
 
curr_dir = os.getcwd()
 
print(curr_dir)
transform = transforms.Compose([
   transforms.Resize((150,150))
])
 
def check_dir_exists(directory):
   curr_path = os.path.join(curr_dir, directory)
   if not os.path.exists(curr_path):
       os.makedirs(curr_path)
 
 
def extract_files(file):
   """Extract image samples from zip files"""
   extracted_files = []
   extract_dest = os.path.join(curr_dir, "data_raw")
   with zipfile.ZipFile(file, "r") as zip_ref:
       list_of_files = zip_ref.namelist()
 
       zip_ref.extractall(extract_dest)
 
       for file in list_of_files:
           basename = os.path.basename(file)
           extracted_files.append(basename)
   return extracted_files
 
def preprocess_images(bucket, filename, dest, img_size=250):
    """Process image samples to decrease image sample size"""
    #download images zipfile
    download_blob(bucket, filename, dest)

    image_samples = extract_files(dest)

    for image_fp in image_samples:
        try:
            image_path = os.path.join(curr_dir,"data_raw", image_fp)
            #print(image_path)
            image_obj = Image.open(image_path)
            resized_image = transform(image_obj)
            resized_image.save(Path('data') / image_fp, 'PNG')
        except Exception as e:
            print(e)
            print(image_fp)
#create directories to hold raw image data and transformed data
check_dir_exists("data_raw")
check_dir_exists("data")

zip_arr = list(range(13,51))
zip_arr.append(54)

def download_process_images(bucket):
    for i in zip_arr:
        
        filename = "{}.zip".format(i)
        newzip_path = os.path.join(curr_dir, filename)

        print("preprocess images from zip {}".format(i))
        preprocess_images(bucket, filename, newzip_path)
        print("finish preprocess from zip {}".format(i))
        #remove zip file
        os.remove(newzip_path)
        #remove previous images in data_raw directory
        raw_data_path = os.path.join(curr_dir, "data_raw")
        
        for f in os.listdir(raw_data_path):
            os.remove(os.path.join(raw_data_path, f))

bucket = "xray_samples"
#download_process_images(bucket)

def resize_images(dir):
    count = 0
    for filename in os.listdir(dir):
        try:
            count += 1
            print(count, filename)
            img_path = os.path.join(dir, filename)
            img_obj = Image.open(img_path)
            resized_image = transform(img_obj)

            resized_image.save(img_path, "PNG")
        
        except Exception as e:
            print(e)
            print(filename)
