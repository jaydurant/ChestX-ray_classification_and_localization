#from gcp_utils import download_blob
import zipfile
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.io import read_image
from pathlib import Path
from PIL import Image
 
curr_dir = os.getcwd()
 
print(curr_dir)
transform = transforms.Compose([
   transforms.Resize((300,300))
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
 
def preproces_images(bucket, filename, dest, img_size=250):
   """Process image samples to decrease image sample size"""
   #download images zipfile
   #download_blob(bucket, filename, dest)
 
   image_samples = extract_files(dest)
 
   for image_fp in image_samples:
       image_path = os.path.join(curr_dir,"data_raw", image_fp)
       print(image_path)
       image_obj = Image.open(image_path)
       resized_image = transform(image_obj)
       resized_image.save(Path('data') / image_fp, 'PNG')
 
check_dir_exists("data_raw")
check_dir_exists("data")
preproces_images("", "","sample.zip")