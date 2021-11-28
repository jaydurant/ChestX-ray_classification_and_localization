from utils.preprocessing_images import  download_process_images


bucket = "xray_samples"

#download zip files from google bucket and download and process images
download_process_images(bucket)