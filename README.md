# ChestX-ray_classification_and_localization

This is a Python3 implementation of Unified Deep CNNs and CheXnet in order to classify X-ray images and generate heatmaps which function as weakly supervised localization of various radiological findings.

## Dataset

Aproximately 200,000 X-ray images are utilzied for this project. The images are sourced from the [PadChest Dataset](https://bimcv.cipf.es/bimcv-projects/padchest/) and the[OpenI](https://openi.nlm.nih.gov/) medical search engine. There are 22 radiological findings which include labels such as Pleural Effusion, Cardiomegaly, Mass, Nodule which are commonly found in other X-ray datasets; in addition, the new labels are introduced such as Electrical Device and NSG Tube. There are multiple views used in this dataset not just the anterior posterior(AP) view which are included in the PadChest dataset. In addition, the files from the OpenI search engine are in the DICOM format which will need to be processed and converted to PNG formats. In total the combined dataset is approximately 1 terrabyte in size which required uploading their zip files to Google Cloud Storage and then downloaded and unzipped on a Virtual Machine.

##  Prerequisites

- Python 3.6 +
- [Pytorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Pandas](https://pandas.pydata.org/)
- [scikit-image](https://scikit-image.org/)
- [scikit-multilearn](http://scikit.ml/)

In order to run this repo, it is assumed the image zip files containing the images have been uploaded to a google cloud storage bucket and have been given read access. IN addition, the repo has been cloned on  GCP VM with access to said bucket containing the zip files.

## Usage
There are several commands which need to be run in order to run this repo. As well, in order to replicate results training will need to be for different epoch counts according to the model being run. Training time can take anywhere from 6 to 20 hours to train.

1. Clone this repository on a Google VM with access to a Nvidia GPU

2. Download the image zip files from PadChest and OpenI search engine. For Padchest they can downloaded from the homepage after agreeing to only this data for research purposes and OpenI image can be used without any significant preconditions.

3. Once the image zip files have ben uploaded to a github bucket, run:

```
python download_images.py
```

You will need to updte the bucket variable name in the download_images.py file.

4. Next take the PadChest label files and move them to the root directory; in addition ensure that the files for the openi labels have been unzipped and placed at the home directory of this repo:

```
python parselabelfile.py
```

5. The padchest file will now be ready but the OpenI labels will not be ready to be used. By utilizing the [chexpert labeler](https://github.com/stanfordmlgroup/chexpert-labeler) a labeled file can be produced. Move the labeled file to the home directory.

6. Now run combine the OpenI and PadChest Label files by running
```
python combinelabels.py
```

7. Finally the test and training runs can be executed with (values for args can be inspected in the traintest file):

```
python traintest.py --is_training 1 --epochs 20 etc...
```







