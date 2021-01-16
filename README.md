# Overview
This repository contains code for classifying video clips of Lake Malawi cichlids.The input is a folder of small video clips of interest and an annotation file that gives a label for each video clip. The 14172 video clips and their annotations can be found at https://data.mendeley.com/datasets/3hspb73m79/draft?a=b72c1f6d-505a-431a-ba3d-824cd148c01e


If fine-tuning is required, an additional saved model file should be provided. This saved model should include the state of the model and the state of the optimizer. This can be retrieved during the training process.

The scripts will first pre-process the video clips to continuous images, randomly split the data into training and validation. After this, a 3D ResNet will be trained and accuracy will be reported along with training.

## TrainModel.py

Master script that runs a 3d convolutional neural network to classify video clips. This script runs 1) Training of the neural network from scratch. 2) Applying a pre_trained neural network to unobserved video clips. 
Arguments for this script include the location of the video clips, labels, temporary directory for storing intermedia files as well as results folder. You can also change parameters for the deep learning model.

### How to use it
#### train from scratch
If train from scratch, follow these steps:
1. In a master directory, put the annotation file and a folder containing all the video clips
2. In ML_train.py, change the default parameter for 'ML_videos_directory' and 'ML_labels' to the location of video clips and annotations
3. Make sure the default parameters are correct 
   Purpose is set to 'train'
   TEST_PROJECT is set to the projects that you want to reserve for testing, separated by comma. For example, 'MC6_5,CV10_2'.
   Results_directory is set to folder at your will.
4. run ML_train.py  

```
python ML_train.py --ML_videos_directory $directory --ML_labels $label_file --Purpose train --Results_directory $model-directory
```

## ClassifyVideo.py

Master script that applies a pre-trained neutral network for a new dataset.  

```
python ClassifyVideo.py --ML_videos_directory $directory --Clips_annotations $video_clip_each_video_belong_to --Purpose classify --Train_json json_file_used_in_training
```




## VideoClassification.yaml

Anaconda environment for running this repository

## Utils/CichlidActionRecognition.py

Master script to run the 3d-resnet model, including training, validation and testing.

## Utils/DataPrepare.py
Master script to prepare the dataset. This script
   1.Convert video clips to images for faster loading
   2.Calculate mean RGB value for each project
   3.Split training, validation and test data
   4.Create a json file containing combining clip annotations


## Utils/data_loader.py
Utility script used by CichlidActionRecognition.py to load images and labels.
## Utils/interpret_annotations.py
Utility script used by DataPrepare.py to convert textual labels to numbers and vice versa.
## Utils/model.py
Utility script used by CichlidActionRecognition.py to construct the 3d resnet model.
## Utils/split_by_animal.py
Utility script used by DataPrepare.py to do special dataset splitting.
## Utils/training_size_test.py
Utility to do ablation studies for the size of the dataset.
## Utils/transforms.py
Utility script used by CichlidActionRecognition.py to do spatial and temporal transformation for each video clip.
## Utils/utils.py
Utility script used by CichlidActionRecognition.py to keep track of accuracy.







 
   

