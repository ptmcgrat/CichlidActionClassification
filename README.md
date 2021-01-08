# Overview
This repository contains code for classifying video clips of Lake Malawi cichlids.The input is a folder of small video clips of interest and an annotation file that gives a label for each video clip. The 14172 video clips and their annotations can be found at https://data.mendeley.com/datasets/3hspb73m79/draft?a=b72c1f6d-505a-431a-ba3d-824cd148c01e


If fine-tuning is required, an additional saved model file should be provided. This saved model should include the state of the model and the state of the optimizer. This can be retrieved during the training process.

The scripts will first pre-process the video clips to continuous images, randomly split the data into training and validation. After this, a 3D ResNet will be trained and accuracy will be reported along with training.

## ML_train.py

Master script that runs a 3d convolutional neural network to classify video clips. This script runs 1) Training of the neural network from scratch. 2) Applying a pre_trained neural network to unobserved video clips. 
To run this file, either change the default arguments or feed in the desired arguments at command line. You can find how to use the arguments using the argument help. 


## VideoClassification.yaml

Anaconda environment for running this repository




# How to use it
## train from scratch
If train from scratch, follow these steps:
1. In a master directory, put the annotation file and a folder containing all the video clips
2. In ML_train.py, change the default parameter for 'ML_videos_directory' and 'ML_labels' to the location of video clips and annotations
3. Make sure the default parameters are correct 
   Purpose is set to 'train'
   TEST_PROJECT is set to the projects that you want to reserve for testing, separated by comma. For example, 'MC6_5,CV10_2'.
   Results_directory is set to folder at your will.
4. run ML_train.py   
   

