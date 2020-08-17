This repository contains code for classifying video clips of Lake Malawi cichlids.The input is a folder of small video clips of interest and an annotation file that gives a label for each video clip. If fine-tuning is required, then a saved model file should be provided.
The scripts will first pre-process the video clips, randomly split the data into training and validation. After this, a 3D ResNet will be trained and accuracy will be reported.

ML_train.py

Master script that runs a 3d convolutional neural network to classify video clips. This script runs 1) Training of the neural network from scratch. 2) Applying a pre_trained neural network to unobserved video clips. 
To run this file, either change the default arguments or feed in the desired arguments at command line.


VideoClassification.yaml

Anaconda environment for running this repository

