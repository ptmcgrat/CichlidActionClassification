# Overview
This repository contains code for classifying video clips of Lake Malawi cichlids.The input is a folder of small video clips of interest and an annotation file that gives a label for each video clip. The 14172 video clips and their annotations can be found at https://data.mendeley.com/datasets/3hspb73m79/1


If fine-tuning is required, an additional saved model file should be provided. This saved model should include the state of the model and the state of the optimizer. This can be retrieved during the training process.

The scripts will first pre-process the video clips to continuous images, randomly split the data into training and validation. After this, a 3D ResNet will be trained and accuracy will be reported along with training.

## TrainModel.py

Master script that runs a 3d convolutional neural network to classify video clips. This script runs 1) Training of the neural network from scratch. 2) Applying a pre_trained neural network to unobserved video clips. 
Arguments for this script include the location of the video clips, labels, temporary directory for storing intermedia files as well as results folder. You can also change parameters for the deep learning model.

### How to use it
#### train from scratch
If train from scratch, follow these steps:
1. In a master directory, put the annotation file and a folder containing all the video clips
Annotation file should contain the following fields:
   1. LID: index 
   2. ClipName: Name of the video clip
   3. ManualLabel: The true label 
   4. (Optional) MLabeler: Identifier for the Labeler
   5. (Optional) MLabelTime : Labeling time
   6. Location: Name of the video clip without the ProjectID and video name
   7. MeanID
   8. (Optional) ProjectID
   9. (Optional) AnalysisID

   Example: 
   ClipName: MC16_2__0001_vid__192__2135__797__238__1036
   Location: 192__2135__797__238__1036
   MeanID: MC16_2:0001_vid
   ProjectID: MC16_2

2. In TrainModel.py, change the default parameter for 'Input_videos_directory' and 'ML_labels' to the location of video clips and annotations
3. Make sure the default parameters are correct 
   Purpose is set to 'train'
   TEST_PROJECT is set to the projects that you want to reserve for testing, separated by comma. For example, 'MC6_5,CV10_2'.
   Results_directory is set to folder at your will.
4. run TrainModel.py  

```
python TrainModel.py --ML_videos_directory $directory --ML_labels $label_file  --Purpose train --Results_directory $model-directory --resume_path $save_model_path --Log $log_file
```

## ClassifyVideo.py

Master script that applies a pre-trained neutral network for a new dataset.  

```
python ClassifyVideos.py --Input_videos_directory $directory --Videos_to_project_file $video_clip_each_video_belong_to --Trained_model $trained_model_path --Training_options $training_commands_file --Purpose classify --Trained_categories $json_file_used_in_training --Temporary_output_directory $temp_output_directory_path --Output_file $output_csv_file_path
```
The Videos_to_project_file should have these columns:
   1. LID: index 
   2. ClipName: Name of the video clip
   3. ManualLabel: The true label 
   4. (Optional) MLabeler: Identifier for the Labeler
   5. (Optional) MLabelTime : Labeling time
   6. Location: Name of the video clip without the ProjectID and video name
   7. MeanID
   8. (Optional) ProjectID
   9. (Optional) AnalysisID
   10. video_name

   Example: 
   ClipName: MC16_2__0001_vid__192__2135__797__238__1036
   Location: 192__2135__797__238__1036
   MeanID: MC16_2:0001_vid
   ProjectID: MC16_2
   video_name: 0001_vid__192__2135__797__238__1036

## Utils/prediction_to_label.py

Convert prediction confidence to labels.

```
python prediction_to_label.py ----confidence_file $confidence_file --prediction_file $prediction_label_file
```
<!-- 
## VideoClassification.yaml

Anaconda environment for running this repository -->

## environment.yml

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







 
   

