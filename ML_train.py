import argparse, subprocess, datetime, os, pdb, sys
from Utils.CichlidActionRecognition import ML_model


parser = argparse.ArgumentParser(description='This script takes video clips and annotations, either train a model from scratch or finetune a model to work on the new animals not annotated')
# Input data
parser.add_argument('--ML_videos_directory',
                    type = str, 
                    default = '/data/home/llong35/data/labeled_videos',
                    required = False, 
                    help = 'Name of directory to hold videos to annotate for machine learning purposes')

parser.add_argument('--Unlabeled_videos_directory',
                    type = str, 
                    default = '/data/home/llong35/data/unlabled_videos/MC16_2/clips/AllClips',
                    required = False, 
                    help = 'Name of directory to hold videos to annotate for machine learning purposes')
                    
parser.add_argument('--ML_labels',
                    type = str, 
                    default = '/data/home/llong35/patrick_code_test/modelAll_34/AnnotationFile.csv',
                    required = False, 
                    help = 'labels given to each ML video')
                    
parser.add_argument('--purpose', 
                    type = str, 
                    default = '/data/home/llong35/data/all_videos',
                    required = False, 
                    help = '(train|finetune), How to use this script? train from scrath or finetune to work on different animals')
                    
parser.add_argument('--Log', 
                    type = str, 
                    default = '/data/home/llong35/data/all_videos',
                    required = False, 
                    help = 'Log file to keep track of versions + parameters used')

# Temp directories that wlil be deleted at the end of the analysis
parser.add_argument('--Clips_temp_directory', 
                    default='/data/home/llong35/data/temp',
                    type = str, 
                    required = False, 
                    help = 'Location for temp files to be stored')

# Output data
parser.add_argument('--Log_directory', 
                    type = str, 
                    required = False, 
                    default = '/data/home/llong35/data/04_03_2020',
                    help = 'directory to store sample prepare logs')
                    
parser.add_argument('--Model_directory', 
                    type = str, 
                    required = False, 
                    default = '/data/home/llong35/data/all_videos',
                    help = 'directory to store models')
                    
parser.add_argument('--Performance_directory', 
                    type = str, 
                    default = '/data/home/llong35/data/all_videos',
                    required = False, 
                    help = 'directory to store accuracy and loss change across training or fineturing')
                    
parser.add_argument('--Prediction_File', 
                    type = str, 
                    help = 'label for new animal clips')


# Parameters for the dataloader
parser.add_argument('--sample_duration',
                    default=96,
                    type=int,
                    help='Temporal duration of inputs')
                    
parser.add_argument('--sample_size',
                    default=60,
                    type=int,
                    help='Height and width of inputs')
                    
parser.add_argument('--n_threads',
                    default=3,
                    type=int,
                    help='Number of threads for multi-thread loading')


# Parameters for the optimizer
parser.add_argument('--learning_rate',default=0.1,type=float,help='Initial learning rate (divided by 10 while training by lr scheduler)')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
parser.add_argument('--weight_decay', default=1e-23, type=float, help='Weight Decay')
parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
parser.set_defaults(nesterov=False)
parser.add_argument('--optimizer',default='sgd',type=str,help='Currently only support SGD')
parser.add_argument('--lr_patience',default=10,type=int,help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')


# Parameters for data augmentation
parser.add_argument('--no_hflip',action='store_true',help='If true holizontal flipping is not performed.')
parser.set_defaults(no_hflip=False)
parser.add_argument('--no_vflip',action='store_true',help='If true vertical flipping is not performed.')
parser.set_defaults(no_hflip=False)


# Parameters for general training
parser.add_argument('--checkpoint',default=10,type=int,help='Trained model is saved at every this epochs.')


# Parameters specific for training from scratch
parser.add_argument('--n_classes',default=10,type=int)
parser.add_argument('--batch_size', default=14, type=int, help='Batch Size')
parser.add_argument('--n_epochs',default=100,type=int,help='Number of total epochs to run')


#Parameters specific for finetuning for other animals
parser.add_argument('--resume_path',default='',type=str,help='Save data (.pth) of previous training')
parser.add_argument('--new_animals',type=str,help='new animals to apply the machine learning model')
parser.add_argument('--finetuning_epochs',default=20,type=int,help='Number of total epochs to run')


args = parser.parse_args()
def check_args(args):
    if os.path.exists(args.Log_directory):
        os.mkdirs(args.Log_directory)

check_args(args)
w = ML_model(args)

# Validate data
# def check_args(args):
#     bad_data = False
#     if '.mp4' not in args.Movie_file:
#         print('Movie_file must be mp4 file')
#         bad_data = True
#     if '.npy' not in args.HMM_transition_filename:
#         print('HMM_transition_filename must have npy extension')
#         bad_data = True
#     if '.npy' not in args.Cl_labeled_transition_filename:
#         print('Cl_labeled_transition_filename must have npy extension')
#         bad_data = True
#     if '.csv' not in args.Cl_labeled_cluster_filename:
#         print('Cl_labeled_cluster_filename must have csv extension')
#         bad_data = True
#     if bad_data:
#         raise Exception('Error in argument input.')
#     else:
#         if args.HMM_temp_directory[-1] != '/':
#             args.HMM_temp_directory += '/'
#         if os.path.exists(args.HMM_temp_directory):
#             subprocess.run(['rm','-rf', args.HMM_temp_directory])
#         os.makedirs(args.HMM_temp_directory)
#         if args.Cl_videos_directory[-1] != '/':
#             args.Cl_videos_directory += '/'
#         if not os.path.exists(args.Cl_videos_directory):
#             os.makedirs(args.Cl_videos_directory)
#         if args.ML_frames_directory[-1] != '/':
#             args.ML_frames_directory += '/'
#         if not os.path.exists(args.ML_frames_directory):
#             os.makedirs(args.ML_frames_directory)
#         if args.ML_videos_directory[-1] != '/':
#             args.ML_videos_directory += '/'
#         if not os.path.exists(args.ML_videos_directory):
#             os.makedirs(args.ML_videos_directory)
# 
#         for ofile in [args.HMM_filename, args.HMM_transition_filename, args.Cl_labeled_transition_filename, args.Cl_labeled_cluster_filename, args.Log]:
#             odir = ofile.split(ofile.split('/')[-1])[0]
#             if not os.path.exists(odir) and odir != '':
#                 os.makedirs(odir)
# 
# check_args(args)
# 
# with open(args.Log, 'w') as f:
#     for key, value in vars(args).items():
#         print(key + ': ' + str(value), file = f)
#     print('PythonVersion: ' + sys.version.replace('\n', ' '), file = f)
#     import pandas as pd
#     print('PandasVersion: ' + pd.__version__, file = f)
#     import numpy as np
#     print('NumpyVersion: ' + np.__version__, file = f)
#     import hmmlearn
#     print('HMMLearnVersion: ' + hmmlearn.__version__, file = f)
#     import scipy
#     print('ScipyVersion: ' + scipy.__version__, file = f)
#     import cv2
#     print('OpenCVVersion: ' + cv2.__version__, file = f)
#     import sklearn
#     print('SkLearnVersion: ' + sklearn.__version__, file = f)
# 
# 
# Filter out HMM related arguments
# HMM_args = {}
# for key, value in vars(args).items():
#     if 'HMM' in key or 'Video' in key or 'Movie' in key or 'Num' in key or 'Filter' in key: 
#         if value is not None:
#             HMM_args[key] = value
# 
# HMM_command = ['python3', 'Utils/calculateHMM.py']
# for key, value in HMM_args.items():
#     HMM_command.extend(['--' + key, str(value)])
# 
# print(HMM_command)
# subprocess.run(HMM_command)
#