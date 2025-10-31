import argparse, subprocess, datetime, os, pdb, sys
from Utils.CichlidActionRecognition import ML_model
from Utils.DataPrepare import DP_worker


parser = argparse.ArgumentParser(description='This script takes video clips and annotations, either train a model from scratch or finetune a model to work on the new animals not annotated')
# Input data
parser.add_argument('--Input_videos_directory',
                    type = str, 
                    required = True, 
                    help = 'Name of directory to hold all video clips')
                    
parser.add_argument('--ML_labels',
                    type = str, 
                    help = 'csv file with labels given to each ML video, it should contain three columns: Location, Label and MeanID')

parser.add_argument('--Temporary_clips_directory',
                    type = str, 
                    required = True, 
                    help = 'Location for temp files to be stored')

parser.add_argument('--Results_directory',
                    type = str,
                    help = 'directory to store sample prepare logs')                    

parser.add_argument('--Purpose',
                    type = str, 
                    default = 'train',
                    help = '(train|finetune), How to use this script? train from scrath or finetune to work on different animals')

parser.add_argument('--TEST_PROJECT',
                    type = str,
                    default = '',
                    help = 'project to be tested on')

parser.add_argument('--Split_mode',
                    type = str,
                    default = 'random',
                    help = 'random|mode1|mode2|mode3')
                    
parser.add_argument('--Log', 
					type = str, 
					required = False, 
					default=os.path.join(os.getenv("HOME"),'temp','test_JAN_24_log'),
					help = 'Log file to keep track of versions + parameters used')
					
parser.add_argument('--n_threads',
                    default=5,
                    type=int,
                    help='Number of threads for multi-thread loading')
                    
parser.add_argument('--gpu',
                    default='0',
                    type=str,
                    help='The index of GPU to use for training')

# Parameters for the dataloader
parser.add_argument('--sample_duration',
                    default=96,
                    type=int,
                    help='Temporal duration of inputs')
                    
parser.add_argument('--sample_size',
                    default=120,
                    type=int,
                    help='Height and width of inputs')
                    
# Parameters for the optimizer
parser.add_argument('--learning_rate',default=0.1,type=float,help='Initial learning rate (divided by 10 while training by lr scheduler)')
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
parser.set_defaults(nesterov=False)
parser.add_argument('--optimizer',default='sgd',type=str,help='Currently only support SGD')
parser.add_argument('--lr_patience',default=10,type=int,help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
parser.add_argument('--resnet_shortcut',default='B',help='Shortcut type of resnet (A | B)')

# Parameters for data augmentation
parser.add_argument('--no_hflip',action='store_true',help='If true holizontal flipping is not performed.')
parser.set_defaults(no_hflip=False)
parser.add_argument('--no_vflip',action='store_true',help='If true vertical flipping is not performed.')
parser.set_defaults(no_hflip=False)

# Parameters for general training
parser.add_argument('--checkpoint',default=10,type=int,help='Trained model is saved at every this epochs.')

# Parameters specific for training from scratch
parser.add_argument('--n_classes',default=10,type=int)
parser.add_argument('--batch_size', default=13, type=int, help='Batch Size')
parser.add_argument('--n_epochs',default=100,type=int,help='Number of total epochs to run')

# Parameters specific for finetuning for other animals
parser.add_argument('--resume_path',default='/data/home/llong35/temp/test_aug_8_2_restricted_total_sampling/save_100.pth',type=str,help='Save data (.pth) of previous training')

args = parser.parse_args()


def check_args(args):
    if not os.path.exists(args.Results_directory):
        os.makedirs(args.Results_directory)
    if not os.path.exists(args.Temporary_clips_directory):
        os.makedirs(args.Temporary_clips_directory)


check_args(args)

with open(args.Log, 'w') as f:
	for key, value in vars(args).items():
		print(key + ': ' + str(value), file = f)
	print('PythonVersion: ' + sys.version.replace('\n', ' '), file = f)
	import pandas as pd
	print('PandasVersion: ' + pd.__version__, file = f)
	import numpy as np
	print('NumpyVersion: ' + np.__version__, file = f)
	import torch
	print('pytorch: ' + torch.__version__, file = f)
	
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
data_worker = DP_worker(args)
data_worker.work()
ML_model = ML_model(args)
ML_model.work()
