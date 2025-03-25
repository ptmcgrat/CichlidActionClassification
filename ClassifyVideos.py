import argparse, subprocess, datetime, os, pdb, sys
from Utils.CichlidActionRecognition import ML_model
from Utils.DataPrepare import DP_worker


parser = argparse.ArgumentParser(description='This script takes a model, and apply this model to new video clips')
# Input data
parser.add_argument('--Input_videos_directory',
                    type = str, 
                    default = '/data/home/llong35/data/labeled_videos',
                    required = False, 
                    help = 'Name of directory to hold all video clips')
                    
parser.add_argument('--Videos_to_project_file',
                    type = str, 
                    default = '/data/home/llong35/patrick_code_test/test.csv',
                    help = 'project each animal belongs to')

parser.add_argument('--Trained_model',
                    default='/data/home/llong35/temp/test_JAN_20_temp/save_50.pth',
                    type=str,
                    help='Save data (.pth) of previous training')
                    
parser.add_argument('--Trained_categories',
                    type = str, 
                    default = '/data/home/llong35/temp/test_JAN_7_temp/source.json',
                    help = 'json file previously used for training')
                    
parser.add_argument('--Training_options',
                    type = str, 
                    default = os.path.join(os.getenv("HOME"),'temp','test_JAN_20_log'),
                    help = 'log file in training')

parser.add_argument('--Output_file',
                    type = str, 
                    default = '/data/home/llong35/temp/output.csv',
                    help = 'csv file that keeps the confidence and label for each video clip')
                    
parser.add_argument('--Purpose',
                    type = str, 
                    default = 'classify',
                    help = 'classify is the only function for this script for now')

parser.add_argument('--batch_size', 
                    default=13, 
                    type=int, help='Batch Size')
                    
parser.add_argument('--n_threads',
                    default=5,
                    type=int,
                    help='Number of threads for multi-thread loading')
                    
parser.add_argument('--gpu_card',
                    default='0',
                    type=str,
                    help='gpu card to use')
# Temp directories that wlil be deleted at the end of the analysis
parser.add_argument('--Temporary_clips_directory',
                    default=os.path.join(os.getenv("HOME"),'clips_temp'),
                    type = str, 
                    required = False, 
                    help = 'Location for temp clips to be stored')

parser.add_argument('--Temporary_output_directory',
                    default=os.path.join(os.getenv("HOME"),'intermediate_temp'),
                    type = str, 
                    required = False, 
                    help = 'Location for temp files to be stored')
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


# Parameters specific for training from scratch
parser.add_argument('--n_classes',default=10,type=int)



args = parser.parse_args()
# Parameters to load from previous training_log

with open(args.Training_options,'r') as input_f:
    for line in input_f:
        # pdb.set_trace()
        key,value = line.rstrip().split(': ')
        if key in ['sample_duration','sample_size','lr_patience','n_classes']:
            vars(args)[key]=int(value)
        elif key in ['optimizer','resnet_shortcut']:
            vars(args)[key]=str(value)
        elif key in ['learning_rate','momentum','dampening','weight_decay',]:
            vars(args)[key]=float(value)
        elif key in ['nesterov']:
            vars(args)[key]= key=='True'
        else:
            pass
vars(args)['Results_directory']= args.Temporary_output_directory
def check_args(args):
    if not os.path.exists(args.Results_directory):
        os.makedirs(args.Results_directory)
    if not os.path.exists(args.Temporary_clips_directory):
        os.makedirs(args.Temporary_clips_directory)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_card
check_args(args)
data_worker = DP_worker(args)
data_worker.work()
ML_model = ML_model(args)
ML_model.work()