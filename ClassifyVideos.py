import argparse, subprocess, datetime, os, pdb, sys, json
from Utils.CichlidActionRecognition import ML_model
from Utils.DataPrepare import DP_worker


parser = argparse.ArgumentParser(description='This script takes a model, and apply this model to new video clips')
# Input data
parser.add_argument('--Clips_directory', type = str, required = True,
                    help = 'Name of directory that holds the mp4 clips')                    
parser.add_argument('--ML_labels', type = str, required = True,
                    help = 'csv file with labels for each ML video, it should contain three columns: ClipName, ManualLabel and ProjectID')
parser.add_argument('--Temp_directory', type = str, required = True,
                    help = 'Location for temp files to be stored')
parser.add_argument('--Results_directory', type = str, required = True,
                    help = 'Location for final files to be stored')
parser.add_argument('--CommandsLog', type = str, required = True,
                    help = 'Logfile to keep track of commands')
parser.add_argument('--JSONLog', type = str, required = True,
                    help = 'Logfile to keep track of data splits and label names')
parser.add_argument('--CondaLog', type = str, required = True,
                    help = 'Logfile to keep track of conda and cuda versions')
parser.add_argument('--Trained_model', type=str, required = True,
                    help='Save data (.pth) of previous training')

parser.add_argument('--Output_file', required = True, type = str, 
                    help = 'csv file that keeps the confidence and label for each video clip')
parser.add_argument('--Purpose', type = str, default = 'Classify',
                    help = 'Leave this alone')

parser.add_argument('--batch_size', default=26, type=int, help='Batch Size')
parser.add_argument('--n_threads', default=5, type=int, help='Number of threads for multi-thread loading')
parser.add_argument('--gpu_card', default='0', type=str, help='gpu card to use')
# Parameters for the dataloader
parser.add_argument('--sample_duration', default=96, type=int, help='Temporal duration of inputs')                   
parser.add_argument('--sample_size', default=120, type=int, help='Height and width of inputs')
                    
# Parameters for the optimizer

args = parser.parse_args()
# Parameters to load from previous training_log

with open(args.CommandsLog,'r') as input_f:
    data = json.load(input_f)

    for key,value in data.items():
        if key in ['sample_duration','sample_size','n_classes']:
            vars(args)[key]=int(value)
        else:
            pass

def check_args(args):
    if not os.path.exists(args.Results_directory):
        os.makedirs(args.Results_directory)
    if not os.path.exists(args.Temp_directory):
        os.makedirs(args.Temp_directory)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_card

data_worker = DP_worker(args.Clips_directory, args.Temp_directory, args.ML_labels)
data_worker.convertVideos(None)
data_worker.calculateMeans()
data_worker.prepareJson('Classify', args.JSONLog, args.n_classes)

ML_model = ML_model('Classify', args.JSONLog, args.Temp_directory, args.Results_directory, args.sample_size, args.sample_duration)
ML_model.createDataLoaders(args.batch_size, args.n_threads)
ML_model.make_predictions(args.Trained_model, args.n_classes, args.Output_file)

