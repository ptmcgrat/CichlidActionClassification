import argparse, os, pdb, subprocess, json
from Utils.CichlidActionRecognition import ML_model
from Utils.DataPrepare import DP_worker


parser = argparse.ArgumentParser(description='This script trains a 3D Resnet from scratch using labeled videos')
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
parser.add_argument('--DataSummaryLog', type = str, required = True,
                    help = 'Logfile to keep track of annotated data')

parser.add_argument('--n_threads', default=5, type=int,
                    help='Number of threads for multi-thread loading')                    
parser.add_argument('--gpu', default='0', type=str, help='The index of GPU to use for training')
# Parameters for the dataloader
parser.add_argument('--sample_duration', default=96, type=int, help='Temporal duration of inputs')                    
parser.add_argument('--sample_size', default=120, type=int, help='Height and width of inputs')
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

args = parser.parse_args()

subprocess.run(['conda','list'], stdout = open(args.CondaLog,'w'))

with open(args.CommandsLog, 'w') as output:
    json.dump(vars(args), output)

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

if not os.path.exists(args.Temp_directory):
    os.makedirs(args.Temp_directory)

data_worker = DP_worker(args.Clips_directory, args.Temp_directory, args.ML_labels)
data_worker.convertVideos(args.DataSummaryLog)
data_worker.calculateMeans()
data_worker.prepareJson('Train', args.JSONLog, args.n_classes)

ML_model = ML_model('Train', args.JSONLog, args.Temp_directory, args.Results_directory, args.sample_size, args.sample_duration, )
ML_model.createDataLoaders(args.batch_size, args.n_threads)
ML_model.train_model(args.n_classes, args.dampening, args.learning_rate, args.momentum, args.weight_decay, args.nesterov, args.lr_patience, args.n_epochs, args.checkpoint, args.ML_labels)
