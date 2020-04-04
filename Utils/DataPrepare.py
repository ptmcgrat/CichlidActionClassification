import os
import subprocess

from skimage import io
import pandas as pd

import pdb
class DP_worker():
    def __init__(self, args):
        self.args = args
        self.means = {}
        
    
    def prepare_domain(self,domain):
        if domain == 'source':
            video_dir = self.args.ML_videos_directory
            meansalll_file = os.path.join(self.args.Log_directory,'source_MeansAll.csv')
            means_file = os.path.join(self.args.Log_directory,'source_Means.csv')
            annotation_file = self.args.ML_labels
            
        else:
            video_dir = self.args.Unlabeled_videos_directory
            annotation_file  = os.path.join(self.args.Log_directory,'target_domain_annotation.csv')
            meansalll_file = os.path.join(self.args.Log_directory,'target_MeansAll.csv')
            means_file = os.path.join(self.args.Log_directory,'target_Means.csv')
            annotation_f=open(annotation_file,'w')
            print('Location,MeanID', file = annotation_f)
        
        videos_temp = os.path.join(self.args.Clips_temp_directory,domain)
        
        for file_name in os.listdir(video_dir):
            if not file_name.endswith('.mp4'):
                continue
            if domain == 'source':
                location = file_name.split('.')[0]
            else:
                tokens = file_name.split('.')[0].split('__')
                location = '_'.join(tokens[-5:])
                MeanID = '_'.join(tokens[:2])
                print(location+','+MeanID,file=annotation_f)
            video_file_path = os.path.join(video_dir,file_name)
            target_folder = os.path.join(videos_temp,location)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            cmd = ['ffmpeg','-i',video_file_path,target_folder+'/image_%05d.jpg']
            subprocess.run(cmd)
            break
        annotation_f.close()
            
        with open(meansalll_file, 'w') as f:
            print('Clip,MeanR,MeanG,MeanB,StdR,StdG,StdB', file = f)
            for video in os.listdir(videos_temp):
                video_folder = os.path.join(videos_temp,video)
                image_indices = []
                frames = []
                for image_file_name in os.listdir(video_folder):
                    image_file_path = os.path.join(video_folder,image_file_name)
                    if 'image' not in image_file_name:
                        continue
                    image_indices.append(int(image_file_name[6:11]))
                    frames.append(image_file_path)
                image_indices.sort(reverse=True)
                n_frames = image_indices[0]
                with open(os.path.join(video_folder, 'n_frames'), 'w') as dst_file:
                    dst_file.write(str(n_frames))
                img = io.imread(frames[0])
                mean = img.mean(axis = (0,1))
                std = img.std(axis = (0,1))
                print(video + ',' + ','.join([str(x) for x in mean]) + ',' + ','.join([str(x) for x in std]), file = f)
                break
        dt = pd.read_csv(meansalll_file,sep=',')
        annotation_df = pd.read_csv(annotation_file,sep=',')
        dt['MeanID'] = dt.apply(lambda row: annotation_df.loc[annotation_df.Location==row.Clip].MeanID, axis = 1)
        means = dt.groupby('MeanID').mean()
        with open(means_file,'w') as f:
            print('meanID,redMean,greenMean,blueMean,redStd,greenStd,blueStd', file = f)
            for row in means.itertuples():
                print(row.Index + ',' + str(row.MeanR) + ',' + str(row.MeanG) + ',' + str(row.MeanB) + ',' + str(row.StdR) + ',' + str(row.StdG) + ',' + str(row.StdB), file = f)
        
    
    
    
    def work(self):
        #convert to jpegs
#         self.prepare_domain('source')
        self.prepare_domain('target')
        
        
        
            
        
        
        