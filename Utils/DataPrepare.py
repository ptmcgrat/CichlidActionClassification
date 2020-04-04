import os
import subprocess

from skimage import io
import pandas as pd

import pdb
class DP_worker():
    def __init__(self, args):
        self.args = args
        self.means = {}
        self.annotation = pd.read_csv(args.ML_labels)
        
    
    def work(self):
        #convert to jpegs
        
        ML_video_dir = self.args.ML_videos_directory
        videos_temp = self.args.Clips_temp_directory
#         for file_name in os.listdir(ML_video_dir):
#             if not file_name.endswith('.mp4'):
#                 continue
#             location = file_name.split('.')[0]
#             video_file_path = os.path.join(ML_video_dir,location+'.mp4')
#             target_folder = os.path.join(self.args.Clips_temp_directory,location)
#             if not os.path.exists(target_folder):
#                 os.makedirs(target_folder)
# #             cmd = 'ffmpeg -i {} {}/image_%05d.jpg'.format(video_file_path, target_folder)
#             cmd = ['ffmpeg','-i',video_file_path,target_folder+'/image_%05d.jpg']
#             subprocess.run(cmd)
#             break
        
        #count number of frames and calculate mean
        with open(os.path.join(self.args.Log_directory,'MeansAll.csv'), 'w') as f:
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
        dt = pd.read_csv(os.path.join(self.args.Log_directory,'MeansAll.csv'), sep = ',')
        pdb.set_trace()
        dt['MeanID'] = dt.apply(lambda row: self.annotation.loc[self.annotation.Location==row.Clip].MeanID, axis = 1)
        print(dt)
                
        
        
            
        
        
        