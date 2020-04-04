import os
import subprocess
import json

from skimage import io
import pandas as pd
import numpy as np

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
        
#         for file_name in os.listdir(video_dir):
#             if not file_name.endswith('.mp4'):
#                 continue
#             if domain == 'source':
#                 location = file_name.split('.')[0]
#             else:
#                 tokens = file_name.split('.')[0].split('__')
#                 location = '_'.join(tokens[-5:])
#                 MeanID = ':'.join(tokens[:2])
#                 print(location+','+MeanID,file=annotation_f)
#             video_file_path = os.path.join(video_dir,file_name)
#             target_folder = os.path.join(videos_temp,location)
#             if not os.path.exists(target_folder):
#                 os.makedirs(target_folder)
#             cmd = ['ffmpeg','-i',video_file_path,target_folder+'/image_%05d.jpg']
#             subprocess.run(cmd)
#         if domain == 'target':
#             annotation_f.close()
            
#         with open(meansalll_file, 'w') as f:
#             print('Clip,MeanR,MeanG,MeanB,StdR,StdG,StdB', file = f)
#             for video in os.listdir(videos_temp):
#                 video_folder = os.path.join(videos_temp,video)
#                 image_indices = []
#                 frames = []
#                 for image_file_name in os.listdir(video_folder):
#                     image_file_path = os.path.join(video_folder,image_file_name)
#                     if 'image' not in image_file_name:
#                         continue
#                     image_indices.append(int(image_file_name[6:11]))
#                     frames.append(image_file_path)
#                 image_indices.sort(reverse=True)
#                 n_frames = image_indices[0]
#                 with open(os.path.join(video_folder, 'n_frames'), 'w') as dst_file:
#                     dst_file.write(str(n_frames))
#                 img = io.imread(frames[0])
#                 mean = img.mean(axis = (0,1))
#                 std = img.std(axis = (0,1))
#                 print(video + ',' + ','.join([str(x) for x in mean]) + ',' + ','.join([str(x) for x in std]), file = f)
        dt = pd.read_csv(meansalll_file,sep=',')
        annotation_df = pd.read_csv(annotation_file,sep=',')
        dt['MeanID'] = dt.apply(lambda row: annotation_df.loc[annotation_df.Location==row.Clip].MeanID, axis = 1)
        means = dt.groupby('MeanID').mean()
        with open(means_file,'w') as f:
            print('meanID,redMean,greenMean,blueMean,redStd,greenStd,blueStd', file = f)
            for row in means.itertuples():
                print(row.Index + ',' + str(row.MeanR) + ',' + str(row.MeanG) + ',' + str(row.MeanB) + ',' + str(row.StdR) + ',' + str(row.StdG) + ',' + str(row.StdB), file = f)
        
        if domain == 'source':
            train_list = os.path.join(self.args.Log_directory,'source_train_list.txt')
            val_list = os.path.join(self.args.Log_directory,'source_val_list.txt')
            test_list = os.path.join(self.args.Log_directory,'source_test_list.txt')
            test_animals = ['MC16_2']
            with open(train_list,'w') as train,open(val_list,'w') as val, open(test_list,'w') as test:
                for index,row in annotation_df.iterrows():
                    animal = row.MeanID.split(':')[0]
                    if animal in test_animals:
                        print(row.Location+','+row.Label,file=test)
                    else:
                        if np.random.uniform()<0.8:
                            print(row.Location+','+row.Label,file=train)
                            try:
                                self.train_count += 1
                            except:
                                self.train_count = 1
                        else:
                            print(row.Location+','+row.Label,file=val)
        else:
            target_list = os.path.join(self.args.Log_directory,'target_list.txt')
            with open(target_list,'w') as target:
                count = 0
                for index,row in annotation_df.iterrows():
                    count += 1
                    print(row.Location+',target',file=target)
                    if count == self.train_count:
                        break
        
        
    def prepare_json(self):
        train_list = os.path.join(self.args.Log_directory,'source_train_list.txt')
        val_list = os.path.join(self.args.Log_directory,'source_val_list.txt')
        test_list = os.path.join(self.args.Log_directory,'source_test_list.txt')
        target_list = os.path.join(self.args.Log_directory,'target_list.txt')
        dst_json_path = os.path.join(self.args.Log_directory,'cichlids.json')
        def convert_csv_to_dict(csv_path, subset):
            keys = []
            key_labels = []
            with open(csv_path,'r') as input:
                for line in input:
                    basename,class_name = line.rstrip().split(',')
                    keys.append(basename)
                    key_labels.append(class_name)
            database = {}
            for i in range(len(keys)):
                key = keys[i]
                database[key] = {}
                database[key]['subset'] = subset
                label = key_labels[i]
                database[key]['annotations'] = {'label': label}
            return database
        train_database = convert_csv_to_dict(train_list, 'training')
        val_database = convert_csv_to_dict(val_list, 'validation')
        test_database = convert_csv_to_dict(test_list, 'testing')
        target_database = convert_csv_to_dict(target_list, 'target')

        dst_data = {}
    
        dst_data['database'] = {}
        dst_data['database'].update(train_database)
        dst_data['database'].update(val_database)
        dst_data['database'].update(test_database)
        dst_data['database'].update(target_database)
        
        with open(dst_json_path, 'w') as dst_file:
            json.dump(dst_data, dst_file)
        
    
    
    
    def work(self):
        #convert to jpegs
        self.prepare_domain('source')
#         self.prepare_domain('target')
#         self.prepare_json()
#         
        
            
        
        
        