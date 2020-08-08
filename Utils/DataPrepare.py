import os
import subprocess
import json

from skimage import io
import pandas as pd
import numpy as np

from collections import defaultdict
import pdb
class DP_worker():
    def __init__(self, args):
        self.args = args
        self.means = {}

    def prepare_data(self):
        video_dir = self.args.ML_videos_directory
        means_all_file = os.path.join(self.args.Results_directory,'MeansAll.csv')
        means_file = os.path.join(self.args.Results_directory,'Means.csv')
        annotation_file = self.args.ML_labels
        videos_temp = self.args.Clips_temp_directory

        if not os.path.exists(videos_temp):
            os.makedirs(videos_temp)
        print('convert video clips to images for faster loading')
        for file_name in os.listdir(video_dir):
            if not file_name.endswith('.mp4'):
                continue
            location = file_name.split('.')[0]
            video_file_path = os.path.join(video_dir,file_name)
            target_folder = os.path.join(videos_temp,location)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
                cmd = ['ffmpeg','-i',video_file_path,target_folder+'/image_%05d.jpg']
                subprocess.run(cmd)
        print('calculate mean file')
        if not os.path.exists(means_file):
            with open(means_all_file, 'w') as f:
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
            dt = pd.read_csv(means_all_file,sep=',')
            annotation_df = pd.read_csv(annotation_file,sep=',')
            dt['MeanID'] = dt.apply(lambda row: annotation_df.loc[annotation_df.Location==row.Clip].MeanID.values[0], axis = 1)
            means = dt.groupby('MeanID').mean()

            with open(means_file,'w') as f:
                print('meanID,redMean,greenMean,blueMean,redStd,greenStd,blueStd', file = f)
                for row in means.itertuples():
                    print(row.Index + ',' + str(row.MeanR) + ',' + str(row.MeanG) + ',' + str(row.MeanB) + ',' + str(row.StdR) + ',' + str(row.StdG) + ',' + str(row.StdB), file = f)


    def split_data(self):
        train_list = os.path.join(self.args.Results_directory,'train_list.txt')
        val_list = os.path.join(self.args.Results_directory,'val_list.txt')
        test_list = os.path.join(self.args.Results_directory,'test_list.txt')
        test_animals = [self.args.TEST_PROJECT]
        if not os.path.exists(train_list):
            with open(train_list,'w') as train,open(val_list,'w') as val, open(test_list,'w') as test:
                if self.args.Split_mode == 'random':
                    for index,row in annotation_df.iterrows():
                        animal = row.MeanID.split(':')[0]
                        if animal in test_animals:
                            print(row.Location+','+row.Label,file=test)
                        else:
                            if np.random.uniform()<0.8:
                                print(row.Location+','+row.Label,file=train)
                            else:
                                print(row.Location+','+row.Label,file=val)
                elif self.args.Split_mode == 'mode1':
                    category_count = defaultdict(list)
                    for index, row in annotation_df.iterrows():
                        animal = row.MeanID.split(':')[0]
                        if animal in test_animals:
                            print(row.Location+','+row.Label,file=test)
                        else:
                            label = row.Label
                            location = row.Location
                            category_count[label].append((location,label))
                            for key,value in category_count.items():
                                training_videos = np.random.choice(value,220)
                                for training_video in training_videos:
                                    print(training_video[0] + ',' + training_video[1], file=train)
                                validation_videos = [item for item in value if item not in training_videos]
                                validation_videos = np.random.choice(validation_videos,50)
                                for validation_video in validation_videos:
                                    print(validation_video[0] + ',' + validation_video[1], file=val)

                elif self.args.Split_mode == 'mode2':
                    pass


    def prepare_json(self):
        train_list = os.path.join(self.args.Results_directory,'train_list.txt')
        val_list = os.path.join(self.args.Results_directory,'val_list.txt')
        test_list = os.path.join(self.args.Results_directory,'test_list.txt')
        source_json_path = os.path.join(self.args.Results_directory,'source.json')
        if os.path.exists(source_json_path):
            return
        def convert_csv_to_dict(csv_path, subset):
            keys = []
            key_labels = []
            classes = []
            with open(csv_path,'r') as input:
                for line in input:
                    basename,class_name = line.rstrip().split(',')
                    keys.append(basename)
                    key_labels.append(class_name)
                    if class_name not in classes:
                        classes.append(class_name)
            database = {}
            for i in range(len(keys)):
                key = keys[i]
                database[key] = {}
                database[key]['subset'] = subset
                label = key_labels[i]
                database[key]['annotations'] = {'label': label}
            return database,classes
        train_database,classes = convert_csv_to_dict(train_list, 'training')
        val_database,_ = convert_csv_to_dict(val_list, 'validation')
        test_database,_ = convert_csv_to_dict(test_list, 'testing')
        
        assert len(classes)==10
        
        dst_data = {}
        dst_data['labels'] = classes
        dst_data['database'] = {}
        
        dst_data['database'].update(train_database)
        dst_data['database'].update(val_database)
        dst_data['database'].update(test_database)
        with open(source_json_path, 'w') as dst_file:
            json.dump(dst_data, dst_file)

    def work(self):
        self.prepare_data()
        print('data conversion done, split data')
        self.split_data()
        print('data split done, prepare json')
        self.prepare_json()
        
        
        