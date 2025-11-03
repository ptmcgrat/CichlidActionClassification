import os, subprocess, json, pdb

from skimage import io
import pandas as pd
import numpy as np
from collections import defaultdict

class DP_worker():
    def __init__(self, args):
        self.inputVideosDir = args.Input_videos_directory
        self.resultsDir = args.Results_directory
        self.tempDir = args.Temporary_clips_directory
        self.manualLabelFile = args.ML_labels
        self.means = {} # Holds mean values for each video

        self.dt = pd.read_csv(self.manualLabelFile, index_col = 0)

        self._convertVideos()

    def _convertVideos(self):
        print('convert video clips to images for faster loading')
        all_videos = os.listdir(self.inputVideosDir)
  
        for lid,mp4_file in self.dt.ClipName:
            pdb.set_trace()

            if not mp4_file.endswith('.mp4'):
                continue

            video_file_path = os.path.join(self.inputVideosDir,mp4_file)
            outputDir = os.path.join(self.tempDir,mp4_file.replace('.mp4',''))

            if not os.path.exists(outputDir):
                # os.makedirs(target_folder)
                if not os.path.exists(video_file_path):
                    print(f"Skipping {video_file_path}: File not found")
                output = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', video_file_path], capture_output = True, encoding = 'utf-8')
                if "moov atom not found" in output.stderr or "invalid data found when processing input" in output.stderr:
                    print(f"Skipping {video_file_path}: Corrupt video (moov atom missing).")
                    continue
                os.makedirs(outputDir)
                cmd = ['ffmpeg','-i',video_file_path,outputDir+'/image_%05d.jpg']
                subprocess.run(cmd, capture_output = True)

    def prepare_data(self):
        
        video_dir = self.args.Input_videos_directory
        means_all_file = os.path.join(self.args.Results_directory,'MeansAll.csv')
        means_file = os.path.join(self.args.Results_directory,'Means.csv')
        if self.args.Purpose == 'classify':
            annotation_file = self.args.Videos_to_project_file
        else:
            annotation_file = self.args.ML_labels
        videos_temp = self.args.Temporary_clips_directory

        if not os.path.exists(videos_temp):
            os.makedirs(videos_temp)

        
            
        # pdb.set_trace()
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
                    # print(video)
                    if not image_indices:
                        pdb.set_trace()
                    #     continue
                    n_frames = image_indices[0]
                    with open(os.path.join(video_folder, 'n_frames'), 'w') as dst_file:
                        # pdb.set_trace()
                        dst_file.write(str(n_frames))
                    img = io.imread(frames[0])
                    mean = img.mean(axis = (0,1))
                    std = img.std(axis = (0,1))
                    print(video + ',' + ','.join([str(x) for x in mean]) + ',' + ','.join([str(x) for x in std]), file = f)
                    # pdb.set_trace()
            dt = pd.read_csv(means_all_file,sep=',')
            annotation_df = pd.read_csv(annotation_file,sep=',')
            # dt['MeanID'] = dt.apply(lambda row: annotation_df.loc[annotation_df.ClipName==row.Clip].MeanID.values[0], axis = 1)
            dt['MeanID'] = dt.apply(lambda row: annotation_df.loc[annotation_df.ClipName == row.Clip, 'MeanID'].iloc[0] if not annotation_df.loc[annotation_df.ClipName == row.Clip].empty else None, axis=1)
            means = dt[['MeanID','MeanR','MeanG','MeanB','StdR', 'StdG', 'StdB']].groupby('MeanID').mean()
            with open(means_file,'w') as f:
                print('meanID,redMean,greenMean,blueMean,redStd,greenStd,blueStd', file = f)
                for row in means.itertuples():
                    print(row.Index + ',' + str(row.MeanR) + ',' + str(row.MeanG) + ',' + str(row.MeanB) + ',' + str(row.StdR) + ',' + str(row.StdG) + ',' + str(row.StdB), file = f)


    def split_data(self):
        train_list = os.path.join(self.args.Results_directory,'train_list.txt')
        val_list = os.path.join(self.args.Results_directory,'val_list.txt')
        test_list = os.path.join(self.args.Results_directory,'test_list.txt')
        

        
        if self.args.Purpose == 'classify':
            annotation_df = pd.read_csv(self.args.Videos_to_project_file, sep=',')
            with open(val_list,'w') as val:
                for index,row in annotation_df.iterrows():
                    print(row.ClipName,file=val)
            return
        test_animals = self.args.TEST_PROJECT.split(',')
        annotation_df = pd.read_csv(self.args.ML_labels, sep=',')
        if not os.path.exists(train_list):
            with open(train_list,'w') as train,open(val_list,'w') as val, open(test_list,'w') as test:
                if self.args.Split_mode == 'random':
                    for index,row in annotation_df.iterrows():
                        #pdb.set_trace()
                        #animal = row.MeanID.split(':')[0]
                        #if animal in test_animals:
                        #    print(row.ClipName+','+row.ManualLabel,file=test)
                        #else:
                        if np.random.uniform()<0.8:
                            # pdb.set_trace()
                            print(row.ClipName+','+row.ManualLabel,file=train)
                        else:
                            print(row.ClipName+','+row.ManualLabel,file=val)
                        # pdb.set_trace()
                elif self.args.Split_mode == 'mode1':
                    category_count = defaultdict(list)
                    for index, row in annotation_df.iterrows():
                        animal = row.MeanID.split(':')[0]
                        if animal in test_animals:
                            print(row.ClipName+','+row.ManualLabel,file=test)
                        else:
                            label = row.ManualLabel
                            location = row.ClipName
                            category_count[label].append(location)
                    for key, value in category_count.items():
                        training_videos = np.random.choice(value,220, replace=False)
                        for training_video in training_videos:
                            print(training_video + ',' + key, file=train)
                        validation_videos = [item for item in value if item not in training_videos]
                        validation_videos = np.random.choice(validation_videos, 50, replace=False)
                        for validation_video in validation_videos:
                            print(validation_video + ',' + key, file=val)

                elif self.args.Split_mode == 'mode2':
                    all_samples = []
                    for index, row in annotation_df.iterrows():
                        animal = row.MeanID.split(':')[0]
                        if animal in test_animals:
                            print(row.ClipName + ',' + row.ManualLabel, file=test)
                        else:
                            label = row.ManualLabel
                            location = row.ClipName
                            all_samples.append((label,location))
                    training_indices = np.random.choice(len(all_samples),2200,replace=False)
                    training_videos = [all_samples[i] for i in training_indices]
                    for training_video in training_videos:
                        print(training_video[1] + ',' + training_video[0], file=train)
                    validation_indices = []
                    for i in range(len(all_samples)):
                        if i not in training_indices:
                            validation_indices.append(i)
                    validation_indices = np.random.choice(validation_indices, 500, replace=False)
                    validation_videos = [all_samples[i] for i in validation_indices]
                    for validation_video in validation_videos:
                        print(validation_video[1] + ',' + validation_video[0], file=val)
                elif self.args.Split_mode == 'mode3':
                    category_count = defaultdict(list)
                    for index, row in annotation_df.iterrows():
                        animal = row.MeanID.split(':')[0]
                        if animal in test_animals:
                            print(row.ClipName + ',' + row.ManualLabel, file=test)
                        else:
                            label = row.ManualLabel
                            location = row.ClipName
                            category_count[label].append(location)
                    for key, value in category_count.items():
                        # if less than 800 samples, 80% for training and rest for validation
                        # otherwise use 640 for training and 160 for validation
                        if len(value) >= 800:
                            training_videos = np.random.choice(value, 640, replace=False)
                            validation_videos = [item for item in value if item not in training_videos]
                            validation_videos = np.random.choice(validation_videos, 160, replace=False)
                        else:
                            training_video_count = int(len(value)*0.8)
                            training_videos = np.random.choice(value, training_video_count, replace=False)
                            validation_videos = [item for item in value if item not in training_videos]
                        for training_video in training_videos:
                            print(training_video + ',' + key, file=train)
                        for validation_video in validation_videos:
                            print(validation_video + ',' + key, file=val)
            




    def prepare_json(self):
        train_list = os.path.join(self.args.Results_directory,'train_list.txt')
        val_list = os.path.join(self.args.Results_directory,'val_list.txt')
        test_list = os.path.join(self.args.Results_directory,'test_list.txt')
        source_json_path = os.path.join(self.args.Results_directory,'source.json')
        if self.args.Purpose == 'classify':
            with open(self.args.Trained_categories,'r') as input_f:
                training_json = json.load(input_f) 
            dst_data = {}
            dst_data['labels'] = training_json['labels']
            
            database={}
            with open(os.path.join(val_list),'r') as input:
                for line in input:
                    key = line.rstrip()
                    database[key] = {}
                    database[key]['subset'] = 'validation'
                    database[key]['annotations'] = {'label': dst_data['labels'][0]}
            dst_data['database'] = database
            with open(source_json_path, 'w') as dst_file:
                json.dump(dst_data, dst_file)
            return
        
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
            # pdb.set_trace()
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
        pdb.set_trace()

        with open(source_json_path, 'w') as dst_file:
            json.dump(dst_data, dst_file)
            

    def work(self):
        self.prepare_data()
        print('data conversion done, split data')
        self.split_data()
        print('data split done, prepare json')
        self.prepare_json()
        
    
        
        
        
