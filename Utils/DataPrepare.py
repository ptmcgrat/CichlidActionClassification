import os, subprocess, json, pdb

from skimage import io
import pandas as pd
import numpy as np

class DP_worker():
    def __init__(self, args):
        self.inputVideosDir = args.Input_videos_directory
        self.resultsDir = args.Results_directory
        self.tempDir = args.Temporary_clips_directory
        self.manualLabelFile = args.ML_labels
        self.purpose = args.Purpose
        self.means = {} # Holds mean values for each video

        self.dt = pd.read_csv(self.manualLabelFile, index_col = 0)

       
    def processData(self):
        print('DP: Converting mp4 clips to jpg images for faster loading')
        self._convertVideos()
        print('DP: Calculating RGB means/stds for normalizing videos')
        self._calculateMeans()
        print('DP: Splitting data into train and validation sets')
        self._splitData()
        self._prepareJson()
        print('DP: Completed')

    def _convertVideos(self):
        all_videos = os.listdir(self.inputVideosDir)
        self.dt['ClipAvailable'] = True
        
        for mp4_file in self.dt.ClipName:
            if not mp4_file.endswith('.mp4'):
                continue

            video_file_path = os.path.join(self.inputVideosDir,mp4_file)
            outputDir = os.path.join(self.tempDir,mp4_file.replace('.mp4',''))
            
            if not os.path.exists(video_file_path):
                #print(f"Skipping {video_file_path}: File not found")
                self.dt.loc[self.dt.ClipName == mp4_file,'ClipAvailable'] = False
                continue

            if not os.path.exists(outputDir):
                # os.makedirs(target_folder)
                output = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', video_file_path], capture_output = True, encoding = 'utf-8')
                if "moov atom not found" in output.stderr or "invalid data found when processing input" in output.stderr:
                    print(f"Skipping {video_file_path}: Corrupt video (moov atom missing).")
                    self.dt.loc[self.dt.ClipName == mp4_file,'ClipAvailable'] = False
                    continue
                os.makedirs(outputDir)
                cmd = ['ffmpeg','-i',video_file_path,outputDir+'/image_%05d.jpg']
                output = subprocess.run(cmd, capture_output = True)
                if output.returncode != 0:
                    pdb.set_trace()
        data_summary = self.dt.groupby(['AnalysisID','ClipAvailable']).count()
        print(data_summary)
        data_summary.to_csv(self.resultsDir + 'DataSummaryByAnalysisID.csv')

    def _calculateMeans(self):
        annotation_file = self.manualLabelFile

        m_dt = pd.DataFrame(columns = ['ClipName','ProjectID','MeanR','MeanG','MeanB','StdR','StdG','StdB'])
            
        print('calculate mean file')
        for i,row in self.dt[self.dt.ClipAvailable == True].iterrows():
            location = row.ClipName.replace('.mp4','')
            projectID = row.ProjectID

            video_folder = os.path.join(self.tempDir,location)
            image_indices = []
            frames = []

            try:
                frames = sorted([os.path.join(video_folder,x) for x in os.listdir(video_folder) if 'image' in x])
            except FileNotFoundError:
                print(video_folder + ' does not exist')
                continue
            n_frames = len(frames)
            with open(os.path.join(video_folder, 'n_frames'), 'w') as dst_file:
                dst_file.write(str(n_frames))
            
            img = io.imread(frames[0])
            mean = img.mean(axis = (0,1))
            std = img.std(axis = (0,1))
            
            m_dt.loc[len(m_dt)] = [location,projectID] + mean.tolist() + std.tolist()
            
        means = m_dt.groupby(['ProjectID']).agg({'MeanR':'mean','MeanG':'mean','MeanB':'mean','StdR':'mean','StdG':'mean','StdB':'mean'}).reset_index()
        means.to_csv(os.path.join(self.resultsDir,'Means.csv'), index = False)

    def _splitData(self):
        train_list = os.path.join(self.resultsDir,'train_list.txt')
        val_list = os.path.join(self.resultsDir,'val_list.txt')
        test_list = os.path.join(self.resultsDir,'test_list.txt')
        
        if self.purpose == 'classify':
            with open(val_list,'w') as val:
                for mp4_file in self.dt.ClipName:
                    print(mp4_file.replace('.mp4',''),file=val)
            return
        else:
            with open(train_list,'w') as train,open(val_list,'w') as val, open(test_list,'w') as test:
                for lid,row in self.dt[self.dt.ClipAvailable==True].iterrows():
                    if np.random.uniform()<0.8:
                        print(row.ClipName.replace('.mp4','') + ',' + row.ManualLabel,file=train)
                    else:
                        print(row.ClipName.replace('.mp4','') + ',' + row.ManualLabel,file=val)

    def convert_csv_to_dict(self,csv_path, subset):
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


    def _prepareJson(self):
        train_list = os.path.join(self.resultsDir,'train_list.txt')
        val_list = os.path.join(self.resultsDir,'val_list.txt')
        test_list = os.path.join(self.resultsDir,'test_list.txt')

        source_json_path = os.path.join(self.resultsDir,'source.json')
        
        if self.purpose == 'classify':
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

        train_database,classes = self.convert_csv_to_dict(train_list, 'training')
        val_database,_ = self.convert_csv_to_dict(val_list, 'validation')
        test_database,_ = self.convert_csv_to_dict(test_list, 'testing')
        assert len(classes)==10
        
        dst_data = {}
        dst_data['labels'] = classes
        dst_data['database'] = {}
        
        dst_data['database'].update(train_database)
        dst_data['database'].update(val_database)
        dst_data['database'].update(test_database)

        with open(source_json_path, 'w') as dst_file:
            json.dump(dst_data, dst_file)
            
    
        
        
        
