import os, subprocess, json, pdb

from skimage import io
import pandas as pd
import numpy as np

class DP_worker():
    def __init__(self, input_videos, temp_directory, manual_label_file):
        self.inputVideosDir = input_videos
        #self.resultsDir = args.Results_directory
        self.tempDir = temp_directory
        self.manualLabelFile = manual_label_file
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

    def convertVideos(self, dataSummaryLog):
        all_videos = os.listdir(self.inputVideosDir)
        self.dt['ClipAvailable'] = True
        
        for mp4_file in self.dt.ClipName:
            # Ensure labeled clip is an mp4 file
            assert mp4_file.endswith('.mp4')
                
            # Define input and output paths
            video_file_path = os.path.join(self.inputVideosDir, mp4_file)
            outputDir = os.path.join(self.tempDir,mp4_file.replace('.mp4',''))
            
            # Ensure mp4 file exists otherwise skip it
            if not os.path.exists(video_file_path):
                self.dt.loc[self.dt.ClipName == mp4_file,'ClipAvailable'] = False
                continue

            if not os.path.exists(outputDir):
                output = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', video_file_path], capture_output = True, encoding = 'utf-8')
                if "moov atom not found" in output.stderr or "invalid data found when processing input" in output.stderr:
                    print(f"Skipping {video_file_path}: Corrupt video (moov atom missing).")
                    self.dt.loc[self.dt.ClipName == mp4_file,'ClipAvailable'] = False
                    continue
                os.makedirs(outputDir)
                output = subprocess.run(['ffmpeg','-i',video_file_path,outputDir+'/image_%05d.jpg'], capture_output = True)
                if output.returncode != 0:
                    pdb.set_trace()

        data_summary = self.dt.groupby(['AnalysisID','ClipAvailable']).count()['ClipName']
        print(data_summary)
        if dataSummaryLog is not None:
            data_summary.to_csv(dataSummaryLog)

    def calculateMeans(self):

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
                pdb.set_trace()
                continue
            if len(frames) != 120:
                print('Problem with ' + row.ClipName)
                self.dt.loc[self.dt.ClipName == row.ClipName,'ClipAvailable'] = False
                continue
            with open(os.path.join(video_folder, 'n_frames'), 'w') as dst_file:
                dst_file.write(str(len(frames)))
            try:
                img = io.imread(frames[0])
            except IndexError:
                print('Problem with ' + row.ClipName)
                self.dt.loc[self.dt.ClipName == row.ClipName,'ClipAvailable'] = False
                continue
            mean = img.mean(axis = (0,1))
            std = img.std(axis = (0,1))
            
            m_dt.loc[len(m_dt)] = [location, row.ProjectID] + mean.tolist() + std.tolist()
            
        means = m_dt.groupby(['ProjectID']).agg({'MeanR':'mean','MeanG':'mean','MeanB':'mean','StdR':'mean','StdG':'mean','StdB':'mean'})
        self.means_dict = {pid:[r.MeanR,r.MeanG,r.MeanB,r.StdR,r.StdG,r.StdB] for pid,r in means.iterrows()}
        #means.to_csv(os.path.join(self.tempDir,'Means.csv'), index = False)

    def prepareJson(self, purpose, json_file, n_classes):
        
        assert purpose in ['Classify','Train']

        dst_data = {}

        if purpose == 'Classify':
            with open(json_file,'r') as input_f:
                training_json = json.load(input_f) 
            dst_data['labels'] = training_json['labels']
        else:
            dst_data['labels'] = list(set(self.dt.ManualLabel))

        clip_data={}
        for lid,row in self.dt[self.dt.ClipAvailable==True].iterrows():
            clip_name = row.ClipName.replace('.mp4','')
            clip_data[clip_name] = {}
            if purpose == 'Classify':
                clip_data[clip_name]['subset'] = 'validation'
                clip_data[clip_name]['annotations'] = {'label': dst_data['labels'][0]}

            else:
                if np.random.uniform()<0.8:
                    clip_data[clip_name]['subset'] = 'train'
                    clip_data[clip_name]['annotations'] = {'label': row.ManualLabel}

                else:
                    clip_data[clip_name]['subset'] = 'validation'
                    clip_data[clip_name]['annotations'] = {'label': row.ManualLabel}
        
            clip_data[clip_name]['projectID'] = row.ProjectID

        dst_data['database'] = clip_data
        dst_data['means'] = self.means_dict

        try:
            assert len(dst_data['labels'])==n_classes
        except AssertionError:
            pdb.set_trace()

        with open(json_file, 'w') as dst_file:
            json.dump(dst_data, dst_file)

        return        
    
        
        
        
