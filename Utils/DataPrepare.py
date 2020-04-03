import os
import subprocess
class DP_worker():
    def __init__(self, args):
        self.args = args
    
    def work(self):
        #convert to jpegs
        ML_video_dir = self.args.ML_videos_directory
        videos_temp = self.args.Clips_temp_directory
        for file_name in os.listdir(ML_video_dir):
            if not file_name.endswith('.mp4'):
                continue
            location = file_name.split('.')[0]
            video_file_path = os.path.join(ML_video_dir,location+'.mp4')
            target_folder = os.path.join(self.args.Clips_temp_directory,location)
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
#             cmd = 'ffmpeg -i {} {}/image_%05d.jpg'.format(video_file_path, target_folder)
            cmd = ['ffmpeg','-i',video_file_path,target_folder+'/image_%05d.jpg']
            subprocess.run(cmd)
            break
        
        #count number of frames
        for video in os.listdir(videos_temp):
            video_folder = os.path.join(videos_temp,video)
            image_indices = []
            for image_file_name in os.listdir(video_folder):
                if 'image' not in image_file_name:
                    continue
                image_indices.append(int(image_file_name[6:11]))
            image_indices.sort(reverse=True)
            n_frames = image_indices[0]
            with open(os.path.join(video_folder, 'n_frames'), 'w') as dst_file:
                dst_file.write(str(n_frames))
            break
        
            
        #calculate means
        
        
        