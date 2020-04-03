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
            cmd = 'ffmpeg -i \"{}\" \"{}/image_%05d.jpg\"'.format(video_file_path, target_folder)
            subprocess.call(cmd)
            break
            
        #calculate means
        
        
        