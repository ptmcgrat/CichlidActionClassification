#transfer
import os
import subprocess
source = '/data/home/llong35/data/annotated_videos'
target = '/data/home/llong35/data/all_videos/'

for folder in os.listdir(source):
    folder_path = os.path.join(source,folder)
    videos = os.listdir(folder_path):
        for video in videos:
            if not video.endswith('.mp4'):
                continue
            video_path = os.path.join(folder_path,video)
            command = ['cp',video_path,target]
            subprocess.run(command)
