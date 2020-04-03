#transfer
import os
source = '/data/home/llong35/data/annotated_videos'
target = '/data/home/llong35/data/all_videos'

for folder in os.listdir(source):
    folder_path = os.path.join(source,folder)
    print(folder_path)
