import os
import subprocess
#transfer
# source = '/data/home/llong35/data/annotated_videos'
# target = '/data/home/llong35/data/all_videos/'
# 
# for folder in os.listdir(source):
#     folder_path = os.path.join(source,folder)
#     videos = os.listdir(folder_path)
#     for video in videos:
#         if not video.endswith('.mp4'):
#             continue
#         video_path = os.path.join(folder_path,video)
#         command = ['cp',video_path,target]
#         subprocess.run(command)


# add other datasets to the video clips
dropbox = 'd'
animals_list = ['MC16_2', 'MC6_5', 'MCxCVF1_12a_1', 'MCxCVF1_12b_1', 'TI2_4', 'TI3_3', 'CV10_3']
#download them to seperate folders
master_path = 'McGrath/Apps/CichlidPiData/'
target_data_folder = '/data/home/llong35/data/unlabled_videos'
for animal in animal_list:
    animal_folder = os.path.join(target_data_folder,animal)
    animal_video_source = dropbox+'/'+master_path+animal+'/AllClips.tar'
    cmd = ['rclone','copy',animal_video_source,animal_folder+'/']
    print(cmd)
    break
