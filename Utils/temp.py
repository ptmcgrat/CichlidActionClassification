import os
import subprocess
#transfer
source = '/data/home/llong35/data/annotated_videos'
target = '/data/home/llong35/data/all_videos/'

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
animals_list = ['MC6_5', 'MCxCVF1_12a_1', 'MCxCVF1_12b_1', 'TI2_4', 'TI3_3', 'CV10_3']
#download them to seperate folders
master_path = 'McGrath/Apps/CichlidPiData/'
target_data_folder = '/data/home/llong35/data/unlabled_videos'
for animal in animals_list:
    animal_folder = os.path.join(target_data_folder,animal)
    if not os.path.exists(animal_folder):
        os.makedirs(animal_folder)
    animal_video_source = dropbox+':'+master_path+animal+'/AllClips.tar'
    cmd = ['rclone','copy',animal_video_source,animal_folder+'/']
    subprocess.run(cmd)
    cmd = ['tar','-xvf',animal_folder+'/AllClips.tar']
    subprocess.run(cmd)



# source = '/data/home/llong35/data/labeled_videos'
# target = '/data/home/llong35/data/annotated_videos'
# annotation = '/data/home/llong35/patrick_code_test/modelAll_34/AnnotationFile.csv'
# 
# with open(annotation,'r') as input:
#     input.readline()
#     for line in input:
#         tokens = line.split(',')
#         file_name = tokens[0]+'.mp4'
#         label = tokens[2]
#         source_file = os.path.join(source,file_name)
#         target_folder = os.path.join(target,label)
#         if not os.path.exists(target_folder):
#             os.makedirs(target_folder)
#         cmd = ['cp',source_file,target_folder+'/']
#         subprocess.run(cmd)
        
        
        
        
        
