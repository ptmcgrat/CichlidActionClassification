import torch, sys, pdb
import torch.utils.data as data
from PIL import Image
import os,json

from Utils.transforms import (Compose, Normalize,CenterCrop, 
                              RandomHorizontalFlip,RandomVerticalFlip, 
                              MultiScaleRandomCenterCrop,ToTensor,
                              TemporalCenterCrop, TemporalCenterRandomCrop)

class cichlids(data.Dataset):
    def __init__(self, clip_directory, json_file, dataset_type):
        self.clip_directory = clip_directory
        self.json_file = json_file
        assert dataset_type in ['train', 'validation']
        self.dataset_type = dataset_type

        self.readDatabase()

    def readDatabase(self):
        with open(self.json_file,'r') as input_f:
            data_dict = json.load(input_f)
        self.labels = data_dict['labels']
        self.labels_to_idx = {x:i for x,i in zip(self.labels,range(len(self.labels)))}
        self.clip_dict = {k:v for k,v in data_dict['database'].items() if v['subset'] == self.dataset_type}
        self.clips = [x for x in self.clip_dict.keys()]
        self.means = data_dict['means']

    def createTransforms(self, sample_size, sample_duration):    
        self.spatial_transforms = {}

        if self.dataset_type == 'train':
            crop_method = MultiScaleRandomCenterCrop([0.99,0.97,0.95,0.93,0.91], sample_size)
            self.temporal_transform = TemporalCenterRandomCrop(sample_duration)

        else:
            crop_method = CenterCrop(sample_size)
            self.temporal_transform = TemporalCenterCrop(sample_duration)

        for pid,norm in self.means.items():
            norm_method = Normalize([float(x) for x in norm[0:3]], [float(x) for x in norm[3:6]]) 
            if self.dataset_type == 'train':
                self.spatial_transforms[pid] = Compose([crop_method, RandomVerticalFlip(),RandomHorizontalFlip(), ToTensor(1), norm_method])
            else:
                self.spatial_transforms[pid] = Compose([crop_method, ToTensor(1), norm_method])

    def video_loader(self, video_dir_path, frame_indices):
        video = []
        for i in frame_indices:
            image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
            if os.path.exists(image_path):
                with open(image_path, 'rb') as f:
                    with Image.open(f) as img:
                        video.append(img.convert('RGB'))
            else:
                return video

        return video

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        clipname = self.clips[index]
        file_location = self.clip_directory + clipname + '/'
        with open(file_location + 'n_frames') as f:
            n_frames = int(f.read())
        frame_indices = [x+1 for x in range(n_frames)]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)

        clip = self.video_loader(file_location, frame_indices)
        if self.spatial_transforms is not None:
            self.spatial_transforms[self.clip_dict[clipname]['projectID']].randomize_parameters()
            clip = [self.spatial_transforms[self.clip_dict[clipname]['projectID']](img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.clip_to_idx[self.data_dict[clipname]['annotations']['label']]
        return clip, target,path

    def __len__(self):
        return len(self.clips)
