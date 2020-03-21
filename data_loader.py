import torch, sys, pdb
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy
from skvideo import io as vp

from utils import load_value_file


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    try:
        import accimage
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def get_default_image_loader():
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader
    else:
        return pil_loader


def video_loader(video_dir_path, frame_indices, image_loader):
    video = []
    for i in frame_indices:
        image_path = os.path.join(video_dir_path, 'image_{:05d}.jpg'.format(i))
        if os.path.exists(image_path):
            video.append(image_loader(image_path))
        else:
            return video

    return video


def get_default_video_loader():
    image_loader = get_default_image_loader()
    return functools.partial(video_loader, image_loader=image_loader)


def load_annotation_data(data_file_path):
    with open(data_file_path, 'r') as data_file:
        return json.load(data_file)


def get_class_labels(data):
    class_labels_map = {}
    index = 0
    for class_label in data['labels']:
        class_labels_map[class_label] = index
        index += 1
    return class_labels_map


def get_video_names_and_annotations(data, subset):
    video_names = []
    annotations = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append('{}/{}'.format(label, key))
            annotations.append(value['annotations'])

    return video_names, annotations


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration):
    
    data = load_annotation_data(annotation_path)
    video_names, annotations = get_video_names_and_annotations(data, subset)
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    dataset = []
    for i in range(len(video_names)):
        #print(video_names[i])
        if i % 10 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))

        video_path = os.path.join(root_path, video_names[i]+'.mp4')
        if not os.path.exists(video_path):
            raise(video_path+'not exist')


        sample = {
            'video': video_path,
            'video_id': video_names[i].split('/')[1],
        }
        if len(annotations) != 0:
            sample['label'] = class_to_idx[annotations[i]['label']]
        else:
            sample['label'] = -1

        dataset.append(sample)
    return dataset, idx_to_class




class cichlids(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self,
                 root_path,
                 annotation_path,
                 subset,
                 n_samples_for_each_video=1,
                 spatial_transforms=None,
                 temporal_transform=None,
                 target_transform=None,
                 annotationDict = None,
                 sample_duration=16,
                 get_loader=get_default_video_loader):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration)

        self.spatial_transforms = spatial_transforms
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        self.annotationDict = annotationDict

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        clip_name = path.rstrip().split('/')[-1].split('.')[0]
        try:
            clip_numpy = vp.vread(path)
        except:
            print(path)
            raise
        n_frames = clip_numpy.shape[0]
        frame_indices = [x for x in range(n_frames)]
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
#         clip = [Image.fromarray(clip_numpy[i]) for i in frame_indices]
        clip = [clip_numpy[i] for i in frame_indices]
#         path = self.data[index]['video']
#         
#         frame_indices = self.data[index]['frame_indices']

#         clip = self.loader(path, frame_indices)
        if self.spatial_transforms is not None:
            self.spatial_transforms[self.annotationDict[clip_name]].randomize_parameters()
            clip = [self.spatial_transforms[self.annotationDict[clip_name]](img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return clip, target, path

    def __len__(self):
        return len(self.data)


def get_training_set(opt, spatial_transforms, temporal_transform,
                     target_transform, annotationDict):
    training_data = cichlids(
    opt.video_path,
    opt.annotation_path,
    'training',
    spatial_transforms=spatial_transforms,
    temporal_transform=temporal_transform,
    target_transform=target_transform, annotationDict = annotationDict)
    return training_data

def get_validation_set(opt, spatial_transforms, temporal_transform,
                       target_transform, annotationDict):
    validation_data = cichlids(
    opt.video_path,
    opt.annotation_path,
    'validation',
    opt.n_val_samples,
    spatial_transforms,
    temporal_transform,
    target_transform,
    sample_duration=opt.sample_duration,annotationDict = annotationDict)
    return validation_data

def get_test_set(opt, spatial_transforms, temporal_transform,
                       target_transform, annotationDict):
    test_data = cichlids(
    opt.video_path,
    opt.annotation_path,
    'test',
    opt.n_val_samples,
    spatial_transforms,
    temporal_transform,
    target_transform,
    sample_duration=opt.sample_duration,annotationDict = annotationDict)
    return test_data
    
# from opts import parse_opts
# import pandas as pd
# from transforms import (
#     Compose, Normalize, Scale, CenterCrop, 
#     RandomHorizontalFlip, FixedScaleRandomCenterCrop, 
#     ToTensor,TemporalCenterCrop, TemporalCenterRandomCrop,
#     ClassLabel, VideoID,TargetCompose)
# opt = parse_opts()
# if opt.root_path != '':
#     opt.video_path = os.path.join(opt.root_path, opt.video_path)
#     opt.annotation_path = os.path.join(opt.root_path, opt.annotation_path)
#     opt.result_path = os.path.join(opt.root_path, opt.result_path)
# 
# print(opt)
# crop_method = FixedScaleRandomCenterCrop(opt.sample_size,2)
# spatial_transforms = {}
# with open(opt.mean_file) as f:
#     for i,line in enumerate(f):
#         if i==0:
#             continue
#         tokens = line.rstrip().split(',')
#         norm_method = Normalize([float(x) for x in tokens[1:4]], [float(x) for x in tokens[4:7]]) 
#         spatial_transforms[tokens[0]] = Compose([crop_method, RandomHorizontalFlip(), ToTensor(opt.norm_value), norm_method])
# annotateData = pd.read_csv(opt.annotation_file, sep = ',', header = 0)
# keys = annotateData[annotateData.Dataset=='Train']['Location']
# values = annotateData[annotateData.Dataset=='Train']['MeanID']
# 
# annotationDictionary = dict(zip(keys, values))
# 
# temporal_transform = TemporalCenterRandomCrop(opt.sample_duration)
# target_transform = ClassLabel()
# training_data = get_training_set(opt, spatial_transforms,
#                                          temporal_transform, target_transform, annotationDictionary)
# pdb.set_trace()
# training_data.__getitem__(0)