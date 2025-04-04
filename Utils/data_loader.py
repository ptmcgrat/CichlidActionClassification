import torch, sys, pdb
import torch.utils.data as data
from PIL import Image
import os
import math
import functools
import json
import copy

from Utils.utils import load_value_file


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
    labels = []

    for key, value in data['database'].items():
        this_subset = value['subset']
        if this_subset == subset:
            label = value['annotations']['label']
            video_names.append(key)
            labels.append(label)
            # pdb.set_trace()
            
    # pdb.set_trace()
    return video_names,labels


def make_dataset(root_path, annotation_path, subset, n_samples_for_each_video,
                 sample_duration,args):
    data = load_annotation_data(annotation_path)
    # pdb.set_trace()
    video_names, labels = get_video_names_and_annotations(data, subset)
    # pdb.set_trace()
    class_to_idx = get_class_labels(data)
    idx_to_class = {}
    for name, label in class_to_idx.items():
        idx_to_class[label] = name
    dataset = []
    video_count = 0
    n_frames_count = 0
    for i in range(len(video_names)):
        if i % 1000 == 0:
            print('dataset loading [{}/{}]'.format(i, len(video_names)))
        
        
        if args.Purpose == 'classify': 
            video_name = "__".join(video_names[i].split("__")[-6:])
        else:
            video_name = video_names[i]
        video_path = os.path.join(root_path, video_name)
        # pdb.set_trace()
        if not os.path.exists(video_path):
            print(video_path+' not exist')
            video_count+=1
            continue

        n_frames_file_path = os.path.join(video_path, 'n_frames')
        if not os.path.exists(n_frames_file_path):
            print(n_frames_file_path+' not exist')
            n_frames_count += 1
            continue
        n_frames = int(load_value_file(n_frames_file_path))
        
        if n_frames <= 0:
            continue

        begin_t = 1
        end_t = n_frames
        sample = {
            'video': video_path,
            'segment': [begin_t, end_t],
            'n_frames': n_frames,
            'video_id': video_name
        }
        if len(labels) != 0:
            if labels[i] != 'target':
                sample['label'] = class_to_idx[labels[i]]
            else:
                sample['label'] = -1
        else:
            sample['label'] = -1

        if n_samples_for_each_video == 1:
            sample['frame_indices'] = list(range(1, n_frames + 1))
            dataset.append(sample)
        else:
            if n_samples_for_each_video > 1:
                step = max(1,
                           math.ceil((n_frames - 1 - sample_duration) /
                                     (n_samples_for_each_video - 1)))
            else:
                step = sample_duration
            for j in range(1, n_frames, step):
                sample_j = copy.deepcopy(sample)
                sample_j['frame_indices'] = list(
                    range(j, min(n_frames + 1, j + sample_duration)))
                dataset.append(sample_j)
    # pdb.set_trace()
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
                 get_loader=get_default_video_loader,
                 args = None):
        self.data, self.class_names = make_dataset(
            root_path, annotation_path, subset, n_samples_for_each_video,
            sample_duration,args)
        
        self.subset = subset

        self.spatial_transforms = spatial_transforms
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        # pdb.set_trace()
        self.annotationDict = annotationDict
        self.args =args
        # pdb.set_trace()
        # pdb.set_trace()
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.data[index]['video']
        if self.args.Purpose == 'classify': 
            clip_name = self.data[index]['video_id'] +'.mp4'
        else:
            clip_name =  "__".join(self.data[index]['video_id'].split("__")[-5:])
        
        # pdb.set_trace()
        frame_indices = self.data[index]['frame_indices']
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        clip = self.loader(path, frame_indices)
        # pdb.set_trace()
        if self.spatial_transforms is not None:
            # pdb.set_trace()
            self.spatial_transforms[self.annotationDict[clip_name]].randomize_parameters()
            clip = [self.spatial_transforms[self.annotationDict[clip_name]](img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)

        target = self.data[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return clip, target,path

    def __len__(self):
        return len(self.data)
