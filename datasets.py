# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
# import jax
# import tensorflow as tf
# import tensorflow_datasets as tfds
import torch
import torchvision

import torchvision.transforms as transforms
# import kornia.augmentation as Ktransforms

import os
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
import warnings
# import imageio

Image.MAX_IMAGE_PIXELS = None
warnings.simplefilter('ignore', Image.DecompressionBombWarning)

import math
import random

from torch.utils.data.sampler import Sampler

import dist

from typing import Any, Callable, Optional, Tuple

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform
from timm.data.transforms_factory import transforms_imagenet_eval
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS

import pickle

try:
    from torchvision.transforms import InterpolationMode

    interpolation = InterpolationMode.BICUBIC
except:
    import PIL

    interpolation = PIL.Image.BICUBIC


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


class UniformDequant(object):
    def __call__(self, x):
        return x + torch.rand_like(x) / 256


class RASampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU).
    Heavily based on 'torch.utils.data.DistributedSampler'.
    This is borrowed from the DeiT Repo:
    https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, repetitions=3):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * float(repetitions) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle
        self.seed = seed
        self.repetitions = repetitions

    def __iter__(self):
        if self.shuffle:
            # Deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add extra samples to make it evenly divisible
        indices = [ele for ele in indices for i in range(self.repetitions)]
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # Subsample
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[: self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, seed=0, filling=False, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.indices = self.gener_indices()

    def gener_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = torch.arange(self.dataset_len).numpy()

        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.filling:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))

        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())

    def __iter__(self):
        self.epoch = 0
        while True:
            self.epoch += 1
            p, q = 0, 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()

    def __len__(self):
        return self.iters_per_ep


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(self, world_size, rank, dataset_len, glb_batch_size, seed=0, repeated_aug=0, filling=False,
                 shuffle=True):
        # from torchvision.models import ResNet50_Weights
        # RA sampler: https://github.com/pytorch/vision/blob/5521e9d01ee7033b9ee9d421c1ef6fb211ed3782/references/classification/sampler.py

        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size

        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.repeated_aug = repeated_aug
        self.epoch = 0
        self.seed = seed
        self.indices = self.gener_indices()

    def gener_indices(self):
        global_max_p = self.iters_per_ep * self.glb_batch_size  # global_max_p % world_size must be 0 cuz glb_batch_size % world_size == 0
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            global_indices = torch.randperm(self.dataset_len, generator=g)
            if self.repeated_aug > 1:
                global_indices = global_indices[
                                 :(self.dataset_len + self.repeated_aug - 1) // self.repeated_aug].repeat_interleave(
                    self.repeated_aug, dim=0)[:global_max_p]
        else:
            global_indices = torch.arange(self.dataset_len)
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.filling:
            global_indices = torch.cat((global_indices, global_indices[:filling]))
        global_indices = tuple(global_indices.numpy().tolist())

        seps = torch.linspace(0, len(global_indices), self.world_size + 1, dtype=torch.int)
        local_indices = global_indices[seps[self.rank]:seps[self.rank + 1]]
        self.max_p = len(local_indices)
        return local_indices


def get_dataset(config, root, uniform_dequantization=False, batch_size=128, is_infinite=True, dist_eval=False):
    """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """

    # Compute batch size for this worker.
    dataset = config.data

    if dataset == 'MNIST':
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=True, transform=
        transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor()] + ([UniformDequant()] if uniform_dequantization else [])))
        testset = torchvision.datasets.MNIST(root=root, train=False, download=True, transform=
        transforms.Compose(
            [transforms.Resize(32), transforms.ToTensor()] + ([UniformDequant()] if uniform_dequantization else [])))

        # train_X, train_y = trainset.data[:,None] / 255, trainset.targets
        # test_X, test_y = testset.data[:,None] / 255, testset.targets
        n_classes = 10
    elif dataset == 'FMNIST':
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=True, transform=
        transforms.Compose([transforms.ToTensor()] + ([UniformDequant()] if uniform_dequantization else [])))
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=True, transform=
        transforms.Compose([transforms.ToTensor()] + ([UniformDequant()] if uniform_dequantization else [])))

        # train_X, train_y = trainset.data[:,None] / 255, trainset.targets
        # test_X, test_y = testset.data[:,None] / 255, testset.targets
        n_classes = 10
    elif dataset == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=
        transforms.Compose([
                               transforms.RandomResizedCrop(config.data.image_size, scale=(0.67, 1.0),
                                                            interpolation=transforms.InterpolationMode.BICUBIC),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), ] + ([UniformDequant()] if uniform_dequantization else [])))
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=
        transforms.Compose([transforms.ToTensor()] + ([UniformDequant()] if uniform_dequantization else [])))

        # train_X = torch.tensor(trainset.data).permute(0,3,1,2) / 255
        # train_y = torch.tensor(trainset.targets, dtype=int)

        # test_X = torch.tensor(testset.data).permute(0,3,1,2) / 255
        # test_y = torch.tensor(testset.targets, dtype=int)
        n_classes = 10
    elif dataset == 'fMoW':
        interpol_mode = transforms.InterpolationMode.BICUBIC

        train_path = os.path.join('/home/amna97', 'train_62classes.csv')
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(FMOW.mean, FMOW.std),
            transforms.RandomResizedCrop(config.input_size, scale=(0.5, 1.0), interpolation=interpol_mode),
            transforms.RandomHorizontalFlip(),
        ] + ([UniformDequant()] if uniform_dequantization else []))
        trainset = FMOW(train_path, train_transform)

        crop_pct = 224/256
        test_path = os.path.join('/home/amna97', 'test_62classes.csv')
        test_transform = transforms.Compose([
            transforms.ToTensor(),  # transforms.Normalize(FMOW.mean, FMOW.std),
            transforms.Resize(int(config.input_size/crop_pct), interpolation=interpol_mode,),
            transforms.CenterCrop(config.input_size),
        ] + ([UniformDequant()] if uniform_dequantization else []))
        testset = FMOW(test_path, test_transform)
        n_classes = 62
    elif dataset == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=
        transforms.Compose([transforms.ToTensor(), ] + ([UniformDequant()] if uniform_dequantization else [])))
        testset = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=
        transforms.Compose([transforms.ToTensor()] + ([UniformDequant()] if uniform_dequantization else [])))

        # train_X = torch.tensor(trainset.data).permute(0,3,1,2) / 255
        # train_y = torch.tensor(trainset.targets, dtype=int)

        # test_X = torch.tensor(testset.data).permute(0,3,1,2) / 255
        # test_y = torch.tensor(testset.targets, dtype=int)
        n_classes = 100
    elif dataset == 'TinyImageNet':
        trainset = TinyImageNetDataset(root_dir=root, mode='train', download=False, transform=
        transforms.Compose([transforms.ToTensor(), ] + ([UniformDequant()] if uniform_dequantization else [])))
        testset = TinyImageNetDataset(root_dir=root, mode='val', download=False, transform=
        transforms.Compose([transforms.ToTensor()] + ([UniformDequant()] if uniform_dequantization else [])))

        n_classes = 200
    elif dataset == 'ImageNet64':

        transform = [
            transforms.RandomResizedCrop(64, scale=(0.67, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        trainset = ImageNetPickleDataset(root=root, train=True, transform=
        transforms.Compose(transform + ([UniformDequant()] if uniform_dequantization else [])))
        testset = ImageNetPickleDataset(root=root, train=False, transform=
        transforms.Compose([transforms.ToTensor()] + ([UniformDequant()] if uniform_dequantization else [])))

        n_classes = 1000
    elif dataset == 'ImageNet':

        transform = [
            transforms.RandomResizedCrop(config.data.image_size, scale=(0.67, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]
        trainset = ImageNetDataset(root=root, train=True, transform=
        transforms.Compose(transform + ([UniformDequant()] if uniform_dequantization else [])))
        testset = ImageNetDataset(root=root, train=False, transform=
        transforms.Compose([transforms.ToTensor()] + ([UniformDequant()] if uniform_dequantization else [])))

        n_classes = 1000
    else:
        raise NotImplementedError()

    if dist.initialized():
        num_tasks = dist.get_world_size()
        global_rank = dist.get_rank()
        # sampler_train = torch.utils.data.DistributedSampler(
        #     trainset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        # )
        if is_infinite:
            train_loader = torch.utils.data.DataLoader(
                dataset=trainset, num_workers=8, pin_memory=True,
                batch_sampler=DistInfiniteBatchSampler(
                    dataset_len=len(trainset), glb_batch_size=batch_size, seed=config.seed,
                    shuffle=True, filling=True, rank=dist.get_rank(), world_size=dist.get_world_size(),
                )
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                dataset=trainset, num_workers=8, pin_memory=True, drop_last=True, batch_size=batch_size // num_tasks,
                sampler=torch.utils.data.DistributedSampler(
                    trainset, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=True
                )
            )

        if dist_eval:
            sampler_test = torch.utils.data.DistributedSampler(
                testset, num_replicas=num_tasks, rank=global_rank, shuffle=True, drop_last=False
            )
        else:
            sampler_test = None
    else:
        num_tasks = 1
        sampler_train = None
        sampler_test = None
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size // num_tasks, sampler=sampler_train, num_workers=2)

    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size // num_tasks, drop_last=False, num_workers=8, pin_memory=True,
        sampler=sampler_test
    )
    return train_loader, test_loader, n_classes


## fMoW
class FMOW(torch.utils.data.Dataset):
    mean = [0.4182007312774658, 0.4214799106121063, 0.3991275727748871]
    std = [0.28774282336235046, 0.27541765570640564, 0.2764017581939697]

    def __init__(self, csv_path, transform):
        """
        Creates Dataset for regular RGB image classification (usually used for fMoW-RGB dataset).
        :param csv_path: csv_path (string): path to csv file.
        :param transform: pytorch transforms for transforms and tensor conversion.
        """
        super().__init__()
        # Transforms
        self.transforms = transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # First column contains the image paths
        self.image_arr = np.asarray(self.data_info.iloc[:, 1])
        # Second column is the labels
        self.label_arr = np.asarray(self.data_info.iloc[:, 0])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform the image
        img_as_tensor = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]

        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len


def download_and_unzip(URL, root_dir):
    error_message = f"Download is not yet implemented. Please, go to {URL} urself."
    raise NotImplementedError(error_message)


def _add_channels(img, total_channels=3):
    while len(img.shape) < 3:  # third axis is the channels
        img = np.expand_dims(img, axis=-1)
    while (img.shape[-1]) < 3:
        img = np.concatenate([img, img[:, :, -1:]], axis=-1)
    return img


class TinyImageNetPaths:
    def __init__(self, root_dir, download=False):
        if download:
            download_and_unzip('http://cs231n.stanford.edu/tiny-imagenet-200.zip',
                               root_dir)
        train_path = os.path.join(root_dir, 'train')
        val_path = os.path.join(root_dir, 'val')
        test_path = os.path.join(root_dir, 'test')

        wnids_path = os.path.join(root_dir, 'wnids.txt')
        words_path = os.path.join(root_dir, 'words.txt')

        self._make_paths(train_path, val_path, test_path,
                         wnids_path, words_path)

    def _make_paths(self, train_path, val_path, test_path,
                    wnids_path, words_path):
        self.ids = []
        with open(wnids_path, 'r') as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, 'r') as wf:
            for line in wf:
                nid, labels = line.split('\t')
                labels = list(map(lambda x: x.strip(), labels.split(',')))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            'train': [],  # [img_path, id, nid, box]
            'val': [],  # [img_path, id, nid, box]
            'test': []  # img_path
        }

        # Get the test paths
        self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      [s for s in os.listdir(test_path) if s.startswith('n')]))

        # Get the validation paths and labels
        with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, 'images', fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths['val'].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + '_boxes.txt')
            imgs_path = os.path.join(train_path, nid, 'images')
            if nid not in self.ids:
                continue
            label_id = self.ids.index(nid)
            with open(anno_path, 'r') as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    if os.path.isdir(fname):
                        continue
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, mode='train', preload=False, load_transform=None,
                 transform=None, download=False, max_samples=None):
        tinp = TinyImageNetPaths(root_dir, download)
        self.mode = mode
        self.label_idx = 1  # from [image, id, nid, box]
        self.preload = preload
        self.transform = transform
        self.transform_results = dict()

        self.IMAGE_SHAPE = (64, 64, 3)

        self.img_data = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

        if self.preload:
            load_desc = "Preloading {} data...".format(mode)
            self.img_data = np.zeros((self.samples_num,) + self.IMAGE_SHAPE,
                                     dtype=np.float32)
            self.label_data = np.zeros((self.samples_num,), dtype=np.int)
            for idx in tqdm(range(self.samples_num), desc=load_desc):
                s = self.samples[idx]
                img = imageio.imread(s[0])
                img = _add_channels(img)
                self.img_data[idx] = img
                if mode != 'test':
                    self.label_data[idx] = s[self.label_idx]

            if load_transform:
                for lt in load_transform:
                    result = lt(self.img_data, self.label_data)
                    self.img_data, self.label_data = result[:2]
                    if len(result) > 2:
                        self.transform_results.update(result[2])

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        if self.preload:
            img = self.img_data[idx]
            lbl = None if self.mode == 'test' else self.label_data[idx]
        else:
            s = self.samples[idx]
            img = imageio.imread(s[0])
            img = _add_channels(img)
            lbl = None if self.mode == 'test' else s[self.label_idx]

        if self.transform:
            img = self.transform(img)
        sample = (img, lbl)

        return sample


############ imagenet


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch(data_folder, img_size=32, filename='train_data_batch_1'):
    d = unpickle(os.path.join(data_folder, filename))
    x = d['data']
    y = d['labels']

    x = x / np.float32(255)
    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i - 1 for i in y]
    data_size = x.shape[0]
    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = np.array(y[0:data_size])

    return X_train, Y_train


class ImageNetPickleDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str,
            train: bool,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            max_cls_id: int = 1000
    ):
        super(ImageNetDataset, self).__init__()
        split = 'train' if train else 'val'

        files = [f for f in os.listdir(root) if f.startswith(split)]
        self.samples = []
        self.targets = []
        for i, f in tqdm(enumerate(files), total=len(files)):
            X_train_b, Y_train_b = load_databatch(root, 64, f)
            self.samples.append(X_train_b[Y_train_b < max_cls_id])
            self.targets.append(Y_train_b[Y_train_b < max_cls_id])

        self.samples = np.concatenate(self.samples, axis=0)
        self.targets = np.concatenate(self.targets, axis=0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class ImageNetDataset(DatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            max_cls_id: int = 1000,
            only=-1,
    ):
        for postfix in (os.path.sep, 'train', 'val'):
            if root.endswith(postfix):
                root = root[:-len(postfix)]

        root = os.path.join(root, 'train' if train else 'val')

        super(ImageNetDataset, self).__init__(
            root,
            # loader=ImageLoader(train),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform, target_transform=target_transform, is_valid_file=is_valid_file
        )

        if only > 0:
            g = torch.Generator()
            g.manual_seed(0)
            idx = torch.randperm(len(self.samples), generator=g).numpy().tolist()

            ws = dist.get_world_size()
            res = (max_cls_id * only) % ws
            more = 0 if res == 0 else (ws - res)
            max_total = max_cls_id * only + more
            if (max_total // ws) % 2 == 1:
                more += ws
                max_total += ws

            d = {c: [] for c in range(max_cls_id)}
            max_len = {c: only for c in range(max_cls_id)}
            for c in range(max_cls_id - more, max_cls_id):
                max_len[c] += 1

            total = 0
            for i in idx:
                path, target = self.samples[i]
                if len(d[target]) < max_len[target]:
                    d[target].append((path, target))
                    total += 1
                if total == max_total:
                    break
            sp = []
            [sp.extend(l) for l in d.values()]

            print(f'[ds] more={more}, len(sp)={len(sp)}')
            self.samples = tuple(sp)
            self.targets = tuple([s[1] for s in self.samples])
        else:
            self.samples = tuple(filter(lambda item: item[-1] < max_cls_id, self.samples))
            self.targets = tuple([s[1] for s in self.samples])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
