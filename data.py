import torchvision
import torch
import numpy as np

import torch_xla.core.xla_model as xm
from torchvision.transforms import ToTensor, Normalize, RandomHorizontalFlip, RandomCrop

def get_dataset():
    dataset_dir = 'data/datasets'
    min_vals = (0.0,0.0,0.0)
    max_vals = (1.0,1.0,1.0)
    offset = [0.5 * (min_val + max_val) for min_val, max_val in zip(min_vals, max_vals)]
    scale = [(max_val - min_val) / 2 for max_val, min_val in zip(max_vals, min_vals)]

    train_transforms = torchvision.transforms.Compose(
        [
            ToTensor(),
            Normalize(offset, scale),
            RandomHorizontalFlip(p=0.5),
            RandomCrop(size=32, padding=4),

        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            ToTensor(),
            Normalize(offset, scale),
        ]
    )

    if not xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    train_dataset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=test_transforms)

    if xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    return train_dataset, test_dataset

def get_datasetv2():
    dataset_dir = 'data/datasets'
    min_vals = (0.0,0.0,0.0)
    max_vals = (1.0,1.0,1.0)
    offset = [0.5 * (min_val + max_val) for min_val, max_val in zip(min_vals, max_vals)]
    scale = [(max_val - min_val) / 2 for max_val, min_val in zip(max_vals, min_vals)]

    train_transforms = torchvision.transforms.Compose(
        [
            ToTensor(),
            Normalize(offset, scale),
            RandomHorizontalFlip(p=0.5),
            RandomCrop(size=32, padding=4),

        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            ToTensor(),
            Normalize(offset, scale),
        ]
    )

    if not xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    train_dataset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=train_transforms)
    test_dataset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=test_transforms)

    if xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    num_test = len(test_dataset)
    indices = list(range(num_test))
    np.random.RandomState(seed=42).shuffle(indices)
    valid_split = int(0.2 * num_test)
    print("valid_split", valid_split)

    test_idx, valid_idx = indices[valid_split:], indices[:valid_split]
    valid_dataset = torch.utils.data.Subset(test_dataset, valid_idx)
    test_dataset = torch.utils.data.Subset(test_dataset, test_idx)

    return train_dataset, test_dataset, valid_dataset
