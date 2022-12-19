from torchvision.transforms import ToTensor, Normalize, RandomHorizontalFlip, RandomCrop
import torchvision

def get_dataset():
    dataset_dir = 'data/datasets'
    min_vals = (0.0,0.0,0.0)
    max_vals = (1.0,1.0,1.0)
    offset = [0.5 * (min_val + max_val) for min_val, max_val in zip(min_vals, max_vals)]
    scale = [(max_val - min_val) / 2 for max_val, min_val in zip(max_vals, min_vals)]
    
    transforms = torchvision.transforms.Compose(
        [
            ToTensor(),
            Normalize(offset, scale),
            RandomHorizontalFlip(p=0.5),
            RandomCrop(size=32, padding=4),
            
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=transforms)
    test_dataset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=transforms)
    return train_dataset, test_dataset
