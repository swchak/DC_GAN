import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torch
import torchvision
import torchvision.transforms as transforms

''' utils.py '''


def display_grid(images: torch.Tensor):
    image_grid = make_grid(images)
    image_grid = image_grid.permute(1, 2, 0)
    image_grid = image_grid.cpu().numpy()
    plt.imshow(image_grid)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.close()


def get_training_dl(root_dir: str, ds_name: str, batch_size, image_size=32, nc=1, mean=0.5, std=0.5, download=True):
    if ds_name == 'letters':
       train_ds = torchvision.datasets.EMNIST(root='data', split='letters', train=True, transform=transforms.Compose([transforms.ToTensor()]), download=download)
    elif ds_name == 'flowers':
        train_ds = torchvision.datasets.Flowers102(root_dir, split='train', transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(nc),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]), download=True)
    elif ds_name == 'fmnist':
         train_ds = torchvision.datasets.FashionMNIST(root_dir, train=True, transform=transforms, download=True)
    elif ds_name == 'pets': 
        train_ds = torchvision.datasets.OxfordIIITPet(root=root_dir, split="trainval", transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.Grayscale(nc),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ]), download=True)
    else:
        train_ds = None
    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

