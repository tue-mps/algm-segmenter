import click
import time
import glob
import os 
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
import pickle
from typing import List, Tuple, Union
from fvcore.nn import FlopCountAnalysis

import torch
import torchvision.transforms.functional as F
import torch.nn.functional as F_I
from torchvision import datasets, transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset


from segm.model.factory import load_model
from segm.model.utils import inference
from segm.data.utils import STATS
from segm.data.factory import create_dataset

import algm



import warnings
warnings.filterwarnings("ignore")


@torch.no_grad()


def get_dataset_validaion_path(dataset_name, root_dir):
    dataset_path_txt = None
    if dataset_name == 'ade20k':
        dataset_path = root_dir + '/ade20k/ADEChallengeData2016/images/validation/'
    elif dataset_name == 'cityscapes':
        dataset_path = root_dir + '/cityscapes/leftImg8bit/val/'
    elif dataset_name == 'pascal_context':
        dataset_path_txt = root_dir + '/pcontext/VOCdevkit/VOC2010/ImageSets/SegmentationContext/val.txt'
        dataset_path = root_dir + '/pcontext/VOCdevkit/VOC2010/JPEGImages/'

    return dataset_path ,dataset_path_txt

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, txt_file=None):
        self.root_dir = root_dir
        self.transform = transform

        # If txt_file is provided, read image names from it
        if txt_file:
            with open(txt_file, 'r') as file:
                self.image_files = [os.path.join(root_dir, line.strip() + '.jpg') for line in file.readlines()]
        else:
            # Otherwise, load all image paths from root_dir and subfolders
            self.image_files = self._load_image_paths(root_dir)

    def _load_image_paths(self, root_dir):
        image_files = []
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                    image_files.append(os.path.join(dirpath, file))
        return image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image
    
def dataset_prepare(dataset_path,dataset_txt_path,stats,batch_size,input_size):

    validation_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),  # Resize the image to input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(stats["mean"], stats["std"]) # Normalize with mean and std
    ])

    if dataset_txt_path is None:

        validation_dataset = InferenceDataset(root_dir=dataset_path, transform=validation_transforms)

    else:
        validation_dataset = InferenceDataset(root_dir=dataset_path, transform=validation_transforms, txt_file=dataset_txt_path)

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    return validation_loader 


def compute_flops_per_image(model,validation_loader,device ,batch_size, resolution):

    gflops = []
    for image in tqdm(validation_loader, position=0, leave=False):
        image = image.to(device)
        flops = FlopCountAnalysis(model,image)
        flops.unsupported_ops_warnings(False).uncalled_modules_warnings(False)
        gflops.append(flops.total() / 1e9)
    
    return np.mean(gflops)





@click.command()
@click.option("--model-dir", type=str)
@click.option("--dataset", type=str)
@click.option("--batch-size", type=int, default=1)
@click.option("--patch-type", default="pure", type=str)
@click.option("--selected-layers", nargs=2, type=int)
@click.option("--merging-window-size", nargs=2, type=int)
@click.option("--threshold", default=0.88, type=float)





def main(model_dir, 
    dataset, 
    batch_size, 
    patch_type, 
    selected_layers, 
    merging_window_size, 
    threshold,):

    device = 'cuda:0'
    
    batch_size = batch_size 
    model_path_list = glob.glob(model_dir + '/checkpoint.pth')
    root_dir = os.getenv('DATASET')

    dataset_path, dataset_txt_path = get_dataset_validaion_path(dataset, root_dir)
    
    for model_path in model_path_list:
        model, variant = load_model(model_path)
        input_size = variant['dataset_kwargs']['crop_size']
        normalization = variant["dataset_kwargs"]["normalization"]
        stats = STATS[normalization]

        if patch_type == "algm":
            algm.patch.algm_segmenter_patch(model, selected_layers, trace_source=True)
            model.encoder.window_size = merging_window_size
            model.encoder.threshold = threshold     

        model.eval()
        model.to(device)
    

    validaion_loader = dataset_prepare(dataset_path, dataset_txt_path, stats, batch_size, input_size)
    GFlops = compute_flops_per_image(model, validaion_loader,device, batch_size, input_size)
    print('GFlops:', GFlops, flush=True)

if __name__ == "__main__":
    main()