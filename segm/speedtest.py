import click
import time
import glob
import pickle
import os 
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import numpy as np
from fvcore.nn import FlopCountAnalysis
from typing import List, Tuple, Union

import torch
import torch.nn.functional as F_I
from torchvision.io import read_image
import torchvision.transforms.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset


from segm.model.factory import load_model
from segm.model.utils import inference
from segm.data.factory import create_dataset
from segm.data.utils import STATS


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
    transforms.Resize((input_size, input_size)),  # Resize the image to the input size
    transforms.ToTensor(),          # Convert image to tensor
    transforms.Normalize(stats["mean"], stats["std"]) # Normalize with mean and std
    ])

    if dataset_txt_path is None:

        validation_dataset = InferenceDataset(root_dir=dataset_path, transform=validation_transforms)

    else:
        validation_dataset = InferenceDataset(root_dir=dataset_path, transform=validation_transforms, txt_file=dataset_txt_path)

    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    return validation_loader 


def compute_throughput(model,validation_loader,device ,batch_size, resolution):

    torch.cuda.empty_cache()
    if not isinstance(device, torch.device):
        device = torch.device(device)
    warmup_iters = 50
   
    timing = []
    i = 0 
    
    torch.cuda.synchronize()
    with torch.no_grad():

        for image in tqdm(validation_loader, position=0, leave=False):
            image = image.to(device)
            image = image.repeat(32, 1, 1, 1)
            if i != warmup_iters:
                i = i+1 
                model(image)
                continue
            if i == warmup_iters:
                torch.cuda.synchronize()
            start = time.time()
            model(image.to(device))
            torch.cuda.synchronize()
            timing.append(time.time() - start)

    timing = torch.as_tensor(timing, dtype=torch.float32)
   
    return round(( (32*batch_size) / timing.mean()).item(), 2)


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
    dataset_path, dataset_txt_path = get_dataset_validaion_path(dataset,root_dir)
    
    for model_path in model_path_list:
        model, variant = load_model(model_path)
        input_size = variant['dataset_kwargs']['crop_size']
        normalization = variant["dataset_kwargs"]["normalization"]
        stats = STATS[normalization]

        if patch_type == "algm":
            algm.patch.algm_segmenter_patch(model,selected_layers,trace_source=True)
            model.encoder.window_size = merging_window_size
            model.encoder.threshold = threshold     

        model.eval()
        model.to(device)
    

    validaion_loader = dataset_prepare(dataset_path, dataset_txt_path, stats, batch_size, input_size)
    fps = compute_throughput(model, validaion_loader,device, batch_size, input_size)

    print('FPS:', fps, flush=True)

if __name__ == "__main__":
    main()