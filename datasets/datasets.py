#!/usr/bin/env python

import collections
import os

import numpy as np
#import PIL.Image
from PIL import Image, ImageOps 
import scipy.io
import torch
from torch.utils import data
import torchvision
from tqdm import tqdm
import pandas as pd
import csv
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from .vggface2 import VGGFaces2

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])


def fetch_ImageNet(args, split='full', transform=None,
                 horizontal_flip=False, upper=None, return_datasets=False, num_cats=1000):
    
    data_path = args.data_path #'/share/data/imagenet-pytorch'

    if args.horizontal_flip:
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(data_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.GaussianBlur(kernel_size = 7, sigma=(0.01, 7.0)),
                # torchvision.transforms.v2.GaussianNoise(),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
    else:
        dataset_train = torchvision.datasets.ImageFolder(
            os.path.join(data_path, 'train'),
            torchvision.transforms.Compose([
                torchvision.transforms.RandomResizedCrop(224),
                torchvision.transforms.GaussianBlur(kernel_size = 7, sigma=(0.01, 7.0)),
                # torchvision.transforms.v2.GaussianNoise(),
                torchvision.transforms.ToTensor(),
                normalize,
            ]))
    
    dataset_val = torchvision.datasets.ImageFolder(
        os.path.join(data_path, 'val'),
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize,
        ]))
    

    if num_cats < 1000:
        idxs = [i for i in range(len(dataset_train)) if dataset_train.imgs[i][1] < num_cats]
        dataset_train = Subset(dataset_train, idxs)

        idxs = [i for i in range(len(dataset_val)) if dataset_val.imgs[i][1] < num_cats]
        dataset_val = Subset(dataset_val, idxs)

    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                batch_size=args.batch_size,
                                                shuffle=True,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    

    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                num_workers=args.num_workers,
                                                pin_memory=True)
    

    if return_datasets:
        return dataset_train, data_loader_train, dataset_val, data_loader_val
    
    return data_loader_train, data_loader_val
        

def fetch_data_loaders(args):
    
    if args.dataset == 'vggface2':
            # VGGface 2
            #save_dir = args.output_dir + 'face_class_all_' + str(args.task_arch) 

            kwargs = {} # {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
            args.num_classes = 3890
            dataset_train = VGGFaces2(args, split='train', num_cats=args.num_classes)            
            dataset_val = VGGFaces2(args, split='val', num_cats=args.num_classes)
        
    elif args.dataset == 'imagenet':

        #save_dir = args.output_dir + 'imagenet_class_8steps_' + str(args.task_arch) 
        args.num_classes = 1000
        dataset_train, _, dataset_val, _  = fetch_ImageNet(args, split='train', return_datasets=True)

    elif args.dataset == 'imagenet_vggface2':
    # half imagenet, half vggface2

        #save_dir = args.output_dir + f'{args.objective}_{args.task_arch}_{args.dataset}/run_{args.run}'
        args.num_classes = 500 + 1920  # 3890 -> 1282172   # 1920 -> 640102  # 4->1134

        kwargs = {} # {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
        
        vggface2_train_dataset = VGGFaces2(args, split='train', num_cats=1920, starting_cat_label=500)
        vggface2_val_dataset = VGGFaces2(args, split='val', num_cats=1920, starting_cat_label=500)
        
        imagenet_train_dataset, _, imagenet_val_dataset, _ = fetch_ImageNet(args, return_datasets=True, num_cats=500)
        
        dataset_train  = torch.utils.data.ConcatDataset([imagenet_train_dataset, vggface2_train_dataset])
        dataset_val  = torch.utils.data.ConcatDataset([imagenet_val_dataset, vggface2_val_dataset])

    elif args.dataset == 'imagenet_face':

        args.num_classes = 999 + 1

        kwargs = {} # {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
        
        vggface2_train_dataset = VGGFaces2(args, split='train', num_cats=4, starting_cat_label=999, just_one_cat=True)
        vggface2_val_dataset = VGGFaces2(args, split='val', num_cats=4, starting_cat_label=999, just_one_cat=True)
        
        imagenet_train_dataset, _, imagenet_val_dataset, _ = fetch_ImageNet(args, return_datasets=True, num_cats=999)
        
        dataset_train = torch.utils.data.ConcatDataset([imagenet_train_dataset, vggface2_train_dataset])
        dataset_val = torch.utils.data.ConcatDataset([imagenet_val_dataset, vggface2_val_dataset])


    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                num_workers=args.num_workers) #collate_fn=utils.collate_fn, 
    val_loader = torch.utils.data.DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, num_workers=args.num_workers) #, collate_fn=utils.collate_fn

    return train_loader, sampler_train, val_loader