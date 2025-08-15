import os, argparse, time, glob, pickle, subprocess, shlex, io, pprint

import numpy as np
import pandas
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo
import torchvision
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

# from models.cornet import get_cornet_model
# from models.blt import get_blt_model
from models.build_model import build_model
from engine import train_one_epoch, evaluate
from datasets.datasets import fetch_data_loaders
import utils 
from pathlib import Path
import os

from PIL import Image
Image.warnings.simplefilter('ignore')

# TODO 
# np.random.seed(0)
# torch.manual_seed(62)

try:
    import wandb
    os.environ['WANDB_MODE'] = 'offline'
except ImportError as e:
    pass 

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

# TODO: add image_size to the fetch_data_loaders
# TODO: add the criterion for contrastive learning
# TODO: make evalute an actual option here where it takes a model and a dataset 
# and returns the accuracy and features for the dataset

def get_args_parser():
    parser = argparse.ArgumentParser(description='ImageNet Training', add_help=False)

    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--output_path', default='./results/', type=str,
                        help='path for storing ')
    
    parser.add_argument('--model', choices=['blt_b', 'blt_b_pm', 'blt_b_top2linear', 'blt_b2', 
                                            'blt_b3', 'blt_bl', 'blt_bl_top2linear', 'blt_b2l',
                                            'blt_b3l', 'blt_bt', 'blt_b2t', 'blt_b3t', 'blt_blt',
                                            'blt_b2lt', 'blt_b3lt', 'blt_bt2', 'blt_b2t2', 
                                            'blt_b3t2', 'blt_blt2', 'blt_b2lt2', 'blt_b3lt2',
                                            'blt_bt3', 'blt_b2t3', 'blt_b3t3', 'blt_blt3', 
                                            'blt_b2lt3', 'blt_b3lt3', 
                                            'cornet_z', 'cornet_s', 'cornet_r', 'cornet_rt',
                                            'resnet50'], 
                        default='blt_bl', type=str)


    parser.add_argument('--pool', choices=['max', 'average', 'blur'],
                        default='max', help='which pooling operation to use')
    
    parser.add_argument('--objective', choices=['classification', 'contrastive'],
                        default='classification', help='which model to train')

    parser.add_argument('--smooth_labels', default=1, type=int,
                        help='whether to smooth the lables for training')
    
    parser.add_argument('--loss_choice', choices=['weighted', 'decay'] 
                        , default='decay', type=str, 
                        help='how to apply loss to earlier readout layers')  
    parser.add_argument('--loss_gamma', default=0, type=float, 
                        help='whether to have loss in earlier steps of the readout')

    parser.add_argument('--recurrent_steps', default=10, type=int,
                        help='number of time steps to run the model for recurrent models')
    parser.add_argument('--num_layers', default=6, type=int,
                        help='number of layers')

    parser.add_argument('--num_workers', default=4, type=int,
                        help='number of data loading num_workers')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', default=.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay ')
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--clip_max_norm', default=0, type=float,
                        help='gradient clipping max norm') #0.1
    
    parser.add_argument('--evaluate', action='store_true', help='just evaluate')
    
    parser.add_argument('--wandb_p', default=None, type=str)
    parser.add_argument('--wandb_r', default=None, type=str)

    # dataset parameters
    parser.add_argument('--dataset', choices=['imagenet', 'vggface2', 'bfm_ids', 'NSD',
                                              'imagenet_vggface2', 'imagenet_face']
                        , default='imagenet', type=str)
    parser.add_argument('--data_path', default='/share/data/imagenet-pytorch',
                         type=str, help='path to ImageNet folder')
    parser.add_argument('--image_size', default=224, type=int, 
                        help='what size should the image be resized to?')
    parser.add_argument('--horizontal_flip', default=False, type=bool,
                    help='wether to use horizontal flip augmentation')
    parser.add_argument('--run', default='1', type=str) 
    
    parser.add_argument('--img_channels', default=3, type=int,
                    help="what should the image channels be (not what it is)?") #gray scale 1 / color 3
    
    parser.add_argument('--save_model', default=0, type=int) 

    parser.add_argument('--distributed', default=1, type=int,
                        help='whether to use distributed training')
    parser.add_argument('--port', default='12382', type=str,
                        help='MASTER_PORT for torch.distributed')

    return parser


class LabelSmoothLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss
    
class SetCriterion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.weight_dict = {'loss_labels': 1}
        self.loss_gamma = args.loss_gamma
        if args.smooth_labels:
            self.loss_func = LabelSmoothLoss() 
        else:
            self.loss_func = nn.CrossEntropyLoss()
        self.loss_choice = args.loss_choice

    def forward(self, outputs, targets):

        loss = 0
        weight_sum = 1
        if isinstance(outputs, list):
            loss = self.loss_func(outputs[-1], targets)

            if self.loss_gamma > 0:
                for s in range(1,len(outputs)):
                    if self.loss_choice == 'decay':
                        weight = self.loss_gamma**s
                    elif self.loss_choice == 'weighted':
                        weight = self.loss_gamma
                    weight_sum += weight
                    loss += weight * self.loss_func(outputs[-(s+1)], targets)

        else:
            loss = self.loss_func(outputs, targets)

        loss = loss / weight_sum
        losses = {'loss_labels': loss}
        return losses

def main(rank, world_size, args):

    if args.distributed == 1:
        args.rank = rank
        args.world_size = world_size
        utils.init_distributed_mode(args)

        args.batch_size = int(args.batch_size / args.world_size)
        args.num_workers = int((args.num_workers + args.world_size - 1) / args.world_size)
        print(f'batch_size:{args.batch_size};   workers:{args.num_workers}')
    else:
        args.gpu = 0

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)
    device = torch.device(args.device)

    args.val_perf = 0

    train_loader, sampler_train, val_loader = fetch_data_loaders(args)
 
    model = build_model(args)
    model = model.cuda() 

    model_ddp = model
    if args.distributed == 1:
        model_ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], 
                                                              find_unused_parameters=True)
        
    criterion = SetCriterion(args)
    
    if args.output_path:
        args.save_dir = args.output_path + f'{args.objective}/{args.dataset}/{args.model}/run_{args.run}'
        if (not os.path.exists(args.save_dir)) and (args.gpu == 0):
            os.makedirs(args.save_dir)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        pretrained_dict = checkpoint['model']
        model.load_state_dict(pretrained_dict)
        
        args.best_val_acc = vars(checkpoint['args'])['val_perf'] #checkpoint['val_acc'] #or read it from the   
        
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            
            train_params = checkpoint['train_params']
            param_dicts = [ { "params" : [ p for n , p in model.named_parameters() if n in train_params ]}, ] 

            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                          weight_decay=args.weight_decay)
            
            # optimizer = torch.optim.SGD(param_dicts, args.lr,
            #                     momentum=args.momentum,
            #                     weight_decay=args.weight_decay)
            optimizer.load_state_dict(checkpoint['optimizer'])
        
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)
            # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            
    else:
        
        param_dicts = [ 
            { "params" : [ p for n , p in model.named_parameters() if p.requires_grad]}, ]  #n not in frozen_params and 
    
        train_params = [ n for n , p in model.named_parameters() if p.requires_grad ]  # n not in frozen_params and

        print('\ntrain_params', train_params)

        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=0.5)

        # optimizer = torch.optim.SGD(param_dicts, args.lr,
        #                         momentum=args.momentum,
        #                         weight_decay=args.weight_decay)
        
        #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90], gamma=0.1)

        args.start_epoch = 0

    # only for one processs
    if args.gpu == 0: 
        if args.wandb_p:
            if args.wandb_p:
                os.environ['WANDB_MODE'] = 'online'
                
            if args.wandb_r:
                wandb_r = args.wandb_r 
            elif args.run:
                wandb_r = f'{args.model}_{args.run}'
            else:
                wandb_r = args.model 

            os.environ["WANDB__SERVICE_WAIT"] = "300"
            #        settings=wandb.Settings(_service_wait=300)
            wandb.init(
                # Set the project where this run will be logged
                # face vggface2 recon  # "face detr dino"
                project= args.wandb_p,   
                # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
                name=wandb_r,   #f"{wandb}"

                # Track hyperparameters and run metadata
                config={
                "learning_rate": args.lr,
                "architecture": f'{args.model}',
                "epochs": args.epochs,
                })

        with open(os.path.join(args.save_dir, 'params.txt'), 'w') as f:
            pprint.pprint(args.__dict__, f, sort_dicts=False)
        
        
        with open(os.path.join(args.save_dir, 'val_results.txt'), 'w') as f:
            f.write(f'validation results: \n') 

    print("Start training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model_ddp, criterion, train_loader, optimizer, args.device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()


        val_ = evaluate(model_ddp, criterion, val_loader, args)
        if args.gpu == 0: 
            if args.wandb_p:
                wandb.log({"val_acc": val_['top1'], "val_loss": val_['loss']})
        val_perf = val_['top1']


        if args.output_path:
            # update best validation acc and save best model to output dir
            if (val_perf > args.val_perf):  
                args.val_perf = val_perf                

                if args.gpu == 0: 
                    with open(os.path.join(args.save_dir, 'val_results.txt'), 'a') as f:
                            f.write(f'epoch {epoch}, val_perf: {val_perf} \n') 

                if args.save_model:
                    checkpoint_paths = [args.save_dir + '/checkpoint.pth']
                    # print('checkpoint_path:',  checkpoint_paths)
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
    #                         'train_params' : train_params,
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            'val_perf': args.val_perf
                        }, checkpoint_path)

    destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_path:
        Path(args.output_path).mkdir(parents=True, exist_ok=True)

    if args.distributed == 1:
        args.world_size = torch.cuda.device_count()
        mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size)
    else:
        main(0, 1, args)