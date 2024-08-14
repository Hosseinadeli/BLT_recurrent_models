import os
import time
import pprint
from pathlib import Path

# import warnings
# warnings.simplefilter('ignore')

import torch
from torch import nn
import torchvision
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

# from models.cornet import get_cornet_model
# from models.blt import get_blt_model
from models.build_model import build_model_from_args
from models.blt_loss import BLTLoss
from engine import train_one_epoch, evaluate
from datasets.datasets import fetch_data_loaders
import utils 

# TODO 
# np.random.seed(0)
# torch.manual_seed(0)

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


def main(rank, world_size, args):
    # args are from run_model.py

    if args.distributed == 1:
        args.rank = rank
        args.world_size = world_size
        utils.init_distributed_mode(args)
    else:
        args.gpu = 0

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device", args.device)

    args.val_perf = 0

    train_loader, val_loader = fetch_data_loaders(args)
 
    model = build_model_from_args(args)
    model.to(args.device)

    if args.distributed == 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        
    criterion = BLTLoss(args)
    
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
            optimizer.load_state_dict(checkpoint['optimizer'])
        
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
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

        args.start_epoch = 0

    # only for one processs
    if args.gpu == 0: 
        if args.wandb_p:
            if args.wandb_p:
                os.environ['WANDB_MODE'] = 'online'
                
            if args.wandb_r:
                wandb_r = args.wandb_r 
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
            pprint.pprint(vars(args), f, sort_dicts=False)
        
        
        with open(os.path.join(args.save_dir, 'val_results.txt'), 'w') as f:
            f.write(f'validation results: \n') 

    print("Start training")
    start_time = time.time()
    
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     sampler_train.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, args.device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()


        val_ = evaluate(model, criterion, val_loader, args)
        if args.gpu == 0: 
            if args.wandb_p:
                wandb.log({"val_acc": val_['top1'], "val_loss": val_['loss']})
        val_perf = val_['top1']


        if args.output_path:
            # update best validation acc and save best model to output dir
            if val_perf > args.val_perf:
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
    from main_args import get_argparser
    args = get_argparser().parse_args()
    
    if args.output_path:
        out = Path(args.output_path).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        args.output_path = str(out)

    if args.distributed == 1:
        args.world_size = torch.cuda.device_count()
        mp.spawn(main, args=(args.world_size, args), nprocs=args.world_size)
    else:
        main(0, 1, args)