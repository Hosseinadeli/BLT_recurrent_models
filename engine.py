import math
import sys
from typing import Iterable
import torch
from tqdm import tqdm

import utils

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    model.train()
    criterion.train()
    print_freq = 100
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=print_freq, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=print_freq))  #, fmt='{value:.2f}'
    header = 'Epoch: [{}]'.format(epoch)
    

    for imgs, targets in metric_logger.log_every(data_loader, print_freq, header):
        
        imgs = imgs.cuda()
        targets = targets.cuda()

        outputs = model(imgs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()
        
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value)  #, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        #metric_logger.update(loss_labels=loss_value) #loss_dict_reduced['loss_recon']
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, args=None, lh_challenge_rois=None, rh_challenge_rois=None, return_all=False):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=100)) #, fmt='{value:.2f}'
    header = 'Test:'

    corr_all = 0
    total_imgs = 0

    record = {'top1': 0, 'top5': 0, 'loss': 0}
    
    for imgs, targets in metric_logger.log_every(data_loader, 100, header):

        imgs = imgs.cuda()
        targets = targets.cuda()
        labels = targets

        outputs = model(imgs)
    
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))
                
        #metric_logger.update(loss_labels=loss_dict_reduced['loss_labels'])

        record['loss'] += loss_dict_reduced['loss_labels']

        # pred_logits = outputs['pred_logits']

        #outputs = outputs[-1]
        p1, p5 = accuracy(outputs, labels, topk=(1, 5))
        record['top1'] += p1
        record['top5'] += p5

        P = outputs.topk(1,dim=1)[1].flatten(0)
        corr = 1* (labels == P)
        corr_all = corr_all + torch.sum(corr)
        total_imgs = total_imgs + len(labels)

    acc = corr_all / total_imgs
        
    for key in record:
        record[key] /= total_imgs
    
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    print(f'acc: {acc}')
    print('top 1 and top 5 accuracies', record)
    
    return record


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = [correct[:k].sum().item() for k in topk]
        return res
    