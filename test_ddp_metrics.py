try:
    from slurmexec import *
    _HAS_SLURMEXEC = True
except ImportError:
    _HAS_SLURMEXEC = False


from main_args import get_argparser

if _HAS_SLURMEXEC and is_this_a_slurm_job():
    import os
    import logging
    
    import torch
    import torchmetrics

    from models.build_model import build_model_from_args
    from models.blt_loss import BLTLoss
    from datasets.datasets import fetch_data_loaders
    import trainer as tr
    import utils

    class LinearClassifier(torch.nn.Module):
        def __init__(self, input_dim, num_classes):
            super(LinearClassifier, self).__init__()
            # A single linear layer
            self.linear = torch.nn.Linear(input_dim, num_classes)

        def forward(self, x):
            # Pass input through the linear layer
            out = self.linear(x)
            return out


def main_trainer(rank, world_size, args):
    if world_size == 1:
        return

    args.rank = rank
    args.world_size = world_size
    utils.init_distributed_mode(args, hide_prints=False)

    level = logging.DEBUG
    logger = utils.get_logger(f"Rank:{rank}", level=level)

    host, port = utils.get_ddp_hostname()
    logger.info(f"Setup DDP on {host}:{port}")

    device = torch.device("cuda", rank)
    logger.info(f"Using device {device}")
    
    input_dim = 20
    num_classes = 4
    model = LinearClassifier(input_dim, num_classes)
    model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    criterion = torch.nn.CrossEntropyLoss()

    # Metrics
    metrics = {tr.TRAIN_DATASET: {}}
    metrics[tr.TRAIN_DATASET]["accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    metrics[tr.TRAIN_DATASET]["accuracy_top2"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=2).to(device)
    metrics[tr.TRAIN_DATASET]["accuracy_top3"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=3).to(device)


    if rank == 0:
        metrics[tr.VALID_DATASET] = {}
        # metrics[tr.VALID_DATASET]["accuracy"] = metrics[tr.TRAIN_DATASET]["accuracy"].clone()
        # metrics[tr.VALID_DATASET]["accuracy_top2"] = metrics[tr.TRAIN_DATASET]["accuracy_top2"].clone()
        # metrics[tr.VALID_DATASET]["accuracy_top3"] = metrics[tr.TRAIN_DATASET]["accuracy_top3"].clone()
        metrics[tr.VALID_DATASET]["accuracy"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, sync_on_compute=False).to(device)
        metrics[tr.VALID_DATASET]["accuracy_top2"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=2, sync_on_compute=False).to(device)
        metrics[tr.VALID_DATASET]["accuracy_top3"] = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes, top_k=3, sync_on_compute=False).to(device)

    
    for epoch in range(1):
        torch.distributed.barrier()
        logger.info(f"Starting epoch {epoch}")

        # Simulate one train batch
        for metric in metrics[tr.TRAIN_DATASET].values():
            metric.reset()

        model.train()
        inputs = torch.randn(32, input_dim, device=device)
        targets = torch.randint(0, num_classes, (32,), device=device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        logger.info(f"Epoch {epoch} loss: {loss.item()}")

        for metric in metrics[tr.TRAIN_DATASET].values():
            metric(outputs, targets)
        logger.info(f"Updated metrics for train dataset")
        logger.info(f'Computed train metrics: {", ".join(f"{metric_name}: {metric.compute()}" for metric_name, metric in metrics[tr.TRAIN_DATASET].items())}')

        # Simulate one validation batch
        if rank == 0:
            for metric in metrics[tr.VALID_DATASET].values():
                metric.reset()

            model.eval()
            with torch.no_grad():
                inputs = torch.randn(32, input_dim, device=device)
                targets = torch.randint(0, num_classes, (32,), device=device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                for metric in metrics[tr.VALID_DATASET].values():
                    metric(outputs, targets)
                logger.info(f"Updated metrics for valid dataset")
                
            logger.info(f'Computed validation metrics: {", ".join(f"{metric_name}: {metric.compute()}" for metric_name, metric in metrics[tr.VALID_DATASET].items())}')

    torch.distributed.destroy_process_group()

    logger.info("Main function complete")


if _HAS_SLURMEXEC:


    @slurm_job
    def main(args):
        args.world_size = torch.cuda.device_count()

        args.port = str(utils.get_open_port())

        # Set num_workers
        args.num_workers = 8

        # Run the main function
        if args.world_size > 1:
            # NCCL optimizations
            # https://github.com/Lightning-AI/pytorch-lightning/issues/7179
            os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"
            os.environ["NCCL_SOCKET_NTHREADS"] = "2"
            os.environ["NCCL_MIN_CHANNELS"] = "32"

            torch.multiprocessing.spawn(main_trainer, args=(args.world_size, args), nprocs=args.world_size, join=True)
        else:
            main_trainer(0, 1, args)

    if __name__ == "__main__":
        slurm_exec(
            func = main,
            argparser = get_argparser(),
            job_name = "TEST",
            pre_run_commands = [
                "conda activate ml",
            ],
            slurm_args = {
                # "--gres": "gpu:a40:4",
                "--gres": "gpu:2",
                # "--mem": "64G",
                "--cpus-per-task": 4,
                "--time": "00:10:00",
            }
        )