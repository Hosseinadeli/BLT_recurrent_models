from slurmexec import *
from main_args import get_argparser

if is_this_a_slurm_job():
    import os
    from pathlib import Path
    import logging
    
    import torch
    import torchmetrics

    from models.build_model import build_model_from_args
    from models.blt_loss import BLTLoss
    from datasets.datasets import fetch_data_loaders
    import trainer as tr
    import utils


def main_trainer(rank, world_size, args):
    is_ddp = world_size > 1
    # args.distributed = args.distributed == 1

    if is_ddp:
        args.rank = rank
        args.world_size = world_size
        utils.init_distributed_mode(args, hide_prints=False)
    else:
        args.gpu = 0

    level = logging.WARN if is_ddp and rank != 0 else logging.INFO
    if args.debug:
        level = logging.DEBUG
    logger = utils.get_logger(f"Rank:{rank}" if is_ddp else None, level=level)

    if is_ddp:
        host, port = utils.get_ddp_hostname()
        logger.info(f"Setup DDP on {host}:{port}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device}")

    # Load the dataset
    # Note this must happen before build_model because this specifies num_classes
    logger.info(f"Loading {args.dataset} dataset...")
    train_dataset, val_dataset = fetch_data_loaders(args, return_loaders=False)
    logger.info(f"Done loading dataset. There are {len(train_dataset):,} training samples and {len(val_dataset):,} validation samples")

    # Build the model
    model = build_model_from_args(args, verbose=False)
    logger.info(f"Model {args.model} has {args.num_layers} layers and {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Build the trainer
    trainer_kwargs = dict(logger=logger, seed=args.seed)
    if is_ddp:
        trainer = tr.DDPTrainer(model, rank, world_size, **trainer_kwargs)
        logger.info(f"Using DDP multi-GPU training ({world_size=}); this process is rank {rank}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(rank)}")

        # Scale learning rate by number of GPUs
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/3706#discussioncomment-238302
        # https://github.com/Lightning-AI/pytorch-lightning/discussions/3706#discussioncomment-3960433
        # With distributed training, the effective batch size is equal to batch_size * num_gpus
        # Want to scale LR so sample variance of gradients is approximately constant
        # Since DDP averages gradients across GPUs, we need to scale the LR by effective batch size:
        #    = batch_size * num_accumulated_batches * num_gpus * num_nodes
        # Specifically gradient is g = 1/N * sum(g_i)
        # So Var(g) = 1/N^2 * sum(Var(g_i)) = 1/N^2 * N * Var(g_i) = Var(g_i) / N = sigma^2 / N
        # So if we scale g_i by sqrt(N) then Var(g_i) = N*sigma^2/N = sigma^2
        # if args.lr_ddp_scale:
        #     from math import sqrt
        #     args.lr *= sqrt(world_size)
    else:
        trainer = tr.Trainer(model, device=device, **trainer_kwargs)

    # Setup wandb
    trainer.set_metadata(args=vars(args))
    if args.wandb_p and not args.debug:
        # Note this method will only run on main process
        trainer.wandb_init(
            project = args.wandb_p,
            name = args.wandb_r,

            config = {
                "learning_rate": args.lr,
                "architecture": f"{args.model}",
                "epochs": args.epochs,
                "run": args.run,
            }
        )

    # Add datasets
    trainer.add_dataset(tr.TRAIN_DATASET, train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.num_workers)
    trainer.add_dataset(tr.VALID_DATASET, val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

    # Setup optimizer
    if args.optimizer == "sgd":
        trainer.set_optimizer(torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.lr_momentum, weight_decay=args.weight_decay))
    elif args.optimizer == "adam":
        trainer.set_optimizer(torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay))
    elif args.optimizer == "adamw":
        trainer.set_optimizer(torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay))
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Setup learning rate scheduler
    trainer.set_lr_scheduler(torch.optim.lr_scheduler.StepLR(
        optimizer = trainer.optimizer,
        step_size = args.lr_drop,
        gamma = args.lr_step_gamma))

    # Setup loss
    trainer.set_loss_criterion(BLTLoss(args, return_dict=False))
    # Note we need a custom "batch_handler" because the BLT model outputs a list of outputs
    def batch_handler(inputs, labels):
        outputs = trainer.model(inputs)
        loss = trainer.criterion(outputs, labels)
        outputs = outputs[-1]  # Only return the output for the final time step
        return outputs, loss
    trainer.set_batch_handler(batch_handler)
    trainer.set_max_grad_norm(args.clip_max_norm)

    # Metrics
    for dataset_name in trainer.get_datasets():
        trainer.add_metric(dataset_name, "accuracy", torchmetrics.Accuracy, task="multiclass", num_classes=args.num_classes)
        trainer.add_metric(dataset_name, "accuracy_top3", torchmetrics.Accuracy, task="multiclass", num_classes=args.num_classes, top_k=3)
        trainer.add_metric(dataset_name, "accuracy_top5", torchmetrics.Accuracy, task="multiclass", num_classes=args.num_classes, top_k=5)
        logger.debug(f"Added metrics for {dataset_name}")

    # Saving
    if args.save_dir:
        logger.info(f"Using save directory {args.save_dir}")
        trainer.set_save_dir(args.save_dir)  # , force_reset=args.force_reset
        trainer.add_save_model_hook(tr.SaveModelAtLatest(), load_saved=True)
        # trainer.add_save_model_hook(tr.SaveModelAtBest(filename="best_train_acc.pt", dataset=tr.TRAIN_DATASET, metric="accuracy", higher_is_better=True))
        trainer.add_save_model_hook(tr.SaveModelAtBest(filename="best_val_acc.pt", dataset=tr.VALID_DATASET, metric="accuracy", higher_is_better=True))
        # trainer.add_save_model_hook(tr.SaveModelAtEachEpoch())

    trainer.train(args.epochs, progress_print=0.1, dry_run=args.debug)
    
    if is_ddp:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

    logger.info("Main function complete")


@slurm_job
def main(args):
    args.world_size = torch.cuda.device_count()

    # This will run Hossein's code
    # from main import main
    # if args.distributed == 1:
    #     torch.multiprocessing.spawn(main, args=(args.world_size, args), nprocs=args.world_size)
    # else:
    #     main(0, 1, args)

    # Format variables
    args.run = str(args.run).format(**vars(args))
    if args.wandb_r:
        args.wandb_r = args.wandb_r.format(**vars(args))

    if args.output_path:
        args.save_dir = Path(args.output_path) / args.objective / args.dataset / args.model / f"run_{args.run}"

    args.port = "auto"  # force auto to make it simple
    if args.port == "auto":
        args.port = str(utils.get_open_port())

    # Set num_workers
    args.num_workers = 4

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
        job_name = "blttrain",
        pre_run_commands = [
            "conda activate ml",
        ],
        slurm_args = {
            # "--gres": "gpu:a40:4",
            "--gres": "gpu:3",
            # "--mem": "64G",
            "--cpus-per-task": 24,
            "--time": "2-00:00:00",
        }
    )