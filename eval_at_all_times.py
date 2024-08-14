from slurmexec import *

from socket import gethostname
if "Chases-MacBook-Pro" in gethostname():
    set_slurm_debug()

if is_this_a_slurm_job():
    from pathlib import Path
    import time
    from functools import partial

    import torch
    import torchmetrics

    import trainer as tr
    from utils import get_logger, format_seconds
    from datasets import datasets
    from models.blt import BLTNet
    from models.blt_loss import BLTLoss
    from cwk_analysis_helpers import load_model

def validate_model(
        save_dir,
        load_from: str,
        loader,
        logger,
        device,
        recurrent_t_min: int,
        recurrent_t_max: int,
        batch_print_idx: list[int],
    ):
    # Load the model
    logger.info(f"Loading model from {save_dir}")
    model, _, metadata = load_model(save_dir, load_from, return_metadata=True, logger=logger)
    model.to(device)
    assert isinstance(model, BLTNet), "Model must be a BLT model"  # TODO: allow other models?
    # criterion = BLTLoss(metadata["args"], return_dict=False)
    
    # logger.info(f"Using model {model.model_name}")
    
    criterion = torch.nn.CrossEntropyLoss()

    metrics_for_timesteps = {
        t: {}
        for t in range(recurrent_t_min, recurrent_t_max)
    }
    # metrics_for_timesteps[-1] = {}  # for benchmarking (essentially to ensure all timesteps return properly, which they do)

    # compute the entropy of a distribution
    def entropy(x, from_logits=True):
        if from_logits:
            x = torch.softmax(x, dim=-1)
        return -torch.sum(x * torch.log2(x), dim=-1)

    # add metrics for each timestep
    for metrics in metrics_for_timesteps.values():
        metrics["loss"] = tr.preprocess_metric()
        metrics["accuracy"] = tr.preprocess_metric(torchmetrics.Accuracy(task="multiclass", num_classes=1000))
        metrics["accuracy_top3"] = tr.preprocess_metric(torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=3))
        metrics["accuracy_top5"] = tr.preprocess_metric(torchmetrics.Accuracy(task="multiclass", num_classes=1000, top_k=5))
        metrics["entropy"] = tr.preprocess_metric(compute_fn=lambda outputs, labels: entropy(outputs, from_logits=True))

        for metric in metrics.values():
            metric.to(device)

    start_time = time.time()

    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            model.num_recurrent_steps = recurrent_t_max
            outputs = model(inputs)
            # ^ shape (num_t, batch_size, num_classes)
            # len(outputs) < num_recurrent_steps because it contains no None values

            t = recurrent_t_max-1  # the last output is t=recurrent_t_max-1
            for out in reversed(outputs):
                # compute loss and metrics; update metrics
                if t not in metrics_for_timesteps:
                    continue
                metrics = metrics_for_timesteps[t]
                loss = criterion(out, labels)
                tr.update_metrics(metrics, inputs, out, labels, loss)
                t = t-1  # decrement time step

            if batch_idx in batch_print_idx:
                logger.info(f" • {(batch_idx+1)/len(loader)*100:.0f}% complete  (batch {batch_idx+1:,}/{len(loader):,})")
    
    # Compute and save the metrics
    save_data = {
        "slurm_job_id": get_slurm_id(),
        "model_name": model.model_name,
        "model_path": str(save_dir / load_from),
        "valid_metrics_at_timesteps": {
            t: {
                metric_name: metric.compute().item()
                for metric_name, metric in metrics.items()
            }
            for t, metrics in metrics_for_timesteps.items()
        }
    }

    save_file = save_dir / "valid_metrics_at_timesteps.pt"
    torch.save(save_data, save_file)
    logger.info(f"Completed in {format_seconds(time.time() - start_time)}")
    logger.info(f"Saved validation metrics to {save_file}")

@slurm_job
def main(
    save_dir: str,
    load_from: str = "latest.pt",
    recurrent_t_min: int = 5,
    recurrent_t_max: int = 15,
    num_workers: int = 4,
    progress_print: float = 0.1,
    # dry_run: bool = False,
):
    logger = get_logger()
    logger.info(f"Starting job on slurm ID {get_slurm_id()}")

    # Load imagenet validation set
    # TODO: Customize dataset?
    valid_dataset = datasets.fetch_ImageNet(only_valid=True)
    loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)
    logger.info(f"Loaded dataset with {len(valid_dataset):,} samples")
    batch_print_idx = tr.get_batch_print_idx(len(loader), progress_print)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available; using CPU")
    
    # Parse directory
    save_dir = Path(save_dir).expanduser()

    # partial function to clear up kwargs
    validate_fn = partial(
        validate_model,
        load_from=load_from, loader=loader, logger=logger, device=device, batch_print_idx=batch_print_idx,
        recurrent_t_min=recurrent_t_min, recurrent_t_max=recurrent_t_max
    )

    if "*" in save_dir.name:  # e.g., ~/models/.../parent/blt*
        dir_format = save_dir.name  # e.g., "blt*"
        parent_save_dir = save_dir.parent  # e.g., ~/models/.../parent

        sub_dirs = [
            save_dir
            for save_dir in parent_save_dir.glob(dir_format + "/**")
            if (save_dir / load_from).exists()  # it is a model directory
        ]
        
        for i, save_dir in enumerate(sub_dirs):
            validate_fn(save_dir=save_dir)
            logger.info(f"Done with {i+1}/{len(sub_dirs)} models")
            logger.info("")
    else:
        validate_fn(save_dir=save_dir)

    logger.info("Main function complete")


if __name__ == "__main__":
    slurm_exec(
        main,
        job_name = "val_{recurrent_t_min}-{recurrent_t_max}",
        pre_run_commands = [
            "conda activate ml",
        ],
        slurm_args = {
            "--time": "01:00:00",
            "--mem": "32G",
            "--gres": "gpu:a40:1",
            "--cpus-per-task": 4,
        }
    )