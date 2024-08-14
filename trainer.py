import sys
import os
from pathlib import Path
import logging
import time
import datetime
from typing import List, Dict, Any, Optional, Type
import warnings
import random

import torch
from torch.utils.data import Dataset, DataLoader
import torchmetrics
from torchmetrics import Metric
import numpy as np

from utils import is_main_process, get_git_commit, format_seconds, get_ddp_hostname

# Keys for datasets
TRAIN_DATASET: str = "train"
VALID_DATASET: str = "valid"
TEST_DATASET: str = "test"

class SamplesSeenMetric(torchmetrics.Metric):
    """Counts the number of samples seen"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        self.total += preds.shape[0]

    def compute(self) -> torch.Tensor:
        return self.total


class SaveModel:
    def __init__(self, filename):
        self.filename = filename

    def __repr__(self) -> str:
        class_vars = ", ".join(f"{k}={v}" for k, v in self.__dict__.items() if not k.startswith("_"))
        return f"{self.__class__.__name__}({class_vars})"
    
    def should_save(self, trainer, latest_epoch_data):
        raise NotImplementedError("Subclasses must implement should_save")


class SaveModelAtEachEpoch(SaveModel):
    """Saves the model after each epoch"""
    def __init__(self, filename="epoch_{epoch}.pt"):
        super().__init__(filename)

    def should_save(self, trainer, latest_epoch_data):
        return True

    
class SaveModelAtLatest(SaveModelAtEachEpoch):
    """Saves the latest model"""
    def __init__(self, filename="latest.pt"):
        super().__init__(filename)


class SaveModelAtBest(SaveModel):
    """Saves the model when a certain metric is at its best value"""
    def __init__(self, filename, dataset, metric, higher_is_better=True):
        super().__init__(filename)
        self.dataset = dataset
        self.metric = metric
        self.higher_is_better = higher_is_better
        self._goal_fn = max if higher_is_better else min

    def should_save(self, trainer, latest_epoch_data):
        values = [data["metrics"][self.dataset][self.metric] for data in trainer.epoch_data]
        best_value = None if len(trainer.epoch_data) <= 1 else self._goal_fn(values[:-1])
        value = values[-1]
        return best_value is None or (value > best_value if self.higher_is_better else value < best_value)


def get_model_state(state, is_ddp=False):
    if is_ddp:
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[7:]] = v
            else:
                new_state[k] = v
        return new_state
    return state

def update_metrics(
        metrics: dict[str, Metric],
        inputs: torch.Tensor,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
        return_metrics: bool = False
    ):
    if return_metrics:
        batch_metrics = {}
    for metric_name, metric in metrics.items():
        # metric(...) updates and returns the metric value on the current batch
        # metric.update(...) just updates (doesn't return anything)
        # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html
        metric_update_fn = metric if return_metrics else metric.update
        has_compute_fn = hasattr(metric, "_compute_fn")

        if has_compute_fn or "loss" in metric_name:
            if has_compute_fn:
                value = metric._compute_fn(outputs, labels)
            else:
                # default to loss; maybe specify more here in the future
                value = loss
            
            if isinstance(metric, torchmetrics.MeanMetric):
                # Weight the loss by the number of samples in the batch
                batch_val = metric_update_fn(value, weight=inputs.shape[0])
            else:
                batch_val = metric_update_fn(value)
        else:
            batch_val = metric_update_fn(outputs, labels)

        if return_metrics:
            batch_metrics[metric_name] = batch_val.item()
    
    if return_metrics:
        return batch_metrics

def preprocess_metric(
        metric: Optional[Metric] = None,
        compute_fn: Optional[callable] = None,
    ) -> Metric:
    """Add a metric to a metrics dict.

    Args:
        metric (Optional[Metric], optional): Metric. Defaults to a new MeanMetric instance.
        compute_fn (Optional[callable], optional): Function mapping (outputs, labels) to metric value. Defaults to None.

    Returns:
        Metric: Preprocessed metric
    """
    if metric is None:
        metric = torchmetrics.MeanMetric()

    if compute_fn is not None:
        setattr(metric, "_compute_fn", compute_fn)

    return metric

def get_batch_print_idx(num_batches: int, progress_print: float = 0.2):
    if not progress_print:
        progress_print = 1
    batch_print_idx = [
        int(num_batches * progress_print * i) - 1
        for i in range(1, int(1/progress_print))
    ]
    return batch_print_idx

def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            logger: logging.Logger = None,
            seed: int = 0,
            device = None
        ):
        self.model: torch.nn.Module = model.to(device) if device is not None else model
        self.logger: logging.Logger = logging.getLogger() if logger is None else logger
        self.device = device
        self.data_loaders: Dict[str, DataLoader] = {}
        self.dataset_metrics: Dict[str, Dict[str, Metric]] = {}  # {dataset_name -> {metric_name -> metric}}
        self.seed: int = seed
        self._using_jit_tracing = False
        self._model_name = f"{model.__module__}.{model.__class__.__name__}"

        # Training stuff
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = None
        self.criterion: callable = None

        # For saving data
        self.save_dir: Path = None
        self.epoch_data: List[Dict[str, Any]] = []
        self._save_model_hooks: List[SaveModel] = []
        self._load_model = None
        self._extra_metadata = {}
        self._current_dataset = TRAIN_DATASET

    def set_optimizer(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer
        return self
    
    def set_lr_scheduler(self, scheduler: torch.optim.lr_scheduler.LRScheduler):
        self.scheduler = scheduler
        return self

    def set_loss_criterion(self, criterion: callable):
        self.criterion = criterion
        return self
    
    def set_batch_handler(self, batch_handler: callable):
        """Sets a custom batch handler, which should return model outputs and loss.

        Args:
            batch_handler (callable): Batch handler function. Takes in (inputs, targets) and returns (outputs, loss)

        Returns:
            self: self
        """
        self._batch_handler = batch_handler
        return self

    def get_datasets(self) -> list[str]:
        return list(self.data_loaders.keys())

    def add_dataset(self, dataset_name, dataset: Dataset, **kwargs):
        if "batch_size" not in kwargs:
            raise ValueError("batch_size must be provided in kwargs")
        dataloader = self._prepare_dataloader(dataset_name, dataset, **kwargs)
        if dataloader is not None:
            self.data_loaders[dataset_name] = dataloader

    def add_metric(self,
            dataset_name: str,
            metric_name: str,
            metric_type: Type[Metric] = torchmetrics.MeanMetric,
            compute_fn = None,
            **metric_kwargs
        ):
        if dataset_name not in self.data_loaders:
            raise ValueError(f"Dataset {dataset_name} not found in Trainer. Make sure to add the dataset before adding metrics.")

        if dataset_name not in self.dataset_metrics:
            # Add default metrics
            self.dataset_metrics[dataset_name] = {}
            self.add_metric(dataset_name, "loss")
            self.add_metric(dataset_name, "_samples_seen", SamplesSeenMetric)
        
        # if metric_name in self.dataset_metrics[dataset_name]:
        #     raise KeyError(f"Metric \"{metric_name}\" already exists in {dataset_name} dataset!")
        
        
        metric = metric_type(**metric_kwargs)
        metric = preprocess_metric(metric, compute_fn=compute_fn)
        metric.to(self.device)
        self.dataset_metrics[dataset_name][metric_name] = metric

        return self

    def reset_metrics(self, dataset_name: Optional[str] = None):
        """Reset and prepare metrics for running.

        Args:
            dataset_name (Optional[str], optional): _description_. Defaults to None.
        """
        # https://lightning.ai/docs/torchmetrics/stable/pages/overview.html
        if dataset_name is None:
            dataset_name = self._current_dataset
        
        if dataset_name not in self.dataset_metrics:
            raise KeyError(f"Metrics not loaded for {dataset_name}. Make sure datasets are added to Trainer before metrics.")

        for metric in self.dataset_metrics[dataset_name].values():
            metric.reset()

    def compute_metrics(self, dataset_name: Optional[str] = None):
        if dataset_name is None:
            dataset_name = self._current_dataset
        if dataset_name not in self.dataset_metrics:
            return None
        return {
            metric_name: metric.compute().item()
            for metric_name, metric
            in self.dataset_metrics[dataset_name].items()
        }
    
    def get_metrics_str(self, dataset_name: Optional[str] = None, metrics: Optional[dict[str, any]] = None, show_internal: bool = True):
        if metrics is None:
            metrics = self.compute_metrics(dataset_name=dataset_name)

        metrics_formatted = []
        internal_metrics_formatted = []

        for metric_name, value in metrics.items():
            is_internal = metric_name.startswith("_")
            if is_internal: metric_name = metric_name[1:]
            s = f"{metric_name}={value:.6f}" if isinstance(value, float) else f"{metric_name}={value}"
            (internal_metrics_formatted if is_internal else metrics_formatted).append(s)
        
        metrics_str = ", ".join(metrics_formatted)

        if show_internal and len(internal_metrics_formatted) > 0:
            metrics_str += " (" + ", ".join(internal_metrics_formatted) + ")"

        return metrics_str


    def to(self, device):
        self.device = device
        self.model.to(device)
        for metrics in self.dataset_metrics.values():
            for metric in metrics.values():
                metric.to(device)
        return self
    

    def set_save_dir(self, save_dir, force_reset=False):
        self.save_dir = Path(save_dir).expanduser()

        if force_reset:
            if self.save_dir.exists() and is_main_process():
                self.logger.warning(f"Force resetting save directory {self.save_dir}")
                from shutil import rmtree
                rmtree(self.save_dir)

            # Make sure directory is deleted before continuing
            self._sync_processes()

        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        epoch_data_path = self.save_dir / "epoch_data.pt"
        if epoch_data_path.exists():
            self.epoch_data = torch.load(epoch_data_path)
            self.logger.info(f"Loaded epoch data from {epoch_data_path}. Model is trained for {self.n_trained_epochs} epochs")


    def add_save_model_hook(self, save_model: SaveModel, load_saved=False):
        if self.save_dir is None:
            raise ValueError("Cannot add save model hook without first calling `set_save_dir`")
        self._save_model_hooks.append(save_model)

        if load_saved and len(self.epoch_data) > 0:
            filename = save_model.filename.format(epoch=self.n_trained_epochs)
            self.load_saved_model(filename)

        return self


    def load_saved_model(self, file=None):
        if isinstance(file, str):
            file = self.save_dir / file

        if file.exists():
            self.logger.info(f"Loading model from {file}")
            checkpoint = torch.load(file)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optim_state"])
            if self.scheduler is not None and checkpoint["scheduler_state"] is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.logger.info(f"Loaded model from epoch {checkpoint['epoch']}")
        else:
            self.logger.error(f"Failed to load model from nonexistent file: {file}")


    def use_jit_tracing(self, input_shape=None, batch_size=1):
        if input_shape is None:
            # Use the first dataset's input shape
            input_shape = self.data_loaders[TRAIN_DATASET].dataset[0][0].shape
        
        # If given batch_size, prepend it to input shape
        if batch_size is not None:
            # Generally fine to use a batch size of 1 for tracing, so I don't think this is needed:
            # batch_size = max(data_loader.batch_size for data_loader in self.data_loaders.values())
            input_shape = (batch_size,) + input_shape
        
        start_time = time.time()
        self.model = torch.jit.trace(self.model, torch.randn(input_shape, device=self.device))
        self.logger.info(f"Using PyTorch JIT tracing (tracing with input shape {input_shape}; done in {time.time()-start_time:.3f}s)")
        self._using_jit_tracing = True

    def set_max_grad_norm(self, max_grad_norm: float):
        self._max_grad_norm = max_grad_norm

    def _get_wandb_config(self):
        return {
            "model_class": self.model.__class__.__name__,
            "optimizer_class": self.optimizer.__class__.__name__,
            "criterion_class": self.criterion.__class__.__name__,
            "trainer_class": self.__class__.__name__,
        }

    def wandb_init(self, project=None, id=None, config=None, **kwargs):
        if not is_main_process():
            # Only master process should run wandb
            return
        
        self.logger.info("Initializing wandb run...")
        import wandb

        wandb_config = self._get_wandb_config()
        if config is not None:
            wandb_config.update(config)

        # Increase wandb wait time to 5m (default 30s)
        os.environ["WANDB__SERVICE_WAIT"] = "300"
        
        self._wandb_run = wandb.init(
            project = project,
            id = id,
            config = wandb_config,
            **kwargs
        )

    def get_metadata(self):
        return {
            "model": {
                "name": self._model_name
            },
            "optimizer": {
                "name": f"{self.optimizer.__module__}.{self.optimizer.__class__.__name__}",
                "defaults": self.optimizer.defaults,
            },
            "scheduler": {
                "name": f"{self.scheduler.__module__}.{self.scheduler.__class__.__name__}",
                "state_dict": {
                    k: v for k, v in self.scheduler.state_dict().items()
                    if not k.startswith("_")
                }
            } if self.scheduler is not None else None,
            "criterion": self.criterion.__class__.__name__ if isinstance(self.criterion, torch.nn.Module) else self.criterion.__name__,
            "max_grad_norm": self._max_grad_norm if hasattr(self, "_max_grad_norm") else None,
            "device": str(self.device),
            "seed": self.seed,
            "using_jit_tracing": self._using_jit_tracing,
            "metrics": {
                # TODO: Allow for different train/valid metrics?
                k: f"{v.__class__.__name__}"
                for k, v in self.dataset_metrics[TRAIN_DATASET].items()
            },
            "save_dir": str(self.save_dir),
            "datasets": {
                dataset_name: {
                    "len": len(data_loader.dataset),
                    "batch_size": data_loader.batch_size,
                    "num_workers": data_loader.num_workers,
                }
                for dataset_name, data_loader in self.data_loaders.items()
            },
            "slurm": {
                # See https://slurm.schedmd.com/sbatch.html#SECTION_OUTPUT-ENVIRONMENT-VARIABLES
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "job_name": os.environ.get("SLURM_JOB_NAME"),
                "node": os.environ.get("SLURM_JOB_NODELIST"),
                "cluster": os.environ.get("SLURM_CLUSTER_NAME"),
                "_env_vars": {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith("SLURM_")
                }
            },
            "runtime": {
                "python_executable": sys.executable,
                "cmdline_filename": sys.argv[0],
                "cmdline_args": " ".join(sys.argv[1:]),
                "ncpus": int(os.environ.get("SLURM_JOB_CPUS_PER_NODE") or os.cpu_count()),
                "ngpus": torch.cuda.device_count(),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "torch_cudnn_version": torch.backends.cudnn.version(),
                "torch_cuda_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                "git_commit": get_git_commit(),
            },
            **self._extra_metadata
        }
    
    def set_metadata(self, **metadata):
        self._extra_metadata.update(metadata)

    @property
    def n_trained_epochs(self):
        if len(self.epoch_data) == 0:
            return 0
        return max(stats["epoch"] for stats in self.epoch_data)

    @property
    def lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def _prepare_dataloader(self, dataset_name: str, dataset: Dataset, **kwargs):
        if "batch_size" not in kwargs:
            raise ValueError("batch_size must be provided in kwargs")
        if "pin_memory" not in kwargs:
            kwargs["pin_memory"] = True

        return DataLoader(
            dataset,
            worker_init_fn = _worker_init_fn,
            **kwargs
            # batch_size = kwargs["batch_size"],
            # shuffle = kwargs.get("shuffle", False),
        )
    
    def _barrier(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def process_batch(self, inputs, labels, should_optim=True, should_update_metrics=True, return_metrics=False, **model_forward_kwargs):
        """Processes a batch of inputs and labels.

        Args:
            inputs (torch.Tensor): Inputs
            labels (torch.Tensor): Labels
            should_optim (bool, optional): Whether to optimize the model. Defaults to True.
            should_update_metrics (bool, optional): Whether to update metrics. Defaults to True.
            return_metrics (bool, optional): Whether to return metrics. Defaults to False.
            model_forward_kwargs: Passed to model.forward

        Returns:
            tuple: model_outputs, loss if return_metrics is False; else model_outputs, loss, metrics
        """
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)

        if should_optim:
            self.optimizer.zero_grad()
        
        # Compute model outputs and loss
        if hasattr(self, "_batch_handler"):
            outputs, loss = self._batch_handler(inputs, labels, **model_forward_kwargs)
        else:
            # Default batch handling
            outputs = self.model(inputs, **model_forward_kwargs)
            loss = self.criterion(outputs, labels)

        if should_optim:
            loss.backward()
            if hasattr(self, "_max_grad_norm"):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self._max_grad_norm)
            self.optimizer.step()

        # Update metrics
        if should_update_metrics:
            metrics = update_metrics(self.dataset_metrics[self._current_dataset], inputs, outputs, labels, loss, return_metrics=return_metrics)
            self.logger.debug(f"Updated metrics on {self._current_dataset} dataset")

        if should_update_metrics and return_metrics:
            return outputs, loss, metrics

        return outputs, loss

    def train_epoch(self, epoch: int, progress_print: float = 0.2, dry_run: bool = False, train_dataset_name: str = TRAIN_DATASET, valid_dataset_name: str = VALID_DATASET, **model_forward_kwargs):
        train_loader = self.data_loaders[train_dataset_name]
        is_main = is_main_process()
        
        # Compute the batch indices at which to print
        if not progress_print:
            progress_print = 1
        num_batches = len(train_loader)
        batch_print_idx = [
            int(len(train_loader) * progress_print * i) - 1
            for i in range(1, int(1/progress_print))
        ]
        if not is_main:
            # Only batch print on main process
            batch_print_idx = []

        # Shorthand for logging
        def log(msg, level=logging.INFO):
            # self.logger.log(level, f"[Train Epoch {epoch}] {msg}")
            self.logger.log(level, msg)

        self._current_dataset = train_dataset_name
        self._barrier()  # ensure all workers are ready
        log(f">>> EPOCH {epoch} START >>>")
        train_time = time.time()
        self.reset_metrics(train_dataset_name)  # reset all metrics

        log(f"Beginning training...   (lr={self.lr:.2e})")
        self.model.train()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            outputs, loss = self.process_batch(inputs, labels, should_optim=True, should_update_metrics=True, **model_forward_kwargs)
            log_batch = batch_idx in batch_print_idx

            if log_batch:
                elapsed_sec = time.time() - train_time
                elapsed = format_seconds(elapsed_sec, ms=False)
                frac_done = (batch_idx+1) / num_batches
                ms_per_batch = elapsed_sec / (batch_idx+1) * 1000
                remaining = format_seconds(elapsed_sec/frac_done - elapsed_sec, ms=False)
                log(f" • {frac_done*100:.0f}% complete  (batch {batch_idx+1:,}/{num_batches:,}, {elapsed} elapsed, {ms_per_batch:.0f}ms/batch, ~{remaining} remaining)  [batch loss: {loss.item():.6f}]")

            if dry_run:
                log("    Dry run (dry_run=True)! Exiting after one batch.", level=logging.WARNING)
                break
        
        self._barrier()  # ensure all workers end training
        self.logger.debug("barrier after train loop")

        # Compute metrics
        train_metrics = self.compute_metrics(train_dataset_name)
        dataset_metrics_computed = {
            train_dataset_name: train_metrics
        }
        train_time = time.time() - train_time
        log(f"Training epoch completed in {format_seconds(train_time)}. Computing validation metrics...")

        # Compute validation metrics if a validation set is given
        should_validate = valid_dataset_name is not None

        self._barrier()
        self.logger.debug("barrier before evaluation")

        valid_time = time.time()
        valid_metrics = self.run_evaluation(valid_dataset_name, dry_run=dry_run, **model_forward_kwargs)

        # Ensure run_validation is complete before continuing
        self._barrier()
        self.logger.debug("Processes synced after validation")
        
        # Broadcast valid_metrics to all processes (not really necessary but just in case)
        # if torch.distributed.is_available() and torch.distributed.is_initialized():
        #     self.logger.debug("broadcasting valid_metrics")
        #     self._barrier()
        #     torch.distributed.broadcast_object_list([valid_metrics], src=0)
        #     self.logger.debug("broadcast done")
        self.logger.debug(f"{valid_metrics = }")

        if should_validate:
            valid_time = time.time() - valid_time
            dataset_metrics_computed[valid_dataset_name] = valid_metrics
            log(f"Validation metrics computed in {format_seconds(valid_time)}")
        
        log("")
        log(f"Stats after epoch {epoch}:")
        log(f" • Train: {self.get_metrics_str(metrics=train_metrics)}")
        if should_validate:
            log(f" • Validation: {self.get_metrics_str(metrics=valid_metrics)}")
        log("")

        # Save model
        data = {
            "epoch": epoch,
            "metrics": dataset_metrics_computed,
            "completion_time": str(datetime.datetime.now()),
            "train_loop_time": train_time,
            "valid_loop_time": valid_time if should_validate else 0,
            "lr": self.lr,
        }

        self.epoch_data.append(data)

        if is_main and not dry_run:  # only main process should save/log
            if self.save_dir is not None:
                # Save epoch data
                torch.save(self.epoch_data, self.save_dir / "epoch_data.pt")

                # Save model if necessary
                print_saved_header = False
                for save in self._save_model_hooks:
                    if save.should_save(self, data):
                        save_checkpoint = {
                            **data,
                            "save": str(save),
                            "model_state": self.model.state_dict(),
                            "optim_state": self.optimizer.state_dict(),
                            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
                        }
                        filename = self.save_dir / save.filename.format(epoch=epoch)
                        torch.save(save_checkpoint, filename)

                        if not print_saved_header:
                            log("Saving model checkpoint to:")
                            print_saved_header = True
                        log(f" • {filename}   (from {save})")

                if print_saved_header:
                    log("")

            if hasattr(self, "_wandb_run"):
                self._wandb_run.log(data)

        # Step the scheduler if given
        if self.scheduler is not None:
            self.scheduler.step()

        self._barrier()
        log(f"<<< EPOCH {epoch} END <<<")
    
    def train(self, n_epochs: int, **kwargs):
        # Save the metadata
        if is_main_process():
            if self.n_trained_epochs == 0:
                metadata = self.get_metadata()
                torch.save(metadata, self.save_dir / "metadata.pt")

        if n_epochs <= self.n_trained_epochs:
            self.logger.warning(f"Given {n_epochs=}, but model is already trained for {self.n_trained_epochs} epochs. Not training further.")
        
        for epoch in range(self.n_trained_epochs+1, n_epochs+1):
            self.train_epoch(epoch, **kwargs)

            if kwargs.get("dry_run", False):
                self.logger.warning("Dry run (dry_run=True)! Exiting after training one epoch.")
                break


    def run_evaluation(self, dataset_name: str = VALID_DATASET, dry_run: bool = False, **model_forward_kwargs):
        self.logger.debug(f"run_evaluation({dataset_name=})")

        if dataset_name not in self.data_loaders:
            self.logger.debug(f"run_evaluation - skipping bc no dataset")
            return None

        # Setup for evaluation
        self._current_dataset = dataset_name
        self.model.eval()
        self.reset_metrics(dataset_name)
        self.logger.info("run_evaluation.reset_metrics done")

        # Process all data in the set
        val_loader = self.data_loaders[dataset_name]
        # self.logger.debug("run_evaluation - starting")
        with torch.no_grad():
            for batch_i, (inputs, labels) in enumerate(val_loader):
                self.process_batch(inputs, labels, should_optim=False, should_update_metrics=True, **model_forward_kwargs)
                if batch_i == 0 or batch_i % 50 == 0:
                    self.logger.debug(f"run_evaluation.batch {batch_i} done")
                if dry_run:
                    self.logger.warning("Evaluation dry run (dry_run=True)! Exiting after one batch.")
                    break

        self.logger.debug("run_evaluation - computing metrics")
        valid_metrics = self.compute_metrics(dataset_name)
        self.logger.debug("run_evaluation - done")
        return valid_metrics




class DDPTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, rank: int, world_size: int, **kwargs):
        if "device" in kwargs: del kwargs["device"]  # ignore any device passed in kwargs
        device = torch.device("cuda", rank)
        model_name = f"{model.__module__}.{model.__class__.__name__}"
        model = torch.nn.parallel.DistributedDataParallel(model.to(device), device_ids=[rank], find_unused_parameters=kwargs.get("find_unused_parameters", False))
        super().__init__(model, device=device, **kwargs)
        self._model_name = model_name
        self.rank = rank
        self.world_size = world_size
        self.evaluate_on_rank_0 = kwargs.get("evaluate_on_rank_0", True)

        # Extra metadata under "ddp" key
        ddp_addr, ddp_port = get_ddp_hostname()
        self.set_metadata(ddp = {
            "world_size": world_size,
            "backend": torch.distributed.get_backend(),
            "addr": ddp_addr,
            "port": ddp_port,
        })

    def add_metric(self,
            dataset_name: str,
            metric_name: str,
            metric_type: Type[Metric] = torchmetrics.MeanMetric,
            compute_fn = None,
            **metric_kwargs
        ):
        extra_kwargs = {}
        if self.evaluate_on_rank_0 and dataset_name != TRAIN_DATASET:
            # Turning sync_on_compute off will prevent torchmetrics from trying to sync when
            # metric.compute() is called (this would otherwise freeze the process)
            extra_kwargs["sync_on_compute"] = False
        return super().add_metric(
            dataset_name, metric_name, metric_type, compute_fn,
            **metric_kwargs, **extra_kwargs
        )


    def _prepare_dataloader(self, dataset_name: str, dataset: Dataset, **kwargs):
        # Validation done on main node
        if self.evaluate_on_rank_0 and dataset_name != TRAIN_DATASET and not is_main_process():
            return None

        # Only use DistributedSampler on train data
        self.logger.debug(f"Using DistributedSampler on data (world_size={self.world_size}, rank={self.rank})")
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas = self.world_size,  # these two params are inferred if not passed
            rank = self.rank,
            shuffle = kwargs.get("shuffle", False),
            drop_last = True,
            seed = self.seed,
        )
        kwargs["shuffle"] = False  # handled in sampler above
        
        # return super()._prepare_dataloader(dataset_name, dataset, **kwargs)
        return DataLoader(
            dataset,
            batch_size = kwargs["batch_size"],
            shuffle = kwargs["shuffle"],
            sampler = sampler,
            pin_memory = kwargs.get("pin_memory", True),
            num_workers = kwargs.get("num_workers", 4),
            worker_init_fn = _worker_init_fn,
            persistent_workers = True,  # supposedly speeds up
        )
    
    def _get_wandb_config(self, **kwargs):
        return {
            **super()._get_wandb_config(**kwargs),
            "ddp": True,
            "world_size": self.world_size,
            "rank": self.rank,
        }





def plot_metric(data, y, x="epoch", dataset=TRAIN_DATASET, ax=None, yscalefunc=None, **kwargs):
    """Plots a metric.

    Args:
        data (list or Trainer): Epoch data from Trainer.
        y (str or tuple): Tuple of (dataset, metric), or metric (and dataset given), or "epoch"
        x (str or tuple, optional): Tuple of (dataset, metric), or metric (and dataset given),  or "epoch". Defaults to "epoch".
        dataset (str, optional): Dataset used for metrics. Defaults to TRAIN_DATASET.
        ax: Matplotlib axis. Defaults to None, in which case a new figure will be created.
        yscalefunc (callable, optional): Function to apply to the y-axis. Defaults to None.
        **kwargs: passed to ax.plot

    Returns:
        tuple: (matplotlib figure, matplotlib axis)
    """
    if isinstance(data, Trainer):
        data = data.epoch_data

    if ax is None:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
    
    x_plot, y_plot = [], []

    x_dataset, x_metric = dataset, (x if isinstance(x, str) else x)
    y_dataset, y_metric = dataset, (y if isinstance(y, str) else y)

    def _getval(d, dataset, key, scalefunc=None):
        if key == "epoch":
            return d["epoch"]
        val = d["metrics"][dataset][key]
        if scalefunc is not None:
            val = scalefunc(val)
        return val

    for d in data:
        x_plot.append(_getval(d, x_dataset, x_metric))
        y_plot.append(_getval(d, y_dataset, y_metric, yscalefunc))

    ax.plot(x_plot, y_plot, **kwargs)
    if x == "epoch":
        ax.set_xlim(-0.5, x_plot[-1]+0.5)
        # ax.set_xticks(range(1, x_plot[-1]+1))
    ax.set_xlabel(x_metric.capitalize())
    ax.set_ylabel(y_metric.capitalize())
    return ax.get_figure(), ax