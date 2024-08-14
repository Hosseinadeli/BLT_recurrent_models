from pathlib import Path
from typing import Optional, Union
from collections import OrderedDict
import warnings

import numpy as np
import torch
import torchvision
import sklearn
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA, NMF
import rsatoolbox

from models.blt import BLTNet
from models.build_model import build_model_from_args
import trainer as tr


def print_feature_shapes(
        layer_features: dict[int, list[Union[None, torch.Tensor]]],
        last_time_only: bool = False
    ):
    for layer, features in layer_features.items():
        print(f"Layer {layer}:")
        if features is not None:
            for t, f in enumerate(features):
                if not last_time_only or t == len(layer_features)-1:
                    print(f" - t={t}: {tuple(f.shape) if f is not None else None}")
        else:
            print(" - None")


class BLTModelAnalyzer:
    def __init__(self, model_dir: Path, device=None):
        """Initializes a new BLT model analyzer from a given save directory.

        Args:
            model_dir (Path): Model save directory
            device (device): Device to use
        """
        self.model_dir = model_dir
        self.model, self.epoch_data, self.metadata = load_model(model_dir, all_epoch_data=True, return_metadata=True)
        self.device = device
        self.model.to(device)

        self.model_name = self.metadata["args"]["model"][4:].upper()
        self.run_name = self.metadata["args"]["run"]
        self.name = f"{self.model_name}/{self.run_name}"

        # Setup analysis dir
        self.analysis_dir = model_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

    def set_sample_data(self, samples: torch.Tensor, sample_labels: torch.Tensor, sample_label_names: dict[int, str] = None):
        self.samples = samples
        self.sample_labels = sample_labels
        self.sample_label_names = sample_label_names

    def load_features(self, num_timesteps: int = 15, recompute: bool = False, filename: str = "features.pt"):
        features_file = self.analysis_dir / filename

        if recompute or not features_file.exists():
            layer_indices = list(range(self.model.num_layers))
            layer_features, outputs = extract_layer_features(self.model, self.samples, layer_indices=layer_indices, num_timesteps=num_timesteps)
            # Note: ^ features and outputs are on CPU
            del outputs  # I don't think we need this right now
            torch.save({
                "layer_features": layer_features,
                # "outputs": outputs
            }, features_file)
        else:
            features_data = torch.load(features_file)
            layer_features = features_data["layer_features"]
            del features_data
            # outputs = features_data["outputs"]
    
        self.layer_features = layer_features


        # Compute RDMs
        self.timestep_rdms = [
            calc_rdm(layer_features, timestep=t, method="euclidean", warn_if_none=False)
            for t in range(num_timesteps)
        ]
        # print(timestep_rdms[-1].get_matrices().shape)  # (num_layers, N, N), where N is number of images


    def run_analyses(self, del_previous_figures: bool = False):
        assert hasattr(self, "layer_features"), "Features not loaded. Call `set_sample_data` and `load_features` first."

        if del_previous_figures:
            # Delete files ending in ".png"
            for file in self.analysis_dir.glob("*.png"):
                file.unlink()

        # self.plot_validation_at_different_timesteps(metric="accuracy")
        # self.plot_last_layer_pca()
        # self.plot_category_distances()
        self.plot_feature_rdms()
        self.plot_feature_trajectory(method="mds", d=2)
        self.plot_feature_trajectory(method="mds", d=3)



    def plot_feature_rdms(self):
        fig, axs = plot_rdm_features(self.model, self.timestep_rdms)
        fig.suptitle(f"{self.name}\nLayer feature RDMs", fontsize=24)
        self._save_fig(fig, "layer_timestep_feature_rdm.png", dpi=500)

    def plot_feature_trajectory(self, method: str = "mds", d: int = 2):
        fig, axs = plot_feature_trajectory(self.model, self.timestep_rdms, method=method, d=d, cmap="rainbow")
        fig.suptitle(f"{self.name}\nLayer feature trajectory (method={method.upper()}, {d=})", fontsize=18)
        self._save_fig(fig, f"layer_feature_trajectory_{method}_{d}d.png", dpi=500)


    def plot_last_layer_pca(self, pca_across_all_timesteps: bool = False):
        layer_index = self.model.num_layers - 1  # last layer
        last_layer_features = OrderedDict()  # timestep -> flattened features
        last_layer_features_pca = OrderedDict()  # timestep -> PCA-transformed features
        pca = PCA(n_components=2)

        if pca_across_all_timesteps:
            # print("Fitting on all timesteps...")
            pca.fit(torch.cat([
                f.flatten(1).cpu()
                for f in self.layer_features[layer_index]
                if f is not None
            ], dim=0))

        for t, feat in enumerate(self.layer_features[layer_index]):
            if feat is None:
                continue
            features_matrix = feat.flatten(1).cpu()  # (num_samples, *)
            last_layer_features[t] = features_matrix
            if pca_across_all_timesteps:
                last_layer_features_pca[t] = pca.transform(features_matrix)
            else:
                last_layer_features_pca[t] = pca.fit_transform(features_matrix)

        fig, axs = plt.subplots(figsize=(15, 6), ncols=5, nrows=2, gridspec_kw=dict(hspace=0.5, wspace=0.5))
        for ax in axs.flatten():
            ax.axis("off")
        for (i, ax), (t, feat_pca) in zip(enumerate(axs.flatten()), last_layer_features_pca.items()):
            for cat, label in self.sample_label_names.items():
                m = self.sample_labels == cat
                ax.scatter(feat_pca[m, 0], feat_pca[m, 1], label=label, s=25)
            ax.set_title(f"${t =}$", color=("red" if t == self.model.num_recurrent_steps-1 else "black"), fontsize=16)
            if i == 0:
                ax.legend(fontsize=12, frameon=True, ncols=len(self.sample_label_names), loc="upper left", columnspacing=0.5, handletextpad=-0.2, bbox_to_anchor=(0, 0), bbox_transform=ax.transAxes)
        fig.suptitle(f"{self.name}\nLast layer feature PCA", fontsize=16)
        self._save_fig(fig, "last_layer_feature_pca.png")

    def plot_category_distances(self, layer_index: int = -1):
        if layer_index < 0:
            layer_index = self.model.num_layers + layer_index
        times = []
        category_distances = [[] for _ in self.sample_label_names]

        for t, features in enumerate(self.layer_features[layer_index]):
            if t-1 < 0 or features is None:
                continue
            prev_features = self.layer_features[layer_index][t-1]
            if prev_features is None:
                continue

            # flatten
            features = features.view(features.shape[0], -1)
            prev_features = prev_features.view(prev_features.shape[0], -1)

            # compute distances
            times.append(t)
            distances = (features-prev_features).square().sum(-1).sqrt()  # shape (num_samples,)
            for i, label in enumerate(self.sample_label_names.keys()):
                m = self.sample_labels == label
                category_distances[i].append(distances[m])

        fig, ax = plt.subplots(figsize=(8, 5))
        for cat_label, cat_distances in zip(self.sample_label_names.values(), category_distances):
            means = [d.mean() for d in cat_distances]
            stds = [d.std() for d in cat_distances]
            ax.errorbar(times, means, yerr=stds, label=cat_label, marker="o")
        ax.legend()
        ax.axvline(x=self.model.num_recurrent_steps-1, color="gray", linestyle="--", zorder=0)
        ax.set_title(f"{self.name}\nMean representation distance from previous timestep")
        self._format_ax(ax)
        
        # fig.savefig(analysis_dir / "mean_representation_distance.png", bbox_inches="tight")


    def plot_validation_at_different_timesteps(self, metric="accuracy"):
        file = self.model_dir / "valid_metrics_at_timesteps.pt"
        if not file.exists():
            return
        valid_metrics_at_timesteps = torch.load(file)["valid_metrics_at_timesteps"]

        fig, ax = plt.subplots(figsize=(9, 4))
        times = sorted(valid_metrics_at_timesteps.keys())
        metric_values = [valid_metrics_at_timesteps[t][metric] for t in times]
        
        ax.plot(times, metric_values, marker="o")

        # Annotate maximum
        max_i = np.argmax(metric_values)
        max_t, max_val = times[max_i], metric_values[max_i]
        ax.text(max_t, max_val, f"{max_val:.4f}", ha="left", va="bottom", fontsize=12)

        ax.set_title(f"{self.name}\nValidation at different timesteps")
        ax.set_xlabel("Prediction timestep")
        ax.set_ylabel(f"Validation {metric}")
        ax.axvline(self.model.num_recurrent_steps-1, color="gray", linestyle="--", zorder=0)
        ax.set_xticks(times)
        ax.set_ylim(0, ax.get_ylim()[1]*1.1)
        self._format_ax(ax)
        self._save_fig(fig, f"validation_{metric}_at_timesteps.png")
    
    def _format_ax(self, ax: plt.Axes):
        ax.set_title(ax.get_title(), fontsize=16)
        ax.set_xlabel(ax.get_xlabel(), fontsize=14)
        ax.set_ylabel(ax.get_ylabel(), fontsize=14)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis="both", labelsize=12)


    def _save_fig(self, fig: plt.Figure, filename: str, close: bool = True, **kwargs):
        if "bbox_inches" not in kwargs:
            kwargs["bbox_inches"] = "tight"
        fig.savefig(self.analysis_dir / filename, **kwargs)
        if close:
            plt.close(fig)
    




def load_model(model_dir: Path, load_from: str = "latest.pt", all_epoch_data: bool = False, return_metadata: bool = False, logger = None):
    """Helper function to load a model from the local model directory."""
    metadata = torch.load(model_dir / "metadata.pt")
    epoch_data = torch.load(model_dir / "epoch_data.pt")
    if not all_epoch_data:
        epoch_data = epoch_data[-1]

    model = build_model_from_args(metadata["args"], verbose=False)
    checkpoint = torch.load(model_dir / load_from, map_location="cpu")
    state = tr.get_model_state(checkpoint["model_state"], is_ddp=("ddp" in metadata))
    load_state = model.load_state_dict(state, strict=False)
    if logger is not None:
        logger.info(f"Loaded model state: {load_state}")

    if return_metadata:
        return model, epoch_data, metadata
    return model, epoch_data


def get_sample_imagenet_validation_images(categories: list[int], num_samples: int = 25, plot_output=False, **kwargs):
    """Get sample ImageNet validation images.

    Args:
        categories (list[int]): List of ImageNet category indices
        num_samples (int, optional): Number of samples per category. Defaults to 25.
        plot_output (bool, optional): Whether to also plot the first image in each category. Defaults to False.
        kwargs: Passed to `datasets.datasetes.fetch_ImageNet`

    Returns:
        tuple:
            - samples (torch.Tensor): ImageNet validation images; first dimension size is len(categories)*num_samples
            - sample_labels (torch.Tensor): 1d tensor of sample labels 
    """
    from datasets.datasets import fetch_ImageNet

    val_dataset = fetch_ImageNet(only_valid=True, **kwargs)
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_classes = torch.tensor([x[1] for x in val_dataset.samples])

    def get_samples(label: int, num_samples: int = 5):
        idx = torch.where(val_classes == label)[0]
        return torch.stack([val_dataset[i][0] for i in idx[:num_samples]])

    samples = torch.cat([get_samples(cat, num_samples) for cat in categories])
    sample_labels = torch.tensor([[cat] * num_samples for cat in categories], dtype=int).flatten()

    # Assuming 'samples' is a tensor of shape (batch_size, channels, height, width)
    # Convert the tensor to a grid of images
    
    if plot_output:
        grid = torchvision.utils.make_grid(samples[0::num_samples], nrow=len(categories), padding=5, pad_value=5)
        # Convert the grid tensor to a numpy array and transpose the dimensions
        grid = grid.permute(1, 2, 0)
        fig, ax = plt.subplots(figsize=(2.5*len(categories), 5))
        ax.imshow(grid)
        ax.axis("off")

    return samples, sample_labels


def extract_layer_features(
        model: BLTNet,
        inputs: torch.Tensor,
        layer_indices: Optional[list[int]] = None,
        num_timesteps: int = 10
    ) -> dict[int, list[torch.Tensor]]:
    """Load layer outputs from a BLT model.

    Args:
        model (BLTNet): Model
        inputs (torch.Tensor): Inputs to the model
        layer_indices (Optional[list[int]], optional): Layer indices to extract features. Defaults to all layers.

    Returns:
        dict[int, list[torch.Tensor]]: Map from layer index to list of tensors of layer outputs, indexed by time step.
        Note list entries may be None, if there is no output. 
        All tensors are on the CPU.
    """
    if layer_indices is None:
        layer_indices = list(range(model.num_layers))
    
    num_inputs = inputs.shape[0]
    layer_features = dict()  # layer_index -> Tensor[num_timepoints, num_inputs, *layer_output_shape]

    # Add model hooks
    hooks = []
    for layer_index in layer_indices:
        layer_features[layer_index] = []

        def _hook_fn(module, input, output, layer_index=layer_index):
            if output is not None:
                output = output.cpu()
            layer_features[layer_index].append(output)

        # get the layer output nn.Module
        layer_output = getattr(model, f"output_{layer_index}")
        hook = layer_output.register_forward_hook(_hook_fn)
        hooks.append(hook)
    
    # Run the inputs through the model
    pre_num_timesteps = model.num_recurrent_steps
    model.num_recurrent_steps = num_timesteps
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
    outputs = [out.cpu() for out in outputs]
    model.num_recurrent_steps = pre_num_timesteps  # reset the number of timesteps to the previous value
    
    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Convert features to tensors
    # for k, feat in features.items():
    #     features[k] = torch.stack(feat)

    return layer_features, outputs



def calc_rdm(
        layer_features: dict[int, list[Optional[torch.Tensor]]],
        timestep: int = -1,
        method: str = "euclidean",
        warn_if_none: bool = True,
        average_conv: bool = True
    ) -> rsatoolbox.rdm.rdms.RDMs:
    """Calculate the representational dissimilarity matrix(es) (RDM) for a set of layer feature(s).

    Args:
        layer_features (dict[int, list[Optional[torch.Tensor]]): Layer features
        timestep (int, optional): Timestep for which to calculate representations. Defaults to -1.
        method (str, optional): RDM method passed to `rsatoolbox.rdm.calc_rdm`. Defaults to "euclidean".
        warn_if_none (bool, optional): Whether to warn if layer features at the given timestep are None. Defaults to True.
        average_conv (bool, optional): Whether to average convolution HxW dimensions. Defaults to True.

    Returns:
        rsatoolbox.rdm.rdms.RDMs: RDM(s) for layer feature(s)
    """
    rsa_datasets = []

    for layer_index, feats in layer_features.items():
        feats = feats[timestep]

        if feats is None:
            if warn_if_none:
                warnings.warn(f"Features for {layer_index} is None at timestep {timestep}")
            continue
        
        if feats.dim() > 2:
            if average_conv:
                feats = feats.mean(dim=(-1, -2))
            else:
                feats = feats.view(feats.size(0), -1)

        # Convert to numpy array
        feats = feats.numpy()

        dataset = rsatoolbox.data.Dataset(measurements=feats, descriptors={"layer_index": layer_index})
        rsa_datasets.append(dataset)
    
    rdm = rsatoolbox.rdm.calc_rdm(rsa_datasets, method=method)
    
    return rdm

def plot_rdm_features(model: BLTNet, timestep_rdms: list[rsatoolbox.rdm.rdms.RDMs], category_boundaries=None):
    nrows = nlayers = timestep_rdms[-1].get_matrices().shape[0]  # number of layers
    ncols = ntimes = len(timestep_rdms)  # number of timesteps
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols*2, nrows*2), gridspec_kw=dict(wspace=0.1, hspace=0.1))

    # Plot each matrix
    for t, rdm in enumerate(timestep_rdms):
        for layer in range(nrows):
            ax = axs[axs.shape[0]-1-layer, t]
            
            # print(timestep, layer, len(rdm.get_matrices())-1)
            if t == layer:
                ax.set_ylabel(model.layer_names[layer], fontsize=22, rotation=0, ha="right", va="center", fontweight="bold")

            if layer == 0:
                ax.set_xlabel(f"$t={t}$", fontsize=18)
            
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            # ax.axis("off")

            if t == model.num_recurrent_steps-1 and layer == nlayers-1:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(5)
                    spine.set_color("green")
                    spine.set_position(("outward", 5))
                

            if layer < len(rdm.get_matrices()):
                im = rdm.get_matrices()[layer]
                im = im / im.max()  # normalize to 0-1 (min will always be zero for self-similarity matrix entries)
                ax.imshow(im, cmap="magma", vmin=0, vmax=1)

                if category_boundaries:
                    for boundary in len(category_boundaries):
                        x = boundary-0.5
                        ax.axvline(x=x, color="black", linewidth=0.5)
                        ax.axhline(y=x, color="black", linewidth=0.5)

    # plt.subplots_adjust(wspace=0, hspace=0)  # Optionally, adjust layout to remove additional spacing
    return fig, axs



def reduce_dim(X: Union[dict, torch.Tensor], method: str = "PCA", n_components: int = 2, return_model=False, random_state=0, **kwargs):
    """Reduce dimension of a matrix.

    Args:
        X (matrix_like or dict[any, matrix_like]): Input matrix
        method (str, optional): Dimensionality reduction method. One of "pca", "mds", "tsne". Defaults to "PCA".
        n_components (int, optional): Number of components. Defaults to 2.
        return_model (bool, optional): Whether to also return the dimensionality reduction model. Defaults to False.
        random_state (int, optional): Random state. Defaults to 0.

    Returns:
        X_transformed if return_model is False. (X_transformed, reduction_model) if return_model is True.
    """
    method = method.lower()
    if method == "pca":
        reduction = PCA(n_components=n_components, **kwargs)
    elif method == "mds":
        if "max_iter" not in kwargs: kwargs["max_iter"] = 2000
        if "n_init" not in kwargs: kwargs["n_init"] = 4
        reduction = sklearn.manifold.MDS(n_components=n_components, random_state=random_state, **kwargs)
    elif method == "tsne":
        if "perplexity" not in kwargs: kwargs["perplexity"] = 40
        if "verbose" not in kwargs: kwargs["verbose"] = 0
        reduction = sklearn.manifold.TSNE(n_components=n_components, random_state=random_state, **kwargs)
    # elif method == "umap":
    #     reduction = sklearn.manifold.UMAP(n_components=n_components, n_neighbors=30)
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")

    if isinstance(X, dict):
        X_transformed = {}
        for k, v in X.items():
            X_transformed[k] = reduction.fit_transform(v)
    else:
        X_transformed = reduction.fit_transform(X)
    
    return (X_transformed, reduction) if return_model else X_transformed


def plot_feature_trajectory(
    model: BLTNet,
    timestep_rdms: list[rsatoolbox.rdm.rdms.RDMs],
    method: str = "mds",
    d: int = 2,
    cmap: str = "viridis",
    colorbar: bool = True,
):
    X = []
    X_times = []
    rdm_layers = []
    for t, rdm in enumerate(timestep_rdms):
        for M in rdm.get_matrices():
            X.append(M[np.triu_indices(M.shape[0], k=1)])  # k=1 ignores the main diagonal
            X_times.append(t)
        rdm_layers.extend(rdm.rdm_descriptors["layer_index"])

    X = np.stack(X)  # shape (n_rdms, n_samples^2)
    X_times = np.array(X_times)
    rdm_layers = np.array(rdm_layers)
    X_reduced, reduction = reduce_dim(X, method=method, n_components=d, return_model=True)

    fig, axs = plt.subplots(figsize=(14, 8), ncols=3, nrows=2, subplot_kw=(dict(projection="3d") if d == 3 else None))

    cmap = mpl.colormaps.get_cmap(cmap)
    norm = mpl.colors.Normalize(vmin=0, vmax=len(timestep_rdms)-1)

    for layer_index, ax in enumerate(axs.flat):
        mask = rdm_layers == layer_index
        X_reduced_mask = X_reduced[mask]
        
        ax.plot(*X_reduced_mask.T, color="black")
        sc = ax.scatter(*X_reduced_mask.T, c=cmap(norm(X_times[mask])), s=100, zorder=10)

        # if d == 3 or layer_index % 3 == 0:
        #     ax.set_ylabel(f"{method}-2", fontsize=14)  # ({reduction.explained_variance_ratio_[1]*100:.0f}%)
        # if d == 3 or layer_index >= 3:
        #     ax.set_xlabel(f"{method}-1", fontsize=14)
        # if d == 3:
        #     ax.set_zlabel(f"{method}-3", fontsize=14)
        
        ax.set_title(f"{model.layer_names[layer_index]} (layer {layer_index+1})", fontsize=18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if d == 3:
            ax.set_zticks([])

    if colorbar:
        cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, pad=0.1, orientation="vertical")
        cax.tick_params(labelsize=14)
        cax.set_ylabel("Timestep", fontsize=14)

    return fig, axs
    
    # fig.savefig(Path.home() / f"Desktop/{model_dir}_time_repr_mds_{D}d.png", dpi=300, bbox_inches="tight")

    # cmap = mpl.colormaps.get_cmap("viridis")
    # norm = mpl.colors.Normalize(vmin=num_timesteps[0], vmax=num_timesteps[-1])

    # for i, t in enumerate(num_timesteps):
        # ax.scatter(*pca_reprs[i], color=cmap(norm(t)), s=100, zorder=10)
        # ax.scatter(*X_reduced_mask[i], color=cmap(norm(t)), s=100, zorder=10)


    # fig.colorbar(sc, ax=ax, shrink=0.7, label="Num timesteps", pad=0.1, orientation="vertical")
    # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7, label="Num timesteps", pad=0.1, orientation="vertical")





