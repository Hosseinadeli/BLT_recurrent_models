
import matplotlib.pyplot as plt
import rsatoolbox
from rsatoolbox.data import Dataset
from rsatoolbox.rdm.calc import calc_rdm
from scipy import stats
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib as mpl

from PIL import Image
from models.cornet import get_cornet_model

from repsim import AngularCKA

import torch
import torch.nn as nn
from models.activations import get_activations_batch
from collections import OrderedDict
import datasets
import os
import pandas as pd
import numpy as np
import csv
import torchvision

def load_vggface2():

    class args_stru:
        def __init__(self):
            
            # dataset info and save directory
            self.image_size =224
            self.img_channels = 3
            self.num_ids = 3929
            
    args = args_stru()
    kwargs = {} # {'num_workers': args.workers, 'pin_memory': True} if cuda else {}
    dt = datasets.VGG_Faces2(args, split='train')
    train_loader = torch.utils.data.DataLoader(dt, batch_size=128, shuffle=True, **kwargs)

    dt = datasets.VGG_Faces2(args, split='val')
    val_loader = torch.utils.data.DataLoader(dt, batch_size=128, shuffle=True, **kwargs)

    return train_loader, val_loader


def sample_vggface2(num_cats=5, per_cat=10, split_folder='test'):

    root = '/scratch/nklab/projects/face_proj/datasets/VGGFace2/'
    #split_folder = 'test'
    split_dir = root + split_folder + '/'
        
    dir_list = os.listdir(split_dir)

    test_bb_path = '/scratch/nklab/projects/face_proj/datasets/VGGFace2/meta/loose_bb_test_wlabels.csv'
    bb_df = pd.read_csv(test_bb_path)

    transform_rgb =  torchvision.transforms.Compose([
    #         transforms.RandomRotation(degrees=(0, 15)),
    #         transforms.RandomCrop(375),
    #         transforms.Resize((225,225)), # resize the images to 224x24 pixels
        torchvision.transforms.ToTensor(), # convert the images to a PyTorch tensor
        torchvision.transforms.Normalize([0.6068, 0.4517, 0.3800], [0.2492, 0.2173, 0.2082]) # normalize the images color channels
    ])

    imgs_o = []
    labels = []

    for i in range(num_cats): #range(len(dir_list))): #range(10): #[1]: #8631

        class_id = dir_list[i]

        im_list = os.listdir(split_dir+class_id)

        for im_ch in range(per_cat): # np.random.choice(len(im_list), 1)[0]
            labels.append(i)
            src_im = im_list[im_ch]
            img_file = os.path.join(split_folder, class_id, src_im)

            img = Image.open(os.path.join(root, img_file))

            img = torchvision.transforms.Resize(224)(img)  #256
            img = torchvision.transforms.CenterCrop(224)(img)

            imgs_o.append(transform_rgb(img))

    imgs = torch.stack(imgs_o).to(device)

    return imgs, labels


def sample_FEI_dataset(num_ids=25, orientation_inds = None, mirror_symmetry = None):
    root = '/engram/nklab/hossein/recurrent_models/face_datasets/'
    split_folder = 'FEI'
    split_dir = root + split_folder + '/'
    dir_list = os.listdir(split_dir)

    

    if orientation_inds is None:
        orientation_inds = [1, 3, 11, 7, 10]
        mirror_symmetry = [0, 1, 2, 1, 0]

    src_im = dir_list[0] 
    img_file = os.path.join(split_dir, src_im)

    id_inds = np.arange(1,num_ids+1)

    transform_rgb =  torchvision.transforms.Compose([
    #         transforms.RandomRotation(degrees=(0, 15)),
    #         transforms.RandomCrop(375),
    #         transforms.Resize((225,225)), # resize the images to 224x24 pixels
        torchvision.transforms.Resize(224),  #256
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(), # convert the images to a PyTorch tensor
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # normalize the images color channels
        #torchvision.transforms.Normalize([0.6068, 0.4517, 0.3800], [0.2492, 0.2173, 0.2082]) # normalize the images color channels
    ])

    imgs_o = []

    labels_o = []
    labels_m = []
    labels_i = []

    
    for o in range(len(orientation_inds)):
        for i in id_inds:
            if mirror_symmetry is not None:
                labels_o.append(o)
                labels_m.append(mirror_symmetry[o])

            labels_i.append(i)
            src_im = f'{i}-{str(orientation_inds[o]).zfill(2)}.jpg'
            img_file = os.path.join(split_dir, src_im)
            img = Image.open(os.path.join(root, img_file))

            img = transform_rgb(img)
            
            #img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=3) 
            imgs_o.append(img)
            

    imgs = torch.stack(imgs_o)

    return imgs, torch.tensor(labels_o), torch.tensor(labels_m), torch.tensor(labels_i)


def FBO_dataset():
    root = '/engram/nklab/hossein/recurrent_models/face_datasets/'
    split_folder = 'FBO'
    split_dir = root + split_folder + '/'
    dir_list = os.listdir(split_dir)

    FBO_info = pd.read_excel(split_dir + 'info.xlsx')  
    FBO_info.head()

    transform_rgb =  torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),  #256
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(), # convert the images to a PyTorch tensor
            torchvision.transforms.Normalize([0.6068, 0.4517, 0.3800], [0.2492, 0.2173, 0.2082]) # normalize the images color channels
        ])

    imgs = []
    labels = []

    for i in range(len(FBO_info)):
        imange_name = FBO_info['image'].iloc[i]
        label = FBO_info['category'].iloc[i]

        if label == 'face':
            label = 1
        elif label == 'body':
            label = 2
        elif label == 'vegifruit':
            label = 3
        elif label == 'artefact':
            label = 4
        elif label == 'scrambled':
            label = 5
        else:
            continue
            
        labels.append(label)

        img = Image.open(split_dir + imange_name)

        img = img.convert('RGB')

        img = transform_rgb(img)
                
        img = torchvision.transforms.functional.rgb_to_grayscale(img, num_output_channels=3) 
        imgs.append(img)
                
    imgs = torch.stack(imgs).to(device)
    #labels = torch.tensor(labels).to(device)

    return imgs, labels



import scipy.io

def kasper_dataset():
    mat = scipy.io.loadmat('/engram/nklab/hossein/recurrent_models/BLT_models/datasets/saved_data/images.mat')
    imarray = mat['imarray']

    neuro_mat = scipy.io.loadmat('/engram/nklab/hossein/recurrent_models/BLT_models/datasets/saved_data/neural.mat')
    neuro_data = neuro_mat['R'].transpose()

    imgs = torch.load('/engram/nklab/hossein/recurrent_models/BLT_models/datasets/saved_data/imgs_kasper.pt')

    # try:
    #     imgs = torch.load('./datasets/saved_data/imgs_kasper.pt')
    # except:
    #     normalize = torchvision.transforms.Compose([
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ])
    #     imgs = torch.stack([normalize(img) for img in np.moveaxis(imarray, -1, 0)])   

    # imgs.shape
    labels = mat['imsets'].squeeze() - 1  #(array([0, 1, 2], dtype=uint8), array([230, 217, 932]))

    return imgs, labels, neuro_data


def calc_rdms(model_features, method='euclidean'):
    """
    Calculates Representational Dissimilarity Matrices (RDMs) from model features.

    Args:
    - model_features (dict): Dictionary containing model features for each layer.
    - method (str): Method used to calculate the RDMs ('euclidean' or 'correlation').

    Returns:
    - rdms (numpy.ndarray): Representational Dissimilarity Matrices Objects.
    - rdms_dict (dict): Dictionary containing RDMs for each layer.
    """
    
    if method == 'cka':
        metric = AngularCKA(m=len(model_features[list(model_features.keys())[0]]))
        rdms_dict = {k: metric.neural_data_to_point(torch.tensor(x)).numpy() for k, x in model_features.items()}
        print(rdms_dict.keys())
        print(rdms_dict['step 4'].shape)
        
        return None, rdms_dict

    ds_list = []
    for l in range(len(model_features)):

        layer = list(model_features.keys())[l]
        feats = model_features[layer]

        if type(feats) is list:
            feats = feats[-1]

        if device == 'cuda':
            feats = feats.cpu()

        if len(feats.shape)>2:
            feats = feats.flatten(1)

        #feats = feats.numpy()

        ds = Dataset(feats, descriptors=dict(layer=layer))
        ds_list.append(ds)

    rdms = calc_rdm(ds_list, method=method)

    rdms_dict = {list(model_features.keys())[i]: rdms.get_matrices()[i] for i in range(len(model_features))}

    return rdms, rdms_dict


def plot_maps(model_features, save=None, cmap = 'magma', add_text=True, add_bar=True):

    fig = plt.figure(figsize=(15, 4))
    # and we add one plot per reference point
    gs = fig.add_gridspec(1, len(model_features))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)

    for l in range(len(model_features)):

        layer = list(model_features.keys())[l]
        map_ = np.squeeze(model_features[layer])

        if len(map_.shape) < 2:
            map_ = map_.reshape( (int(np.sqrt(map_.shape[0])), int(np.sqrt(map_.shape[0]))) )

        map_ = map_ / np.max(map_)

        ax = plt.subplot(gs[0,l])
        ax_ = ax.imshow(map_, cmap=cmap)
        if add_text:
            ax.text(0.5, 1.15, f'{layer}', size=12, ha="center", transform=ax.transAxes)
            #ax.set_title(f'{layer}')
        ax.axis("off")
        
    
    if add_bar:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.95, 0.25, 0.01, 0.5])
        fig.colorbar(ax_, cax=cbar_ax)

    # if save:
    #     fig.savefig(f'{save}.svg', format='svg', dpi=300, bbox_inches='tight')

    return fig

def compare_rdms(model, imgs, layers, neuro_data, method='corr', num_steps=5):

    neuro_rdms, _ = calc_rdms(neuro_data)

    pred_models = []
    pred_model_names = []

    for layer in layers: # ['output_0', 'output_1', 'output_2', 'output_3']:
        features = extract_features(model, imgs, layer, num_steps=num_steps)
        model_rdms, rdms_dict = calc_rdms(features)

        for model_name in list(features.keys()):
            rdm_m = model_rdms.subset('layer', model_name)
            m = rsatoolbox.model.ModelFixed(f'{layer} {model_name}', rdm_m)
            pred_model_names.append(f'{layer} {model_name}')
            pred_models.append(m)

    N = 100
    if 'cov' in method:
        N=5
    results = rsatoolbox.inference.eval_bootstrap_pattern(pred_models, neuro_rdms, method=method, N=N)
    rsatoolbox.vis.plot_model_comparison(results)
    return results, pred_model_names



def plot_recurrent_rdms(model, imgs, layer, num_steps=7, save=None):

    try:
        model = model.module
    except:
        model = model

    output = get_activations_batch(model, imgs, layer=layer, sublayer='output')
    output = output.reshape(*output.shape[:3], -1).mean(-1)

    output.shape

    recurrent_features = {}

    for t in range(len(output)-num_steps,len(output)):
        recurrent_features[str(t)] = output[t]

    recurrent_features

    rdms, rdms_dict = calc_rdms(recurrent_features)
    plot_maps(rdms_dict, save)


def reduce_dim(features, transformer = 'MDS', n_components = 2):

    if transformer == 'PCA': transformer_func = PCA(n_components=n_components)
    if transformer == 'MDS': transformer_func = manifold.MDS(n_components = n_components, max_iter=200, n_init=4)
    if transformer == 't-SNE': transformer_func = manifold.TSNE(n_components = n_components, perplexity=40, verbose=0)

    return_layers = list(features.keys())
    feats_transformed = {}
    
    for layer in return_layers:
        feats = transformer_func.fit_transform(features[layer])
        feats_transformed[layer] = feats

    return feats_transformed

def plot_dim_reduction_one(features, labels=None, transformer='MDS', save=None, add_text=True, add_bar=True):

    return_layers = list(features.keys())    

    fig = plt.figure(figsize=(3*len(return_layers), 4))
    # and we add one plot per reference point
    gs = fig.add_gridspec(1, len(return_layers))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    if add_text:
        fig.text(0.55, 0.95, transformer, size=14, ha="center")

    for l in range(len(return_layers)):
        layer =  return_layers[l]
        feats = features[layer]

        ax = plt.subplot(gs[0,l])
        ax.set_aspect('equal', adjustable='box')

        amin, amax = feats.min(), feats.max()
        amin, amax = (amin + amax) / 2 - (amax - amin) * 5/8, (amin + amax) / 2 + (amax - amin) * 5/8
        ax.set_xlim([amin, amax])
        ax.set_ylim([amin, amax])
        
        # for d in range(2):
        #     feats[:, d] = feats[:, d] / (np.max(feats[:, d]) - np.mean(feats[:, d]))
        # ax.set_ylim(-1.3, 1.3)
        # ax.set_xlim(-1.3, 1.3)
        if add_text:
            ax.text(0.5, 1.1, f'{layer}', size=12, ha="center", transform=ax.transAxes) 
        ax.axis("off")
        #if l == 0: 
        # these lines are to create a discrete color bar
        if labels is None:
            labels = np.zeros(len(feats[:, 0]))

        num_colors = len(np.unique(labels))
        cmap = plt.get_cmap('viridis_r', num_colors) # 10 discrete colors
        norm = mpl.colors.BoundaryNorm(np.arange(-0.5,num_colors), cmap.N) 
        ax_ = ax.scatter(feats[:, 1], feats[:, 0], c=labels, cmap=cmap, norm=norm, s=3)
    
    
    if add_bar:
        fig.subplots_adjust(right=0.9, top=0.9)
        cbar_ax = fig.add_axes([0.95, 0.3, 0.01, 0.45])
        fig.colorbar(ax_, cax=cbar_ax, ticks=np.linspace(0,9,10))

    # if save:
    #     fig.savefig(f'{save}.svg', format='svg', dpi=300, bbox_inches='tight')

    return fig



def plot_rdm_mds(model, imgs, labels, layers, rdm_method='euclidean', num_steps=5, plot='rdm mds', cmap = 'magma', add_text= True, add_bar=True, save= None, format='png', filter_units=None, clip_acts=None):

    if 'rdm' in plot:
        for layer in layers:
            # if save:
            #     save = f'results/figures/{layer}_{save}_rdms'
            features = extract_features(model, imgs.to(device), layer, num_steps=num_steps, filter_units=filter_units, clip_acts=clip_acts)
            _, rdms_dict = calc_rdms(features, method=rdm_method)
            #if layer is not 'IT':
            fig = plot_maps(rdms_dict, save=save, add_text=add_text, add_bar=add_bar, cmap=cmap)
            if save:
                save_path = f'results/figures/{layer}_{save}_rdms'
                fig.savefig(f'{save_path}.{format}', format=format, dpi=300, bbox_inches='tight')

    if 'mds' in plot:
        for layer in layers:
            # if save:
            #     save = f'results/figures/{layer}_{save}_mds'
            features = extract_features(model, imgs.to(device), layer, num_steps=num_steps, filter_units=filter_units, clip_acts=clip_acts)
            features_transformed = reduce_dim(features)

            fig = plot_dim_reduction_one(features_transformed, labels, transformer='MDS', save=save, add_text=add_text, add_bar=add_bar)
            if save:
                save_path = f'results/figures/{layer}_{save}_mds'
                fig.savefig(f'{save_path}.{format}', format=format, dpi=300, bbox_inches='tight')

from models.build_model import build_model
from scipy import stats
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calc_dprime_from_dataset(model, output_layer='output_5', threshold = 0.2, dataset_name='FBO', num_model_steps=5):

    if dataset_name == 'FBO':
        # extract features for FBO dataset
        imgs, labels = FBO_dataset()

    outputs = extract_features(model, imgs.to(device), output_layer, num_steps=num_model_steps)

    model_resp_steps = []
    for key, value in outputs.items():
        model_resp_steps.append(outputs[key].transpose())

    model_resp_steps = np.array(model_resp_steps)
    model_resp_steps.shape
    #model_resp = model_resp[np.where(np.mean(model_resp, axis=1))[0],:]

    #model_resp = z_score_feats(model_resp_steps)
    model_resp = stats.zscore(model_resp_steps, axis=2)

    # averatge over time-steps for calculating selectivity
    model_resp = np.mean(model_resp_steps, axis=0)

    face_resp = model_resp[:,np.where(np.array(labels)==1)[0]]
    non_face_resp = model_resp[:,np.where(np.array(labels)!=1)[0]]

    d_prime = np.mean(face_resp, axis=1) - np.mean(non_face_resp, axis=1)

    d_prime = d_prime / np.sqrt(np.var(face_resp, axis=1)/2 + np.var(non_face_resp, axis=1)/2)
    face_cell_resp_steps = model_resp_steps[:,np.where(d_prime > threshold)[0], :] 

    return d_prime, face_cell_resp_steps




def load_model_path(model_path, print_model=False):

    checkpoint = torch.load(model_path + 'checkpoint.pth', map_location='cpu')

    pretrained_dict = checkpoint['model']
    args = checkpoint['args']
    model = build_model(args, verbose=print_model)
    model.load_state_dict(pretrained_dict, strict=False)
    model.to(device)

    model.eval()

    try:
        model = model.module
    except:
        model = model

    gap = nn.Sequential(OrderedDict([
                    ('avgpool', nn.AdaptiveAvgPool2d(1)),
                    # ('linear', nn.Linear(512, 1000))
                ]))
        
    return model, gap, args


def load_pretrained_models(model_name):
    
    if 'cornet' in model_name:
        model = get_cornet_model('s', pretrained=True)

    model.to(device)
    model.eval()

    try:
        model = model.module
    except:
        model = model

    return model


def extract_features(model, imgs, layer, num_steps=5, normalize=False, filter_units=None, clip_acts = None):

    try:
        model = model.module
    except:
        model = model

    num_chunks = len(imgs) // 64
    chunks = torch.chunk(imgs, num_chunks, dim=0)

    outputs = []
    for chunk in chunks:
        output = get_activations_batch(model, chunk, layer=layer, sublayer='output')
        #print(output.reshape(*output.shape[:3], -1).shape) (5, 276, 512, 49)

        if clip_acts is not None:
            output = np.clip(output, a_min=0, a_max=clip_acts)

        print(output.shape)
        if filter_units is not None:
            output = output[:, :, filter_units, :] # only take the last time-step
            print(output.shape)

        # average pooling over spatial dimensions in higher layers
        if layer == 'output_4' or layer == 'output_5':
            output = output.reshape(*output.shape[:3], -1).mean(-1)
        else:
            output = output.reshape(*output.shape[:2], -1)

        outputs.append(output)

    output = np.concatenate(outputs, axis=1)

    features = {}
    average_features = []
    for t in range(len(output)-num_steps,len(output)):
        features[f'step {t}'] = output[t]
        if normalize:
            average_features.append(output[t] / np.max(output[t]))
        else:
            average_features.append(output[t])

    #features['average'] = np.mean(average_features, axis=0)
    return features

# def extract_features_model(model, imgs, layer, num_steps=5):

#     try:
#         model = model.module
#     except:
#         model = model

#     num_chunks = len(imgs) // 64
#     chunks = torch.chunk(imgs, num_chunks, dim=0)

#     outputs = []
#     for chunk in chunks:
#         output = get_activations_batch(model, chunk, layer=layer, sublayer='output')

#         # average pooling over spatial dimensions in higher layers
#         if layer == 'output_4' or layer == 'output_5':
#             output = output.reshape(*output.shape[:3], -1).mean(-1)
#         else:
#             output = output.reshape(*output.shape[:2], -1)

#         outputs.append(output)

#     output = np.concatenate(outputs, axis=1)

#     features = {}
#     average_features = []
#     for t in range(len(output)-num_steps,len(output)):
#         features[f'step {t}'] = output[t]
#         average_features.append(output[t]) # / np.max(output[t]))

#     #features['average'] = np.mean(average_features, axis=0)
#     return features