from typing import Union, Dict
from argparse import Namespace

from models.cornet import get_cornet_model
from models.blt import get_blt_model


def build_model(
        model_name: str,
        recurrent_steps: int = 10,
        img_channels: int = 3,
        num_layers: int = 6,
        num_classes: int = 1000,
        pretrained: bool = False,
        verbose: bool = True,
    ):
    if 'blt' in model_name:
        # if args.model[4:] == 'b':
        #     kwargs = {'in_channels': args.img_channels, 'times': args.recurrent_steps} #
        # else:
        model = get_blt_model(
            model_name[4:],
            pretrained=pretrained,
            map_location=None,
            in_channels = img_channels,
            num_recurrent_steps = recurrent_steps,
            num_layers = num_layers,
            num_classes = num_classes
        )
    elif 'cornet' in model_name:
        if model_name[7:] == 'z' or model_name[7:] == 's':
            kwargs = {'in_channels': img_channels, 'num_classes': num_classes} #
        elif model_name[7:] == 'r' or model_name[7:] == 'rt':
            kwargs = {'in_channels': img_channels, 'times': recurrent_steps, \
                      'num_classes': num_classes} #

        model = get_cornet_model(model_name[7:], pretrained=pretrained, map_location=None, **kwargs) #
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if verbose:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of model parameters: {num_parameters}")
        print(model)

    return model

def build_model_from_args(args: Union[Dict, Namespace], pretrained=False, verbose=True):
    if not isinstance(args, dict):
        args = vars(args)

    return build_model(
        model_name = args["model"],
        recurrent_steps = args["recurrent_steps"],
        img_channels = args["img_channels"],
        num_layers = args["num_layers"],
        num_classes = args["num_classes"],
        pretrained = pretrained,
        verbose = verbose
    )
