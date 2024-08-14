from collections import OrderedDict
from typing import Optional

import torch
from torch import nn
    
class Unsqueeze4d(nn.Module):
    """Unsqueezes a 2d input tensor by adding two singleton dimensions at the end."""
    def forward(self, x: torch.Tensor):
        return x.view(x.size(0), x.size(1), 1, 1)

# Feedforward layer i -> j connection is when j > i
FEEDFORWARD_CONV_SCALE_KWARGS = {
    1: dict(kernel_size=3, stride=1, padding=1),
    2: dict(kernel_size=3, stride=2, padding=1),
    4: dict(kernel_size=5, stride=4, padding=2),
    8: dict(kernel_size=9, stride=8, padding=4),
    16: dict(kernel_size=17, stride=16, padding=8),
}

# Recurrent layer i -> j connection is when j = i
RECURRENT_CONV_KWARGS = dict(kernel_size=3, stride=1, padding=1)

# Feedback layer i -> j connection is when j < i
# If "output_padding" is defined, then it will be a ConvTranspose2d, otherwise Conv2d
FEEDBACK_CONV_SCALE_KWARGS = {
    1: dict(kernel_size=3, stride=1, padding=1),
    2: dict(kernel_size=3, stride=2, padding=1, output_padding=1),
    4: dict(kernel_size=5, stride=4, padding=2, output_padding=3),
    8: dict(kernel_size=9, stride=8, padding=4, output_padding=7),
    16: dict(kernel_size=17, stride=16, padding=8, output_padding=15),
}

MAXPOOL_KWARGS = dict(kernel_size=3, stride=2, padding=1)

# The layer connection attribute name
# Used to be conv_{pre}_{post} but changing it to make it more readable
# LAYER_CONNECTION_ATTR_NAME = "conv_{pre}_{post}"
LAYER_CONNECTION_ATTR_NAME = "conn_{pre}_to_{post}"

class BLTNet(nn.Module):
    def __init__(self, model_name, conn_matrix, num_classes, layer_channels, out_shape, input_shape: int = 224, num_recurrent_steps: int = 10, use_maxpool: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_recurrent_steps = num_recurrent_steps
        self.num_classes = num_classes
        self.connections = {}
        self.non_lins = {}
        self.layer_channels = layer_channels
        self.out_shape = out_shape
        self.conn_matrix = conn_matrix
        self.num_layers = len(conn_matrix)
        self.input_size = input_shape
        self.input_channels = layer_channels["inp"]
        self.use_maxpool = use_maxpool
        self._t = 0  # current time step

        if self.num_layers == 6:
            self.layer_names = ["V1", "V2", "V4", "ML", "AL", "AM"]
        else:
            self.layer_names = [f"L{i}" for i in range(self.num_layers)]

        # layers_inputting_to[L] is the list of layers that feed into layer L
        self.layers_inputting_to = [
            [
                pre for pre in range(self.num_layers)
                if conn_matrix[pre, post]
            ]
            for post in range(self.num_layers)
        ]

        # Setup the input --> first layer connections
        l1_channels = layer_channels['0']
        stride_maxpool_div = MAXPOOL_KWARGS["stride"] if use_maxpool else 1
        if out_shape["0"] == 56:
            conv_kwargs = dict(kernel_size=7, stride=4, padding=3)
        elif out_shape["0"] == 112:
            conv_kwargs = dict(kernel_size=5, stride=2, padding=2) 
        else:
            raise ValueError(f"Can't handle first layer reduction: {input_shape} -> {out_shape['0']}")
        conv_kwargs["stride"] //= stride_maxpool_div
        self.conv_input = nn.Conv2d(self.input_channels, l1_channels, **conv_kwargs)
        if use_maxpool:
            self.maxpool_input = nn.MaxPool2d(**MAXPOOL_KWARGS)

        # Define all the connections between the layers
        for layer in range(self.num_layers):
            size, channels = out_shape[str(layer)], layer_channels[str(layer)]
            
            setattr(self, f'norm_{layer}', nn.GroupNorm(32, channels))
            setattr(self, f'non_lin_{layer}', nn.ReLU(inplace=True))
            setattr(self, f'output_{layer}', nn.Identity())  # useful to have identity layer to get intermediate representations
            
            for pre_layer in self.layers_inputting_to[layer]:
                pre_size, pre_channels = out_shape[str(pre_layer)], layer_channels[str(pre_layer)]

                if pre_layer < layer:
                    # Bottom-up/feedforward connection
                    if size == 1:
                        # Once we get to size 1 we switch from conv to linear layers
                        # This size representation is used in the networks where the last layers are linear
                        if pre_size == 1:
                            # This layer's input is 1x1 and its output is 1x1
                            conn = nn.Sequential(OrderedDict([
                                ("flatten", nn.Flatten()),
                                ("linear", nn.Linear(pre_channels, channels)),
                                ("unsqueeze", Unsqueeze4d()),
                            ]))
                        else:
                            # input is NxN and output is 1x1
                            # print(f"pre: {pre_channels}, {pre_size}  |  post: {channels}, {size}")
                            conn = nn.Sequential(OrderedDict([
                                ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                                ("flatten", nn.Flatten()),
                                
                                # In practice this is equal to 512*4*4 = 8192. The previous layer
                                # has an output shape of 7x7 which goes through the above maxpool,
                                # resulting in a 4x4
                                ("linear", nn.Linear(pre_channels*4*4, channels)),  # TODO: dynamic first dim value?
                                ("unsqueeze", Unsqueeze4d()),
                            ]))
                    else:
                        # Output is >1x1; use a conv layer
                        shape_factor = pre_size // size  # note pre_size >= size
                        conv_kwargs = FEEDFORWARD_CONV_SCALE_KWARGS[shape_factor]
                        if use_maxpool and conv_kwargs["stride"] >= 2:
                            conv_kwargs = conv_kwargs.copy()
                            conv_kwargs["stride"] //= stride_maxpool_div
                        conv = nn.Conv2d(pre_channels, channels, **conv_kwargs)

                        if shape_factor > 1 and use_maxpool:
                            # There is some shrinkage which warrants a maxpool layer
                            conn = nn.Sequential(OrderedDict([
                                ("maxpool", nn.MaxPool2d(**MAXPOOL_KWARGS)),
                                ("conv", conv),
                            ]))
                        else:
                            conn = conv
                elif pre_layer == layer:
                    # Lateral/recurrent connection
                    conv_kwargs = RECURRENT_CONV_KWARGS
                    conn = nn.Conv2d(pre_channels, channels, **conv_kwargs)
                else:
                    # Top-down/feedback connection
                    conv_kwargs = FEEDBACK_CONV_SCALE_KWARGS[size // pre_size]  # note pre_size <= size
                    conv_class = nn.ConvTranspose2d if "output_padding" in conv_kwargs else nn.Conv2d
                    conn = conv_class(pre_channels, channels, **conv_kwargs)
                
                setattr(self, LAYER_CONNECTION_ATTR_NAME.format(pre=pre_layer, post=layer), conn)

        self.read_out = nn.Sequential(OrderedDict([
            ("avgpool", nn.AdaptiveAvgPool2d(1)),
            ("flatten", nn.Flatten()),
            ("linear", nn.Linear(512, self.num_classes))
        ]))

    def _layer_activation(self, layer: int, x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """Computs the output of a layer by applying normalization and non-linearity:
            y = output(non_lin(norm(x)))   if x is not None
            y = output(x)                  otherwise
        where output is an identity layer.

        Args:
            layer (int): Layer index
            x (Optional[torch.Tensor]): Pre-activation of layer. Can be None.

        Returns:
            Optional[torch.Tensor]: Layer output. Will be None if input `x` is None.
        """
        if x is not None:
            # apply norm -> nonlinearity
            x = getattr(self, f"norm_{layer}")(x)
            x = getattr(self, f"non_lin_{layer}")(x)

        # NOTE: We always call `output_X` for each layer so it gets a representation at each time step
        # This representation may be None.
        x = getattr(self, f"output_{layer}")(x)
        return x

    def forward(self, inp):
        # layer_outputs[L] is the output of layer L (at the current time step)
        layer_outputs = [None for _ in range(self.num_layers)]

        # Simulate t=0 by passing the input through to the first layer
        # (mainly to save computation by convolving the input only once)
        self._t = 0  # set current time step
        inp = self.conv_input(inp)
        if self.use_maxpool:
            inp = self.maxpool_input(inp)
        layer_outputs[0] = self._layer_activation(0, inp)  # first layer state at t=0
        # other layers at t=0 have no input
        # run _layer_activation to match outputs below
        for layer in range(1, self.num_layers):
            layer_outputs[layer] = self._layer_activation(layer, None)

        # If the model is strictly feedforward, then we don't need to iterate over time steps
        if self.model_name in ("b", "b_pm", "b_top2linear"):
            for layer in range(1, self.num_layers):  # skip the first layer because we processed it above
                self._t += 1
                pre_layer = self.layers_inputting_to[layer][0]  # only one pre layer for feedforward
                assert pre_layer == layer-1  # should be a feedforward model
                conn = getattr(self, LAYER_CONNECTION_ATTR_NAME.format(pre=pre_layer, post=layer))
                x = layer_outputs[pre_layer]
                x = conn(x)  # feedforward connection
                x = self._layer_activation(layer, x)
                layer_outputs[layer] = x

            out = layer_outputs[-1]
            return [self.read_out(out)]

        # Otherwise, we will need to iterate over time steps
        # Use this list to temporarily store the outputs of each layer at the next timestep
        layer_outputs_next_timestep = [None for _ in range(self.num_layers)]
        all_outs = []

        # Iterate over time steps (note we already did t=0 above)
        for t in range(1, self.num_recurrent_steps):
            self._t = t  # set current time step
            for layer in range(self.num_layers):
                # Determine the input to this layer at the current time step
                layer_input = 0

                # The first layer also gets external input
                if layer == 0:
                    layer_input = inp

                # Iterate over layers that project to this layer
                for pre_layer in self.layers_inputting_to[layer]:
                    pre_layer_output = layer_outputs[pre_layer]
                    if pre_layer_output is not None:
                        conn = getattr(self, LAYER_CONNECTION_ATTR_NAME.format(pre=pre_layer, post=layer))
                        x = conn(pre_layer_output)
                        # x = getattr(self, f'norm_{pre_layer}_{layer}')(x)
                        # x = getattr(self, f'non_lin_{pre_layer}_{layer}')(x)
                        layer_input = layer_input + x  # add the input from this pre_layer
                
                # If input is not a Tensor (e.g., if it == 0), then this layer has no input
                if not isinstance(layer_input, torch.Tensor):
                    layer_input = None
                # else:
                #     print(f"{t=} {layer=}, {layer_input.shape=}")

                # Always call _layer_activation so `output_X` is called at each time step
                # (This gives None representations for layers at certain time steps)
                layer_output = self._layer_activation(layer, layer_input)
                layer_outputs_next_timestep[layer] = layer_output

            # At the end of this timestep, update the layer outputs    
            for layer, new_output in enumerate(layer_outputs_next_timestep):
                layer_outputs[layer] = new_output
            
            # Store the last layer output, if available
            last_layer_output = layer_outputs[-1]
            if last_layer_output is not None:
                all_outs.append(self.read_out(last_layer_output))

        return all_outs


def get_blt_model(model_name, pretrained=False, map_location=None, num_layers=6, in_channels=3, num_recurrent_steps=10, num_classes=1000):
    if num_layers == 4:
        layer_channels  = {'inp':in_channels, '0':64, '1':128, '2':256, '3':512}
        out_shape  = {'0':56, '1':28, '2':14, '3':7}
        # layer_channels  = {'inp':img_channels, '0':128, '1':384, '2':512, '3':512}
        # out_shape  = {'0':112, '1':56, '2':28, '3':14}

    elif num_layers == 5:
        layer_channels = {'inp':in_channels, '0':64, '1':128, '2':128, '3':256, '4':512}
        out_shape  = {'0':56, '1':28, '2':14, '3':7, '4':7}

    elif num_layers == 6:
        layer_channels = {'inp':in_channels, '0':64, '1':64, '2':128, '3':128, '4':256, '5':512}
        out_shape  = {'0': 56, '1': 56, '2': 28, '3': 28, '4': 14, '5': 7}

        # if we have two linear layer after 4 conv layers
        if 'top2linear' in model_name:
            layer_channels  = {'inp':in_channels, '0':64, '1':128, '2':256, '3':512, '4':1024 , '5':512}
            out_shape  = {'0':56, '1':28, '2':14, '3':7, '4':1, '5':1}

        # we can paramter match a 6 layer b model with a 4 layer bl model (~ 6.5 m)
        if 'b_pm' in model_name:
            layer_channels = {'inp':in_channels, '0':64, '1':128, '2':256, '3':256, '4':512, '5':512}
            out_shape  = {'0':112, '1':56, '2':28, '3':28, '4':14, '5':7}

    elif num_layers == 8:
        # layer_channels = {'inp':img_channels, '0':64, '1':64, '2':128, '3':128, '4':256, '5':512}
        # out_shape  = {'0':56, '1':28, '2':14, '3':14, '4':7, '5':7}
        layer_channels = {'inp':in_channels, '0':64, '1':128, '2':128, '3':256, '4':256, '5':256, '6':256, '7':512}
        out_shape  = {'0':112, '1':56, '2':28, '3':28, '4':7, '5':7, '6':7, '7':7}

    conn_matrix = torch.zeros((num_layers, num_layers), dtype=bool)

    # make the model name only the architecture name
    model_name = model_name.split('_')[0]

    shift = [-1]  # this corresponds to bottom up connections -- always present
    if 'l' in model_name: shift.extend([0])
    if 'b2' in model_name: shift.extend([-2])
    if 'b3' in model_name: shift.extend([-2, -3])
    if 't' in model_name: shift.extend([1])
    if 't2' in model_name: shift.extend([1, 2])
    if 't3' in model_name: shift.extend([2, 3])
    for i in range(num_layers):
        for j in range(num_layers):
            for s in shift:
                # just add other connections for the last 4 layers
                # if (s != -1) and ((i<(num_layers-4)) or (j<(num_layers-4))): continue 
                if i == (j+s):
                    conn_matrix[i, j] = True

    model = BLTNet(model_name, conn_matrix, num_classes, layer_channels, out_shape=out_shape, num_recurrent_steps=num_recurrent_steps)
    return model


def get_all_blt_model_names():
    from itertools import product
    b_combinations = ["b", "b2", "b3"]
    l_combinations = ["", "l"]
    t_combinations = ["", "t", "t2", "t3"]
    blt_combinations = ["".join(x) for x in product(b_combinations, l_combinations, t_combinations)]
    return blt_combinations
