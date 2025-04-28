import numpy as np
import torch
from torch import nn
from collections import OrderedDict
import torch.utils.model_zoo
import antialiased_cnns

class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name
    """
    def forward(self, x):
        return x

class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """
    def __init__(self, unsqueeze=False):
        super().__init__()
        self.unsqueeze = unsqueeze

    def forward(self, x):
        if self.unsqueeze:
            return x.view(x.size(0), -1).unsqueeze(-1)
        else:
            return x.view(x.size(0), -1)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.view(x.size(0), x.size(1), 1, 1)


# BLT

class blt(nn.Module):

    def __init__(self, model_name, conn_matrix, num_classes, layer_channels, out_shape, times=5, pooling_function='max'):
        super().__init__()
        self.model_name = model_name
        self.times = times
        self.num_classes = num_classes
        self.connections = {}
        self.non_lins = {}
        self.layer_channels = layer_channels 
        self.conn_matrix = conn_matrix
        num_layers = len(conn_matrix)

        if out_shape['0'] == 56:
            self.conv_input = nn.Conv2d(self.layer_channels['inp'], self.layer_channels['0'], 
                                        kernel_size=7, stride=2, padding=3) # 5/2  7/4
            
        elif out_shape['0'] == 112:
            self.conv_input = nn.Conv2d(self.layer_channels['inp'], self.layer_channels['0'], 
                                        kernel_size=5, stride=1, padding=2) 

        self.maxpool_input = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # self.non_lin_input =  nn.ReLU(inplace=True)
        # self.norm_input = nn.GroupNorm(32, self.layer_channels['0'])

        # define all the connections between the layers
        for i in range(num_layers):
            setattr(self, f'output_prenorm_{i}', Identity())
            setattr(self, f'non_lin_{i}', nn.ReLU(inplace=True))
            setattr(self, f'norm_{i}', nn.GroupNorm(32, self.layer_channels[f'{i}']))
            setattr(self, f'output_{i}', Identity())
            for j in range(num_layers):
                if not conn_matrix[i,j]: continue

                # setattr(self, f'norm_{i}_{j}', nn.GroupNorm(32, self.layer_channels[f'{j}']))
                # setattr(self, f'non_lin_{i}_{j}', nn.ReLU(inplace=True))

                # bottom-up or lateral connection
                if i <= j:
                    shape_factor = out_shape[f'{i}'] // out_shape[f'{j}'] 

                    if out_shape[f'{i}'] == 1 or out_shape[f'{j}'] == 1:
                        cnn_kwargs = dict(kernel_size=1, stride=1, padding=0)

                    else:
                        
                        if shape_factor == 1:
                            cnn_kwargs = dict(kernel_size=3, stride=1, padding=1)
                        elif shape_factor == 2:
                            cnn_kwargs = dict(kernel_size=3, stride=1, padding=1)
                        elif shape_factor == 4:
                            cnn_kwargs = dict(kernel_size=5, stride=2, padding=2)
                        elif shape_factor == 8:
                            cnn_kwargs = dict(kernel_size=9, stride=4, padding=4)
                        elif shape_factor == 16:
                            cnn_kwargs = dict(kernel_size=17, stride=8, padding=8)


                    conv =  nn.Conv2d(self.layer_channels[f'{i}'], 
                                      self.layer_channels[f'{j}'], 
                                      **cnn_kwargs)
                    
                    # if shape_factor == 1:
                    #     cnn_kwargs = dict(kernel_size=3, stride=1, padding=1)
                    # elif shape_factor == 2:
                    #     cnn_kwargs = dict(kernel_size=5, stride=2, padding=2)
                    # elif shape_factor == 4:
                    #     cnn_kwargs = dict(kernel_size=9, stride=4, padding=4)
                    # elif shape_factor == 8:
                    #     cnn_kwargs = dict(kernel_size=13, stride=8, padding=6)
                    # elif shape_factor == 16:
                    #     cnn_kwargs = dict(kernel_size=17, stride=16, padding=8)

                    #print(out_shape[f'{i}'], out_shape[f'{j}'], shape_factor)
                                        
                    
                    if shape_factor > 1:
                        # apply avgpooling if the output shape is 1 and the input shape is not 1
                        if out_shape[f'{i}'] != 1 and out_shape[f'{j}'] == 1:
                            conn = nn.Sequential(OrderedDict([
                                ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                                ('flatten', Flatten()),
                                ('linear', nn.Linear(512*4*4, self.layer_channels[f'{j}'])),
                                ('Unsqueeze', Unsqueeze(2)),
                                # ('avgpool', nn.AdaptiveAvgPool2d(1)),
                                # ('conv', conv)
                            ]))

                        else:
                            if pooling_function == 'max':
                                conn = nn.Sequential(OrderedDict([
                                    ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                                    ('conv', conv),
                                ]))
                            elif pooling_function == 'avg':
                                conn = nn.Sequential(OrderedDict([
                                    ('avgpool', nn.AvgPool2d(kernel_size=3, stride=2, padding=1)),
                                    ('conv', conv),
                                ]))
                            elif pooling_function == 'blur':
                                conn = nn.Sequential(OrderedDict([ 
                                    ('blurpool', antialiased_cnns.BlurPool(self.layer_channels[f'{i}'], stride=2)),
                                    ('conv', conv),
                                ]))
                    else:
                        # if out_shape[f'{i}'] == 1 and out_shape[f'{j}'] == 1:
                        #     conn = nn.Sequential(OrderedDict([
                        #         ('flatten', Flatten()),
                        #         ('linear', nn.Linear(self.layer_channels[f'{i}'], self.layer_channels[f'{j}'])),
                        #         ('Unsqueeze', Unsqueeze(2)),
                        #     ]))
                        # else:
                        conn = conv

                    
                    setattr(self, f'conv_{i}_{j}', conn)

                # top-down connections
                elif i > j:

                    shape_factor = out_shape[f'{j}'] // out_shape[f'{i}'] 

                    if shape_factor == 1:
                        #cnn_kwargs = dict(kernel_size=3, stride=1, padding=1, output_padding=1)
                        cnn_kwargs = dict(kernel_size=3, stride=1, padding=1)
                        conn =  nn.Conv2d(self.layer_channels[f'{i}'], 
                                      self.layer_channels[f'{j}'], 
                                      **cnn_kwargs)
                    else: 
                    # output_shape = (input_shape - 1) * stride - 2*padding + kernel_size + output_padding
                        if shape_factor == 2:
                            cnn_kwargs = dict(kernel_size=3, stride=2, padding=1, output_padding=1)
                        elif shape_factor == 4:
                            cnn_kwargs = dict(kernel_size=5, stride=4, padding=1, output_padding=1)
                        elif shape_factor == 8:
                            cnn_kwargs = dict(kernel_size=9, stride=8, padding=1, output_padding=1)
                        elif shape_factor == 16:
                            cnn_kwargs = dict(kernel_size=17, stride=16, padding=1, output_padding=1)

                        conn = nn.ConvTranspose2d(self.layer_channels[f'{i}'],
                                                self.layer_channels[f'{j}'],
                                                **cnn_kwargs)

                    setattr(self, f'conv_{i}_{j}', conn)
                    
        # self.gap = nn.AdaptiveAvgPool2d(1)

        self.read_out = nn.Sequential(OrderedDict([
            ('gap', nn.AdaptiveAvgPool2d(1)),
            ('flatten', Flatten()),
            ('linear', nn.Linear(512, self.num_classes))
        ]))


    def forward(self, inp):
        outputs = {} #'inp': inp
        states = {}
        blocks = list(self.layer_channels.keys()) #['inp', '0', '1', '2', '3', '4', '5']

        inp = self.conv_input(inp)
        inp = self.maxpool_input(inp)
        outputs[blocks[1]] = getattr(self, f'output_{blocks[1]}')(self.non_lin_0(self.norm_0(inp)))
        for block in blocks[2:]:
            outputs[block] = None

        # if the model is b then we don't need to iterate over time steps
        if self.model_name=='b' or self.model_name=='b_pm':
            for block in blocks[2:]:
                in_blocks =  self.conn_matrix[:,int(block)] 
                i = np.where(in_blocks)[0][0]
                input = getattr(self, f'conv_{i}_{block}')(outputs[f'{i}'])
                new_output = getattr(self, f'norm_{block}')(input)
                new_output = getattr(self, f'non_lin_{block}')(new_output)
                new_output = getattr(self, f'output_{block}')(new_output)
                outputs[block] = new_output

            out = outputs[blocks[-1]]
            return self.read_out(out) # make a list to make it cosistent with the recurrent model?

        all_outs = []
        # iterate over time steps
        for t in range(1, self.times):
            new_outputs = {blocks[1]: outputs[blocks[1]]}  # {'0': inp}
            for block in blocks[1:]:

                #output_prev_step = outputs[block]
                
                conn_input = 0
                in_blocks =  self.conn_matrix[:,int(block)] 
                for i in np.where(in_blocks)[0]: 
                    if outputs[f'{i}'] is not None:
                        input = getattr(self, f'conv_{i}_{block}')(outputs[f'{i}'])
                        # input = getattr(self, f'norm_{i}_{block}')(input)
                        # input = getattr(self, f'non_lin_{i}_{block}')(input)
                        conn_input += input

                if block == '0':
                    conn_input += inp
                
                # if output_prev_step is not None:
                #     new_output = output_prev_step + conn_input
                
                if conn_input is not 0:
                    # This will only be False before a layer gets its first input
                    new_output = conn_input
                else:
                    new_output = None

                # apply relu here   
                if new_output is not None:
                    new_output = getattr(self, f'output_prenorm_{block}')(new_output)
                    new_output = getattr(self, f'norm_{block}')(new_output)
                    new_output = getattr(self, f'non_lin_{block}')(new_output)
                    new_output = getattr(self, f'output_{block}')(new_output)

                new_outputs[block] = new_output
                
            outputs = new_outputs
            if outputs[blocks[-1]] is not None:
                all_outs.append(self.read_out(outputs[blocks[-1]]))

        return all_outs


def get_blt_model(model_name, pretrained=False, map_location=None, **kwargs):

    num_layers = kwargs['num_layers']
    img_channels = kwargs['in_channels']
    times = kwargs['times']
    num_classes = kwargs['num_classes']
    pooling_function = kwargs['pooling_function']
    
    if num_layers == 4:
        layer_channels  = {'inp':img_channels, '0':64, '1':128, '2':256, '3':512}
        out_shape  = {'0':56, '1':28, '2':14, '3':7}
        # layer_channels  = {'inp':img_channels, '0':128, '1':384, '2':512, '3':512}
        # out_shape  = {'0':112, '1':56, '2':28, '3':14}

    elif num_layers == 5:
        layer_channels = {'inp':img_channels, '0':64, '1':128, '2':128, '3':256, '4':512}
        out_shape  = {'0':56, '1':28, '2':14, '3':7, '4':7}

    elif num_layers == 6:
        layer_channels = {'inp':img_channels, '0':64, '1':64, '2':128, '3':128, '4':256, '5':512}
        out_shape  = {'0':56, '1':56, '2':28, '3':28, '4':14, '5':7}

        # if we have two linear layer after 4 conv layers
        if 'top2linear' in model_name:
            layer_channels  = {'inp':img_channels, '0':64, '1':128, '2':256, '3':512, '4':1024 , '5':512}
            out_shape  = {'0':56, '1':28, '2':14, '3':7, '4':1, '5':1}

        # we can paramter match a 6 layer b model with a 4 layer bl model (~ 6.5 m)
        if 'b_pm' in model_name:
            layer_channels = {'inp':img_channels, '0':64, '1':128, '2':256, '3':256, '4':512, '5':512}
            out_shape  = {'0':112, '1':56, '2':28, '3':28, '4':14, '5':7}

    elif num_layers == 8:
        # layer_channels = {'inp':img_channels, '0':64, '1':64, '2':128, '3':128, '4':256, '5':512}
        # out_shape  = {'0':56, '1':28, '2':14, '3':14, '4':7, '5':7}
        layer_channels = {'inp':img_channels, '0':64, '1':128, '2':128, '3':256, '4':256, '5':256, '6':256, '7':512}
        out_shape  = {'0':112, '1':56, '2':28, '3':28, '4':7, '5':7, '6':7, '7':7}


    num_layers = len(list(layer_channels.keys())) - 1                         
    conn_matrix = np.zeros((num_layers, num_layers))

    # make the model name only the architecture name
    model_name = model_name.split('_')[0]

    shift = [-1]  # this corresponds to bottom up connections -- always present
    if 'l' in model_name: shift = shift + [0]
    if 'b2' in model_name: shift = shift + [-2]
    if 'b3' in model_name: shift = shift + [-2, -3]
    if 't' in model_name: shift = shift + [1] 
    if 't2' in model_name: shift = shift + [2]
    if 't3' in model_name: shift = shift + [2, 3]
    for i in range(num_layers):
        for j in range(num_layers):
            for s in shift:
                # just add other connections for the last 4 layers
                # if (s != -1) and ((i<(num_layers-4)) or (j<(num_layers-4))): continue 
                if i == (j+s):
                    conn_matrix[i, j] = 1

    model = blt(model_name, conn_matrix, num_classes, layer_channels, out_shape, times, pooling_function)
    return model
