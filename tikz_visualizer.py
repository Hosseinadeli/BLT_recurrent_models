from torch import nn
from tikz import *
# ^ pip install git+https://github.com/allefeld/pytikz

from models.blt import blt as BLTNet

def abbrev_num(num: int) -> str:
    if num >= 1_000_000:
        return f"{num / 1_000_000:.3g}M"
    elif num >= 10_000:
        return f"{num // 1_000}k"
    elif num >= 1_000:
        return f"{num / 1_000:.2g}k"
    else:
        return str(num)

class TikzVisualizerBase():
    def __init__(self, layer_names, add_input_output=True, scale=2):
        self.pic = Picture(thick=True, auto=True, node_distance="2cm")
        self.model_layer_names = layer_names
        self.add_input_output = add_input_output

        # Setup pic styles
        self.pic.style(
            "layerstyle",
            rectangle=True,
            rounded_corners="15pt",
            fill="white",
            draw="black",
            line_width="1pt",
            minimum_width="1cm",
            minimum_height="1cm",
            text_centered=True,
            font=r"{\Large \bfseries \sffamily}",
            inner_xsep="1em",  # L/R padding
            outer_sep="0",
            scale=scale,  # for clarity
        )

        self.pic.style(
            "graybg",
            fill="lightgray"
        )

        self.pic.style(
            "conn",
            opt="->, >=stealth",  # customize arrowhead (filled in > vs open ->)
            line_width="3pt",
            draw=True,
            inner_sep="10pt"
        )

        self.pic.style(
            "code",
            font=r'{\Large \tt}',
        )

        self.pic.definecolor("ff1", "rgb", "0, 0, 0")  # fading black
        self.pic.definecolor("ff2", "rgb", "0.5, 0.5, 0.5")
        self.pic.definecolor("ff3", "rgb", "0.75, 0.75, 0.75")
        self.pic.definecolor("fb1", "rgb", "0.8, 0.2, 0.2")  # fading red
        self.pic.definecolor("fb2", "rgb", "0.9, 0.6, 0.6")
        self.pic.definecolor("fb3", "rgb", "0.95, 0.8, 0.8")
        self.pic.definecolor("rec", "rgb", "0.1, 0.5, 0.8")  # blue

        self.layer_names = layer_names.copy()
        if add_input_output:
            self.layer_names.insert(0, "Input")
            self.layer_names.append("Readout")

    def get_picture(self) -> Picture:
        return self.pic
    
    def _repr_mimebundle_(self, include, exclude, **kwargs):
        """Display the picture in Jupyter notebooks"""
        return self.pic._repr_mimebundle_(include, exclude, **kwargs) 

    def _model_layer_index_to_pic_index(self, model_layer_index: int):
        return model_layer_index + 1 if self.add_input_output else model_layer_index

    def _model_layer_index_to_node_id(self, model_layer_index: int):
        return self.layer_node_ids[self._model_layer_index_to_pic_index(model_layer_index)]

    def _connection_to_type(self, post_layer_index: int, pre_layer_index: int):
        diff = post_layer_index - pre_layer_index  # positive if feedforward, negative if feedback
        if diff > 0:
            return f"ff{min(diff, 3)}"
        elif diff < 0:
            return f"fb{min(-diff, 3)}"
        else:
            return "rec"



class TikzModelVisualizer(TikzVisualizerBase):
    def __init__(self, layer_names, add_input_output=True, scale=2):
        super().__init__(layer_names, add_input_output, scale)
        
        self.layer_node_ids = [f"{layer}" for layer in self.layer_names]  # TODO: idk if any custom id is necessary; this seems to work fine
        
        # Add layers as nodes in graph
        last_layer_id = None
        for i in range(len(self.layer_names)-1, -1, -1):
            layer = self.layer_names[i]
            layer_id = self.layer_node_ids[i]
            opt = "layerstyle"
            if layer == "Input" or layer == "Readout":
                opt += ", graybg"
            self.pic.node(contents=f"{layer}", name=layer_id, opt=opt, below_of=(None if last_layer_id is None else last_layer_id))
            last_layer_id = layer_id

        # For keeping track of annotations
        self.layer_text_annotation_nodes = {}  # layer_index: int -> (num_annos: int, node_name: str)
    
    def draw_layer_connection(self, post_layer_index: int, pre_layer_index: int, has_xshift=True, shiftpt=10):
        """Draw a connection between two layers in the model

        Args:
            post_layer_index (int): Model index of the receiving layer
            pre_layer_index (int): Model index of the sending layer
            has_xshift (bool, optional): _description_. Defaults to True.
            shiftpt (int, optional): _description_. Defaults to 10.
        """
        preshiftpt = [0, 0]
        postshiftpt = [0, 0]
        arrow_params = ""
        
        pre_node_id = self._model_layer_index_to_node_id(pre_layer_index)
        post_node_id = self._model_layer_index_to_node_id(post_layer_index)
        type = self._connection_to_type(post_layer_index, pre_layer_index)

        if type[:2] == "ff":
            # Feedforward connection, draw up
            if has_xshift and type == "ff1":
                # FF1 connections shifted slightly left
                pre_node_id = f"{pre_node_id}.north"
                post_node_id = f"{post_node_id}.south"
                preshiftpt[0] -= shiftpt
                postshiftpt[0] -= shiftpt
        elif type[:2] == "fb":
            # Feedback connection, draw down
            if has_xshift and type == "fb1":
                # FF1 connections shifted slightly right
                pre_node_id = f"{pre_node_id}.south"
                post_node_id = f"{post_node_id}.north"
                preshiftpt[0] += shiftpt
                postshiftpt[0] += shiftpt
        elif type == "rec":
            # Recurrent connection
            arrow_params = "[loop right, looseness=4]"

        if type[2] == "2":
            arrow_params = "[bend left=50]"
        elif type[2] == "3":
            arrow_params = "[bend left=60]"

        if preshiftpt[0] != 0 or preshiftpt[1] != 0:
            pre_node_id = f"({pre_node_id}) ++ ({preshiftpt[0]}pt, {preshiftpt[1]}pt)"
        else:
            pre_node_id = f"({pre_node_id})"
        
        if postshiftpt[0] != 0 or postshiftpt[1] != 0:
            # in f-strings, the {{ is a single {
            post_node_id = f"([shift={{ ( {postshiftpt[0]}pt, {postshiftpt[1]}pt) }}] {post_node_id})"
        else:
            post_node_id = f"({post_node_id})"

        self.pic.draw(f"{pre_node_id} to{arrow_params} {post_node_id}", opt="conn", color=type)
    
    def annotate_layer_text(self, layer_index: int, text: str, **kwargs):
        if layer_index in self.layer_text_annotation_nodes:
            num_annos, last_node_name = self.layer_text_annotation_nodes[layer_index]
            node_kwargs = dict(
                at = f"({last_node_name}.base west)",
                anchor = "north west",
            )
        else:
            num_annos = 0
            layer_node_id = self._model_layer_index_to_node_id(layer_index)
            node_kwargs = dict(
                at = f"({layer_node_id}.north)",
                xshift = "15em",
                yshift = "-1em",  # move it a bit down
                anchor = "west",
            )

        node_name = f"anno-{layer_index}-{num_annos}"
        self.layer_text_annotation_nodes[layer_index] = (num_annos+1, node_name)

        if "scale" not in kwargs:
            kwargs["scale"] = 1.1

        self.pic.node(
            contents = text,
            name = node_name,
            opt = "code",
            **node_kwargs,
            **kwargs
        )
    
    def annotate_layer_param(self, layer_index: int, param: nn.Module, post_pre: tuple[int] = None, show_param_count=True, abbrev_param_counts=True, **kwargs):
        if isinstance(param, (nn.Conv2d, nn.ConvTranspose2d)):
            conn_str = "Conv"
            is_transpose = isinstance(param, nn.ConvTranspose2d)
            if is_transpose:
                conn_str += "$^\\top$"
            conn_str += f"(ch={param.in_channels}$\\to${param.out_channels}, ker={param.kernel_size[0]}$\\times${param.kernel_size[1]}, strd={param.stride[0]}, pad={param.padding[0]}"
            if is_transpose:
                conn_str += f", outpad={param.output_padding[0]}"
            conn_str += ")"
        elif isinstance(param, nn.Linear):
            conn_str = f"Linear(feat={param.in_features}$\\to${param.out_features})"
        else:
            conn_str = str(param).replace("_", " ")  # tikz can't handle underscores

        text = ""

        if post_pre is not None:
            conn_type = self._connection_to_type(*post_pre)
            pre_layer_name = self.layer_names[self._model_layer_index_to_pic_index(post_pre[1])]
            text += f"{pre_layer_name} {conn_type.upper()}: "
            kwargs["color"] = conn_type

        text += conn_str

        if show_param_count:
            num_params = sum(p.numel() for p in param.parameters())
            if abbrev_param_counts:
                num_params = abbrev_num(num_params)
            else:
                num_params = f"n={num_params:,}"
            text += f"  [{num_params}]"
        
        self.annotate_layer_text(layer_index, text, **kwargs)
    

    from typing import Union, Tuple
    def annotate_layer_shape(self, layer_index: int, shape: Union[str, int, Tuple[int]], **kwargs):
    #def annotate_layer_shape(self, layer_index: int, shape: str | int | tuple[int], **kwargs):
        layer_node_id = self._model_layer_index_to_node_id(layer_index)

        if isinstance(shape, tuple):
            shape_prod = 1
            for dim in shape: shape_prod *= dim
            if len(shape) == 3:
                # assume channels x height x width
                shape = f"{shape[1]}$\\times${shape[2]} [$\\times${shape[0]}{{\\normalsize ch}}]"
            else:
                shape = r"$\times$".join(shape)
        elif isinstance(shape, int):
            shape = str(shape)
            shape_prod = None

        if shape_prod is not None:
            shape += r"\\" + f"{{\\normalsize ({shape_prod:,} units)}}"

        self.pic.node(
            contents = shape,
            name = f"shape-{layer_node_id}",
            at = f"({layer_node_id})",
            anchor = "east",
            opt = "code",
            xshift = kwargs.pop("xshift", "-15em"),
            scale = kwargs.pop("scale", 1.4),
            align = "right",
            **kwargs
        )


def visualize_blt(model: BLTNet) -> TikzModelVisualizer:
    layer_shape_scale = 1.75
    layer_param_scale = 1.15
    pic = TikzModelVisualizer(model.layer_names)

    for i in range(-1, model.num_layers+1):
        if i == -1:
            # Input layer
            shape = (model.input_channels, model.input_size, model.input_size)

            # Model details
            details = [
                f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
                f"Timesteps: {model.times}"
            ]
            for text in details:
                pic.annotate_layer_text(i, text, scale=1.75)
        elif i == model.num_layers:
            # Readout layer
            shape = model.num_classes

            # Readout connection
            pic.draw_layer_connection(i, i-1, has_xshift=False)

            # Readout param
            pic.annotate_layer_param(i, model.read_out[-1], post_pre=(i, i-1), scale=layer_param_scale)
        else:
            # Model layer
            shape = (model.layer_channels[str(i)], model.out_shape[str(i)], model.out_shape[str(i)])

            for pre in model.layers_inputting_to[i]:
                # Layer connections
                # xshift only for FF1/FB1 connections where there is also a connection in the different direction
                has_xshift = abs(pre-i) == 1 and i in model.layers_inputting_to[pre]
                pic.draw_layer_connection(i, pre, has_xshift=has_xshift)

                # Layer params
                conv = getattr(model, f"conv_{pre}_{i}")
                pic.annotate_layer_param(i, conv, post_pre=(i, pre), scale=layer_param_scale)
        
        pic.annotate_layer_shape(i, shape, scale=layer_shape_scale)

    return pic



class TikzComputationGraphVisualizer(TikzVisualizerBase):
    def __init__(self, model: BLTNet, **kwargs):
        super().__init__(model.layer_names, **kwargs)
        self.num_timesteps = model.times

        xshiftcm = 2.3  # needs to be big enough so that nodes don't overlap

        # layer_node_ids[timestep][node_id]
        layer_node_ids = []

        # Annotate the nodes
        num_layers_at_t_prev = None
        for t in range(self.num_timesteps):
            if num_layers_at_t_prev is None:
                num_layers_at_t = 2  # at the first timestep show input and V1
            else:
                # At subsequent time steps, show all layers that receive input form the previous ones
                for layer_i in range(model.num_layers):
                    if (num_layers_at_t_prev-2) in model.layers_inputting_to[layer_i]:
                        num_layers_at_t = layer_i+2

                if num_layers_at_t == len(self.layer_names)-1:
                    # If info reaches the final layer, add the readout node
                    num_layers_at_t += 1
            
            num_layers_at_t_prev = num_layers_at_t
            
            layer_node_ids.append([f"layer_{layer-1}_time_{t}" for layer in range(num_layers_at_t)])  # layer-1 so input is -1
            
            for layer_index in range(num_layers_at_t):
                layer_name = self.layer_names[layer_index]
                node_id = layer_node_ids[t][layer_index]
                node_kwargs = {}
                if layer_index == 0:
                    if t > 0:
                        node_kwargs["at"] = f"({layer_node_ids[t-1][0]}.east)"
                        node_kwargs["xshift"] = f"{xshiftcm}cm"
                else:
                    node_kwargs["above_of"] = layer_node_ids[t][layer_index-1]
                    # node_kwargs["xshift"] = f"{xshiftcm}cm"
                
                opt = "layerstyle"
                if layer_name == "Input":  # or layer_name == "Readout":
                    opt += ", graybg"

                self.pic.node(f"{layer_name}", name=node_id, opt=opt, **node_kwargs)

                if layer_index == 0:
                    # Add text below
                    self.pic.node(
                        contents = f"$t = {t}$",
                        name = f"label_time_{t}",
                        at = f"({node_id}.south)",
                        anchor = "north",
                        # opt = "code",
                        yshift = "-0.5cm",
                        scale = 3,
                        # align = "right",
                    )
                
                if t == self.num_timesteps - 1 and layer_index == len(self.layer_names)-1:
                    # Final prediction node
                    final_pred_node = "final_pred_node"
                    self.pic.node(
                        contents = f"Final Prediction",
                        name = final_pred_node,
                        opt = "layerstyle, graybg",
                        above_of = node_id
                    )
        
        # Now annotate the connections
        def draw_conn(pre_layer, pre_node, post_layer, post_node, upward=False):
            arrow_params = ""
            conn_type = "ff1" if upward else self._connection_to_type(post_layer, pre_layer)
            diff = post_layer - pre_layer
            diffpt = -10*diff
            if upward:
                pre_node = f"({pre_node}.north)"
                post_node = f"({post_node}.south)"
            elif conn_type.startswith("ff") or conn_type.startswith("fb"):
                pre_node = f"([shift={{ ( 0pt, {5*diff}pt) }}] {pre_node}.east)"
                post_node = f"([shift={{ ( 0pt, {-10*diff}pt) }}] {post_node}.west)"  # order so FF1 is above FF2 (diff > 0)
            elif conn_type == "rec":
                pre_node = f"({pre_node}.east)"
                post_node = f"({post_node}.west)"
            self.pic.draw(f"{pre_node} to{arrow_params} {post_node}", opt="conn", color=conn_type)

        for t in range(self.num_timesteps):
            for post_layer, post_node_id in enumerate(layer_node_ids[t]):
                if post_layer == 0:
                    continue  # external input doesn't get any input...

                if post_layer == len(self.layer_names)-1:
                    # Readout layer; connect to previous layer
                    draw_conn(post_layer-1, layer_node_ids[t][post_layer-1], post_layer, post_node_id, upward=True)
                    continue

                # Otherwise intermediate layer
                # Note we have to add [-1] for the first layer because the first layer receives input
                for pre_layer_model_indexing in model.layers_inputting_to[post_layer-1] + ([-1] if post_layer == 1 else []):
                    pre_layer = pre_layer_model_indexing + 1
                    
                    if pre_layer_model_indexing == -1:
                        # This layer receives input
                        draw_conn(pre_layer, layer_node_ids[t][pre_layer], post_layer, post_node_id, upward=True)
                        pass
                    else:
                        # this connection needs to come from the previous time step
                        # validate that this is possible (i.e., pre node exists at prev. time step)
                        if t > 0 and pre_layer < len(layer_node_ids[t-1]):
                            draw_conn(pre_layer, layer_node_ids[t-1][pre_layer], post_layer, post_node_id)
        
        # add connection to final prediction
        draw_conn(len(self.layer_names)-1, layer_node_ids[self.num_timesteps-1][len(self.layer_names)-1], len(self.layer_names), final_pred_node, upward=True)



"""
This does the half slant rather than full 45 degree slant

class TikzComputationGraphVisualizer(TikzVisualizerBase):
    def __init__(self, layer_names, num_timesteps: int, **kwargs):
        super().__init__(layer_names, **kwargs)
        self.num_timesteps = num_timesteps

        # self.node_ids[layer_index][timestep]
        self.node_ids = [
            [
                f"layer_{layer_name}_time_{t}"
                for t in range(model.num_timesteps)
            ]
            for layer_name in self.layer_names
        ]

        xshiftcm = 0.4

        for t in range(num_timesteps):
            for layer_index, layer_name in enumerate(self.layer_names):
                node_kwargs = {}
                if layer_index == 0:
                    if t > 0:
                        node_kwargs["right_of"] = self.node_ids[0][t-1]
                        node_kwargs["xshift"] = f"{xshiftcm*(len(layer_names)+3)}cm"
                else:
                    node_kwargs["above_of"] = self.node_ids[layer_index-1][t]
                    node_kwargs["xshift"] = f"{xshiftcm}cm"
                
                node_id = self.node_ids[layer_index][t]
                self.pic.node(
                    f"{layer_name}",
                    name=node_id,
                    opt="layerstyle",
                    **node_kwargs
                )

                if layer_index == 0:
                    # Add text below
                    self.pic.node(
                        contents = f"$t = {t}$",
                        name = f"shape_for_time_{t}",
                        at = f"({node_id}.south)",
                        anchor = "north",
                        # opt = "code",
                        yshift = "-0.5cm",
                        scale = 3,
                        # align = "right",
                    )
"""