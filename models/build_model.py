from models.cornet import get_cornet_model
from models.blt import get_blt_model
from models.ResNet import ResNet

def build_model(args, pretrained=False, verbose=True):

    args.img_channels = 3
    if 'blt' in args.model:
        # if args.model[4:] == 'b':
        #     kwargs = {'in_channels': args.img_channels, 'times': args.recurrent_steps} #
        # else:
        if not hasattr(args, 'pool'):
            args.pool = 'max'

        kwargs = {'in_channels': args.img_channels, 'times': args.recurrent_steps, \
                  'num_layers': args.num_layers, 'num_classes': args.num_classes, \
                  'pooling_function': args.pool} #
        model = get_blt_model(args.model[4:], pretrained=pretrained, map_location=None, **kwargs) #
        
    elif 'cornet' in args.model:
        if args.model[7:] == 'z' or args.model[7:] == 's':
            kwargs = {'in_channels': args.img_channels, 'num_classes': args.num_classes} #
        elif args.model[7:] == 'r' or args.model[7:] == 'rt':
            kwargs = {'in_channels': args.img_channels, 'times': args.recurrent_steps, \
                      'num_classes': args.num_classes} #

        model = get_cornet_model(args.model[7:], pretrained=pretrained, map_location=None, **kwargs) #
        
    elif 'resnet' in args.model:
        model = ResNet()


    num_parameters =  sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"Number of model parameters: {num_parameters}")
        print(model)
    return model