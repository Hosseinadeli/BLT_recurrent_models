from argparse import ArgumentParser

def get_argparser():
    parser = ArgumentParser(description="BLT network training")

    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--output_path', default='./results/', type=str,
                        help='path for storing ')
    
    parser.add_argument('--model', choices=['blt_b', 'blt_b_pm', 'blt_b_top2linear', 'blt_b2', 
                                            'blt_b3', 'blt_bl', 'blt_bl_top2linear', 'blt_b2l',
                                            'blt_b3l', 'blt_bt', 'blt_b2t', 'blt_b3t', 'blt_blt',
                                            'blt_b2lt', 'blt_b3lt', 'blt_bt2', 'blt_b2t2', 
                                            'blt_b3t2', 'blt_blt2', 'blt_b2lt2', 'blt_b3lt2',
                                            'blt_bt3', 'blt_b2t3', 'blt_b3t3', 'blt_blt3', 
                                            'blt_b2lt3', 'blt_b3lt3', 
                                            'cornet_z', 'cornet_s', 'cornet_r', 'cornet_rt'], 
                        default='blt_bl', type=str)
    
    parser.add_argument('--objective', choices=['classification', 'contrastive'], default='classification', help='which model to train')
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adamw", help="optimizer to use")
    parser.add_argument('--loss_choice', choices=['weighted', 'decay'], default='decay', type=str, help='how to apply loss to earlier readout layers')  
    parser.add_argument('--loss_gamma', default=.5, type=float, help='whether to have loss in earlier steps of the readout')

    parser.add_argument('--recurrent_steps', default=10, type=int, help='number of time steps to run the model (only R model)')
    parser.add_argument('--num_layers', default=4, type=int, help='number of layers in the network')
    parser.add_argument('--seed', default=0, type=int, help='seed for initializing training')

    parser.add_argument('--num_workers', default=4, type=int, help='number of data loading num_workers')
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=.0005, type=float, help='initial learning rate')
    parser.add_argument("--lr_momentum", default=0.9, type=float, help="momentum for SGD optimizer")
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--lr_drop', default=100, type=int)
    parser.add_argument('--lr_step_gamma', default=0.5, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    parser.add_argument('--evaluate', action='store_true', help='just evaluate')
    
    parser.add_argument('--wandb_p', default=None, type=str)
    parser.add_argument('--wandb_r', default=None, type=str)

    # dataset parameters
    parser.add_argument('--dataset', choices=['imagenet', 'vggface2', 'bfm_ids', 'NSD',
                                              'imagenet_vggface2', 'imagenet_face']
                        , default='imagenet', type=str)
    parser.add_argument('--data_path', default='/share/data/imagenet-pytorch',
                         type=str, help='path to ImageNet folder')
    parser.add_argument('--image_size', default=224, type=int, 
                        help='what size should the image be resized to?')
    parser.add_argument('--horizontal_flip', default=False, type=bool, help='whether to use horizontal flip augmentation')
    parser.add_argument('--run', default="1", type=str, help="run identifier")
    
    parser.add_argument('--img_channels', default=3, type=int,
                    help="what should the image channels be (not what it is)?") #gray scale 1 / color 3
    
    parser.add_argument('--save_model', default=0, type=int) 

    parser.add_argument('--distributed', default=1, type=int, help='whether to use distributed training')
    parser.add_argument('--port', default='12382', type=str, help='MASTER_PORT for torch.distributed')

    parser.add_argument("--debug", action="store_true", help="debug and only one batch for training")

    return parser    
