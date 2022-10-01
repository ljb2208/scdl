import argparse

def getPredictionArgs():

    # Prediction settings
    parser = argparse.ArgumentParser(description='SCDL Prediction args')
    parser.add_argument('--maxdisp', type=int, default=192, 
                        help="max disp")
    parser.add_argument('--crop_height', type=int, default=288, 
                        help="crop height")
    parser.add_argument('--crop_width', type=int, default=576, 
                        help="crop width")
    parser.add_argument('--resume', type=str, default='/home/lbarnett/development/scdl/run/kitti15/best/checkpoints/best.pth', 
                        help="resume from saved model")
    parser.add_argument('--cuda', type=int, default=1, 
                        help='use cuda? Default=True')
    parser.add_argument('--threads', type=int, default=1, 
                        help='number of threads for data loader to use')
    parser.add_argument('--seed', type=int, default=2019, 
                        help='random seed to use. Default=123')
    parser.add_argument('--shift', type=int, default=0, 
                        help='random shift of left image. Default=0')
    parser.add_argument('--save_path', type=str, default='/home/lbarnett/development/scdl/run/kitti15/best/predictions/', 
                        help="location to save predictions")    
    parser.add_argument('--stage', type=str, default='train', choices=['search', 'train'])
    parser.add_argument('--dataset', type=str, default='kitti15', 
                        choices=['sceneflow', 'kitti15', 'kitti12', 'middlebury'], help='dataset name')





    ######### LEStereo params ##################
    parser.add_argument('--fea_num_layers', type=int, default=6)
    parser.add_argument('--mat_num_layers', type=int, default=12)
    parser.add_argument('--fea_filter_multiplier', type=int, default=8)
    parser.add_argument('--mat_filter_multiplier', type=int, default=8)
    parser.add_argument('--fea_block_multiplier', type=int, default=4)
    parser.add_argument('--mat_block_multiplier', type=int, default=4)
    parser.add_argument('--fea_step', type=int, default=3)
    parser.add_argument('--mat_step', type=int, default=3)
    parser.add_argument('--net_arch_fea', default='/home/lbarnett/development/scdl/run/kitti15/best/architecture/feature_network_path.npy', type=str)
    parser.add_argument('--cell_arch_fea', default='/home/lbarnett/development/scdl/run/kitti15/best/architecture/feature_genotype.npy', type=str)
    parser.add_argument('--net_arch_mat', default='/home/lbarnett/development/scdl/run/kitti15/best/architecture/matching_network_path.npy', type=str)
    parser.add_argument('--cell_arch_mat', default='/home/lbarnett/development/scdl/run/kitti15/best/architecture/matching_genotype.npy', type=str)

    args = parser.parse_args()
    return args


