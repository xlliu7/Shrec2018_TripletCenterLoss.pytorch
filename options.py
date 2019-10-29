import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="Triplet center loss for sketch-based-shape retrieval")
    parser.add_argument('-b', '--batch-size', type=int, default=8)
    parser.add_argument('-j', '--workers', type=int, default=0)
    # parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int,
                        help="input height, default: 256 for resnet*, "
                             "144 for inception")
    parser.add_argument('--width', type=int,
                        help="input width, default: 128 for resnet*, "
                             "56 for inception")
    parser.add_argument('--features', type=int, default=4096)
    # parser.add_argument('--dropout', type=float, default=0.5)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.01,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    # parser.add_argument('--momentum', type=float, default=0.9)
    # parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--start_save', type=int, default=15,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--wn', action='store_true', help='weight normalization for centers')
    parser.add_argument('--w1', type=float, default=1, help='weight for classification loss')
    parser.add_argument('--margin', type=float, default=5, help='margin for triplet center loss')
    parser.add_argument('--init', action='store_true', help='initial the norm of centers')
    parser.add_argument('--norm', action='store_true', help='feature normalizations')
    # clamp parameters into a cube
    parser.add_argument('--gradient_clip', type=float, default=0.05) # previous i set it to be 0.01
    #parser.add_argument('--pool-idx', type=int, default=13) # 13 is for alexnet
    parser.add_argument('--pool-idx', type=int)
    # parser.add_argument('--arch', type=str, default='alexnet')
    parser.add_argument('--w2', type=float, default=0.1)
    parser.add_argument('--sf', action='store_true')
    parser.add_argument('--pk-flag', action='store_true')
    parser.add_argument('--num-instances',type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--balance', action='store_true')
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--backbone', choices=['alexnet','vgg11_bn','resnet50','resnet101','vgg13_bn', 'vgg16_bn'],default='alexnet')
    parser.add_argument('--sketch_finetune', action='store_true')
    
    # specify data folders 
    # test and train shapes/sketches reside in the same folder 
    parser.add_argument('--train_shape_views_folder', type=str, default='/home/lxl/dataset/shape/SceneSBR2018_Dataset/Views')
    parser.add_argument('--test_shape_views_folder', type=str, default='/home/lxl/dataset/shape/SceneSBR2018_Dataset/Views')
    parser.add_argument('--train_shape_flist', type=str, default='/home/lxl/dataset/shape/SceneSBR2018_Dataset/lists/model_train_pair.txt')
    parser.add_argument('--test_shape_flist', type=str, default='/home/lxl/dataset/shape/SceneSBR2018_Dataset/lists/test_pair.txt')

    parser.add_argument('--train_sketch_folder', type=str, default='/home/lxl/dataset/shape/SceneSBR2018_Dataset/Sketches')
    parser.add_argument('--test_sketch_folder', type=str, default='/home/lxl/dataset/shape/SceneSBR2018_Dataset/Sketches')
    parser.add_argument('--train_sketch_flist', type=str, default='/home/lxl/dataset/shape/SceneSBR2018_Dataset/lists/sketch_train_pair.txt')
    parser.add_argument('--test_sketch_flist', type=str, default='/home/lxl/dataset/shape/SceneSBR2018_Dataset/lists/sketch_test_pair.txt')
    
    # save folder 
    parser.add_argument('--checkpoint_folder', type=str, default='./models_checkpoint/Sketch')
    return parser.parse_args()
