GPU=$1
DATE=`date +%Y-%m-%d_%H-%M-%S`
backbone=$2

action=$3

ckptfolder=$4
dataroot=/home/lxl/dataset/shape/SceneSBR2018_Dataset
data2dtrain=$dataroot/lists/sketch_all.txt
data3dtrain=$dataroot/lists/model_all.txt
data3dtest=/home/lxl/dataset/shape/SceneSBR2018_Dataset/lists/test_pair.txt

# standard train
if [ $action -eq 1 ]; then 
    CUDA_VISIBLE_DEVICES=${GPU} python sketch2shape_main.py --checkpoint_folder $ckptfolder --train_shape_flist $data3dtrain --train_sketch_flist $data2dtrain --backbone $backbone --margin 1 --w1 1 --w2 0.1  --print-freq 10 --max_epochs 150 | tee logs/sketch_${backbone}_${DATE}.txt
fi 

# standard test
if [ $action -eq 2 ]; then 
    CUDA_VISIBLE_DEVICES=${GPU} python extract_features.py --checkpoint_folder $ckptfolder --backbone $backbone --margin 1 --w1 1 --w2 0.1 --test_shape_flist $data3dtest --print-freq 10 --max_epochs 150 | tee logs/sketch_${backbone}_eval_${DATE}.txt
fi

