GPU=$1
backbone=$2
DATE=`date +%Y-%m-%d_%H-%M-%S`
action=$3

ckptfolder=$4
data3dtest=/home/lxl/dataset/shape/SceneSBR2018_Dataset/lists/test_pair.txt

if [ $action -eq 1 ]; then
    Data2dRoot=/home/lxl/dataset/shape/SceneIBR2018_Dataset
    Data2dFolder=${Data2dRoot}/Images/Scene
    Data3dTrain=/home/lxl/dataset/shape/SceneSBR2018_Dataset/lists/model_all.txt
    Data2dTrain=${Data2dRoot}/lists/image_train_pair.txt
    Data2dTest=${Data2dRoot}/lists/image_test_pair.txt
    CUDA_VISIBLE_DEVICES=$GPU python sketch2shape_main.py --backbone $backbone --max_epoch 150 --print-freq 10 --margin 1 --w1 1 --w2 0.1 --train_sketch_folder $Data2dFolder --test_sketch_folder $Data2dFolder --train_sketch_flist $Data2dTrain --test_sketch_flist $Data2dTest --train_shape_flist $Data3dTrain --checkpoint_folder $ckptfolder | tee logs/image_${backbone}_${DATE}.txt
fi

if [ $action -eq 2 ]; then 
    Data2dRoot=/home/lxl/dataset/shape/SceneIBR2018_Dataset
    Data2dFolder=${Data2dRoot}/Images/Scene
    Data2dTrain=${Data2dRoot}/lists/image_train_pair.txt
    Data2dTest=${Data2dRoot}/lists/image_test_pair.txt
    CUDA_VISIBLE_DEVICES=$GPU python extract_features.py --backbone $backbone -b 4 --max_epoch 150 --print-freq 10 --margin 1 --w1 1 --w2 0.1 --train_sketch_folder $Data2dFolder --test_sketch_folder $Data2dFolder --test_shape_flist $data3dtest --train_sketch_flist $Data2dTrain --test_sketch_flist $Data2dTest --checkpoint_folder $ckptfolder | tee logs/image_${backbone}_eval_${DATE}.txt
fi
