# Triplet Center Loss (TCL) Solution for SHREC2018 IBR & SBR
This repo holds the code for our method _Triple Center Loss(TCL)_ at the [SHREC 2018](http://www2.projects.science.uu.nl/shrec/index2018-cfparticipation.html) challenges:
- [Sketch-Based 3D Scene Retrieval(SBR)](http://orca.st.usm.edu/~bli/SceneSBR2018/)
- [Image-Based 3D Scene Retrieval(IBR)](http://orca.st.usm.edu/~bli/SceneIBR2018/).

The TCL method is based on the CVPR 2018 work
> Xinwei He, Yang Zhou, Zhichao Zhou, Song Bai, Xiang Bai. **Triplet-Center Loss for Multi-View 3D Object Retrieval**. _CVPR 2018_. [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/He_Triplet-Center_Loss_for_CVPR_2018_paper.pdf)

For a detailed description of our submitted method, you can also refer to the technical report and find our method scription in Sec 4.2.
> SHREC’18 Track: 2D Image-Based 3D Scene Retrieval. [[pdf]](http://orca.st.usm.edu/~bli/SceneIBR2018/SHREC18_Track_2D_Scene_Image-Based_3D_Scene_Retrieval.pdf) 

> SHREC’18 Track: 2D Scene Sketch-Based 3D Scene Retrieval. [[pdf]](http://orca.st.usm.edu/~bli/SceneSBR2018/SHREC18_Track_2D_Scene_Sketch-Based_3D_Scene_Retrieval.pdf)

## Prerequisites
Our code has been tested with Python2 + PyTorch 0.3. It should work with higher versions after minor modifications.

## Usage
The code works for both the IBR and the SBR tasks.

For the IBR task, run the following command
```
bash run_image_based.sh $gpu $backbone $action $suffix $output_dir
```
Arguments:
- gpu: the GPU ID
- backbone: the backbone architecture to use. e.g. vgg11_bn, resnet50.
- action: 1 or 2. choose **1 for training** and **2 for inference**.
- output_dir: where to save the outputs

For the SBR task, use the script `run_sketch_based.sh` instead. The arguments are the same.

## Related Projects
- [cvpr_2018_TCL.pytorch](https://github.com/eriche2016/cvpr_2018_TCL.pytorch)
