# PBECount: Prompt-Before-Extract Paradigm for Class-Agnostic Counting
This repository is the official implementation of our paper: [PBECount: Prompt-Before-Extract Paradigm for Class-Agnostic Counting](https://temp), which is accepted by AAAI2025.

<img src="https://github.com/volcanoYcc/PBECount/raw/master/README_images/1.png" width="390px" /><img src="https://github.com/volcanoYcc/PBECount/raw/master/README_images/2.png" width="390px" />  
<details>
<summary>click to show detection results</summary>
<img src="https://github.com/volcanoYcc/PBECount/raw/master/README_images/3.png" width="750px" />
</details>

## Project setup
1. To setup our project on your own device, you need to download all the following files:  
(1). All the code in this repository, the 'PBECount' folder include the source code of our proposed PBECount method for CAC, and the 'FSC147_384_V2' folder include the code for ground truth data generation.  
(2). The [FSC147 dataset](https://github.com/cvlab-stonybrook/LearningToCountEverything).  
(3). Our [pretrained model weights](https://pan.baidu.com/s/1mzpNd8hXpy6xrg0XBMYROg)(code:7mw5).

2. Extract the pretrained model weights in the 'PBECount' folder, and put the images in the FSC147 dataset into the 'FSC147_384_V2' folder, make sure the paths on your own device looks like follows:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;FSC147_384_V2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──images_384_VarV2  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──images_384_VarV2_probmap  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──annotation_FSC147_384.json  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──ImageClasses_FSC147.txt  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──make_dataset_probmap.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──Train_Test_Val_FSC_147.json  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;PBECount  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──run  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├──model_paper  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├──best_similarity1.pth.tar  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└──best1.pth.tar  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└──train  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──data_utils.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──dataloader.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──demo_ui.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──model_init.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──model34.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├──train_eval.py  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└──train_utils.py  

3. Run 'make_dataset_probmap.py' in the 'FSC147_384_V2' folder to generate ground truth probmaps in the 'FSC147_384_V2/images_384_VarV2_probmap' folder.

## Quick Demo
We provide a quick demo to check the detection performance of model weight file 'PBECount/run/model_paper/best_similarity1.pth.tar' with the best classification performence, which is obtained by changing the 'crop_aug' parameter in 'PBECount/train_eval.py' to 'False' in training.
1. Run 'PBECount/demo_ui.py'.
2. Select the target image.
3. Draw exemplars on the target image, press the left mouse button to start draw, release the left mouse button to finish draw, press 'Enter' to detect and show the detect result.
4. Press 'ESC' or click the close button of the window to stop.

## Evaluation
1. Change the 'config' parameter in 'PBECount/train_eval.py' to 'config_eval'.
2. Run 'PBECount/train_eval.py' to test the model weight file 'PBECount/run/model_paper/best1.pth.tar' with the best counting performence.

## Training
1. Change the 'config' parameter in 'PBECount/train_eval.py' to 'config_train_stage1'.
2. Run 'PBECount/train_eval.py' to train the model for stage one.
3. Change the 'config' parameter in 'PBECount/train_eval.py' to 'config_train_stage2', switch the 'pre_trained' parameter in 'config_train_stage2' to the path to the model weight file with the best counting performance in stage one training process.
4. Run 'PBECount/train_eval.py' to train the model for stage two.

