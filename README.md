# Probability Map–based Grape Detection and Counting
This repository is the official implementation of our paper: [PBECount: Prompt-Before-Extract Paradigm for Class-Agnostic Counting](https://temp), which is accepted by AAAI2025.

## Project setup
1. To setup our project on your own device, you need to download all the following files:  
(1). All the code in this repository, the 'PBECount' folder include the source code of our proposed PBECount method for CAC, and the 'FSC147_384_V2' folder include the code for ground truth data generation.  
(2). The [FSC147 dataset](https://github.com/cvlab-stonybrook/LearningToCountEverything).  
(3). Our [pretrained model weights](https://temp).

2. Then, extract the pretrained model weights in the 'PBECount' folder, and put the images in the FSC147 dataset in the 'FSC147_384_V2' folder, make sure the paths on your own device looks like follows:

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

## Evaluation


