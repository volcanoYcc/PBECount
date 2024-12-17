# Probability Map–based Grape Detection and Counting
This repository is the official implementation of our paper: [PBECount: Prompt-Before-Extract Paradigm for Class-Agnostic Counting](https://temp), which is accepted by AAAI2025.

## Project setup
To setup our project on your own device, you need to download all the following files:
1. All the code in this repository, the 'PBECount' folder include the source code of our proposed PBECount method for CAC, and the 'FSC147_384_V2' folder include the code for ground truth data generation.
2. The [FSC147 dataset](https://github.com/cvlab-stonybrook/LearningToCountEverything).
3. Our [pretrained model weights](https://temp).

Then, extract the pretrained model weights in the 'PBECount' folder, and put the images in the FSC147 dataset in the 'FSC147_384_V2' folder, make sure the paths on your own device looks like follows:

<details>
<summary>FSC147_384_V2</summary>
├──images_384_VarV2  
├──images_384_VarV2_probmap  
├──annotation_FSC147_384.json  
├──ImageClasses_FSC147.txt  
├──make_dataset_probmap.py  
└──Train_Test_Val_FSC_147.json  
</details>
<details>
<summary>PBECount</summary>
├──run  
│&nbsp;&nbsp;&nbsp;&nbsp;├──model_paper  
│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├──best_similarity1.pth.tar  
│&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└──best1.pth.tar  
│&nbsp;&nbsp;&nbsp;&nbsp;└──train  
├──data_utils.py  
├──dataloader.py  
├──demo_ui.py  
├──model_init.py  
├──model34.py  
├──train_eval.py  
└──train_utils.py  
</details>
