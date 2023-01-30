# AMD_loss
Leveraging angular distributions for improved knowledge distillation

## Overview
This is re-implementation of the AMD loss described in:
Jeon, E. S., Choi, H., Shukla, A., & Turaga, P. (2023). Leveraging angular distributions for improved knowledge distillation. Neurocomputing, 518, 466-481.
https://www.sciencedirect.com/science/article/pii/S0925231222014096

## Requirements
* pytorch>=1.4.0
* python>=3.6.0
* resnet34.pth: 'https://download.pytorch.org/models/resnet34-333f7ec4.pth'

## Image Classification
We use ImageNet classification as an example with a simple architecture. In order to reproduce the results described on the paper, please modify the hyperparameters. The users can also change the data to other dataset at their interest.
To run the code, please download pre-trained resnet34.pth and then run with the model.

## Sample
- global feature distillation:
python3 train_student_imagenet_amd.py -b 256 --gpu 0 --distill amd
- global + local feature distillation:
python3 train_student_imagenet_amd4p.py -b 256 --gpu 0 --distill amd
