# linearlinearmodel
These Jupyter Notebook and MATLAB codes have been prepared as a part of research publication entitled:

Deep Learning for Quantification of Basilar Artery Morphology Using Intracranial Vessel Wall MRI: A Feasibility Study

Authors: Chien-Hung Tsou, Hon-Man Liu, Adam Huang

# How to run the Jupyter Notebook example

1. Download the Jupyter Notebook file and images (image.png and mask.png)

2. Try the Jupyter Notebook file on Google Colab platform (with both image.png and mask.png)

# How to run the MATLAB example code

1. Download all MATLAB code files and images (image.png and mask.png)

2. Run "main.m"

# function [M1]=findVesselLumen(I,M,num)

% input:

%   I: cross sectional vessel image

%   M: outer vessel wall boundary mask

%   num: number of probing directions

% ouput:

%   M1: vessel lumen (inner wall) mask

% Initial Submission Date: 2024/11/28
