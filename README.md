## Description
Repository of Propensity Score Synthetic Augmentation Matchingusing Generative Adversarial Networks (PSSAM-GAN) - Paper under review at AAAI 2021.

## Introduction
Propensity Score Synthetic Augmentation Matching using Generative Adversarial Net- works (PSSAM-GAN) - a novel deep learning training algorithm which attempts to make an unbalanced dataset bal- anced to train DCN-PD and TARNET to estimate ITE.

## Block Diagram
<embed src="https://github.com/Shantanu48114860/PSSAM-GAN/blob/master/Figures/Block.pdf" type="application/pdf">



## Requirements and versions
pytorch - 1.3.1 <br/>
numpy - 1.17.2 <br/>
pandas - 0.25.1 <br/>
scikit - 0.21.3 <br/>
matplotlib - 3.1.1 <br/>
python -  3.7.4 <br/>


## Keywords
causal AI, biomedical informatics, deep learning, multitask learning, GAN

## Dependencies
[python 3.7.7](https://www.python.org/downloads/release/python-374/)

[pytorch 1.3.1](https://pytorch.org/get-started/previous-versions/)


## How to run
To reproduce the experiments mentioned in the paper for IHDP dataset
command: <br/>
<b>
  cd IHDP <br/>
  python3 main_PM_GAN.py
</b>

To reproduce the experiments mentioned in the paper for Jobs dataset
command:<br/>
<b>
  cd Jobs <br/>
  python3 main_PM_GAN.py
</b>

By default it will run for 1000 and 10 iterations for IHDP and Jobs dataset respectively.

## Results
The default results mentioned in the paper is avaliable at the following locations:

<b> [IHDP](https://github.com/Shantanu48114860/PSSAM-GAN/tree/master/Stats/IHDP_Random/1000_iter) </b> 

<b> [Jobs](https://github.com/Shantanu48114860/PSSAM-GAN/tree/master/Stats/Jobs/--%3E%3EBest!!90_val__80-20_split_Early_stopping_Tarnet_elu_GAN_10000)</b>

## Hyperparameters
The codebase is setup with the default hyperparameters depicted in the paper. However, if one wishes to change the hyperparameters, please visit the following files for IHDP and Jobs respectively:

<b> [IHDP](https://github.com/Shantanu48114860/PSSAM-GAN/blob/master/IHDP/Constants.py) </b> 

<b> [Jobs](https://github.com/Shantanu48114860/PSSAM-GAN/blob/master/Jobs/Constants.py)</b>

## Output
After the run, the outputs will be generated in the following location:

<b>[IHDP](https://github.com/Shantanu48114860/PSSAM-GAN/tree/master/IHDP/MSE) </b>

<b>[Jobs](https://github.com/Shantanu48114860/PSSAM-GAN/tree/master/Jobs/MSE) </b>

Consolidated results will be available in textfile in /IHDP/Details_original.txt and /Jobs/Details_original.txt files.

The details of each run will be avalable in csv files in the following locations:

1) IHDP - /IHDP/MSE/Results_consolidated_NN.csv

2) Jobs - /Jobs/MSE/Results_consolidated_NN.csv

## Plots
The plots for each run will be found at the following location:

1) IHDP - /IHDP/Plots

2) Jobs - /Jobs/Plots

## License & copyright

Licensed under the [MIT License](LICENSE)
