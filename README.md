## Description
Repository of Propensity Score Synthetic Augmentation Matching using Generative Adversarial Networks (PSSAM-GAN) - Paper accepted at the Journal of Computers in Biology and Medicine.

## Abstract
Understanding causality is of crucial importance in biomedical sciences, where developing prediction models is insufficient because the models need to be actionable. However, data sources, such as electronic health records, are observational and often plagued with various types of biases, e.g. confounding. Although randomized controlled trials are the gold standard to estimate the causal effects of treatment interventions on health outcomes, they are not always possible. Propensity score matching (PSM) is a popular statistical technique for observational data that aims at balancing the characteristics of the population assigned either to a treatment or to a control group, making treatment assignment and outcome independent upon these characteristics. However, matching subjects can reduce the sample size. Inverse probability weighting (IPW) maintains the sample size, but extreme values can lead to instability. While PSM and IPW have been historically used in conjunction with linear regression, machine learning methods –including deep learning with propensity dropout– have been proposed to account for nonlinear treatment assignments. In this work, we propose a novel deep learning approach –the Propensity Score Synthetic Augmentation Matching using Generative Adversarial Networks (PSSAM-GAN)– that aims at keeping the sample size, without IPW, by generating synthetic matches. PSSAM-GAN can be used in conjunction with any other prediction method to estimate treatment effects. Experiments performed on both semi-synthetic (perinatal interventions) and real-world observational data (antibiotic treatments, and job interventions) show that the PSSAM-GAN approach effectively creates balanced datasets, relaxing the weighting/dropout needs for downstream methods, and providing competitive performance in effects estimation as compared to simple GAN and in conjunction with other deep counterfactual learning architectures, e.g. TARNet


## Requirements and versions
pytorch - 1.3.1 <br/>
numpy - 1.17.2 <br/>
pandas - 0.25.1 <br/>
scikit - 0.21.3 <br/>
matplotlib - 3.1.1 <br/>
python -  3.7.4 <br/>

## Citation
@article{ghosh2021propensity,
  title={Propensity Score Synthetic Augmentation Matching using Generative Adversarial Networks (PSSAM-GAN)},
  author={Ghosh, Shantanu and Boucher, Christina and Bian, Jiang and Prosperi, Mattia},
  journal={Computer Methods and Programs in Biomedicine Update},
  pages={100020},
  year={2021},
  publisher={Elsevier}
}
{"mode":"full","isActive":false}


## Keywords
causal AI; causal inference; deep learning; biomedical informatics; generative
adversarial networks; propensity score; treatment effect; electronic health record; big
data

## Contributors
[Shantanu Ghosh](https://www.linkedin.com/in/shantanu-ghosh-b369783a/)

[Christina Boucher](https://christinaboucher.com/)

[Jiang Bian](http://jiangbian.me/)

[Mattia Prosperi](https://epidemiology.phhp.ufl.edu/profile/prosperi-mattia/)

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
