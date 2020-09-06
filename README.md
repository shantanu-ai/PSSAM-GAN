## Description
Repository of Propensity Score Synthetic Augmentation Matchingusing Generative Adversarial Networks (PSSAM-GAN)

## Requirements and setup
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

## License & copyright
© DISL, University of Florida

Licensed under the [MIT License](LICENSE)
