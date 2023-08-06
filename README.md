




# 

This repository is for "Diversified Ensemble of Independent Sub-Networks for
Robust Self-Supervised Representation Learning" paper. It contains the example codes for different task descriptions. 



## Table of contents
* [Badges](#general-information)
* [Installation](#Installation)
* [Usage/Examples](#Usage/Examples)
* [Acknowledgements](#Acknowledgements)

## Badges

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt)

### Dependency

![pytorch_lightning](https://img.shields.io/badge/Pytorch_lightning-1.5.10-brightgreen)
![torch](https://img.shields.io/badge/Torch-1.10.1-brightgreen)

## Installation

Install requirments:
```python
pip install -r requirements.txt
```


## Usage/Examples


### Pretraining

You need to specify the path for dataset and also saved_models for the following codes.  the You can run the pretraining of model for CIFAR-10 for UA-SSL with following code.

```python
python main_pretrain.py --cosine --nh 10 --dataset cifar10 --lamda1 1 --lamda2 0.08 --epoch 800
``` 
### Linear evaluation

You can run the linear-evaluation of the model for CIFAR-10 for UA-SSL with the following code. To get the results for semi-supervised, you need to use --semi and also specify what percccentage of data to be used for linear evaaluation for example --semi_percent 10 means it uses 10 percent of data for linear evaluation.

```python
python main_linear.py --nh 10 --dataset cifar10 --lamda1 1 --lamda2 0.08
``` 

### Uncertainty evaluation 

You can calculate different metrics (e.g. accuracy, NLL, ECE, OE, ...) for the model with the following code. You should specify paths for the pretrained and linear evaluation models.

```python
python uncertainty_metric.py --nh 10 --dataset cifar10 --lamda1 1 --lamda2 0.08
```
 
### Out of distribtuion detection

You can calculate results of out of distribution detection for CIFAR-10 with the followingg code. You should give the path for datasets and also paths for pretrained and linear evaluation models.

```python
python main_execute_method.py --nh 10 --dataset cifar10 --lamda1 1 --lamda2 0.08
```
To get the results for covariant shift (corrupted datasets) you should add --c to the code.


## Acknowledgements
Base Simclr adapted from following repository:

 - [Simclr](https://github.com/HobbitLong/SupContrast)

Out of distribution code adapted from following repository:

 - [Out of distribution](https://github.com/kobybibas/pnml_ood_detection)

Metrics for uncertainty analysis are taken from following repository:

 - [Uncertainty_metrics](https://github.com/bicycleman15/KD-calibration/blob/f436583f4458c89971414e972686c55596d5950d/calibration_library/metrics.py)






