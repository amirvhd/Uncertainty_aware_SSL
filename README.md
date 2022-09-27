![Logo](https://user-images.githubusercontent.com/65691404/192393358-8170f550-638f-4fbd-933e-e538cc9fdb7a.png)





# 

This repository is for . It contains the example codes for different task descriptions. 



## Table of contents
* [Badges](#general-information)
* [Installation](#Installation)
* [Usage/Examples](#Usage/Examples)
* [Acknowledgements](#Acknowledgements)
* [Feedback](#Feedback)

## Badges

Add badges from somewhere like: [shields.io](https://shields.io/)

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)](https://opensource.org/licenses/)
[![AGPL License](https://img.shields.io/badge/license-AGPL-blue.svg)](http://www.gnu.org/licenses/agpl-3.0)

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

```python
python main_pretrain.py --cosine --nh 10 --dataset cifar10 --lamda1 1 --lamda2 0.08 --epoch 800
``` 
### Linear evaluation

```python
python main_linear.py --nh 10 --dataset cifar10 --lamda1 1 --lamda2 0.08
``` 

### Uncertainty evaluation 

```python
python uncertainty_metric.py --nh 10 --dataset cifar10 --lamda1 1 --lamda2 0.08
```
 
### Out of distribtuion detection

```python
python main_execute_method.py --nh 10 --dataset cifar10 --lamda1 1 --lamda2 0.08
```
 


## Acknowledgements
Base Simclr adapted from following repository:

 - [Simclr](https://github.com/HobbitLong/SupContrast)

Out of distribution code adapted from following repository:

 - [Out of distribution](https://github.com/kobybibas/pnml_ood_detection)

Metrics for uncertainty analysis are taken from following repository:

 - [Uncertainty_metrics](https://github.com/bicycleman15/KD-calibration/blob/f436583f4458c89971414e972686c55596d5950d/calibration_library/metrics.py)






