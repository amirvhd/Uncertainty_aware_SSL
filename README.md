![Logo](https://user-images.githubusercontent.com/65691404/192392997-5dc4b220-38d9-4f1a-b99c-d756eab877a5.png)



# 

This repository is for . It contains the example codes for different task descriptions. 



## Table of contents
* [Badges](#general-information)
* [Installation](#Installation)
* [Datasets](#Datasets)
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
python main_pretrain.py --cosine
``` 
### Linear evaluation

```python
python main_linear.py
``` 

### Uncertainty evaluation 

```python
python uncertainty_metric.py
```
 
### Out of distribtuion detection

```python
python main_execute_method.py
```
 


## Acknowledgements
Base Simclr adapted from following repository:

 - [Simclr](https://github.com/MaartenGr/BERTopic](https://github.com/HobbitLong/SupContrast))

Out of distribution code adapted from following repository:

 - [Out of distribution](https://github.com/kobybibas/pnml_ood_detection)

Metrics for uncertainty analysis are taken from following repository:

 - [Uncertainty_metrics](https://github.com/bicycleman15/KD-calibration/blob/f436583f4458c89971414e972686c55596d5950d/calibration_library/metrics.py)



## Feedback

If you have any feedback, please reach out to us at A.vahidi@campus.lmu.de




