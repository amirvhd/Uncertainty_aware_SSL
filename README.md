![Logo](https://blog.rwth-aachen.de/forschungsdaten/files/2019/09/BERD@NFDI-Logo.png)


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
This code is adapted based on the following links:

 - [Topic modeling](https://github.com/MaartenGr/BERTopic)
 - [Multi-label-Text-classification](https://curiousily.com/posts/multi-label-text-classification-with-bert-and-pytorch-lightning/)
 - [Image Segmentation](https://colab.research.google.com/github/qubvel/segmentation_models.pytorch/blob/master/examples/binary_segmentation_intro.ipynb#scrollTo=PeGCIYNlVx5y)



## Feedback

If you have any feedback, please reach out to us at A.vahidi@campus.lmu.de


## Acknowledgements
This code is adapted based on the following repositories:

 - [Simclr](https://github.com/HobbitLong/SupContrast)
 - [Out of disstribution detection](https://github.com/kobybibas/pnml_ood_detection)
 - [Laplace](https://github.com/AlexImmer/Laplace)




## Feedback

If you have any feedback, please reach out to us at A.vahidi@campus.lmu.de

