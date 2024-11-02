# Deep-Probability-Aggregation-Clustering
PyTorch implementation for ECCV 2024 paper： '**Deep Probability Aggregation Clustering**' official code
## Introduction

## Requirement
>-python>=3.7\
>-pytorch>=1.6.0\
>-torchvision>=0.8.1\
>-munkres>=1.1.4\
>-numpy>=1.19.2\
>-cudatoolkit>=11.0

## Usage
>Simply run `python pretrain_step.py` to strat the **contrastive pre-training**.\
>Then run `python clustering_step.py` to deploy **Deep Probability Aggregation Clustering**.\
>We also provide the Python implementation `python probability_aggregation_clustering.py`. for machine clustering algorithm: '**Probability Aggregation Clustering**'.

## Results 
Method (ACC %) | CIFAR-10 | CIFAR-100  | STL-10
---- |---- | ----- | ------  
K-menas + SimCLR | 76.8 | 41.8 | 66.8
PAC + SimCLR | 87.1 | 43.8 | 74.9
DPAC | 93.4 | 55.5 | 93.4 
