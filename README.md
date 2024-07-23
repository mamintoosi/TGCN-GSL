**Graph Structure Learning for Traffic Prediction**

This repository provides the implementation of the paper "Graph Structure Learning for Traffic Prediction" by MAHMOOD AMINTOOSI.

**Overview**

Accurate traffic forecasting is crucial for smart urban traffic control systems. This repository presents a novel approach that leverages graph structure learning (GSL) to estimate the adjacency matrix of urban roads based on traffic data. By integrating GSL with graph convolutional networks (GCNs), we demonstrate improved traffic forecasting performance compared to traditional proximity-based graph construction methods.

**Key Features**

* Implementation of Graph Structure Learning (GSL) for traffic prediction
* Integration of GSL with Graph Convolutional Networks (GCNs) for traffic forecasting
* Code for experiments conducted on real-world traffic datasets
* Comparison with traditional proximity-based graph construction methods

## Requirements:

* tensorflow, version 1.15
* scipy
* numpy
* matplotlib
* pandas
* math
* dagma
* networkx

## Run the demo

- Jupyter book 'est_adj_dagma' uses [DAGMA](https://github.com/kevinsbello/dagma) for learning graph from the data.
- The methods described in the paper can be run from main.ipynb.

The GCN and GRU models were in gcn.py and gru.py respective.
The T-GCN model was in the tgcn.py

## Data Description

SZ-taxi. This dataset was the taxi trajectory of Shenzhen from Jan. 1 to Jan. 31, 2015. We selected 156 major roads of Luohu District as the study area.

In order to use the model, we need
* A N by N adjacency matrix, which describes the spatial relationship between roads, 
* A N by D feature matrix, which describes the speed change over time on the roads.

In this paper, we set time interval as 15 minutes.

**Acknowledgments**

This repository heavily borrows from and builds upon the excellent work of [T-GCN](https://github.com/lehaifeng/T-GCN) and [DAGMA](https://github.com/kevinsbello/dagma), and we would like to express our gratitude to the authors for making their code publicly available.
