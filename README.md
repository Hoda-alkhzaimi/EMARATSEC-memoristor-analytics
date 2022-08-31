# MR-PUF Instructions

This is a repository for the instructions for the MR-PUF. Below there is a description of data as well as instructions for how to run the analysis code. If you have feeback, please use github issues or open a pull request, further improvement is appreciated. In the future we hope to expand the dataset and add more models.

## Table of Contents

- [MR-PUF Instructions](#mr-puf-instructions)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Data](#data)
  - [Running the Analysis](#running-the-analysis)

## Introduction

Physical unclonable functions (PUFs) have emerged as a favorable hardware security primitive, they exploit the process variations to provide unique signatures or secret keys amidst other critical cryptographic applications. CMOS based PUFs are the most popular type, they generate unique bit strings using process variations in semiconductor fabrication. However, most existing CMOS PUFs are considered vulnerable to modeling attacks based on machine learning (ML) algorithms. Memristors leveraging nanotechnology fabrication process and highly nonlinear behavior became an interesting alternative to the existing PUF CMOS technology introducing cryptographic and resilient randomness outputs. Memristor-based PUFs are emerging due to the inherent randomness at both the memristor level, due to the C2C programming variation of the device and the fabrication process level such as the cross-sectional area and variations. Our study focuses on building a machine learning analysis and attack framework of tools on our $Cu/HfO_{2-x}/p^{++}Si$  memristor-based PUF (MR-PUF). Our objective is  to test the resiliency of the security margins of the presented PUF using machine learning analysis tools, on-top of holistic NIST cryptographic randomness testing initially provided, with the goal to provide high-level of certainty in predicting the randomness output of the verified memristor-based PUF.  The main contribution of this work is the holistic study that focuses on attacking the randomness output resiliency based on building randomness predictors using logistic regression (LR), support vector machine (SVM), Gaussian mixture models (GMM), K-means , K-means++, random forest and optimized distributed gradient boosting model (XGBoost) within efficient time, and data complexity. Our results yield low accuracy and ROC results of within $0.56-0.44$ and $0.53-0.49$ respectively which indicates failure in predicting random data and demonstrates efficient randomness prediction resiliency of our PUF. The efficient time and data complexities of these attacks are illustrated in this study are yielded to be linear and quadratic resulting in attack execution time in seconds and 857 training samples combined with 423 testing samples to verify the randomness of our PUF.

The code in this repository is divided between **/data** and **/src** folders. The data folder contains the raw data and the src folder contains the code for processing and analysis.
Matlab portion of the code is used to process raw sets, however for convenience we provided already processed sets to be used with the analysis code.

## Data

Located in **/data** folder, you will find `processed_set` folder with the following files: `set9.mat`, `set10.mat`, `set11.mat`, `set12.mat`, `set17.mat`, `set18.mat`, `set25.mat`, `set26.mat`, `set27.mat`, `set28.mat`,  `set33.mat`. These contain extrated data from the raw data files. For ease of access you can open them with python and use the `loadmat` function to load the data like so:

```python
from scipy import io
set27 = io.loadmat('set27.mat')
```

`set27['V']` will give you all voltage values in a cycle and `set27['C']` will give you all current values, `set27['Z']` contains **!approximate** binary values and in general should not be used for prediction.

`truebinary.txt` contains the true binary values for the data set. This is the set you want to use for your predictions. Each line there corresponds to a set in `processed_set` folder. The order of the lines in this file is the same as the order of the sets in `processed_set` folder.

## Running the Analysis

To run the analysis, it is advised you create a virtual enviroenemnt in python, next you will need to install the dependencies:

```python
pip install -r requirements.txt
```

Then run the following command:

```python
python analysis.py
```

To see all the plots we recommend using jupyter lab. `cd` into project directory first.  Next, you can run the following command to start jupyter lab:

```python
jupyter-lab
```
