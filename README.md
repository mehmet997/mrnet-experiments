# MRNet replication

This repository reproduces (to some extent) the results proposed in the paper ["Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet"](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699) by Bien, Rajpurkar et al.

## Data

Data must be downloaded from MRNet's [official website](https://stanfordmlgroup.github.io/competitions/mrnet/) and put anywhere into your machine. Then, edit the  ``` train_mrnet.sh``` script file by expliciting the full path to ```MRNet-v1.0``` directory into the ```DATA_PATH``` variable.

## Execution
To run an experiment just run
```
bash train_mrnet.sh
```

This will train three models for each view (sagittal, axial, coronal) of each task (acl tear recogntion, meniscal tear recognition, abnormalities recognition), for a total of 9 models. After that, a logistic regression is trained to combine the predictions of the different view model for each task.
All checkpoints, training and validation logs, and results will be saved inside the ```experiment``` folder (it will be created if it doesn't exists).
 
Training and evaluation code is based on PyTorch and scikit-learn.