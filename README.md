# lung_TCGA_prediction
## Installation 
pandas, numpy, matplotlib, scikit-learn, pytorch 1.1.0
## Data
TCGA lung datasets: TCGA-LUAD, TCGA-LUSC

Considering patients who received external beam radiotherapy at primary tumor field as adjuvant.

Select those have both dose and endpoint (local control) information.
## Model 
Adopt a fully-connected 2-hidden layer neural network (NN) to predict local control probability. 
##
