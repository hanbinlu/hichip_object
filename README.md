# Overdispersion and Bias Jointly Emulating and Correcting Tool (OBJECT) for HiChIP

## Introduction

HiChIP data contains bias from both Hi-C (distance dependent contact frequencies, <img src="https://latex.codecogs.com/svg.latex?\normal&space;L" />) and ChIP (varing coverages at anchors, <img src="https://latex.codecogs.com/svg.latex?\normal&space;D" />). 
Calling significant loops from HiChIP dataset generally breakdown into two stages (they can be done simultaneously, see below). First, the background models, that correct the above bias, are constructed from the contact data. The models are then employed to estimate expected counts (probabilities) of potential loops (all possible anchor pairs).
Second, significancies of potential loops are quantified via comparing the assumed count distribution (parameterized on expected counts given by the background models) with the observed counts.

## OBJECT
### Overdispersion

We performed mean-variance analyses on published HiChIP data and our newly-developed MID HiChIP data. We found that both datasets were affected by overdispersion, which might violate poisson distribution or binomial distribution that are used in other tools. Therefore, we explored to use zero-inflated poisson (<img src="https://latex.codecogs.com/svg.latex?\normal&space;ZIP"/>) distribution in loop calling.

<p align="center"><img width="220" height="180" src="misc/overdisperse.png"></p>

### General Linear Model (GLM)

To simutaneously correct HiChIP bias as well as overdispersion, we take advantage of the GLM framework. Specificially, background signal, poisson mean <img src="https://latex.codecogs.com/svg.latex?\normal&space;\lambda_i" />, is determined by independent variables, <img src="https://latex.codecogs.com/svg.latex?\normal&space;L_i"/> and <img src="https://latex.codecogs.com/svg.latex?\normal&space;D_i"/> (equation 1).
The expected count (<img src="https://latex.codecogs.com/svg.latex?\normal&space;y_i" title="\large y_i" />) distribution is described by <img src="https://latex.codecogs.com/svg.latex?\normal&space;ZeroInflatedPoisson(\lambda_i,&space;\pi)">, which is combined from the distributions of poisson process and being structual zeros, determined by the parameters <img src="https://latex.codecogs.com/svg.latex?\normal&space;\lambda_i"/> and <img src="https://latex.codecogs.com/svg.latex?\normal&space;\pi"/> respectively.
Fit the data to find optimal solutions (MLE) for the coefficients and <img src="https://latex.codecogs.com/svg.latex?\normal&space;\pi"/>. 

<p align="center"><img src="https://latex.codecogs.com/svg.latex?ln(\lambda_i)&space;=&space;\beta_0&space;&plus;&space;\beta_1ln(L_i)&space;&plus;&space;\beta_2ln(D_i)" /></p>
<p align="center"><img src="https://latex.codecogs.com/svg.latex?y_i&space;\sim&space;ZeroInflatedPoisson(\lambda_i,&space;\pi)" /></p>

## Installation
Clone the repo and run `pip` to install
```
pip install hichip_obj
```
