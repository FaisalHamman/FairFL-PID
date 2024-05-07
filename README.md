<<<<<<< HEAD
# FairFL-PID
=======
# Demystifying Local & Global Fairness Trade-offs in Federated Learning


This repository contains the code accompanying the paper ["Demystifying Local & Global Fairness Trade-offs in Federated Learning Using Partial Information Decomposition"](https://arxiv.org/abs/2307.11333) by [Faisal Hamman](https://www.faisalhamman.com/) and [Sanghamitra Dutta](https://sites.google.com/site/sanghamitraweb/), presented at ICLR 2024.


Our research investigates the trade-offs between local and global fairness in federated learning environments through the lens of Partial Information Decomposition (PID). 



## Experiments

### Experiment A: Accuracy-Global-Local-Fairness Trade-off Pareto Front
- **Objective**: To study the trade-offs between model accuracy and different fairness constraints.
- **Files**: Run `Adult-tradeoff.ipynb` for the Adult dataset and `Synthetic-tradeoff.ipynb` for synthetic data analysis.

### Experiment B: Demonstrating Disparities in Federated Learning Settings
- **Objective**: Investigate the PID of disparities on the Adult dataset trained within a federated learning (FL) framework using the FedAvg algorithm (McMahan et al., 2017).
- **Files**: Code implementations are located in the `FedAvg` directory.


To run the experiments, ensure your environment is properly set up by installing the required packages:

```bash
conda env create -f environment.yml
```

Ensure your Federated Learnining environment is configured correctly by referring to `config.yaml` for detailed settings.

Run for experiment B:
```bash
python main.py
```


## Acknowledgments

The implementation of the FedAvg algorithm in this project was adapted from [Federated Learning in PyTorch](https://github.com/vaseline555/Federated-Learning-in-PyTorch)
and [Federated Averaging PyTorch](https://github.com/ijgit/Federated-Averaging-PyTorch/tree/main). The trade-off analysis code was adapted from [FACT](https://github.com/wnstlr/FACT).

Please consider citing our work:
```bibtex
@inproceedings{
hamman2024demystifying,
title={Demystifying Local \& Global Fairness Trade-offs in Federated Learning Using Partial Information Decomposition},
author={Faisal Hamman and Sanghamitra Dutta},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=SBj2Qdhgew}
}
```

>>>>>>> b80b16d (Initial commit of FairFL-PID project)
