# Causal Discovery from Poisson Branching Structural Causal Model Using High-Order Cumulant with Path Analysis

The Python implementation of paper [Causal Discovery from Poisson Branching Structural Causal Model Using High-Order Cumulant with Path Analysis](https://arxiv.org/abs/2403.16523). (AAAI 2024)

# Usage

The running example of PBSCM is given below.

```python
from PB_SCM import PB_SCM
from util import *

param_dict = {
    "n": 10,
    "seed": 2024,
    "in_degree_rate": 3.0,
    "sample_size": 30000,
    "alpha_range_str": "0.1,0.5",
    "mu_range_str": "1,3",
}

data, edge_mat, alpha_mat, mu = data_generate(**param_dict)

model = PB_SCM(data, seed=2024)
skeleton = model.Hill_Climb_search()

causal_graph = learning_causal_direction(data, skeleton)
```



# Real World Experiment

The real world experiment is implemented in the script `foot_ball_event_causal_discovery.py`. The dataset used for this experiment, `event.csv`, can be found on Kaggle: [Football Events Dataset](https://www.kaggle.com/datasets/secareanualin/football-events?select=events.csv).

To run the real experiment, make sure to download the dataset and place `event.csv` in the appropriate directory. Additionally, we provide a processed version of the data: `foot_ball_event_table.csv`.



# Requirements

The requirements are given in `requirements.txt`. You can install them using the following command: 

```shell
pip install -r requirements.txt
```



# Citation

If you find this useful for your research, we would appreciate it if you could cite the following papers:

```
@inproceedings{qiao2024causal,
  title={Causal Discovery from Poisson Branching Structural Causal Model Using High-Order Cumulant with Path Analysis},
  author={Qiao, Jie and Xiang, Yu and Chen, Zhengming and Cai, Ruichu and Hao, Zhifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={18},
  pages={20524--20531},
  year={2024}
}
```