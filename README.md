# CHEM

Official implementation of **CHEM: Causally and Hierarchically Explaining Molecules** using PyTorch.

> **NOTE:** This repository supports [WandB](https://wandb.ai/) MLOps platform!

---

### Arguments

|       Argument               | Description                                                                 | Options                                         |
|----------------------|-----------------------------------------------------------------------------|------------------------------------------------------------------|
| `--dataset`         | Dataset to use                                                              | `mutag`, `bbbp`, `bace`, `clintox`, `tox21`, `sider`, `syn`      |
| `--target_col`     | For multi-task dataset, assign a single task.                                      | `clintox`: `0, 1`, `tox21`: `0 ~ 12`, `sider`: `0 ~ 26`  |
| `--bias`          | For synthetic dataset `syn`, assign bias of data.                                    | `0.5`, `0.7`, `0.9`                                                   |

---

## Imputation & Evaluation

Run the following command to perform train and evaluation:

```bash
python main.py --dataset mutag
python main.py --dataset bbbp
python main.py --dataset bace
python main.py --dataset clintox --target_col 0
python main.py --dataset tox21 --target_col 0
python main.py --dataset sider --target_col 0
python main.py --dataset syn --bias 0.5
