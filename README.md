# Multi-Drug-Transfer-Learning
Pre-clinical drug discovery (PDD) faces the low efficiency dilemma. One of the reasons is the lack of cross-drug efficacy evaluation infrastructure at the patient level. Here we propose Patient Multi-Drug Learning(P-MDL) task, and construct the P-MDL dataset and model zoo. The best P-MDL model DSN-adv achieve SOTA in all of the 13 tumor types. 


## News!

- [07/12] 🔥The code has been updated in Github!



## Features

### P-MDL task
Artificial intelligence (AI) models used for drug response prediction (DRP) tasks are generally classified into Single-Drug learning (SDL) and Multi-Drug Learning (MDL) paradigms. SDL paradigms have been adapted to the patient level and evaluate within-drug response, disregarding tumor types. However, there exist substantial differences in treatment response and survival outcomes among different tumor types, indicating that tumor type is a crucial confounding factor that can not be overlooked when predicting drug response. Additionally, SDL paradigms fail to assess cross-drug response, while MDL paradigms are currently limited to the cell line level. Therefore, we propose the P-MDL approach, which aims to achieve a comprehensive view of drug response at the patient level.

<p align="center">
    <img src="./png/1_AI_DRP_model.png" alt="DRP paradigms and limitations" width="500">
</p>

<!-- ![DRP paradigms and limitations](./png/1_AI_DRP_model.png){: width="400px" height="300px"} -->

### P-MDL dataset
We constructed the first P-MDL dataset from publicly available data. Tumor types with relatively sufficient data were filtered out. Finally, 13 tumor types were selected for the P-MDL dataset.

<!-- ![P-MDL dataset spanning 13 tumor types](./png/2_P-MDL_dataset.png) -->
<p align="center">
    <img src="./png/2_P-MDL_dataset.png" alt="P-MDL dataset spanning 13 tumor types" width="600">
</p>


### P-MDL model zoo

P-MDL model zoo includes eight models employing different transfer learning methods:
| **P-MDL models**  | **Description** | 
| ------- | ------- | 
| **ae**    |  **Autoencoder** used for encoding both of the gene expression profiles (GEPs) of cell lines and patients.   |
| **ae-mmd** | **ae** model added with another **mmd-loss**. | 
| **ae-adv** | **ae** model added with another **adv-loss**. | 
| **dsn**| **Domain seperation network** has been successfully applied in computer vision. Here, it is used for the GEPs encoding of cell lines and patients. | 
| **dsn-adv**| **dsn** model added with another **adv-loss**. | 
| **dsrn** | An variant of **dsn** model.| 
| **dsrn-mmd** | **dsrn** model added with another **mmd-loss**. | 
| **dsrn-adv**| **dsrn** model added with another **adv-loss**. | 

<!-- - **ae**: **Autoencoder** used for encoding both of the gene expression profiles (GEPs) of cell lines and patients.
- **ae-mmd**: **ae** model added with another **mmd-loss**.
- **ae-adv**: **ae** model added with another **adv-loss**.
- **dsn**: **Domain seperation network** has been successfully applied in computer vision. Here, it is used for the GEPs encoding of cell lines and patients.
- **dsn-mmd**: **dsn** model added with another **mmd-loss**.
- **dsn-adv**: **dsn** model added with another **adv-loss**.
- **dsrn**: An variant of **dsn** model.
- **dsrn-mmd**: **dsrn** model added with another **mmd-loss**.
- **dsrn-adv**: **dsrn** model added with another **adv-loss**. -->

<!-- ![P-MDL model zoo architecture](./png/3_P-DML_model_zoo.png) -->

### Patient Drug Response (PDR) prediction for PDD
To further validate the P-MDL models and demonstrate their potential in PDD applications, the test-pairwise pre-trained DSN-adv model was used to screen 233 small molecules for patients of 13 tumor types.
Take tumor type COAD as an example, most drugs were inefficient, but a few drugs showed potential efficacy for over half of COAD patients.

<!-- ![PDR score and analysis](./png/5_PDR_result.png){: width="200px" height="100px"} -->
<p align="center">
    <img src="./png/5_PDR_result.png" alt="PDR score and analysis" width="400">
</p>


## Installation

To support basic usage of P-MDL task, run the following commands:

```bash
conda create -n P-MDL python=3.8
conda activate P-MDL
conda install -c conda-forge rdkit

pip install torch

```

## Quick Start

Here, we provide a quick example.

This quick example requires installation of an additional package:
```bash
git clone https://github.com/wuys13/Multi-Drug-Transfer-Learning.git
cd code
python setup.py install
cd ..
```

### Step 1: Data Preparation

Download the P-MDL dataset [here](https://zenodo.org/record/8021167).

<!-- ```
mkdir datasets
cd datasets
mkdir dti
mv [your_path_of_davis] ./dti/davis
``` -->

### Step 2: Training and Evaluation

<!-- Run:

```bash
cd ../open_biomed
bash scripts/dti/train_baseline_regression.sh
```

The results will look like the following (running takes around 40 minutes on an NVIDIA A100 GPU):

```bash
INFO - __main__ - MSE: 0.2198, Pearson: 0.8529, Spearman: 0.7031, CI: 0.8927, r_m^2: 0.6928
``` -->

## Contact Us

As a pre-alpha version release, we are looking forward to user feedback to help us improve our framework. If you have any questions or suggestions, please open an issue or contact [wuyushuai@mail.tsinghua.edu.cn](wuyushuai@mail.tsinghua.edu.cn).


## Cite Us

If you find our open-sourced code & models helpful to your research, please consider giving this repo a star🌟 and citing📑 the following article. Thank you for your support!
```
@misc{P_MDL_code,
  author={Yushuai Wu},
  title={Code of Multi-Drug-Transfer-Learning},
  year={2023},
  howpublished={\url{https://github.com/wuys13/Multi-Drug-Transfer-Learning.git}}
}
```

## Contributing

If you encounter problems, feel free to create an issue! 