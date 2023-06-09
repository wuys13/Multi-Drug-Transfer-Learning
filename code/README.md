# CODE-AE


This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].


[cc-by-nc]: http://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## CONTEXT-AWARE DECONFOUNDING AUTOENCODER
-----------------------------------------------------------------

## 1. Introduction
**CODE-AE** is a Python implementation of Context-aware Deconfounding Autoencoder (CODE-AE) that can extract common biological signals masked by context-specific patterns and confounding factors.

## 2. Installation

**CODE-AE** depends on Numpy, SciPy, PyTorch (CUDA toolkit if use GPU), scikit-learn, pandas. 
You must have them installed before using **CODE-AE**.

The simple way to install them is using Docker: the Dockerfile is given within folder ``code/``

## 3. Usage

### 3.1. Data

Benchmark datasets available at Zenodo http://doi.org/10.5281/zenodo.4776448  (version 2.0)

### 3.2 Basic Usage 
#### 3.2.1 Pre-train encoders
```sh
    $ python pretrain_hyper_main.py
```
Arguments in this script:
* ``--method``:       method to be used for encoder pre-training, choose from \['code_adv', 'dsna', 'dsn', 'code_base', 'code_mmd', 'adae', 'coral', 'dae', 'vae','vaen', 'ae'\]
* ``--train``:        retrain the encoders
* ``--no-train``:     no re-training
* ``--norm``:        use L2 normalization for embedding
* ``--no-norm``:     don't use L2 normalization for embedding

#### 3.2.2 Fine-tune encoders for different drug response predictive models
```sh
    $ python drug_ft_hyper_main.py
```
Arguments in this script:
* ``--method``:       method to be used for encoder pre-training, choose from \['code_adv', 'dsna', 'dsn', 'code_base', 'code_mmd', 'adae', 'coral', 'dae', 'vae','vaen', 'ae'\]
* ``--train``:        retrain the encoders
* ``--no-train``:     no re-training
* ``--norm``:        use L2 normalization for embedding
* ``--no-norm``:     don't use L2 normalization for embedding
* ``--metric``:     metric used in early stopping
*  ``--pdtc``:  fine-tune PDTC drugs (target therapy agents)
*  ``--no-pdtc``:  fine-tune chemotherapy drugs

#### 3.2.3 Reproduce the final results for TCGA drugs ( Fig4 in the paper)
```sh 
    $ python reproduce_fig4.py
    please use the default settings to reproduce the results.
    
### Additional scripts
- adae_hyper_main.py: run ADAE de-confounding experiments
- drug_inference_main.py: generate TCGA sample prediction
- generate_encoded_features.py: generate encoded features by pre-trained encoders
- ml_baseline.py: get benchmark experiments' results using random forest and elastic net
- mlp_main.py: get benchmark experiments' results using simple multi-layer neural network
- tcrp_main.py: get benchmark experiments' results using TCRP
- vaen_main.py: get benchmark experiments' results using VAEN (VAE+elastic net)
