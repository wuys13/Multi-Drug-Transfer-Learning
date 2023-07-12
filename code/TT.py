# nohup python -u drug_ft_hyper_main.py 1>../record/finetune.txt 2>&1 &
import pandas as pd
import torch
import json
import os
import argparse
import random
import pickle
from collections import defaultdict
import itertools
import numpy as np
# https://www.nature.com/articles/s42256-022-00541-0#Abs1

import data
import data_config
import wuys.pdr2.code.train_dsrn_adv as train_dsrn_adv
import fine_tuning
from copy import deepcopy



def generate_encoded_features(encoder, dataloader, normalize_flag=False):
    """

    :param normalize_flag:
    :param encoder:
    :param dataloader:
    :return:
    """
    encoder.eval()
    raw_feature_tensor = dataloader.dataset.tensors[0].cpu()
    label_tensor = dataloader.dataset.tensors[1].cpu()

    encoded_feature_tensor = encoder.cpu()(raw_feature_tensor)
    if normalize_flag:
        encoded_feature_tensor = torch.nn.functional.normalize(encoded_feature_tensor, p=2, dim=1)
    return encoded_feature_tensor, label_tensor


def load_pickle(pickle_file):
    data = []
    with open(pickle_file, 'rb') as f:
        try:
            while True:
                data.append(pickle.load(f))
        except EOFError:
            pass

    return data


def wrap_training_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])

    return aux_dict


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')


def dict_to_str(d):
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])




def main(args, TT):
    print('current Tumor type is {}'.format(TT))

    train_fn = train_dsrn_adv.train_code_adv

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)

    with open(os.path.join('model_save/train_params.json'), 'r') as f:
        training_params = json.load(f)
    if not args.norm_flag:
        method_save_folder = os.path.join('model_save', args.method)
    else:
        method_save_folder = os.path.join('model_save' ,f'{args.method}_norm')
   
    training_params.update(
        {
            'device': device,
            'input_dim': gex_features_df.shape[-1],
            'es_flag': False,
            'retrain_flag': args.retrain_flag,
            'norm_flag': args.norm_flag
        })
    if args.pdtc_flag:
        task_save_folder = os.path.join(f'{method_save_folder}', args.metric, 'pdtc', TT)
    else:
        task_save_folder = os.path.join(f'{method_save_folder}', args.metric, TT)
    safe_make_dir(task_save_folder)

    random.seed(2020)

    s_dataloaders, t_dataloaders = data.get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2020,
        batch_size=training_params['unlabeled']['batch_size'],
        ccle_only=True
    )

    # start unlabeled training
    ##先pretrain code_ae_base，再用adv（GAN） train；pretrain可以没有，train都有
    encoder, historys = train_fn(s_dataloaders=s_dataloaders, #若no-train，就根据training_params中的save_folder找到对应Pretrained_model
                                 t_dataloaders=t_dataloaders, #[0]是train,[1]是test
                                 **wrap_training_params(training_params, type='unlabeled')) 
    if args.retrain_flag:
        with open(os.path.join(training_params['model_save_folder'], f'unlabel_train_history.pickle'),
                  'wb') as f:
            for history in historys:
                pickle.dump(dict(history), f)

    print('start eval.... ')
    ft_evaluation_metrics = defaultdict(list)
    labeled_dataloader_generator = data.get_TT_labeled_generator(  ## need to modify
        gex_features_df=gex_features_df,
        tcga_cancer_type = TT,
        batch_size = 1
)
    fold_count = 0
   
    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, labeled_tcga_dataloader in labeled_dataloader_generator:
        ft_encoder = deepcopy(encoder)
        print(labeled_tcga_dataloader.dataset.tensors[1].sum())
      
        ft_historys = fine_tuning.TT_result(   ## need to modify
            encoder=ft_encoder,
            train_dataloader=train_labeled_ccle_dataloader,
            val_dataloader=test_labeled_ccle_dataloader,
            test_dataloader=labeled_tcga_dataloader,
            seed=fold_count,
            normalize_flag=args.norm_flag,
            metric_name=args.metric,
            task_save_folder=task_save_folder,
            **wrap_training_params(training_params, type='labeled')
        )
        
        ft_evaluation_metrics[args.metric].append(ft_historys[args.metric][0])
                
        fold_count += 1
        
    print('the final metrics are {}'.format(ft_evaluation_metrics) )
    final_report[TT].append(ft_evaluation_metrics[args.metric])
    with open(f'../results/CODEAE_Fig4_{args.metric}', 'w') as f:
        json.dump(final_report, f)        



if __name__ == '__main__':
    parser = argparse.ArgumentParser('CODE-AE reproducing result')
    parser.add_argument('--method', dest='method', nargs='?', default='code_adv',
                        choices=['code_adv', 'dsn', 'dsna','code_base', 'code_mmd', 'adae', 'coral', 'dae', 'vae', 'ae'])
    parser.add_argument('--metric', dest='metric', nargs='?', default='auroc', choices=['auroc', 'auprc'])

    parser.add_argument('--measurement', dest='measurement', nargs='?', default='AUC', choices=['AUC', 'LN_IC50'])
    parser.add_argument('--a_thres', dest='a_thres', nargs='?', type=float, default=None)
    parser.add_argument('--d_thres', dest='days_thres', nargs='?', type=float, default=None)

    parser.add_argument('--n', dest='n', nargs='?', type=int, default=5)

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=False)

    train_group.add_argument('--pdtc', dest='pdtc_flag', action='store_true')
    train_group.add_argument('--no-pdtc', dest='pdtc_flag', action='store_false')
    parser.set_defaults(pdtc_flag=False)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)
    args = parser.parse_args()
    
    params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000],
        "dop": [0.0, 0.1]
    }
    if args.method not in ['code_adv', 'adsn', 'adae', 'dsnw']:
        params_grid.pop('pretrain_num_epochs')

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]



    print(args.pdtc_flag)
    print(args.method)
    TT_num = 0
    for tt in ['ACC',...]:
        final_report = defaultdict(list)
        param_num = 0
        TT_num = TT_num + 1
        print(' ')
        print(' ')
        for param_dict in update_params_dict_list:
            param_num = param_num + 1
            print(' ')
            print('#######      Tumor_type_num: {0}/20 ; Param_num {1}/60       #######'.format(TT_num,param_num))
            print('Drug: {0}; Param_dict: {1}'.format(tt,param_dict) )       
            main(args=args, TT=tt, update_params_dict=param_dict)


