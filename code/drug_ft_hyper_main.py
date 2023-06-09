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
import wuys.pdr2.code.train_dsrn as train_dsrn
import wuys.pdr2.code.train_ae_adv as train_ae_adv
import wuys.pdr2.code.train_dsrn_adv as train_dsrn_adv
import train_coral
import train_dae
import train_vae
import train_ae
import wuys.pdr2.code.train_dsrn_mmd as train_dsrn_mmd
import wuys.pdr2.code.train_dsn_mmd as train_dsn_mmd
import wuys.pdr2.code.train_dsn_adnn as train_dsn_adnn

import fine_tuning
import ml_baseline
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


def main(args, drug, update_params_dict):
    if args.method == 'dsn':
        train_fn = train_dsn_mmd.train_dsn
    elif args.method == 'adae':
        train_fn = train_ae_adv.train_adae
    elif args.method == 'coral':
        train_fn = train_coral.train_coral
    elif args.method == 'dae':
        train_fn = train_dae.train_dae
    elif args.method == 'vae':
        train_fn = train_vae.train_vae
    elif args.method == 'ae':
        train_fn = train_ae.train_ae
    elif args.method == 'code_mmd': 
        train_fn = train_dsrn_mmd.train_code_mmd
    elif args.method == 'code_base':
        train_fn = train_dsrn.train_code_base
    elif args.method == 'dsna':
        train_fn = train_dsn_adnn.train_dsna
    else:
        train_fn = train_dsrn_adv.train_code_adv
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    #for unlabel training
    if args.source_dataset == 'tcga':
        gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)  #11114TCGA_sample * 1427HVG(overlap)
    elif args.source_dataset == 'brca':    
        gex_features_df = pd.read_csv(data_config.brca_feature_file, index_col=0)
    elif args.source_dataset == 'brca_pseudo_random':    
        gex_features_df = pd.read_csv(data_config.brca_pseudo_random_feature_file, index_col=0)
    elif args.source_dataset == 'brca_pseudo_pam50':    
        gex_features_df = pd.read_csv(data_config.brca_pseudo_pam50_feature_file, index_col=0)
    elif args.source_dataset == 'tcga_pseudo_random':    
        gex_features_df = pd.read_csv(data_config.tcga_pseudo_random_feature_file, index_col=0)
    elif args.source_dataset == 'tcga_pseudo_TT':    
        gex_features_df = pd.read_csv(data_config.tcga_pseudo_TT_feature_file, index_col=0)

    # #for label training
    # all_gex_feature_df = pd.read_csv(data_config.gex_feature_file, index_col=0)  #11114TCGA_sample * 1427HVG(overlap)


    with open(os.path.join('model_save/train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params['unlabeled'].update(update_params_dict)   ##unlabeled一开始就更新成dict的队列了，labeled一直没有更新，都是json读进来的2000
    param_str = dict_to_str(update_params_dict)  #给这一版的文件命名，从dict转为str

    if not args.norm_flag:
        method_save_folder = os.path.join('model_save', args.method)
    else:
        method_save_folder = os.path.join('model_save', f'{args.method}_norm')

    training_params.update(
        {
            'device': device,
            'input_dim': gex_features_df.shape[-1],
            'model_save_folder': os.path.join(method_save_folder, param_str),
            'es_flag': False,
            'retrain_flag': args.retrain_flag,
            'norm_flag': args.norm_flag
        })
    if args.pdtc_flag:
        task_save_folder = os.path.join(f'{method_save_folder}', args.measurement, 'pdtc', drug)
    else:
        task_save_folder = os.path.join(f'{method_save_folder}', args.measurement, drug)

    safe_make_dir(training_params['model_save_folder'])
    safe_make_dir(task_save_folder) #unlabel result save dir

    random.seed(2020)

    s_dataloaders, t_dataloaders = data.get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2020,
        batch_size=training_params['unlabeled']['batch_size'],
        ccle_only=False
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

    # labeled_ccle_dataloader, labeled_tcga_dataloader = data.get_labeled_dataloaders(
    #     gex_features_df=gex_features_df,
    #     seed=2020,
    #     batch_size=training_params['labeled']['batch_size'],
    #     drug=drug,
    #     threshold=args.a_thres,
    #     days_threshold=args.days_thres,
    #     ccle_measurement=args.measurement,
    #     ft_flag=False,
    #     pdtc_flag=args.pdtc_flag
    # )
    # ml_baseline_history = defaultdict(list)
    # ccle_encoded_feature_tensor, ccle_label_tensor = generate_encoded_features(encoder, labeled_ccle_dataloader,
    #                                                                            normalize_flag=args.norm_flag)
    # tcga_encoded_feature_tensor, tcga_label_tensor = generate_encoded_features(encoder, labeled_tcga_dataloader,
    #                                                                            normalize_flag=args.norm_flag)
    # ml_baseline_history['enet'].append(
    #     ml_baseline.n_time_cv(
    #         model_fn=ml_baseline.classify_with_enet,
    #         n=int(args.n),
    #         train_data=(
    #             ccle_encoded_feature_tensor.detach().cpu().numpy(),
    #             ccle_label_tensor.detach().cpu().numpy()
    #         ),
    #         test_data=(
    #             tcga_encoded_feature_tensor.detach().cpu().numpy(),
    #             tcga_label_tensor.detach().cpu().numpy()
    #         ),
    #         metric=args.metric
    #     )[1]
    # )
    #
    # with open(os.path.join(task_save_folder, f'{param_str}_ft_baseline_results.json'), 'w') as f:
    #     json.dump(ml_baseline_history, f)

    ft_evaluation_metrics = defaultdict(list)

    if args.single_drug_flag:
        labeled_dataloader_generator = data.get_labeled_dataloader_generator(
            gex_features_df=gex_features_df,
            seed=2020,
            batch_size=training_params['labeled']['batch_size'],
            drug=drug,
            ccle_measurement=args.measurement,
            threshold=args.a_thres,
            days_threshold=args.days_thres,
            pdtc_flag=args.pdtc_flag,
            n_splits=args.n)
        fold_count = 0
        # 5折开始，但用的是z-score作为均匀划分？：每个药物对于所有cell line做Normalize
        for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, labeled_tcga_dataloader in labeled_dataloader_generator:
            ft_encoder = deepcopy(encoder)
            print('Fold count = {}'.format(fold_count))
            print('CCLE_train : Pos_num = {0} ; Dataset_num = {1}'.format(
                    train_labeled_ccle_dataloader.dataset.tensors[1].sum(), #根据AUC的均值在划分？
                    train_labeled_ccle_dataloader.dataset.tensors[1].size(0)
            )) #label有多少个正的
            print('CCLE_test : Pos_num = {0} ; Dataset_num = {1}'.format(
                    test_labeled_ccle_dataloader.dataset.tensors[1].sum(),
                    test_labeled_ccle_dataloader.dataset.tensors[1].size(0)
            ))
            print('TCGA_test : Pos_num = {0} ; Dataset_num = {1}'.format(
                labeled_tcga_dataloader.dataset.tensors[1].sum(),
                labeled_tcga_dataloader.dataset.tensors[1].size(0)
            ))

            target_classifier, ft_historys = fine_tuning.fine_tune_encoder( #有时候要这个finetune的模型，故返回target_classifier
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
            ft_evaluation_metrics['best_index'].append(ft_historys[-2]['best_index'])
            for metric in ['auroc', 'acc', 'aps', 'f1', 'auprc']:
                ft_evaluation_metrics[metric].append(ft_historys[-1][metric][ft_historys[-2]['best_index']])
            fold_count += 1
    else:  ##多个药之间互相比
        labeled_dataloader_generator = data.get_finetune_dataloader_generator(
            gex_features_df = gex_features_df,
            seed=2020,
            batch_size=training_params['labeled']['batch_size'],
            dataset = 'gdsc_raw',
            ccle_measurement=args.measurement,
            pdtc_flag=args.pdtc_flag,
            n_splits=args.n,
            q=2,
            tumor_type = args.tumor_type)
        fold_count = 0
        # 5折开始，但用的是z-score作为均匀划分？：每个药物对于所有cell line做Normalize
        for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, labeled_tcga_dataloader in labeled_dataloader_generator:
            ft_encoder = deepcopy(encoder)
            print('Fold count = {}'.format(fold_count))
            
            target_classifier, ft_historys = fine_tuning.fine_tune_encoder_drug( #有时候要这个finetune的模型，故返回target_classifier
                encoder=ft_encoder,
                train_dataloader=train_labeled_ccle_dataloader,
                val_dataloader=test_labeled_ccle_dataloader,
                test_dataloader=labeled_tcga_dataloader,
                seed=fold_count,
                normalize_flag=args.norm_flag,
                metric_name=args.metric,
                task_save_folder=task_save_folder,
                drug_emb_dim=128,
                **wrap_training_params(training_params, type='labeled')
            )
            ft_evaluation_metrics['best_index'].append(ft_historys[-2]['best_index'])
            for metric in ['auroc', 'acc', 'aps', 'f1', 'auprc']:
                ft_evaluation_metrics[metric].append(ft_historys[-1][metric][ft_historys[-2]['best_index']])
            fold_count += 1
 
    with open(os.path.join(task_save_folder, f'{param_str}_ft_evaluation_results.json'), 'w') as f:
        json.dump(ft_evaluation_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADSN training and evaluation')
    parser.add_argument('--num',default = None)
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
    parser.set_defaults(retrain_flag=True)
    # parser.set_defaults(retrain_flag=False)


    train_group.add_argument('--pdtc', dest='pdtc_flag', action='store_true')
    train_group.add_argument('--no-pdtc', dest='pdtc_flag', action='store_false')
    parser.set_defaults(pdtc_flag=False)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    single_drug_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--single', dest='single_drug_flag', action='store_true')
    norm_group.add_argument('--no-single', dest='single_drug_flag', action='store_false')
    parser.set_defaults(single_drug_flag=False)

    parser.add_argument('--source_dataset',default = "tcga",
        choices=["tcga", 'brca', 'brca_pseudo_random','brca_pseudo_pam50','tcag_pseudo_random','tcag_pseudo_TT'])
    parser.add_argument('--tumor_type',default = None,
        choices=[None,"GBM", 'LGG', 'PAAD','HNSC','LIHC','KIRC','SARC','PRAD','OV',
        'BRCA','STAD','CESC','SKCM','BLCA','LUSC','LUAD','UCEC','READ','COAD']) #19 PFS >10


    args = parser.parse_args()

    params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000],
        "dop": [0.0, 0.1]
    }
    # params_grid = {
    #     "pretrain_num_epochs": [0,100],  # encoder、decoder
    #     "train_num_epochs": [100,300],   # GAN
    #     "dop":  [0.0,0.1]
    # }   

    if args.num :
        Num = int(args.num)
        print(Num)
        args.method = ['code_adv', 'dsn', 'dsna','code_base', 'code_mmd', 'adae', 'coral', 'dae', 'vae', 'ae'][Num]
        print(args.method)
   
    if args.method not in ['code_adv', 'adsn', 'adae', 'dsnw']:
        params_grid.pop('pretrain_num_epochs')

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    if args.pdtc_flag:
        drug_list = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0).index.tolist()
        print(drug_list)
    else:
        drug_list = ['tgem', 'tfu', 'tem', 'gem', 'cis', 'sor', 'fu']

    print(args.pdtc_flag)
    print(args.method)
    drug_num = 0
    for drug in drug_list:
        param_num = 0
        drug_num = drug_num + 1
        print(' ')
        print(' ')
        for param_dict in update_params_dict_list:
            param_num = param_num + 1
            print(' ')
            print('#######      Drug_num: {0}/7 ; Param_num {1}/60       #######'.format(drug_num,param_num))
            print('Drug: {0}; Param_dict: {1}'.format(drug,param_dict) )       
            main(args=args, drug=drug, update_params_dict=param_dict)
