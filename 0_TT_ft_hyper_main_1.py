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
import data
import data_config

import train_ae
import train_ae_mmd
import train_ae_adv 

import train_dsn_mmd
import train_dsn_adv
import train_dsn_adnn 

import train_dsrn 
import train_dsrn_mmd 
import train_dsrn_adv

import fine_tuning
from copy import deepcopy

def dict_to_str(d):
    
    return "_".join(["_".join([k, str(v)]) for k, v in d.items()])

def wrap_training_params(training_params, type='unlabeled'):
    aux_dict = {k: v for k, v in training_params.items() if k not in ['unlabeled', 'labeled']}
    aux_dict.update(**training_params[type])

    return aux_dict


def safe_make_dir(new_folder_name):
    if not os.path.exists(new_folder_name):
        os.makedirs(new_folder_name)
    else:
        print(new_folder_name, 'exists!')



def main(args, update_params_dict):
    if args.method == 'ae':
        train_fn = train_ae.train_ae
    elif args.method == 'ae_mmd':
        train_fn = train_ae_mmd.train_ae_mmd
    elif args.method == 'ae_adv':
        train_fn = train_ae_adv.train_ae_adv
    
    elif args.method == 'dsn':
        train_fn = train_dsn.train_dsn
    elif args.method == 'dsn_mmd':
        train_fn = train_dsn_mmd.train_dsn_mmd
    elif args.method == 'dsn_adv':
        train_fn = train_dsn_adv.train_dsn_adv

    elif args.method == 'dsrn':
        train_fn = train_dsrn.train_dsrn
    elif args.method == 'dsrn_mmd':
        train_fn = train_dsrn_mmd.train_dsrn_mmd
    elif args.method == 'dsrn_adv':
        train_fn = train_dsrn_adv.train_dsrn_adv
    
    else:
        raise NotImplementedError("Not true method supplied!")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    with open(os.path.join('model_save/train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params['unlabeled'].update(update_params_dict)   
    param_str = dict_to_str(update_params_dict)  

    source_dir = os.path.join(args.pretrain_dataset,args.tcga_construction,
                              args.select_gene_method,
                              CCL_tumor_type,args.CCL_construction,
                              tumor_type,
                              args.CCL_dataset,args.select_drug_method)

    store_dir = os.path.join('../results',args.store_dir)
    method_save_folder = os.path.join(store_dir, f'{args.method}_norm',source_dir)

    training_params.update(
        {
            'device': device,
            'input_dim': gex_features_df.shape[-1],
            'model_save_folder': os.path.join(method_save_folder, param_str),  #一个模型就存一个unlabel model
            'es_flag': False,
            'retrain_flag': args.retrain_flag,
            'norm_flag': args.norm_flag
        })
    if args.pdtc_flag:
        task_save_folder = os.path.join(f'{method_save_folder}', args.measurement, 'pdtc')
    else:
        task_save_folder = os.path.join(f'{method_save_folder}', args.measurement)

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
    print(' ')
    # print("Trained SE:",encoder)

    if args.retrain_flag:
        with open(os.path.join(training_params['model_save_folder'], f'unlabel_train_history.pickle'),
                  'wb') as f:
            for history in historys:
                pickle.dump(dict(history), f)

    
    ft_evaluation_metrics = defaultdict(list)

    ##多个药之间互相比
    labeled_dataloader_generator = data.get_finetune_dataloader_generator(
            gex_features_df = gex_features_df,
            all_ccle_gex = all_ccle_gex,
            ccl_match = args.ccl_match,
            seed=2020,
            batch_size=training_params['labeled']['batch_size'],
            dataset = args.CCL_dataset, #finetune_dataset
            sample_size = 0.006,
            ccle_measurement=args.measurement,
            pdtc_flag=args.pdtc_flag,
            n_splits=args.n,
            q=2, #二分类
            tumor_type = tumor_type,
            label_type = args.label_type,
            select_drug_method = args.select_drug_method)

    fold_count = 0
    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, labeled_tcga_dataloader in labeled_dataloader_generator:
            ft_encoder = deepcopy(encoder)
            print(' ')
            print('Fold count = {}'.format(fold_count))
            if args.select_drug_method == "overlap" : 
                save_folder = method_save_folder
            elif args.select_drug_method == "all" : 
                save_folder = training_params['model_save_folder']
            target_classifier, ft_historys = fine_tuning.fine_tune_encoder_drug( 
                encoder=ft_encoder,
                train_dataloader=train_labeled_ccle_dataloader,
                val_dataloader=test_labeled_ccle_dataloader,
                test_dataloader=labeled_tcga_dataloader,
                fold_count=fold_count,
                normalize_flag=args.norm_flag,
                metric_name=args.metric,
                task_save_folder = save_folder,
                drug_emb_dim=300,
                class_num = args.class_num, #CCL_dataset几分类，需要和train_labeled_ccle_dataloader, test_labeled_ccle_dataloader匹配
                store_dir = method_save_folder, #store_dir,#：所有癌种都一样，所有method都一样
                **wrap_training_params(training_params, type='labeled')
            )
            if fold_count == 0 :
                print(f"Save target_classifier {param_dict}.")
                torch.save(target_classifier.state_dict(),
                           os.path.join(method_save_folder, f'save_classifier_{fold_count}.pt'))
            ft_evaluation_metrics['best_index'].append(ft_historys[-2]['best_index'])
            for metric in ['auroc']:
                ft_evaluation_metrics[metric].append(ft_historys[-1][metric][ft_historys[-2]['best_index']])
            fold_count += 1
            print(' ')
            print(' ')
            
 
    with open(os.path.join(task_save_folder, f'{param_str}_ft_evaluation_results.json'), 'w') as f:
        json.dump(ft_evaluation_metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADSN training and evaluation')
    parser.add_argument('--params_num',default = None,type=int)
    parser.add_argument('--pretrain_num',default = None,type=int)
    parser.add_argument('--zero_shot_num',default = None,type=int)
    parser.add_argument('--method_num',default = None,type=int)
    
    parser.add_argument('--method', dest='method', nargs='?', default='dsrn_adv',
                        choices=['ae','dae','vae','ae_mmd','ae_adv', #ae_mmd:3
                       'dsn_mmd','dsn_adv','dsn_adnn',
                       'dsrn','dsrn_mmd','dsrn_adv','dsrn_adnn'])
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

    
    parser.add_argument('--label_type', default = "PFS",choices=["PFS","Imaging"])

    parser.add_argument('--select_drug_method', default = "overlap",choices=["overlap","all","random"])
    
    parser.add_argument('--store_dir',default = "test") 
    parser.add_argument('--select_gene_method',default = "Percent_sd",choices=["Percent_sd","HVG"])
    parser.add_argument('--select_gene_num',default = 1000,type=int)

    parser.add_argument('--pretrain_dataset',default = "tcga",
        choices=["tcga", "blca", "brca", "cesc", "coad", "gbm", "hnsc", "kich", "kirc", 
        "kirp", "lgg", "lihc", "luad", "lusc", "ov", "paad", "prad", "read", "sarc", "skcm", "stad", "ucec",
        "esca","meso","ucs","acc"])
    parser.add_argument('--tcga_construction',default = "raw",
        choices=["raw","pseudo_random"])
    parser.add_argument('--CCL_type',default = "all_CCL",
        choices=["all_CCL","single_CCL"])
    parser.add_argument('--CCL_construction',default = "raw",
        choices=["raw","pseudo_random"])
    parser.add_argument('--tumor_type',default = "BRCA",
        choices=["TCGA","ESCA","MESO", "UCS", "ACC","GBM", 'LGG', 'PAAD','HNSC','LIHC','KIRC','SARC','PRAD','OV',
        'BRCA','STAD','CESC','SKCM','BLCA','LUSC','LUAD','UCEC','READ','COAD']) 
    parser.add_argument('--CCL_dataset',default = 'gdsc1_raw',
        choices=['gdsc1_raw','gdsc1_rec','gc_combine','gp_combine','gcp_combine','GDSC1_raw','GDSC1_rec','gdsc1_rebalance','gdsc1_rebalance_regression'])
    parser.add_argument('--class_num',default = 0,type=int)
    parser.add_argument('--ccl_match',default = "yes",
        choices=["yes","no","match_zs"])
    
    args = parser.parse_args()
    if args.class_num == 1:
        # args.CCL_dataset = f"{args.CCL_dataset}_regression"
        print("Regression task.  Use dataset:",args.CCL_dataset)

    if args.select_drug_method == "all" :
         params_grid = {
        "pretrain_num_epochs": [0,100,300],  # encoder、decoder
        "train_num_epochs": [100,1000,2000,3000],   # GAN
        "dop":  [0.0,0.1]
    } 
    else:
        params_grid = {
            "pretrain_num_epochs": [0, 100, 300],
            "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000],
            "dop": [0.0, 0.1]
        }

    params_grid = {
            "pretrain_num_epochs": [0,200,400,600,800,1000], #[0, 100, 300],
            "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000],
            "dop": [0.0, 0.1]
        }
    
    Tumor_type_list = ["tcga", "blca", "brca", "cesc", "coad", "gbm", "hnsc", "kich", "kirc", 
            "kirp", "lgg", "lihc", "luad", "lusc", "ov", "paad", "prad", "read", "sarc", "skcm", "stad", #"ucec",
            "esca","meso","ucs","acc"] #1+20+4
    
    if args.pretrain_num :
        args.pretrain_dataset = Tumor_type_list[args.pretrain_num] #24
    if args.zero_shot_num :
        args.tumor_type = [element.upper() for element in Tumor_type_list][args.zero_shot_num]
        # print(f'Tumor type:  Select zero_shot_num: {Num}. Zero-shot dataset: {args.tumor_type}')
    if args.method_num : 
        # args.method = ['ae','dae','vae','ae_mmd','ae_adv', #ae_mmd:3
        #                'dsn_mmd','dsn_adv','dsn_adnn',
        #                'dsrn','dsrn_mmd','dsrn_adv','dsrn_adnn'][args.method_num] #0-11
        args.method = ['dsn_adnn',  #'dsrn_adnn',
                       'ae','dae','vae','dsrn', 
                       'ae_mmd','dsn_mmd','dsrn_mmd',
                       'ae_adv','dsn_adv','dsrn_adv'][args.method_num] #0-11
    
    #tcga pretrain需要根据args.tumor_type去指定finetune哪个癌种，其他的默认先是原癌种
    tumor_type = args.pretrain_dataset.upper() #需要交叉的时候注释掉这行，直接让【tumor_type = args.tumor_type】自由选择
    if tumor_type == "TCGA" : 
        tumor_type = args.tumor_type

    if args.method not in ['dsrn_adv', 'dsn_adv', 'ae_adv']:
        params_grid.pop('pretrain_num_epochs')

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]


    #构建4级文件夹结构，并生成用于unlabel training需要的gex_features_df------
    gex_features_df,CCL_tumor_type,all_ccle_gex,_ = data.get_pretrain_dataset(
        patient_tumor_type = args.pretrain_dataset,
        CCL_type = args.CCL_type,
        tumor_type = tumor_type,
        tcga_construction = args.tcga_construction,
        CCL_construction = args.CCL_construction,
        gene_num = args.select_gene_num,select_gene_method = args.select_gene_method
        )

    print(f'Pretrain dataset: Patient({args.pretrain_dataset} {args.pretrain_num} {args.tcga_construction}) CCL({args.CCL_type} {CCL_tumor_type} {args.CCL_construction}). Select_gene_method: {args.select_gene_method}')
    print(f'Zero-shot dataset: {tumor_type}({args.zero_shot_num})')
    print(f'CCL_dataset: {args.CCL_dataset}  Select_drug_method: {args.select_drug_method}')
    print(f'Store_dir: {args.store_dir} ')
    
    print(f'pdtc_flag: {args.pdtc_flag}. method: {args.method}({args.method_num}). label_type: {args.label_type}')
    param_num = 0

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # update_params_dict_list.reverse()
    if args.params_num : 
        update_params_dict_list = update_params_dict_list[args.params_num:]
        print(f'params: {update_params_dict_list}.')

    for param_dict in update_params_dict_list:
            param_num = param_num + 1
            print(' ')
            print('##############################################################################')
            print(f'#######    Param_num {param_num}/{len(update_params_dict_list)}       #######')
            print('Param_dict: {}'.format(param_dict) )    
            print('##############################################################################')   
            main(args=args, update_params_dict=param_dict)
    print("Finsh All !!!!!!!!!!!!!!!")
