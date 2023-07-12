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

import train_coral
import train_ae
import train_dae
import train_vae
import train_ae_mmd
import train_ae_adv 

import train_dsn_mmd
import train_dsn_adv
import train_dsn_adnn 

import train_dsrn 
import train_dsrn_mmd 
import train_dsrn_adv
import train_dsrn_adnn


import fine_tuning1
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
    # label_tensor = dataloader.dataset.tensors[1].cpu()

    encoded_feature_tensor = encoder.cpu()(raw_feature_tensor)
    if normalize_flag:
        encoded_feature_tensor = torch.nn.functional.normalize(encoded_feature_tensor, p=2, dim=1)
    return encoded_feature_tensor #, label_tensor


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

def str_to_dict(s): #Not for adv now
    d = {}
    if "pretrain_num_epochs" in s:
        i = s.split('pretrain_num_epochs_')[1]
        d['pretrain_num_epochs'] = int(i.split('_train_num_epochs_')[0])
        j = i.split('train_num_epochs_')[1].split('_dop_')
    else:
        j = s.split('train_num_epochs_')[1].split('_dop_')
    d['train_num_epochs'] = int(j[0])
    d['dop'] = float(j[1])
    return d

def main(args):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # pretrain_dataset_file = os.path.join(data_config.every_tumor_type_folder,args.select_gene_method,f"{args.pretrain_dataset}_uq1000_feature.csv")
    # gex_features_df = pd.read_csv(pretrain_dataset_file, index_col=0)

    
    with open(os.path.join('model_save/train_params.json'), 'r') as f:
        training_params = json.load(f)

    for TT_num in range(param_list.shape[1]):
        # if TT_num>0:
        #     break
        # method_num = 7
        param_str = param_list.iloc[method_num,TT_num].rstrip('_ft_evaluation_results.json')
        print(param_str)
        update_params_dict = str_to_dict(param_str)
        training_params['unlabeled'].update(update_params_dict) 
        print(training_params)
        # training_params['unlabeled'].update(update_params_dict)   ##unlabeled一开始就更新成dict的队列了，labeled一直没有更新，都是json读进来的2000
        
        pretrain_dataset = param_list.columns[TT_num].lower()
        select_method = param_list.index[method_num]
        select_method = select_method[:-5]
        print(pretrain_dataset,select_method)
        
        if select_method == 'ae':
            train_fn = train_ae.train_ae
        elif select_method == 'dae':
            train_fn = train_dae.train_dae
        elif select_method == 'vae':
            train_fn = train_vae.train_vae

        elif select_method == 'ae_mmd':
            train_fn = train_ae_mmd.train_ae_mmd
        elif select_method == 'ae_adv':
            train_fn = train_ae_adv.train_ae_adv
        
        elif select_method == 'dsn_mmd':
            train_fn = train_dsn_mmd.train_dsn_mmd
        elif select_method == 'dsn_adv':
            train_fn = train_dsn_adv.train_dsn_adv
        elif select_method == 'dsn_adnn':
            train_fn = train_dsn_adnn.train_dsn_adnn

        elif select_method == 'dsrn':
            train_fn = train_dsrn.train_dsrn
        elif select_method == 'dsrn_mmd':
            train_fn = train_dsrn_mmd.train_dsrn_mmd
        elif select_method == 'dsrn_adv':
            train_fn = train_dsrn_adv.train_dsrn_adv
        elif select_method == 'dsrn_adnn':
            train_fn = train_dsrn_adnn.train_dsrn_adnn
        
        else:
            raise NotImplementedError("Not true method supplied!")

        CCL_tumor_type = "all_CCL"
        gex_features_df = pd.read_csv(f'../data/align_data/{pretrain_dataset}_pretrain_dataset.csv',index_col=0)

        source_dir = os.path.join(pretrain_dataset,args.tcga_construction,
                                args.select_gene_method,
                                CCL_tumor_type,args.CCL_construction,
                                pretrain_dataset.upper(),
                                args.CCL_dataset,args.select_drug_method)
        #  pretrain数据，pretrain数据基因挑选方法，zero-shot什么癌种，用的什么CDR数据（GDSC？），怎么挑选Drug数据

        #后面就存在results中指定的文件夹内了
        
       
        method_save_folder = os.path.join(store_dir, f'{select_method}_norm',source_dir)

        training_params.update(
            {
                'device': device,
                'input_dim': gex_features_df.shape[-1],
                'model_save_folder': os.path.join(method_save_folder, param_str),  #一个模型就存一个unlabel model
                'es_flag': False,
                'retrain_flag': args.retrain_flag,
                'norm_flag': args.norm_flag
            })
        
        task_save_folder = f'../data/align_data/{select_pds}/{select_method}'
        safe_make_dir(task_save_folder)

        print(training_params['model_save_folder'])
        print(task_save_folder)

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
        # print("Trained SE:",encoder)

        # generate encoded features
        ccle_encoded_feature_tensor = generate_encoded_features(encoder, s_dataloaders[0],
                                                                normalize_flag=args.norm_flag)
        tcga_encoded_feature_tensor = generate_encoded_features(encoder, t_dataloaders[0],
                                                                normalize_flag=args.norm_flag)
        a = pd.concat([pd.DataFrame(ccle_encoded_feature_tensor.detach().cpu().numpy()),
                       pd.DataFrame(tcga_encoded_feature_tensor.detach().cpu().numpy())
                       ])
        a=a.assign(label = ["CCLE"] * ccle_encoded_feature_tensor.shape[0]+["TCGA"] * tcga_encoded_feature_tensor.shape[0])
        a.to_csv(os.path.join(task_save_folder,
                              f'{pretrain_dataset}.csv'))
        # pd.DataFrame(ccle_encoded_feature_tensor.detach().cpu().numpy()).to_csv(
        #     os.path.join(task_save_folder, f'{param_str}_ccle_encoded_feature.csv'))

        # pd.DataFrame(tcga_encoded_feature_tensor.detach().cpu().numpy()).to_csv(
        #     os.path.join(task_save_folder, f'{param_str}_tcga_encoded_feature.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADSN training and evaluation')
    parser.add_argument('--pretrain_num',default = None,type=int)
    parser.add_argument('--zero_shot_num',default = None,type=int)
    parser.add_argument('--method_num',default = None,type=int)
    parser.add_argument('--pretrain_mode',default = 0,type=int)
    
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
    # parser.set_defaults(retrain_flag=True)
    parser.set_defaults(retrain_flag=False)


    train_group.add_argument('--pdtc', dest='pdtc_flag', action='store_true')
    train_group.add_argument('--no-pdtc', dest='pdtc_flag', action='store_false')
    parser.set_defaults(pdtc_flag=False)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    
    parser.add_argument('--label_type', default = "PFS",choices=["PFS","Imaging"])

    # CDR挑选TCGA检测中要用的药物作为finetune数据
    parser.add_argument('--select_drug_method', default = "overlap",choices=["overlap","all","random"])
    
    parser.add_argument('--store_dir',default = "classification") 
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
        'BRCA','STAD','CESC','SKCM','BLCA','LUSC','LUAD','UCEC','READ','COAD']) #19 PFS >10
    parser.add_argument('--CCL_dataset',default = 'gdsc1_raw',
        choices=['gdsc1_raw','gdsc1_rec','gc_combine','gp_combine','gcp_combine','GDSC1_raw','GDSC1_rec','gdsc1_rebalance'])
    parser.add_argument('--class_num',default = 0,type=int)
    parser.add_argument('--ccl_match',default = "yes",
        choices=["yes","no","match_zs"])
    
    args = parser.parse_args()
    
    Tumor_type_list = ["tcga", "blca", "brca", "cesc", "coad", "gbm", "hnsc", "kich", "kirc", 
            "kirp", "lgg", "lihc", "luad", "lusc", "ov", "paad", "prad", "read", "sarc", "skcm", "stad", #"ucec",
            "esca","meso","ucs","acc"] #1+20+4
    
    # if args.pretrain_num :
    #     args.pretrain_dataset = Tumor_type_list[args.pretrain_num] #24
    # if args.zero_shot_num :
    #     args.tumor_type = [element.upper() for element in Tumor_type_list][args.zero_shot_num]
    #     # print(f'Tumor type:  Select zero_shot_num: {Num}. Zero-shot dataset: {args.tumor_type}')
    # if args.method_num : 
    #     # args.method = ['ae','dae','vae','ae_mmd','ae_adv', #ae_mmd:3
    #     #                'dsn_mmd','dsn_adv','dsn_adnn',
    #     #                'dsrn','dsrn_mmd','dsrn_adv','dsrn_adnn'][args.method_num] #0-11
    #     args.method = ['dsn_adnn',  #'dsrn_adnn',
    #                    'ae','dae','vae','dsrn', 
    #                    'ae_mmd','dsn_mmd','dsrn_mmd',
    #                    'ae_adv','dsn_adv','dsrn_adv'][args.method_num] #0-11
    
    # #tcga pretrain需要根据args.tumor_type去指定finetune哪个癌种，其他的默认先是原癌种
    # tumor_type = args.pretrain_dataset.upper() #需要交叉的时候注释掉这行，直接让【tumor_type = args.tumor_type】自由选择
    # if tumor_type == "TCGA" : 
    #     tumor_type = args.tumor_type

    # #构建4级文件夹结构，并生成用于unlabel training需要的gex_features_df------
    # gex_features_df,CCL_tumor_type,all_ccle_gex,_ = data.get_pretrain_dataset(
    #     patient_tumor_type = args.pretrain_dataset,
    #     CCL_type = args.CCL_type,
    #     tumor_type = tumor_type,
    #     tcga_construction = args.tcga_construction,
    #     CCL_construction = args.CCL_construction,
    #     gene_num = args.select_gene_num,select_gene_method = args.select_gene_method
    #     )
    
    # print(f'Pretrain dataset: Patient({args.pretrain_dataset} {args.pretrain_num} {args.tcga_construction}) CCL({args.CCL_type} {CCL_tumor_type} {args.CCL_construction}). Select_gene_method: {args.select_gene_method}')
    # print(f'Zero-shot dataset: {tumor_type}({args.zero_shot_num})')
    # print(f'CCL_dataset: {args.CCL_dataset}  Select_drug_method: {args.select_drug_method}')
    # print(f'Store_dir: {args.store_dir} ')
    
    # print(f'pdtc_flag: {args.pdtc_flag}. method: {args.method}({args.method_num}). label_type: {args.label_type}')
    select_pds = ["tcga","match"][args.pretrain_mode]
    param_num = 0

    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    # update_params_dict_list.reverse()
    store_dir = os.path.join('../results',args.store_dir)
    param_list = pd.read_csv(os.path.join(store_dir,f'param_list_{select_pds}.csv'),index_col = 0)
    
    for method_num in range(param_list.shape[0]):
        param_num = param_num + 1
        # args.method = ['ae','dae','vae','dsrn', 
        #             'ae_mmd','dsrn_mmd','dsn_mmd',
        #             'ae_adv','dsrn_adv','dsn_adv'][method_num] 
        print(' ')
        print('##############################################################################')
        print(f'#######    Param_num {param_num}/{param_list.shape[0]}       #######')
        # print('Select method: {}'.format(args.method) )    
        print('##############################################################################')   
        # if method_num not in [0,2,5]:
        # if method_num not in [5]:
        if method_num < 7 :
            continue
        main(args=args)
    print("Finsh All !!!!!!!!!!!!!!!")
