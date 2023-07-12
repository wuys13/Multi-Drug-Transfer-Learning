import gzip
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader,Dataset
from rdkit import RDLogger 

import data_config
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def unique_percent(x):
        return len(x.unique()) / len(x)

def select_gene(gex,gene_num,method = "Percent_sd"):
    if method == "Percent_sd":
        cancer_unique = gex.apply(unique_percent)
        cancer_sd = gex.apply(np.std)
        percent_sd = cancer_unique*cancer_sd
    else: #目前暂不实现 To do:
        raise NotImplementedError("Only support Percent_sd now")
    return percent_sd.sort_values(ascending=False).index[:gene_num].values

def get_pretrain_dataset(patient_tumor_type,CCL_type,tumor_type,tcga_construction,CCL_construction,gene_num,select_gene_method):
    print("Starting generating pretrain dataset...")
    #Step1:Load gex and sample info file
    patient_gex_raw = pd.read_csv("../data/preprocessed_dat/xena_gex_raw.csv", index_col=0)  #9808TCGA_sample * 18966HVG(overlap)
    ccle_gex_raw = pd.read_csv("../data/preprocessed_dat/ccle_gex_raw.csv", index_col=0)  #1305CCL_sample * 18966HVG(overlap)

    patient_sample_info = pd.read_csv("../data/preprocessed_dat/xena_sample_info.csv", index_col=0) 
    ccle_sample_info = pd.read_csv("../data/preprocessed_dat/ccle_sample_info.csv", index_col=0) 
    print(f"Step1 load in dim: patient({patient_sample_info.shape[0]}); ccle({ccle_sample_info.shape[0]})")
    
    #Step2:Locate selected tumor type for patient and CCL
    patient_tumor_type = patient_tumor_type.upper() 
    if patient_tumor_type == "TCGA":
        patient_gex = patient_gex_raw
    else:
        patient_samples = patient_gex_raw.index.intersection(patient_sample_info.loc[patient_sample_info.tumor_type == patient_tumor_type].index)
        patient_gex = patient_gex_raw.loc[patient_samples]

    if CCL_type == "all_CCL":
        ccle_gex = ccle_gex_raw
        CCL_tumor_type = CCL_type
    else:
        if patient_tumor_type == "TCGA":
            CCL_tumor_type = tumor_type
        else:
            CCL_tumor_type = patient_tumor_type
        ccle_samples = ccle_gex_raw.index.intersection(ccle_sample_info.loc[ccle_sample_info.tumor_type == CCL_tumor_type].index)
        ccle_gex = ccle_gex_raw.loc[ccle_samples]
        CCL_tumor_type = CCL_tumor_type.lower()
    print(f"Step2 after locate dim: patient({patient_gex.shape[0]}); ccle({ccle_gex.shape[0]})")
    print(f"Define CCL_tumor_type: {CCL_tumor_type}")

    #Step3:Construct pseudo samples
    if tcga_construction == "raw":
        patient_gex = patient_gex
    elif tcga_construction == "pseudo_random":
        #To do:伪样本算法
        pass
    if CCL_construction == "raw":
        ccle_gex = ccle_gex
    elif CCL_construction == "pseudo_random":
        #To do:伪样本算法
        pass
    print(f"Step3 after pseudo construct dim: patient({patient_gex.shape[0]}); ccle({ccle_gex.shape[0]})")

    #Step4:Select HVGs for AE features
    patient_gene = select_gene(patient_gex,gene_num,method = select_gene_method)
    ccle_gene = select_gene(ccle_gex,gene_num,method = select_gene_method)
    union_gene = list(set(patient_gene).union(set(ccle_gene)))
    print(f"Step4 after select HVGs: patient_gene({len(patient_gene)}):ccle_gene({len(ccle_gene)}):union_gene({len(union_gene)})")

    #Step5:Combine patient_gex with ccle_gex
    patient_gex_new = patient_gex.loc[:,union_gene]
    patient_gex_nor = pd.DataFrame(preprocessing.scale(patient_gex_new),index=patient_gex_new.index,columns=patient_gex_new.columns)
    ccle_gex_new = ccle_gex.loc[:,union_gene]
    ccle_gex_nor = pd.DataFrame(preprocessing.scale(ccle_gex_new),index=ccle_gex_new.index,columns=ccle_gex_new.columns)

    combine_gex = pd.concat([patient_gex_nor,ccle_gex_nor]) #patient_gex_nor.append(ccle_gex_nor)
    print(f"Step5 after combination dim: {combine_gex.shape[0]} samples  *  {combine_gex.shape[1]} genes")
    print('')
    
    ccle_gex_new = ccle_gex_raw.loc[:,union_gene]
    all_ccle_gex = pd.DataFrame(preprocessing.scale(ccle_gex_new),
                                index=ccle_gex_new.index,columns=ccle_gex_new.columns)
    patient_gex_new = patient_gex_raw.loc[:,union_gene]
    all_patient_gex = pd.DataFrame(preprocessing.scale(patient_gex_new),
                                index=patient_gex_new.index,columns=patient_gex_new.columns)
    return combine_gex,CCL_tumor_type,all_ccle_gex,all_patient_gex


def get_unlabeled_dataloaders(gex_features_df, seed, batch_size, ccle_only=False):
    """
    CCLE as source domain, thus s_dataloaders
    Xena(TCGA) as target domain, thus t_dataloaders
    :param gex_features_df:
    :param seed:
    :param batch_size:
    :return:
    """
    set_seed(seed)
    
    ccle_sample_info_df = pd.read_csv(data_config.ccle_sample_file, index_col=0)
    ccle_sample_info_df = ccle_sample_info_df.reset_index().drop_duplicates(subset="Depmap_id",keep='first').set_index("Depmap_id")
    # with gzip.open(data_config.xena_sample_file) as f:
    #     xena_sample_info_df = pd.read_csv(f, sep='\t', index_col=0)
    xena_sample_info_df = pd.read_csv(data_config.xena_sample_file, index_col=0)
    xena_samples = xena_sample_info_df.index.intersection(gex_features_df.index)   #gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')].index
    ccle_samples = ccle_sample_info_df.index.intersection(gex_features_df.index)
    xena_sample_info_df = xena_sample_info_df.loc[xena_samples]
    ccle_sample_info_df = ccle_sample_info_df.loc[ccle_samples.intersection(ccle_sample_info_df.index)]

    xena_df = gex_features_df.loc[xena_samples]
    ccle_df = gex_features_df.loc[ccle_samples]

    excluded_ccle_samples = []
    excluded_ccle_samples.extend(ccle_df.index.difference(ccle_sample_info_df.index))
    excluded_ccle_diseases = ccle_sample_info_df.primary_disease.value_counts()[
        ccle_sample_info_df.primary_disease.value_counts() < 2].index
    excluded_ccle_samples.extend(
        ccle_sample_info_df[ccle_sample_info_df.primary_disease.isin(excluded_ccle_diseases)].index)

    to_split_ccle_df = ccle_df[~ccle_df.index.isin(excluded_ccle_samples)]
    train_ccle_df, test_ccle_df = train_test_split(to_split_ccle_df, test_size=0.1,
                                                   stratify=ccle_sample_info_df.loc[
                                                       to_split_ccle_df.index].primary_disease)
    test_ccle_df = test_ccle_df.append(ccle_df.loc[excluded_ccle_samples])
    train_xena_df, test_xena_df = train_test_split(xena_df, test_size=0.1, #len(test_ccle_df) / len(xena_df),
                                                   #stratify=xena_sample_info_df['_primary_disease'],
                                                   random_state=seed)
    print(' ')
    print(f"Pretrain dataset: {xena_df.shape[0]}(TCGA) {to_split_ccle_df.shape[0]}(Cell line)")
    print(' ')
    #train用所有，test取10%
    xena_dataset = TensorDataset(
        torch.from_numpy(xena_df.values.astype('float32'))
    )

    ccle_dataset = TensorDataset(
        torch.from_numpy(ccle_df.values.astype('float32'))
    ) 

    train_xena_dateset = TensorDataset(
        torch.from_numpy(train_xena_df.values.astype('float32')))
    test_xena_dateset = TensorDataset(
        torch.from_numpy(test_xena_df.values.astype('float32')))
    train_ccle_dateset = TensorDataset(
        torch.from_numpy(train_ccle_df.values.astype('float32')))
    test_ccle_dateset = TensorDataset(
        torch.from_numpy(test_ccle_df.values.astype('float32')))

    xena_dataloader = DataLoader(xena_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)
    train_xena_dataloader = DataLoader(train_xena_dateset,
                                       batch_size=batch_size,
                                       shuffle=True)
    test_xena_dataloader = DataLoader(test_xena_dateset,
                                      batch_size=batch_size,
                                      shuffle=True)

    ccle_data_loader = DataLoader(ccle_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True
                                  )

    train_ccle_dataloader = DataLoader(train_ccle_dateset,
                                       batch_size=batch_size,
                                       shuffle=True, drop_last=True)
    test_ccle_dataloader = DataLoader(test_ccle_dateset,
                                      batch_size=batch_size,
                                      shuffle=True)
    if ccle_only:
        return (ccle_data_loader, test_ccle_dataloader), (ccle_data_loader, test_ccle_dataloader)
    else:
        return (ccle_data_loader, test_ccle_dataloader), (xena_dataloader, test_xena_dataloader)



def get_pdtc_labeled_dataloaders(drug, batch_size, threshold=None, measurement='AUC'):
    pdtc_features_df = pd.read_csv(data_config.pdtc_gex_file, index_col=0)       # 40PDTC*1427genes（预处理好的）
    target_df = pd.read_csv(data_config.pdtc_target_file, index_col=0, sep='\t') #1637条（PDC*drug）AUC用于判断二分类
    drug_target_df = target_df.loc[target_df.Drug == drug]
    labeled_samples = drug_target_df.index.intersection(pdtc_features_df.index)
    drug_target_vec = drug_target_df.loc[labeled_samples, measurement]
    drug_feature_df = pdtc_features_df.loc[labeled_samples]

    assert all(drug_target_vec.index == drug_target_vec.index)

    if threshold is None:
        threshold = np.median(drug_target_vec)

    drug_label_vec = (drug_target_vec < threshold).astype('int')

    labeled_pdtc_dateset = TensorDataset(
        torch.from_numpy(drug_feature_df.values.astype('float32')),
        torch.from_numpy(drug_label_vec.values))

    labeled_pdtc_dataloader = DataLoader(labeled_pdtc_dateset,
                                         batch_size=batch_size,
                                         shuffle=True)

    return labeled_pdtc_dataloader


class FTDataset(Dataset):
    def __init__(self,gex_df,gex_index, smiles_index,  label):
        # gex_df.to_csv("../gex_df.csv")
        # print("gex_index:",gex_index,"\n")
        # print("smiles_index:",smiles_index,"\n")
        # print("label:",label,"\n")

        drug_emb = pd.read_csv("../data/preprocessed_dat/drug_embedding/CCL_dataset/supervised_infomax.pth300_results.csv",index_col=1)
        drug_emb = drug_emb.iloc[:,1:]
        self.gex = gex_df.loc[gex_index].values.astype('float32')
        self.smiles= drug_emb.loc[smiles_index].values.astype('float32')
        self.label = label.astype('float32')

    def __getitem__(self, item):
        Gex = self.gex[item]
        Smiles = self.smiles[item]
        Label = self.label[item]
        # print(item,"*"*30)
        # print("gex_index:",len(Gex),Gex,"\n")
        # print("smiles_index:",len(Smiles),Smiles,"\n")
        # print("label:",Label,"\n")
        # print(item,"*"*30)
        return Gex,Smiles, Label

    def __len__(self):
        return len(self.label)


# To replace get_labeled_dataloader_generator
def get_finetune_dataloader_generator(gex_features_df,all_ccle_gex,ccl_match,label_type, sample_size,dataset = 'gdsc1_raw', seed = 2020 , batch_size = 64, ccle_measurement='AUC',                                  
                                     pdtc_flag=False,
                                     n_splits=5,q=2,
                                     tumor_type = "TCGA",
                                     select_drug_method = True):
    """
    sensitive (responder): 1
    resistant (non-responder): 0

    """
    # if pdtc_flag:
    #     drug_mapping_df = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0)
    # else:
    #     drug_mapping_df = pd.read_csv(data_config.gdsc_tcga_mapping_file, index_col=0)

    # print(f'Drug: {drug}, TCGA (PDTC): {drug_name}, GDSC: {gdsc_drug}')
    
    #
    RDLogger.DisableLog('rdApp.*')

    if pdtc_flag: #To do : does not work now!
        test_labeled_dataloaders = get_pdtc_labeled_dataloaders(
                                                                batch_size=batch_size,
                                                                measurement=ccle_measurement)
    else: #index是TCGA_id，smiles(0列),gex,label(-1列)
        test_labeled_dataloaders,cid_list = get_tcga_ZS_dataloaders(gex_features_df=gex_features_df, 
                                                                    label_type = label_type,
                                                                    batch_size=batch_size,
                                                                    q=2,
                                                                    tumor_type = tumor_type)

    ccle_labeled_dataloader_generator = get_ccl_labeled_dataloader_generator(gex_features_df=gex_features_df,
                                                                               all_ccle_gex= all_ccle_gex,
                                                                               ccl_match = ccl_match,
                                                                               tumor_type = tumor_type,
                                                                              seed=seed,
                                                                              sample_size = sample_size,
                                                                              dataset = dataset,
                                                                              batch_size=batch_size,
                                                                              measurement=ccle_measurement,
                                                                              n_splits=n_splits,q=q,
                                                                              cid_list = cid_list,
                                                                              select_drug_method = select_drug_method)
    
    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader in ccle_labeled_dataloader_generator:
        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, test_labeled_dataloaders

def get_ccl_labeled_dataloader_generator(gex_features_df, all_ccle_gex,ccl_match,tumor_type,cid_list,select_drug_method,sample_size,dataset = 'gdsc_raw', batch_size = 64, seed=2020, 
                                          measurement='AUC', n_splits=5,q=2):
    # measurement = 'Z_score'       'AUC'      'IC50'
    # dataset = 'gdsc_raw'   'gdsc_recompute'  'gc_combine'  'gp_combine'  'gcp_combine' 'all_combine'
    # Dataset_path = '../data/raw_dat/CCL_dataset/all_drug/{}.csv'.format(dataset)
    Dataset_path = '../data/raw_dat/CCL_dataset/{}.csv'.format(dataset)
    # if select_drug_method == "all":
    #     Dataset_path = '../data/raw_dat/CCL_dataset/gdsc1_rebalance.csv'
    sensitivity_df = pd.read_csv(Dataset_path,index_col=1)
    # sensitivity_df.dropna(inplace=True)

    # target_df = sensitivity_df.groupby(['Depmap_id', 'Drug_smiles']).mean()
    # target_df = target_df.reset_index()
    ccle_sample_with_gex = pd.read_csv("../data/preprocessed_dat/ccle_sample_with_gex.csv",index_col=0)
    sensitivity_df = sensitivity_df.loc[sensitivity_df.index.isin(ccle_sample_with_gex.index)]
    if select_drug_method == "all":
        target_df = sensitivity_df
    elif select_drug_method == "overlap":
        target_df = sensitivity_df.loc[sensitivity_df['cid'].isin(cid_list)]
        print("Drug num: target(TCGA) / source(GDSC) / overlap = {0} {1} / {2} / {3} {4}".format(
            pd.Series(cid_list).unique() , pd.Series(cid_list).nunique(),
            sensitivity_df['Drug_name'].nunique(),
            target_df['Drug_name'].unique() , target_df['Drug_name'].nunique()
        ))
    elif select_drug_method == "random":
        target_df = sensitivity_df.sample(n = round(sample_size*sensitivity_df.shape[0])) #抽取 %1 的样本
    print(' ')
    
    print("Select {0} dataset {1} / {2}".format(dataset, 
        # round(sample_size*sensitivity_df.shape[0]),#
        target_df.shape[0],
        sensitivity_df.shape[0]
        ))
    print(' ')

    #似乎没用
    # ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=0)
    # ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()] 
    if ccl_match == "yes": #use match
        ccl_gex_df = gex_features_df  #如果希望finetune的ccl和pretrain的癌种匹配就用构建好的pretrain_dataset来match
    elif ccl_match == "no": #use all
        ccl_gex_df = all_ccle_gex 
    elif ccl_match == "match_zs": #
         #根据tumor_type从all_ccle_gex筛选出对应的ccl
         ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=0)
         select_ccl = ccle_sample_info.loc[ccle_sample_info.tumor_type == tumor_type].index
         ccl_gex_df = all_ccle_gex.loc[select_ccl]
    else:
        raise NotImplementedError('Not true ccl_match supplied!')
    keep_samples = target_df.index.isin(ccl_gex_df.index)

    ccle_labeled_feature_df = target_df.loc[keep_samples][["Drug_smile","label"]]
    ccle_labeled_feature_df.dropna(inplace=True)
    ccle_labeled_feature_df = ccle_labeled_feature_df.merge(ccl_gex_df,left_index=True,right_index=True)

    drug_emb = pd.read_csv("../data/preprocessed_dat/drug_embedding/CCL_dataset/supervised_infomax.pth300_results.csv",index_col=0)
    ccle_labeled_feature_df =  ccle_labeled_feature_df.merge(drug_emb,left_on="Drug_smile",right_on="Drug_smile") #gex,label(1),drug(300)
    del ccle_labeled_feature_df['Drug_smile'] 
    ccle_labeled_feature_df.dropna(inplace=True)

    # ccle_labeled_feature_df.drop_duplicates(keep='first',inplace=True)
    # ccle_labeled_feature_df.to_csv("../r_plot/cesc_ccl_data.csv")  #需要时把fintune筛选得到的ccl_feature输出来看看
    # if select_drug_method == "all":
        # ccle_labeled_feature_df['label'] = 0
        # ccle_labeled_feature_df.loc[ccle_labeled_feature_df[measurement]<0.55,'label'] = 1
        # print("Label distribution before: ",ccle_labeled_feature_df['label'].value_counts())
        # ccle_labeled_feature_df = Rebalance_dataset_for_every_drug(ccle_labeled_feature_df)
        # ccle_labeled_feature_df = ccle_labeled_feature_df['label'].astype('float32')
        # ccle_labeled_feature_df['label'] = ccle_labeled_feature_df['label'].astype('int')
        # print("Label distribution after: ",ccle_labeled_feature_df['label'].value_counts())
    
    ccle_labels = ccle_labeled_feature_df['label'].values
    # ccle_labeled_feature_df.to_csv("../see.csv")
    # del ccle_labeled_feature_df["Drug_name"] 
    if max(ccle_labels) < 1 and min(ccle_labels) > 0:
        ccle_labels = (ccle_labels < np.median(ccle_labels)).astype('int')


    s_kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels):
        train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.values[train_index], \
                                                      ccle_labeled_feature_df.values[test_index]
        # train_ccle_labels, test_ccle_labels = ccle_labels.values[train_index], ccle_labels.values[test_index]
        
        train_labeled_ccle_dateset = TensorDataset(
            # torch.from_numpy(train_labeled_ccle_df[:,0:train_labeled_ccle_df.shape[1]-1].astype('float32')),
            torch.from_numpy(train_labeled_ccle_df[:,1:(train_labeled_ccle_df.shape[1]-300)].astype('float32')), #gex:length-301
            torch.from_numpy(train_labeled_ccle_df[:,(train_labeled_ccle_df.shape[1]-300):(train_labeled_ccle_df.shape[1])].astype('float32')), #drug:300
            torch.from_numpy(train_labeled_ccle_df[:,0])          
        )
        test_labeled_ccle_dateset = TensorDataset(
            # torch.from_numpy(test_labeled_ccle_df[:,0:test_labeled_ccle_df.shape[1]-1].astype('float32')),
            torch.from_numpy(test_labeled_ccle_df[:,1:(test_labeled_ccle_df.shape[1]-300)].astype('float32')),
            torch.from_numpy(test_labeled_ccle_df[:,(test_labeled_ccle_df.shape[1]-300):(test_labeled_ccle_df.shape[1])].astype('float32')),
            torch.from_numpy(test_labeled_ccle_df[:,0])          
        )
        
        train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset,
                                                   batch_size=batch_size,
                                                #    num_workers = 8,
                                                #    pin_memory=True,
                                                   shuffle=True)

        test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_dateset,
                                                  batch_size=batch_size,
                                                #   num_workers = 8,
                                                #    pin_memory=True,
                                                   shuffle=True)

        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader

def get_finetune_dataloader_generator_1(gex_features_df,all_ccle_gex,ccl_match,label_type, sample_size,dataset = 'gdsc1_raw', seed = 2020 , batch_size = 64, ccle_measurement='AUC',                                  
                                     pdtc_flag=False,
                                     n_splits=5,q=2,
                                     tumor_type = "TCGA",
                                     select_drug_method = True):
    """
    sensitive (responder): 1
    resistant (non-responder): 0

    """
    # if pdtc_flag:
    #     drug_mapping_df = pd.read_csv(data_config.gdsc_pdtc_drug_name_mapping_file, index_col=0)
    # else:
    #     drug_mapping_df = pd.read_csv(data_config.gdsc_tcga_mapping_file, index_col=0)

    # print(f'Drug: {drug}, TCGA (PDTC): {drug_name}, GDSC: {gdsc_drug}')
    
    #
    RDLogger.DisableLog('rdApp.*')

    if pdtc_flag: #To do : does not work now!
        test_labeled_dataloaders = get_pdtc_labeled_dataloaders(
                                                                batch_size=batch_size,
                                                                measurement=ccle_measurement)
    else: #index是TCGA_id，smiles(0列),gex,label(-1列)
        test_labeled_dataloaders,cid_list = get_tcga_ZS_dataloaders(gex_features_df=gex_features_df, 
                                                                    label_type = label_type,
                                                                    batch_size=batch_size,
                                                                    q=2,
                                                                    tumor_type = tumor_type)

    ccle_labeled_dataloader_generator = get_ccl_labeled_dataloader_generator_1(gex_features_df=gex_features_df,
                                                                               all_ccle_gex= all_ccle_gex,
                                                                               ccl_match = ccl_match,
                                                                               tumor_type = tumor_type,
                                                                              seed=seed,
                                                                              sample_size = sample_size,
                                                                              dataset = dataset,
                                                                              batch_size=batch_size,
                                                                              measurement=ccle_measurement,
                                                                              n_splits=n_splits,q=q,
                                                                              cid_list = cid_list,
                                                                              select_drug_method = select_drug_method)
    
    for train_labeled_ccle_dataloader, test_labeled_ccle_dataloader in ccle_labeled_dataloader_generator:
        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader, test_labeled_dataloaders

def get_ccl_labeled_dataloader_generator_1(gex_features_df, all_ccle_gex,ccl_match,tumor_type,cid_list,select_drug_method,sample_size,dataset = 'gdsc_raw', batch_size = 64, seed=2020, 
                                          measurement='AUC', n_splits=5,q=2):
    # measurement = 'Z_score'       'AUC'      'IC50'
    # dataset = 'gdsc_raw'   'gdsc_recompute'  'gc_combine'  'gp_combine'  'gcp_combine' 'all_combine'
    # Dataset_path = '../data/raw_dat/CCL_dataset/all_drug/{}.csv'.format(dataset)
    Dataset_path = '../data/raw_dat/CCL_dataset/{}.csv'.format(dataset)
    if select_drug_method == "all":
        Dataset_path = '../data/raw_dat/CCL_dataset/gdsc1_rebalance.csv'
    sensitivity_df = pd.read_csv(Dataset_path,index_col=1)
    # sensitivity_df.dropna(inplace=True)

    # target_df = sensitivity_df.groupby(['Depmap_id', 'Drug_smiles']).mean()
    # target_df = target_df.reset_index()
    ccle_sample_with_gex = pd.read_csv("../data/preprocessed_dat/ccle_sample_with_gex.csv",index_col=0)
    sensitivity_df = sensitivity_df.loc[sensitivity_df.index.isin(ccle_sample_with_gex.index)]
    if select_drug_method == "all":
        target_df = sensitivity_df
    elif select_drug_method == "overlap":
        target_df = sensitivity_df.loc[sensitivity_df['cid'].isin(cid_list)]
        print("Drug num: target(TCGA) / source(GDSC) / overlap = {0} {1} / {2} / {3} {4}".format(
            pd.Series(cid_list).unique() , pd.Series(cid_list).nunique(),
            sensitivity_df['Drug_name'].nunique(),
            target_df['Drug_name'].unique() , target_df['Drug_name'].nunique()
        ))
    elif select_drug_method == "random":
        target_df = sensitivity_df.sample(n = round(sample_size*sensitivity_df.shape[0])) #抽取 %1 的样本
    print(' ')
    
    print("Select {0} dataset {1} / {2}".format(dataset, 
        # round(sample_size*sensitivity_df.shape[0]),#
        target_df.shape[0],
        sensitivity_df.shape[0]
        ))
    print(' ')

    #似乎没用
    # ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=0)
    # ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()] 
    if ccl_match == "yes": #use match
        ccl_gex_df = gex_features_df  #如果希望finetune的ccl和pretrain的癌种匹配就用构建好的pretrain_dataset来match
    elif ccl_match == "no": #use all
        ccl_gex_df = all_ccle_gex 
    elif ccl_match == "match_zs": #
         #根据tumor_type从all_ccle_gex筛选出对应的ccl
         ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=0)
         select_ccl = ccle_sample_info.loc[ccle_sample_info.tumor_type == tumor_type].index
         ccl_gex_df = all_ccle_gex.loc[select_ccl]
    else:
        raise NotImplementedError('Not true ccl_match supplied!')
    keep_samples = target_df.index.isin(ccl_gex_df.index)

    ccle_labeled_feature_df = target_df.loc[keep_samples][["Drug_smile","label"]]
    ccle_labeled_feature_df.dropna(inplace=True)
    # ccle_labeled_feature_df = ccle_labeled_feature_df.merge(ccl_gex_df,left_index=True,right_index=True)
    ccl_gex_df = ccl_gex_df.loc[ccl_gex_df.index.str.startswith('ACH')] #替换上一行
    
    # drug_emb = pd.read_csv("../data/preprocessed_dat/drug_embedding/CCL_dataset/supervised_infomax.pth300_results.csv",index_col=0)
    # ccle_labeled_feature_df =  ccle_labeled_feature_df.merge(drug_emb,left_on="Drug_smile",right_on="Drug_smile") #gex,label(1),drug(300)
    # del ccle_labeled_feature_df['Drug_smile'] 
    # ccle_labeled_feature_df.dropna(inplace=True)

    ccle_labels = ccle_labeled_feature_df['label']
    # ccle_labeled_feature_df.to_csv("../see.csv")
    # del ccle_labeled_feature_df["Drug_name"] 
    
    drug_emb = pd.read_csv("../data/preprocessed_dat/drug_embedding/CCL_dataset/supervised_infomax.pth300_results.csv",index_col=1)
    drug_emb = drug_emb.iloc[:,1:]

    s_kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
    for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
        # train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.values[train_index], \
        #                                               ccle_labeled_feature_df.values[test_index]
        
        train_labeled_ccle_dateset = TensorDataset(
            torch.from_numpy(pd.DataFrame(data=ccl_gex_df,index=ccle_labeled_feature_df.index.values[train_index]).values.astype('float32')),
            torch.from_numpy(pd.DataFrame(data=drug_emb,index=ccle_labeled_feature_df.Drug_smile.values[train_index]).values.astype('float32')),
            torch.from_numpy(ccle_labeled_feature_df.label.values[train_index].astype('float32'))
        )

        test_labeled_ccle_dateset = TensorDataset(
            torch.from_numpy(pd.DataFrame(data=ccl_gex_df,index=ccle_labeled_feature_df.index.values[test_index]).values.astype('float32')),
            torch.from_numpy(pd.DataFrame(data=drug_emb,index=ccle_labeled_feature_df.Drug_smile.values[test_index]).values.astype('float32')),
            torch.from_numpy(ccle_labeled_feature_df.label.values[test_index].astype('float32'))
        )
        
        train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset,
                                                   batch_size=batch_size,
                                                #    num_workers = 2,
                                                #    pin_memory=True,
                                                   shuffle=True)

        test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_dateset,
                                                  batch_size=batch_size,
                                                #   num_workers = 2,
                                                #   pin_memory=True,
                                                   shuffle=True)

        yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader

# def get_ccl_labeled_dataloader_generator_1(gex_features_df, all_ccle_gex,ccl_match,tumor_type,cid_list,select_drug_method,sample_size,dataset = 'gdsc_raw', batch_size = 64, seed=2020, 
#                                           measurement='AUC', n_splits=5,q=2):
#     # measurement = 'Z_score'       'AUC'      'IC50'
#     # dataset = 'gdsc_raw'   'gdsc_recompute'  'gc_combine'  'gp_combine'  'gcp_combine' 'all_combine'
#     # Dataset_path = '../data/raw_dat/CCL_dataset/all_drug/{}.csv'.format(dataset)
#     Dataset_path = '../data/raw_dat/CCL_dataset/{}.csv'.format(dataset)
#     if select_drug_method == "all":
#         Dataset_path = '../data/raw_dat/CCL_dataset/gdsc1_rebalance.csv'
#     sensitivity_df = pd.read_csv(Dataset_path,index_col=1)
#     # sensitivity_df.dropna(inplace=True)

#     # target_df = sensitivity_df.groupby(['Depmap_id', 'Drug_smiles']).mean()
#     # target_df = target_df.reset_index()
#     ccle_sample_with_gex = pd.read_csv("../data/preprocessed_dat/ccle_sample_with_gex.csv",index_col=0)
#     sensitivity_df = sensitivity_df.loc[sensitivity_df.index.isin(ccle_sample_with_gex.index)]
#     if select_drug_method == "all":
#         target_df = sensitivity_df
#     elif select_drug_method == "overlap":
#         target_df = sensitivity_df.loc[sensitivity_df['cid'].isin(cid_list)]
#     elif select_drug_method == "random":
#         target_df = sensitivity_df.sample(n = round(sample_size*sensitivity_df.shape[0])) #抽取 %1 的样本
#     print(' ')
#     print("Drug num: target(TCGA) / source(GDSC) / overlap = {0} / {1} / {2}".format(
#         pd.Series(cid_list).nunique(),
#         sensitivity_df['cid'].nunique(),
#         target_df['cid'].nunique()
#         ))
#     print("Select {0} dataset {1} / {2}".format(dataset, 
#         # round(sample_size*sensitivity_df.shape[0]),#
#         target_df.shape[0],
#         sensitivity_df.shape[0]
#         ))
#     print(' ')

#     #似乎没用
#     # ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=0)
#     # ccle_sample_info = ccle_sample_info.loc[ccle_sample_info.index.dropna()] 
#     if ccl_match == "yes": #use match
#         ccl_gex_df = gex_features_df  #如果希望finetune的ccl和pretrain的癌种匹配就用构建好的pretrain_dataset来match
#     elif ccl_match == "no": #use all
#         ccl_gex_df = all_ccle_gex 
#     elif ccl_match == "match_zs": #
#          #根据tumor_type从all_ccle_gex筛选出对应的ccl
#          ccle_sample_info = pd.read_csv(data_config.ccle_sample_file, index_col=0)
#          select_ccl = ccle_sample_info.loc[ccle_sample_info.tumor_type == tumor_type].index
#          ccl_gex_df = all_ccle_gex.loc[select_ccl]
#     else:
#         raise NotImplementedError('Not true ccl_match supplied!')
#     keep_samples = target_df.index.isin(ccl_gex_df.index)

#     ccle_labeled_feature_df = target_df.loc[keep_samples][["Drug_smile","label"]]
#     ccle_labeled_feature_df.dropna(inplace=True)
#     # ccle_labeled_feature_df = ccle_labeled_feature_df.merge(ccl_gex_df,left_index=True,right_index=True)
#     ccl_gex_df = ccl_gex_df.loc[ccl_gex_df.index.str.startswith('ACH')] #替换上一行
    
#     # drug_emb = pd.read_csv("../data/preprocessed_dat/drug_embedding/CCL_dataset/supervised_infomax.pth300_results.csv",index_col=0)
#     # ccle_labeled_feature_df =  ccle_labeled_feature_df.merge(drug_emb,left_on="Drug_smile",right_on="Drug_smile") #gex,label(1),drug(300)
#     # del ccle_labeled_feature_df['Drug_smile'] 
#     # ccle_labeled_feature_df.dropna(inplace=True)

#     ccle_labels = ccle_labeled_feature_df['label']
#     # ccle_labeled_feature_df.to_csv("../see.csv")
#     # del ccle_labeled_feature_df["Drug_name"] 
    

#     s_kfold = StratifiedKFold(n_splits=5, random_state=seed, shuffle=True)
#     for train_index, test_index in s_kfold.split(ccle_labeled_feature_df.values, ccle_labels.values):
#         train_labeled_ccle_df, test_labeled_ccle_df = ccle_labeled_feature_df.iloc[train_index], \
#                                                       ccle_labeled_feature_df.iloc[test_index]
        
#         train_labeled_ccle_dateset = FTDataset(
#             ccl_gex_df,
#             train_labeled_ccle_df.index.values,
#             train_labeled_ccle_df.Drug_smile.values,
#             train_labeled_ccle_df.label.values
#         )

#         test_labeled_ccle_dateset = FTDataset(
#             ccl_gex_df,
#             test_labeled_ccle_df.index.values,
#             test_labeled_ccle_df.Drug_smile.values,
#             test_labeled_ccle_df.label.values
#         )
        
#         train_labeled_ccle_dataloader = DataLoader(train_labeled_ccle_dateset,
#                                                    batch_size=batch_size,
#                                                 #    num_workers = 8,
#                                                 #    pin_memory=True,
#                                                    shuffle=True)

#         test_labeled_ccle_dataloader = DataLoader(test_labeled_ccle_dateset,
#                                                   batch_size=batch_size,
#                                                 #   num_workers = 8,
#                                                 #    pin_memory=True,
#                                                    shuffle=True)

#         yield train_labeled_ccle_dataloader, test_labeled_ccle_dataloader


#根据预训练的癌种所筛选的基因，选择其他癌种的表达谱
def get_all_tcga_ZS_dataloaders(all_patient_gex, batch_size,Tumor_type_list, label_type = "PFS",q=2):
    patient_sample_info = pd.read_csv("../data/preprocessed_dat/xena_sample_info.csv", index_col=0) 
   
    for TT in Tumor_type_list:
        print(f'Generating Zero-shot dataset: {TT}')
        patient_samples = all_patient_gex.index.intersection(patient_sample_info.loc[patient_sample_info.tumor_type == TT].index)
        gex_features_df = all_patient_gex.loc[patient_samples]
        # print(gex_features_df)
        TT_ZS_dataloaders,_ = get_tcga_ZS_dataloaders(gex_features_df, 
                                                    batch_size, label_type = label_type,q=q,tumor_type = TT)
        yield TT_ZS_dataloaders, TT


def get_tcga_ZS_dataloaders(gex_features_df, batch_size, label_type = "PFS",q=2,tumor_type = "TCGA"):
    # Return: Dataloader 内部是dataframe，index是TCGA_id，smiles,gex,label
    # label_type: 'PFS'    'Imaging'
    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()

    response_df = pd.read_csv('../data/tcga/PDR_data/tcga698_single_drug_response_df.csv',index_col=1)
    response_type_df = pd.read_csv('../data/tcga/PDR_data/tcga667_single_drug_response_type_df.csv',index_col=1)

    if tumor_type != "TCGA":
        response_df = response_df.loc[response_df['tcga_project'] == tumor_type]
        response_type_df = response_type_df[response_type_df['tcga_project'] == tumor_type]
   
    response_df = response_df[['days_to_new_tumor_event_after_initial_treatment','smiles','cid']]
    response_type_df = response_type_df[['treatment_best_response','smiles','cid']]

    if label_type == "PFS":
        tcga_drug_gex = response_df.merge(tcga_gex_feature_df,left_index=True,right_index=True)
    elif label_type == "Imaging":
        tcga_drug_gex = response_type_df.merge(tcga_gex_feature_df,left_index=True,right_index=True)
        
    #读入所有TCGA药物的embed
    drug_emb = pd.read_csv("../data/preprocessed_dat/drug_embedding/TCGA_dataset/supervised_infomax.pth300_results.csv",index_col=0)

    #替换drug embed
    # print(tcga_drug_gex.shape)
    # print(tcga_drug_gex['smiles'].isin(drug_emb['Drug_smile']))
    tcga_drug_gex = tcga_drug_gex.merge(drug_emb,left_on="smiles",right_on="Drug_smile") #gex,label(1),drug(300)
    # print(tcga_drug_gex.shape)
    # exit()

    del tcga_drug_gex['smiles']
    del tcga_drug_gex['Drug_smile']
    cid_list = tcga_drug_gex.iloc[:,1].values
    # print("cid_list: {}".format(cid_list))
    del tcga_drug_gex['cid']
    # tcga_drug_gex.dropna(inplace=True)

    if label_type == "PFS":
        # tcga_drug_gex = response_df.merge(tcga_gex_feature_df,left_index=True,right_index=True)
        drug_label = pd.qcut(tcga_drug_gex['days_to_new_tumor_event_after_initial_treatment'],q,labels = range(0,q))
        del tcga_drug_gex['days_to_new_tumor_event_after_initial_treatment']
    elif label_type == "Imaging":
        # tcga_drug_gex = response_type_df.merge(tcga_gex_feature_df,left_index=True,right_index=True)
        drug_label = np.array(tcga_drug_gex['treatment_best_response'].apply(
            lambda s: s in ['Complete Response']), dtype='int32')
        del tcga_drug_gex['treatment_best_response']
    
    tcga_drug_gex['label'] = drug_label
    # # #加上index
    # tcga_label_df = pd.DataFrame(index = labeled_df.index,data = drug_label,columns=['label'])
    # tcga_label_df.to_csv(f'label_index/{drug}_index.csv')
    
    tcga_drug_gex = tcga_drug_gex.values
    # print(' ')
    print("Zero-shot to {0} num: {1}".format(tumor_type,tcga_drug_gex.shape[0]))
    # print(tcga_drug_gex.shape[1])
    # print(tcga_drug_gex)

    labeled_tcga_dateset = TensorDataset(
        torch.from_numpy(tcga_drug_gex[:,0:tcga_drug_gex.shape[1]-301].astype('float32')), #gex:length-301
        torch.from_numpy(tcga_drug_gex[:,tcga_drug_gex.shape[1]-301:tcga_drug_gex.shape[1]-1].astype('float32')), #drug:300
        torch.from_numpy(tcga_drug_gex[:,tcga_drug_gex.shape[1]-1])   
        )
    # labeled_tcga_dateset = TCGADataset(tcga_drug_gex[:,0], #smiles
    #         tcga_drug_gex[:,2:tcga_drug_gex.shape[1]-1], #gex
    #         tcga_drug_gex[:,tcga_drug_gex.shape[1]-1] #label
    #         )

    labeled_tcga_dataloader = DataLoader(labeled_tcga_dateset,
                                         batch_size=batch_size,
                                         shuffle=False) #保持Patient原本的顺序

    return labeled_tcga_dataloader,cid_list

###

def get_pdr_data_dataloaders(gex_features_df, batch_size, tumor_type = "TCGA"):
    # Return: Dataloader 内部是dataframe，index是TCGA_id，smiles,gex,label
    # label_type: 'PFS'    'Imaging'
    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()
    # print(tcga_gex_feature_df.shape)

    patient_sample_info = pd.read_csv("../data/preprocessed_dat/xena_sample_info.csv", index_col=0) 
    
    # print(tcga_gex_feature_df.index)
    select_index = patient_sample_info.loc[patient_sample_info['tumor_type'] == tumor_type].index.intersection(tcga_gex_feature_df.index)
    # print(select_index)
    if tumor_type != "TCGA":
        tcga_gex_feature_df = tcga_gex_feature_df.loc[select_index]
    
    #读入所有CCL药物的embed
    drug_emb = pd.read_csv("../data/preprocessed_dat/drug_embedding/CCL_dataset/supervised_infomax.pth300_results.csv",index_col=0)
    drug_emb['Drug_smile']="a"

    tcga_gex_feature_df['Drug_smile'] = "a"
    sample_id = tcga_gex_feature_df.index
    tcga_gex_feature_df.reset_index(inplace=True)
    #替换drug embed
    # print(tcga_drug_gex.shape)
    # print(tcga_drug_gex['smiles'].isin(drug_emb['Drug_smile']))
    tcga_drug_gex = tcga_gex_feature_df.merge(drug_emb,left_on="Drug_smile",right_on="Drug_smile") #gex,label(1),drug(300)
    tcga_drug_gex.set_index("index",inplace=True)
    print(f"{tumor_type} Sample_num: {tcga_gex_feature_df.shape[0]}; PDR_test_num: {tcga_drug_gex.shape[0]}")
    # exit()

    del tcga_drug_gex['Drug_smile']

    pdr_data_dateset = TensorDataset(
        torch.from_numpy(tcga_drug_gex.loc[:,~tcga_drug_gex.columns.str.startswith('drug_embedding')].values.astype('float32')), #gex:length-301
        torch.from_numpy(tcga_drug_gex.loc[:,tcga_drug_gex.columns.str.startswith('drug_embedding')].values.astype('float32')) #drug:300
        )

    pdr_data_dataloader = DataLoader(pdr_data_dateset,
                                         batch_size=batch_size,
                                         shuffle=False) #保持Patient原本的顺序

    return (pdr_data_dataloader,sample_id)


def Rebalance_dataset_for_every_drug(Original_dataset):
    drug_list = Original_dataset.Drug_name.drop_duplicates().values
    all_drug = pd.DataFrame()
    # drug = "5-Fluorouracil"
    for drug in drug_list:
        # print(drug)
        drug_data = Original_dataset.loc[Original_dataset.Drug_name == drug]
        drug_data_0 = drug_data.loc[drug_data.label == 0]
        drug_data_1 = drug_data.loc[drug_data.label == 1]
        ratio = drug_data_1.shape[0]/drug_data_0.shape[0]
        if ratio > 1 : 
            ratio = math.floor(ratio)
            drug_data_0 = pd.DataFrame(np.repeat(drug_data_0.values,ratio,axis=0),columns=drug_data.columns)
        elif ratio == 0:
            drug_data_0 = drug_data_0
        else:
            ratio = math.floor(1/ratio)
            drug_data_1 = pd.DataFrame(np.repeat(drug_data_1.values,ratio,axis=0),columns=drug_data.columns)
        drug_data = pd.concat([drug_data_0,drug_data_1])
        # drug_data.label.value_counts()
        all_drug = pd.concat([all_drug,drug_data])
    print(all_drug.iloc[1:5])
    # all_drug.groupby('Drug_name')['label'].value_counts()

    return all_drug