import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader,Dataset
from rdkit import RDLogger 
import math

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)

def get_unlabeled_dataloaders(gex_features_df, seed, batch_size, ccle_only=False):
    """
    Cancer cell lines as source domain, thus s_dataloaders
    Patients as target domain, thus t_dataloaders
    """
    set_seed(seed)
    
    ccle_sample_info_df = pd.read_csv('../data/supple_info/sample_info/ccle_sample_info.csv', index_col=0)
    ccle_sample_info_df = ccle_sample_info_df.reset_index().drop_duplicates(subset="Depmap_id",keep='first').set_index("Depmap_id")
    
    xena_sample_info_df = pd.read_csv('../data/supple_info/sample_info/xena_sample_info.csv', index_col=0)
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



def get_finetune_dataloader_generator(gex_features_df,label_type, sample_size = 0.006,dataset = 'gdsc1_raw', seed = 2020 , batch_size = 64, ccle_measurement='AUC',                                  
                                     n_splits=5,q=2,
                                     tumor_type = "TCGA",
                                     select_drug_method = True):
    """
    sensitive (responder): 1
    resistant (non-responder): 0

    """
    RDLogger.DisableLog('rdApp.*')

    # TCGA_id(index)，smiles(column 0),gex,label(column -1)
    test_labeled_dataloaders,cid_list = get_tcga_ZS_dataloaders(gex_features_df=gex_features_df, 
                                                                    label_type = label_type,
                                                                    batch_size=batch_size,
                                                                    q=2,
                                                                    tumor_type = tumor_type)

    ccle_labeled_dataloader_generator = get_ccl_labeled_dataloader_generator(gex_features_df=gex_features_df,
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

def get_ccl_labeled_dataloader_generator(gex_features_df,tumor_type,cid_list,select_drug_method,sample_size,dataset = 'gdsc_raw', batch_size = 64, seed=2020, 
                                          measurement='AUC', n_splits=5,q=2):
    # measurement = 'Z_score'       'AUC'      'IC50'
    # dataset = 'gdsc1_raw'   'gdsc1_rebalance.csv'

    Dataset_path = '../data/finetune_data/{}.csv'.format(dataset)
    # if select_drug_method == "all":
    #     Dataset_path = '../data/finetune_data/gdsc1_rebalance.csv'
    sensitivity_df = pd.read_csv(Dataset_path,index_col=1)
    # sensitivity_df.dropna(inplace=True)

    # target_df = sensitivity_df.groupby(['Depmap_id', 'Drug_smiles']).mean()
    # target_df = target_df.reset_index()
    ccle_sample_with_gex = pd.read_csv("../data/supple_info/ccle_sample_with_gex.csv",index_col=0)
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
        target_df = sensitivity_df.sample(n = round(sample_size*sensitivity_df.shape[0])) # sample some data randomly
    print(' ')
    
    print("Select {0} dataset {1} / {2}".format(dataset, 
        # round(sample_size*sensitivity_df.shape[0]),#
        target_df.shape[0],
        sensitivity_df.shape[0]
        ))
    print(' ')


    # if ccl_match == "yes": #use match
    #     ccl_gex_df = gex_features_df  
    # elif ccl_match == "no": #use all
    #     ccl_gex_df = all_ccle_gex 
    # elif ccl_match == "match_zs": #
    #      ccle_sample_info = pd.read_csv('../data/supple_info/sample_info/ccle_sample_info.csv', index_col=0)
    #      select_ccl = ccle_sample_info.loc[ccle_sample_info.tumor_type == tumor_type].index
    #      ccl_gex_df = all_ccle_gex.loc[select_ccl]
    # else:
    #     raise NotImplementedError('Not true ccl_match supplied!')
    ccl_gex_df = gex_features_df
    keep_samples = target_df.index.isin(ccl_gex_df.index)

    ccle_labeled_feature_df = target_df.loc[keep_samples][["Drug_smile","label"]]
    ccle_labeled_feature_df.dropna(inplace=True)
    ccle_labeled_feature_df = ccle_labeled_feature_df.merge(ccl_gex_df,left_index=True,right_index=True)

    drug_emb = pd.read_csv("../data/supple_info/drug_embedding/drug_embedding_for_cell_line.csv",index_col=0)
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


def get_tcga_ZS_dataloaders(gex_features_df, batch_size, label_type = "PFS",q=2,tumor_type = "TCGA"):
    '''
    gex_features_df: pd.DataFrame, index: patient_id, columns: gene_id
    label_type: str, "PFS" 
    q: int
    '''
    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()

    response_df = pd.read_csv('../data/zeroshot_data/tcga698_single_drug_response_df.csv',index_col=1)
    response_type_df = pd.read_csv('../data/zeroshot_data/tcga667_single_drug_response_type_df.csv',index_col=1)

    if tumor_type != "TCGA":
        response_df = response_df.loc[response_df['tcga_project'] == tumor_type]
        response_type_df = response_type_df[response_type_df['tcga_project'] == tumor_type]
   
    response_df = response_df[['days_to_new_tumor_event_after_initial_treatment','smiles','cid']]
    response_type_df = response_type_df[['treatment_best_response','smiles','cid']]

    if label_type == "PFS":
        tcga_drug_gex = response_df.merge(tcga_gex_feature_df,left_index=True,right_index=True)
    elif label_type == "Imaging":
        tcga_drug_gex = response_type_df.merge(tcga_gex_feature_df,left_index=True,right_index=True)
        
    # Get the drug embedding of each drug for TCGA patients
    drug_emb = pd.read_csv("../data/supple_info/drug_embedding/drug_embedding_for_patient.csv",index_col=0)

    # print(tcga_drug_gex.shape)
    # print(tcga_drug_gex['smiles'].isin(drug_emb['Drug_smile']))
    tcga_drug_gex = tcga_drug_gex.merge(drug_emb,left_on="smiles",right_on="Drug_smile") #gex,label(1),drug(300)
    # print(tcga_drug_gex.shape)

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
                                         shuffle=False) # set shuffle Flase

    return labeled_tcga_dataloader,cid_list


# Generating Zero-shot dataset for specific tumor type
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

# Generating PDR dataset for specific tumor type
def get_pdr_data_dataloaders(gex_features_df, batch_size, tumor_type = "TCGA"):
    # Return: Dataloader: TCGA_id，smiles,gex,label
    tcga_gex_feature_df = gex_features_df.loc[gex_features_df.index.str.startswith('TCGA')]
    tcga_gex_feature_df.index = tcga_gex_feature_df.index.map(lambda x: x[:12])
    tcga_gex_feature_df = tcga_gex_feature_df.groupby(level=0).mean()
    # print(tcga_gex_feature_df.shape)

    patient_sample_info = pd.read_csv("../data/supple_info/sample_info/xena_sample_info.csv", index_col=0) 
    
    # print(tcga_gex_feature_df.index)
    select_index = patient_sample_info.loc[patient_sample_info['tumor_type'] == tumor_type].index.intersection(tcga_gex_feature_df.index)
    # print(select_index)
    if tumor_type != "TCGA":
        tcga_gex_feature_df = tcga_gex_feature_df.loc[select_index]
    
    drug_emb = pd.read_csv("../data/supple_info/drug_embedding/drug_embedding_for_cell_line.csv",index_col=0)
    drug_emb['Drug_smile']="a"

    tcga_gex_feature_df['Drug_smile'] = "a"
    sample_id = tcga_gex_feature_df.index
    tcga_gex_feature_df.reset_index(inplace=True)
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
                                         shuffle=False) # not shuffle to save the order of patient

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