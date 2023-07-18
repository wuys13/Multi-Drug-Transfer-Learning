import pandas as pd
import numpy as np
import torch

from collections import defaultdict
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, \
    log_loss, auc, precision_recall_curve
from scipy.stats import pearsonr  #pearsonr(self.y_val, y_pred_val[:,0])[0]



def auprc(y_true, y_score):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true=y_true, probas_pred=y_score)
    return auc(lr_recall, lr_precision)


def model_save_check(history, metric_name, tolerance_count=5, reset_count=1):
    save_flag = False
    stop_flag = False
    if 'best_index' not in history:
        history['best_index'] = 0
    if metric_name.endswith('loss'):
        if history[metric_name][-1] <= history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1
    else:
        if history[metric_name][-1] >= history[metric_name][history['best_index']]:
            save_flag = True
            history['best_index'] = len(history[metric_name]) - 1

    # print(history['best_index'],history[metric_name][history['best_index']])
    if len(history[metric_name]) - history['best_index'] > tolerance_count * reset_count and history['best_index'] > 0:
        stop_flag = True
        a,b,c,d = history['best_index'],len(history[metric_name]) - 1,history[metric_name][0],history[metric_name][-1]
        print(f'The best epoch: {a} / {b}')
        print(f'Metric from first to stop: {c} to {d}')

    return save_flag, stop_flag


def eval_ae_epoch(model, data_loader, device, history):
    model.eval()
    avg_loss_dict = defaultdict(float)
    for x_batch in data_loader:
        x_batch = x_batch[0].to(device)
        with torch.no_grad():
            loss_dict = model.loss_function(*(model(x_batch)))
            for k, v in loss_dict.items():
                avg_loss_dict[k] += v.cpu().detach().item() / len(data_loader)

    for k, v in avg_loss_dict.items():
        history[k].append(v)
    return history

def evaluate_target_classification_epoch(classifier, dataloader, device, history,class_num,test_flag = False):
    y_truths = np.array([])
    classifier.eval()

    if class_num == 0:
         y_preds = np.array([])
         for x_gex ,x_smiles, y_batch in dataloader:
            x_gex = x_gex.to(device)
            x_smiles = x_smiles.to(device)
            y_batch = y_batch.to(device)
            # print("x_gex:",x_gex)
            # print("x_smiles",x_smiles)
            # exit()
            with torch.no_grad():
                y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
                # print("y_pred0",classifier(x_smiles,x_gex))
                y_pred = torch.sigmoid(classifier(x_smiles,x_gex)).detach()
                # print("y_pred",y_pred)
                # exit()
                y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])
    elif class_num == 1:
        y_preds = []
        for x_gex ,x_smiles, y_batch in dataloader:
            x_gex = x_gex.to(device)
            x_smiles = x_smiles.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
                # y_pred = torch.sigmoid(classifier(x_smiles,x_gex)).detach() 
                y_pred = classifier(x_smiles,x_gex).detach() 
                y_preds.append( y_pred.cpu().detach().tolist() )
        y_preds = [token for st in y_preds for token in st]
        y_preds = np.array(y_preds)

    else :
        y_preds = []
        for x_gex ,x_smiles, y_batch in dataloader:
            x_gex = x_gex.to(device)
            x_smiles = x_smiles.to(device)
            y_batch = y_batch.to(device)
            with torch.no_grad():
                y_truths = np.concatenate([y_truths, y_batch.cpu().detach().numpy().ravel()])
                # y_pred = torch.sigmoid(classifier(x_smiles,x_gex)).detach() 
                y_pred = nn.functional.softmax(classifier(x_smiles,x_gex),dim=1).detach() 
                y_preds.append( y_pred.cpu().detach().tolist() )
        y_preds = [token for st in y_preds for token in st]
        y_preds = np.array(y_preds)

        if class_num == 2:
            # y_preds = y_preds.gather(-1,nn.init.ones_(torch.zeros(y_preds.size(0), 1))) #for torch
            # print(y_preds)
            y_preds = y_preds[:,1]
            # print(y_preds)
            # exit()
        elif class_num > 2 :
            if test_flag :#做zero shot
                a = torch.from_numpy(y_preds)
                _, max_class = torch.max(a, -1)
                y_preds = torch.sigmoid(max_class).numpy() #x->3->[0,1]
                #numpy
                # max_class = np.argmax(y_preds,-1)
                # y_preds = sigmoid(max_class)
            else:
                y_preds = y_preds
        else:
            raise Exception("class_num error")
    
    # print("y_truths: ",y_truths)
    # print("y_preds: ",y_preds)
    if class_num == 1:
        if test_flag :
            # history['auroc'].append(1-roc_auc_score(y_true=y_truths, y_score=sigmoid(y_preds))) #AUC越大越差
            history['auroc'].append(roc_auc_score(y_true=y_truths, y_score=1-y_preds)) #AUC本身就在01之间，加个负号就行
        else:
            history['auroc'].append(pearsonr(y_truths, y_preds)[0])
    else:
        history['auroc'].append(roc_auc_score(y_true=y_truths, y_score=y_preds,multi_class='ovo')) #暂时只有auroc，多分类下面可能会有问题
        history['auprc'].append(auprc(y_true=y_truths, y_score=y_preds))
        history['acc'].append(accuracy_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
        history['f1'].append(f1_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
        history['aps'].append(average_precision_score(y_true=y_truths, y_score=y_preds))
        history['ce'].append(log_loss(y_true=y_truths, y_pred=y_preds))
    

    return history


def evaluate_adv_classification_epoch(classifier, s_dataloader, t_dataloader, device, history):
    y_truths = np.array([])
    y_preds = np.array([])
    classifier.eval()

    for s_batch in s_dataloader:
        s_x = s_batch[0].to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, np.zeros(s_x.shape[0]).ravel()])
            s_y_pred = torch.sigmoid(classifier(s_x)).detach()
            y_preds = np.concatenate([y_preds, s_y_pred.cpu().detach().numpy().ravel()])

    for t_batch in t_dataloader:
        t_x = t_batch[0].to(device)
        with torch.no_grad():
            y_truths = np.concatenate([y_truths, np.ones(t_x.shape[0]).ravel()])
            t_y_pred = torch.sigmoid(classifier(t_x)).detach()
            y_preds = np.concatenate([y_preds, t_y_pred.cpu().detach().numpy().ravel()])

    history['acc'].append(accuracy_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['auroc'].append(roc_auc_score(y_true=y_truths, y_score=y_preds))
    history['aps'].append(average_precision_score(y_true=y_truths, y_score=y_preds))
    history['f1'].append(f1_score(y_true=y_truths, y_pred=(y_preds > 0.5).astype('int')))
    history['bce'].append(log_loss(y_true=y_truths, y_pred=y_preds))
    history['auprc'].append(auprc(y_true=y_truths, y_score=y_preds))

    return history


# patient drug score matrix
def predict_pdr_score(classifier, pdr_dataloader ,device):
    test_dataloader,patient_index = pdr_dataloader
    y_preds = np.array([])
    classifier.eval()
    for x_gex ,x_smiles in test_dataloader:
            x_gex = x_gex.to(device)
            x_smiles = x_smiles.to(device)
            # print("x_gex:",x_gex)
            # print("x_smiles",x_smiles)
            # exit()
            with torch.no_grad():
                y_pred = torch.sigmoid(classifier(x_smiles,x_gex)).detach()
                # print("y_pred",y_pred)
                # exit()
                y_preds = np.concatenate([y_preds, y_pred.cpu().detach().numpy().ravel()])

    drug_list = pd.read_csv('../data/preprocessed_dat/drug_embedding/CCL_dataset/drug_smiles.csv')
    patient_num = len(patient_index)
    y_preds = y_preds.reshape(patient_num,-1)
    output_df = pd.DataFrame(y_preds,index=patient_index,columns=drug_list['Drug_name'])

    return output_df

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
