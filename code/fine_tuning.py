import sys
sys.path.append('./model_train')

from evaluation_utils import evaluate_target_classification_epoch,model_save_check,predict_pdr_score
from collections import defaultdict
from itertools import chain
from mlp import MLP
from encoder_decoder import EncoderDecoder 
import os
import torch
import torch.nn as nn


def fine_tune_encoder_drug(encoder, train_dataloader, val_dataloader, fold_count,store_dir, task_save_folder,test_dataloader=None,
                      metric_name='auroc',
                      class_num = 2,
                      normalize_flag=False,
                      break_flag=False,
                      test_df=None,
                      drug_emb_dim=128,
                      to_roughly_test = False,
                      **kwargs):
    finetune_output_dim = class_num
    if finetune_output_dim == 0:
        finetune_output_dim = 1
    target_decoder = MLP(input_dim=kwargs['latent_dim'] + drug_emb_dim,
                         output_dim=finetune_output_dim,
                         hidden_dims=kwargs['classifier_hidden_dims']).to(kwargs['device'])
    
    target_classifier = EncoderDecoder(encoder=encoder, decoder=target_decoder, 
                                        normalize_flag=normalize_flag).to(kwargs['device'])
    # target_decoder load to re-train faster
    target_classifier_file = os.path.join(store_dir, 'save_classifier_0.pt')
    if os.path.exists(target_classifier_file) and fold_count>0:
        print("Loading ",target_classifier_file)
        target_classifier.load_state_dict( 
                    torch.load(target_classifier_file))
    else:
        to_roughly_test = False
        print("No target_classifier_file stored for use. Generate first!")
    # print(' ')

    if class_num == 0:
        classification_loss = nn.BCEWithLogitsLoss()
    elif class_num == 1:
        classification_loss = nn.MSELoss()  ## nn.MAELoss()
    else:
        classification_loss = nn.CrossEntropyLoss()

    target_classification_train_history = defaultdict(list)
    target_classification_eval_train_history = defaultdict(list)
    target_classification_eval_val_history = defaultdict(list)
    target_classification_eval_test_history = defaultdict(list)

    encoder_module_indices = [i for i in range(len(list(encoder.modules())))
                              if str(list(encoder.modules())[i]).startswith('Linear')]

    reset_count = 1
    lr = kwargs['lr']

    # target_classification_params = [target_classifier.decoder.parameters(),target_classifier.smiles_encoder.parameters()] #原来只更新decoder的层
    target_classification_params = [target_classifier.decoder.parameters()] #原来只更新decoder的层

    target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params),
                                                        lr=lr)

    stop_flag_num = 0
    for epoch in range(kwargs['train_num_epochs']):
        # if epoch % 50 == 0:
        #     print(f'Fine tuning epoch {epoch}')
        for step, batch in enumerate(train_dataloader):
            target_classification_train_history = classification_train_step_drug(model=target_classifier,
                                                                            batch=batch,
                                                                            loss_fn=classification_loss,
                                                                            device=kwargs['device'],
                                                                            optimizer=target_classification_optimizer,
                                                                            history=target_classification_train_history,
                                                                            class_num = class_num)
        target_classification_eval_train_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                        dataloader=train_dataloader,
                                                                                        device=kwargs['device'],
                                                                                        history=target_classification_eval_train_history,
                                                                                      class_num = class_num)
        target_classification_eval_val_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                      dataloader=val_dataloader,
                                                                                      device=kwargs['device'],
                                                                                      history=target_classification_eval_val_history,
                                                                                      class_num = class_num)

        if test_dataloader is not None:
            target_classification_eval_test_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                           dataloader=test_dataloader,
                                                                                           device=kwargs['device'],
                                                                                           history=target_classification_eval_test_history,
                                                                                           class_num = class_num,
                                                                                           test_flag=True)
        save_flag, stop_flag = model_save_check(history=target_classification_eval_val_history,
                                                metric_name=metric_name,
                                                tolerance_count=5, # stop_flag once 5 epochs not better
                                                reset_count=reset_count)
        test_metric = target_classification_eval_test_history[metric_name][target_classification_eval_val_history['best_index']]
        if epoch % 50 == 0:
            print(f'Fine tuning epoch {epoch}. stop_flag_num: {stop_flag_num}. {test_metric}')
            # print(save_flag, stop_flag,stop_flag_num)
        if save_flag: 
            torch.save(target_classifier.state_dict(),
                       os.path.join(task_save_folder, f'target_classifier_{fold_count}.pt'))  # save model
            # print("Save model. ",test_metric,epoch)
            if to_roughly_test:
                print("To roughly test pass, just get the zero-shot metric at the begining for judge.")
                break
            pass
        if stop_flag: 
            stop_flag_num = stop_flag_num+1
            print(' ')
            try:
                ind = encoder_module_indices.pop()
                print(f'Unfreezing Linear {ind} in the epoch {epoch}. {test_metric}')
                target_classifier.load_state_dict(
                    torch.load(os.path.join(task_save_folder, f'target_classifier_{fold_count}.pt')))

                target_classification_params.append(list(target_classifier.encoder.modules())[ind].parameters())
                lr = lr * kwargs['decay_coefficient']
                target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params), lr=lr)
                reset_count += 1
            except Exception as e:
                print(e)
                print(test_metric)
                break
        # if stop_flag and not break_flag:
        #     print(f'Unfreezing {epoch}')
        #     target_classifier.load_state_dict(
        #         torch.load(os.path.join(task_save_folder, f'target_classifier_{fold_count}.pt')))
        #
        #     target_classification_params.append(target_classifier.encoder.shared_encoder.parameters())
        #     target_classification_params.append(target_classifier.encoder.private_encoder.parameters())
        #
        #     lr = lr * kwargs['decay_coefficient']
        #     target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params), lr=lr)
        #     break_flag = True
        #     stop_flag = False
        # if stop_flag and break_flag:
        #     break

    target_classifier.load_state_dict(
        torch.load(os.path.join(task_save_folder, f'target_classifier_{fold_count}.pt')))


    return target_classifier, (target_classification_train_history, target_classification_eval_train_history,
                               target_classification_eval_val_history, target_classification_eval_test_history)#, prediction_df


def classification_train_step_drug(model, batch, loss_fn, device, optimizer, history, class_num,scheduler=None, clip=None):
    model.zero_grad()
    model.train()

    x_smiles = batch[1].to(device)
    x_gex = batch[0].to(device)
    y = batch[2].to(device)
    # print("smiles:",x_smiles)
    # print("gex:",x_gex)
    # print("label:",y)

    if class_num == 0: 
        loss = loss_fn(model(x_smiles,x_gex), y.double().unsqueeze(1)) #
    else :
        loss = loss_fn(model(x_smiles,x_gex), y.float().unsqueeze(1)) #

    optimizer.zero_grad()
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['ce'].append(loss.cpu().detach().item())
    # history['bce'].append(loss.cpu().detach().item())

    return history


#predict pdr result
def predict_pdr(encoder,  fold_count, task_save_folder, test_dataloader=None,
                      metric_name='auroc',
                      class_num = 2,
                      normalize_flag=False,
                      break_flag=False,
                      pdr_dataloader=None,
                      drug_emb_dim=128,
                      **kwargs):
    finetune_output_dim = class_num
    if finetune_output_dim == 0:
        finetune_output_dim = 1
    target_decoder = MLP(input_dim=kwargs['latent_dim'] + drug_emb_dim,
                         output_dim=finetune_output_dim,
                         hidden_dims=kwargs['classifier_hidden_dims']).to(kwargs['device'])

    target_classifier = EncoderDecoder(encoder=encoder, decoder=target_decoder, 
                                        normalize_flag=normalize_flag).to(kwargs['device'])
    target_classifier.load_state_dict(
            torch.load(os.path.join(task_save_folder, f'target_classifier_{fold_count}.pt')))
    print('Sucessfully loaded target_classifier_{}'.format(fold_count))   
    
    target_classification_eval_test_history = defaultdict(list)
    if test_dataloader is not None:
        target_classification_eval_test_history = evaluate_target_classification_epoch_1(classifier=target_classifier,
                                                                                           dataloader=test_dataloader,
                                                                                           device=kwargs['device'],
                                                                                           history=target_classification_eval_test_history,
                                                                                           class_num = class_num,
                                                                                           test_flag=True)
        

    prediction_df = None
    if pdr_dataloader is not None:
        prediction_df = predict_pdr_score(classifier=target_classifier, pdr_dataloader=pdr_dataloader,
                                           device=kwargs['device'])

    return  target_classification_eval_test_history, prediction_df