from evaluation_utils import evaluate_target_classification_epoch, evaluate_target_classification_epoch_new,model_save_check, predict_target_classification
from collections import defaultdict
from itertools import chain
from mlp import MLP
from encoder_decoder import EncoderDecoder , EncoderDecoder_drug
import os
import torch
import torch.nn as nn


def classification_train_step(model, batch, loss_fn, device, optimizer, history, scheduler=None, clip=None):
    model.zero_grad()
    model.train()

    x = batch[0].to(device)
    y = batch[1].to(device)
    loss = loss_fn(model(x), y.double().unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['bce'].append(loss.cpu().detach().item())

    return history


def fine_tune_encoder(encoder, train_dataloader, val_dataloader, seed, task_save_folder, test_dataloader=None,
                      metric_name='auroc',
                      normalize_flag=False,
                      break_flag=False,
                      test_df=None,
                      **kwargs):
    target_decoder = MLP(input_dim=kwargs['latent_dim'],
                         output_dim=1,
                         hidden_dims=kwargs['classifier_hidden_dims']).to(kwargs['device'])
    target_classifier = EncoderDecoder(encoder=encoder, decoder=target_decoder, normalize_flag=normalize_flag).to(
        kwargs['device'])
    classification_loss = nn.BCEWithLogitsLoss()

    target_classification_train_history = defaultdict(list)
    target_classification_eval_train_history = defaultdict(list)
    target_classification_eval_val_history = defaultdict(list)
    target_classification_eval_test_history = defaultdict(list)

    encoder_module_indices = [i for i in range(len(list(encoder.modules())))
                              if str(list(encoder.modules())[i]).startswith('Linear')]

    reset_count = 1
    lr = kwargs['lr']

    target_classification_params = [target_classifier.decoder.parameters()] #原来只更新decoder的层
    target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params),
                                                        lr=lr)

    for epoch in range(kwargs['train_num_epochs']):
        if epoch % 50 == 0:
            print(f'Fine tuning epoch {epoch}')
        for step, batch in enumerate(train_dataloader):
            target_classification_train_history = classification_train_step(model=target_classifier,
                                                                            batch=batch,
                                                                            loss_fn=classification_loss,
                                                                            device=kwargs['device'],
                                                                            optimizer=target_classification_optimizer,
                                                                            history=target_classification_train_history)
        target_classification_eval_train_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                        dataloader=train_dataloader,
                                                                                        device=kwargs['device'],
                                                                                        history=target_classification_eval_train_history)
        target_classification_eval_val_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                      dataloader=val_dataloader,
                                                                                      device=kwargs['device'],
                                                                                      history=target_classification_eval_val_history)

        if test_dataloader is not None:
            target_classification_eval_test_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                           dataloader=test_dataloader,
                                                                                           device=kwargs['device'],
                                                                                           history=target_classification_eval_test_history)
        save_flag, stop_flag = model_save_check(history=target_classification_eval_val_history,
                                                metric_name=metric_name,
                                                tolerance_count=10, #10次更新没有提高就stop_flag一次
                                                reset_count=reset_count)
        if save_flag: #这个epoch比之前的都好，就存；10次之后就stop，stop用逐层解冻3次encoder的全连接层
            torch.save(target_classifier.state_dict(),
                       os.path.join(task_save_folder, f'target_classifier_{seed}.pt'))  #存下所有超参（pre[train]epoch,dop）中AUC最大的模型
        if stop_flag: #全连接层为3层，pop4次会IndexError（ind异常）退出 【for epoch in range(kwargs['train_num_epochs']): 】的循环
            try:
                ind = encoder_module_indices.pop()
                print(f'Unfreezing Linear {ind} in the epoch {epoch}')
                target_classifier.load_state_dict(
                    torch.load(os.path.join(task_save_folder, f'target_classifier_{seed}.pt')))

                target_classification_params.append(list(target_classifier.encoder.modules())[ind].parameters())
                lr = lr * kwargs['decay_coefficient']
                target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params), lr=lr)
                reset_count += 1
            except IndexError:
                break
        # if stop_flag and not break_flag:
        #     print(f'Unfreezing {epoch}')
        #     target_classifier.load_state_dict(
        #         torch.load(os.path.join(task_save_folder, f'target_classifier_{seed}.pt')))
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
        torch.load(os.path.join(task_save_folder, f'target_classifier_{seed}.pt')))

    prediction_df = None
    if test_df is not None:
        prediction_df = predict_target_classification(classifier=target_classifier, test_df=test_df,
                                                      device=kwargs['device'])

    return target_classifier, (target_classification_train_history, target_classification_eval_train_history,
                               target_classification_eval_val_history, target_classification_eval_test_history)#, prediction_df

def reproduce_result(train_dataloader, val_dataloader, seed, task_save_folder, test_dataloader=None,
                      metric_name='auroc',
                      normalize_flag=False,
                      break_flag=False,
                      test_df=None,
                      **kwargs):
    encoder = MLP(input_dim=kwargs['input_dim'],
                        output_dim=kwargs['latent_dim'],
                        hidden_dims=kwargs['encoder_hidden_dims'],
                        dop=kwargs['dop']).to(kwargs['device'])

    target_decoder = MLP(input_dim=kwargs['latent_dim'],
                         output_dim=1,
                         hidden_dims=kwargs['classifier_hidden_dims']).to(kwargs['device'])
    target_classifier = EncoderDecoder(encoder=encoder, decoder=target_decoder, normalize_flag=normalize_flag).to(
        kwargs['device'])
    target_classifier.load_state_dict(
            torch.load(os.path.join(task_save_folder, f'target_classifier_{seed}.pt')))
    print('sucessfully loaded target_classifier_{}'.format(seed))
    # prediction_df = None
    # if test_df is not None:
    #     prediction_df = predict_target_classification(classifier=target_classifier, test_df=test_df,
    #                                                   device=kwargs['device'])


    target_classification_eval_test_history = defaultdict(list)
        
    target_classification_eval_test_history, prediction_df = evaluate_target_classification_epoch_new(classifier=target_classifier,
                                                                                           dataloader=test_dataloader,
                                                                                           device=kwargs['device'],
                                                                                           history=target_classification_eval_test_history)
    print(target_classification_eval_test_history[metric_name])
    return target_classification_eval_test_history , prediction_df


#
def TT_result(seed, task_save_folder, test_dataloader=None,
                      metric_name='auroc',
                      normalize_flag=False,
                      break_flag=False,
                      test_df=None,
                      **kwargs):
    encoder = MLP(input_dim=kwargs['input_dim'],
                        output_dim=kwargs['latent_dim'],
                        hidden_dims=kwargs['encoder_hidden_dims'],
                        dop=kwargs['dop']).to(kwargs['device'])

    target_decoder = MLP(input_dim=kwargs['latent_dim'],
                         output_dim=1,
                         hidden_dims=kwargs['classifier_hidden_dims']).to(kwargs['device'])
    target_classifier = EncoderDecoder(encoder=encoder, decoder=target_decoder, normalize_flag=normalize_flag).to(
        kwargs['device'])
    target_classifier.load_state_dict(
            torch.load(os.path.join(task_save_folder, f'target_classifier_{seed}.pt')))
    print('sucessfully loaded target_classifier_{}'.format(seed))
    # prediction_df = None
    # if test_df is not None:
    #     prediction_df = predict_target_classification(classifier=target_classifier, test_df=test_df,
    #                                                   device=kwargs['device'])


    target_classification_eval_test_history = defaultdict(list)
        
    target_classification_eval_test_history, prediction_df = evaluate_target_classification_epoch_new(classifier=target_classifier,
                                                                                           dataloader=test_dataloader,
                                                                                           device=kwargs['device'],
                                                                                           history=target_classification_eval_test_history)
    print(target_classification_eval_test_history[metric_name])
    return target_classification_eval_test_history , prediction_df


def fine_tune_encoder_drug(encoder, train_dataloader, val_dataloader, seed, task_save_folder, test_dataloader=None,
                      metric_name='auroc',
                      normalize_flag=False,
                      break_flag=False,
                      test_df=None,
                      drug_emb_dim=128,
                      **kwargs):
    target_decoder = MLP(input_dim=kwargs['latent_dim'] + drug_emb_dim,
                         output_dim=1,
                         hidden_dims=kwargs['classifier_hidden_dims']).to(kwargs['device'])

    target_classifier = EncoderDecoder_drug(encoder=encoder, decoder=target_decoder, drug_emb_dim=drug_emb_dim, 
                                        device=kwargs['device'], normalize_flag=normalize_flag).to(kwargs['device'])
        
    classification_loss = nn.BCEWithLogitsLoss()

    target_classification_train_history = defaultdict(list)
    target_classification_eval_train_history = defaultdict(list)
    target_classification_eval_val_history = defaultdict(list)
    target_classification_eval_test_history = defaultdict(list)

    encoder_module_indices = [i for i in range(len(list(encoder.modules())))
                              if str(list(encoder.modules())[i]).startswith('Linear')]

    reset_count = 1
    lr = kwargs['lr']

    target_classification_params = [target_classifier.decoder.parameters(),target_classifier.smiles_encoder.parameters()] #原来只更新decoder的层
    # target_classification_params = [target_classifier.decoder.parameters()] #原来只更新decoder的层

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
                                                                            history=target_classification_train_history)
        target_classification_eval_train_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                        dataloader=train_dataloader,
                                                                                        device=kwargs['device'],
                                                                                        history=target_classification_eval_train_history)
        target_classification_eval_val_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                      dataloader=val_dataloader,
                                                                                      device=kwargs['device'],
                                                                                      history=target_classification_eval_val_history)

        if test_dataloader is not None:
            target_classification_eval_test_history = evaluate_target_classification_epoch(classifier=target_classifier,
                                                                                           dataloader=test_dataloader,
                                                                                           device=kwargs['device'],
                                                                                           history=target_classification_eval_test_history)
        save_flag, stop_flag = model_save_check(history=target_classification_eval_val_history,
                                                metric_name=metric_name,
                                                tolerance_count=10, #10次更新没有提高就stop_flag一次
                                                reset_count=reset_count)
        
        if epoch % 50 == 0:
            print(f'Fine tuning epoch {epoch}. stop_flag_num: {stop_flag_num}')
            # print(save_flag, stop_flag,stop_flag_num)
        if save_flag: #这个epoch比之前的都好，就存；10次之后就stop，stop用逐层解冻3次encoder的全连接层
            torch.save(target_classifier.state_dict(),
                       os.path.join(task_save_folder, f'target_classifier_{seed}.pt'))  #存下所有超参（pre[train]epoch,dop）中AUC最大的模型
        if stop_flag: #全连接层为3层，pop4次会IndexError（ind异常）退出 【for epoch in range(kwargs['train_num_epochs']): 】的循环
            stop_flag_num = stop_flag_num+1
            print(' ')
            try:
                ind = encoder_module_indices.pop()
                print(f'Unfreezing Linear {ind} in the epoch {epoch}')
                target_classifier.load_state_dict(
                    torch.load(os.path.join(task_save_folder, f'target_classifier_{seed}.pt')))

                target_classification_params.append(list(target_classifier.encoder.modules())[ind].parameters())
                lr = lr * kwargs['decay_coefficient']
                target_classification_optimizer = torch.optim.AdamW(chain(*target_classification_params), lr=lr)
                reset_count += 1
            except Exception as e:
                print(e)
                print(target_classification_eval_test_history[metric_name][target_classification_eval_val_history['best_index']])
                break
        # if stop_flag and not break_flag:
        #     print(f'Unfreezing {epoch}')
        #     target_classifier.load_state_dict(
        #         torch.load(os.path.join(task_save_folder, f'target_classifier_{seed}.pt')))
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
        torch.load(os.path.join(task_save_folder, f'target_classifier_{seed}.pt')))

    prediction_df = None
    if test_df is not None:
        prediction_df = predict_target_classification(classifier=target_classifier, test_df=test_df,
                                                      device=kwargs['device'])

    return target_classifier, (target_classification_train_history, target_classification_eval_train_history,
                               target_classification_eval_val_history, target_classification_eval_test_history)#, prediction_df


def classification_train_step_drug(model, batch, loss_fn, device, optimizer, history, scheduler=None, clip=None):
    model.zero_grad()
    model.train()

    x_smiles = batch[0]
    x_gex = batch[1].to(device)
    y = batch[2].to(device)
    loss = loss_fn(model(x_smiles,x_gex), y.double().unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    if clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    history['bce'].append(loss.cpu().detach().item())

    return history