import torch
import os
from evaluation_utils import eval_ae_epoch, model_save_check
from collections import defaultdict
from ae import AE
from mlp import MLP
from loss_and_metrics import mmd_loss
from encoder_decoder import EncoderDecoder


def ae_train_step(ae, s_batch, t_batch, device, optimizer, history, scheduler=None):
    ae.zero_grad()
    ae.train()

    s_x = s_batch[0].to(device)
    t_x = t_batch[0].to(device)

    s_loss_dict = ae.loss_function(*ae(s_x))
    t_loss_dict = ae.loss_function(*ae(t_x))

    optimizer.zero_grad()
    m_loss = mmd_loss(source_features=ae.encode(s_x), target_features=ae.encode(t_x), device=device)
    loss = s_loss_dict['loss'] + t_loss_dict['loss'] + m_loss
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    loss_dict = {k: v.cpu().detach().item() + t_loss_dict[k].cpu().detach().item() for k, v in s_loss_dict.items()}

    for k, v in loss_dict.items():
        history[k].append(v)
    history['mmd_loss'].append(m_loss.cpu().detach().item())

    return history


def train_ae_mmd(s_dataloaders, t_dataloaders, **kwargs):
    """

    :param s_dataloaders:
    :param t_dataloaders:
    :param kwargs:
    :return:
    """
    s_train_dataloader = s_dataloaders[0]
    s_test_dataloader = s_dataloaders[1]

    t_train_dataloader = t_dataloaders[0]
    t_test_dataloader = t_dataloaders[1]

    autoencoder = AE(input_dim=kwargs['input_dim'],
                     latent_dim=kwargs['latent_dim'],
                     hidden_dims=kwargs['encoder_hidden_dims'],
                     dop=kwargs['dop']).to(kwargs['device'])
    classifier = MLP(input_dim=kwargs['latent_dim'],
                     output_dim=1,
                     hidden_dims=kwargs['classifier_hidden_dims'],
                     dop=kwargs['dop']).to(kwargs['device'])
    confounder_classifier = EncoderDecoder(encoder=autoencoder.encoder, decoder=classifier).to(kwargs['device'])

    ae_eval_train_history = defaultdict(list)
    ae_eval_val_history = defaultdict(list)

    if kwargs['retrain_flag']:
        ae_optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=kwargs['lr'])


        # start autoencoder pretraining
        for epoch in range(int(kwargs['pretrain_num_epochs'])):
            if epoch % 50 == 0:
                print(f'----Autoencoder  Pre-Training Epoch {epoch} ----')
            for step, s_batch in enumerate(s_train_dataloader):
                t_batch = next(iter(t_train_dataloader))
                ae_eval_train_history = ae_train_step(ae=autoencoder,
                                                      s_batch=s_batch,
                                                      t_batch=t_batch,
                                                      device=kwargs['device'],
                                                      optimizer=ae_optimizer,
                                                      history=ae_eval_train_history)

            ae_eval_val_history = eval_ae_epoch(model=autoencoder,
                                                data_loader=s_test_dataloader,
                                                device=kwargs['device'],
                                                history=ae_eval_val_history
                                                )
            ae_eval_val_history = eval_ae_epoch(model=autoencoder,
                                                data_loader=t_test_dataloader,
                                                device=kwargs['device'],
                                                history=ae_eval_val_history
                                                )
            for k in ae_eval_val_history:
                if k != 'best_index':
                    ae_eval_val_history[k][-2] += ae_eval_val_history[k][-1]
                    ae_eval_val_history[k].pop()
            # print some loss/metric messages
            if kwargs['es_flag']:
                save_flag, stop_flag = model_save_check(history=ae_eval_val_history, metric_name='loss',
                                                        tolerance_count=10)
                if save_flag:
                    torch.save(autoencoder.state_dict(), os.path.join(kwargs['model_save_folder'], 'ae.pt'))
                if stop_flag:
                    break

        if kwargs['es_flag']:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'ae.pt')))

    else:
        try:
            autoencoder.load_state_dict(torch.load(os.path.join(kwargs['model_save_folder'], 'ae.pt')))
        except FileNotFoundError:
            raise Exception("No pre-trained encoder")

    return autoencoder.encoder, (ae_eval_train_history,
                                 ae_eval_val_history)
