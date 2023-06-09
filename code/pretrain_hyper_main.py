import pandas as pd
import torch
import json
import os
import argparse
import random
import pickle
import itertools

import data
import data_config
import wuys.pdr2.code.train_dsrn_adv as train_dsrn_adv
import wuys.pdr2.code.train_ae_adv as train_ae_adv
import wuys.pdr2.code.train_dsrn as train_dsrn
import train_coral
import train_dae
import train_vae
import train_ae
import wuys.pdr2.code.train_dsrn_mmd as train_dsrn_mmd
import wuys.pdr2.code.train_dsn_mmd as train_dsn_mmd
import wuys.pdr2.code.train_dsn_adnn as train_dsn_adnn


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


def main(args, update_params_dict):
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
    elif args.method == 'vaen':
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
    gex_features_df = pd.read_csv(data_config.gex_feature_file, index_col=0)

    with open(os.path.join('model_save/train_params.json'), 'r') as f:
        training_params = json.load(f)

    training_params['unlabeled'].update(update_params_dict)
    param_str = dict_to_str(update_params_dict)

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

    safe_make_dir(training_params['model_save_folder'])
    random.seed(2020)

    s_dataloaders, t_dataloaders = data.get_unlabeled_dataloaders(
        gex_features_df=gex_features_df,
        seed=2020,
        batch_size=training_params['unlabeled']['batch_size'],
        ccle_only=True
    )

    # start unlabeled training
    encoder, historys = train_fn(s_dataloaders=s_dataloaders,
                                 t_dataloaders=t_dataloaders,
                                 **wrap_training_params(training_params, type='unlabeled'))
    with open(os.path.join(training_params['model_save_folder'], f'unlabel_train_history.pickle'),
              'wb') as f:
        for history in historys:
            pickle.dump(dict(history), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ADSN training and evaluation')
    parser.add_argument('--method', dest='method', nargs='?', default='code_adv',
                        choices=['code_adv', 'dsna', 'dsn', 'code_base', 'code_mmd', 'adae', 'coral', 'dae', 'vae',
                                 'vaen', 'ae'])

    train_group = parser.add_mutually_exclusive_group(required=False)
    train_group.add_argument('--train', dest='retrain_flag', action='store_true')
    train_group.add_argument('--no-train', dest='retrain_flag', action='store_false')
    parser.set_defaults(retrain_flag=True)

    train_group.add_argument('--pdtc', dest='pdtc_flag', action='store_true')
    train_group.add_argument('--no-pdtc', dest='pdtc_flag', action='store_false')
    parser.set_defaults(pdtc_flag=False)

    norm_group = parser.add_mutually_exclusive_group(required=False)
    norm_group.add_argument('--norm', dest='norm_flag', action='store_true')
    norm_group.add_argument('--no-norm', dest='norm_flag', action='store_false')
    parser.set_defaults(norm_flag=True)

    args = parser.parse_args()
    print(f'current config is {args}')
    params_grid = {
        "pretrain_num_epochs": [0, 100, 300],
        "train_num_epochs": [100, 200, 300, 500, 750, 1000, 1500, 2000, 2500, 3000],
        "dop": [0.0, 0.1]
    }

    if args.method not in ['code_adv', 'adsn', 'adae', 'dsnw']:
        params_grid.pop('pretrain_num_epochs')

    keys, values = zip(*params_grid.items())
    update_params_dict_list = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(args.method)
    for param_dict in update_params_dict_list:
        print('Param_dict: %s' % param_dict)
        main(args=args, update_params_dict=param_dict)
