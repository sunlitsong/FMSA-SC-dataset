import os
import json
import random
from easydict import EasyDict as edict


def get_config(
        model_name,
        config_path
):
    with open(config_path, 'r', encoding='utf-8') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = config_all[model_name]['datasetParams']
    dataset_args = config_all['datasetCommonParams']

    config = {'model_name': model_name, 'dataset_name': 'fgmsa'}
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config = edict(config)

    return config


def get_config_tune(
        model_name, config_path,random_choice=True
):
    with open(config_path, 'r') as f:
        config_all = json.load(f)
    model_common_args = config_all[model_name]['commonParams']
    model_dataset_args = {}
    model_debug_args = config_all[model_name]['debugParams']
    dataset_args = config_all['datasetCommonParams']

    if random_choice:
        for item in model_debug_args['d_paras']:
            if type(model_debug_args[item]) == list:
                model_debug_args[item] = random.choice(model_debug_args[item])
            elif type(model_debug_args[item]) == dict:
                for k, v in model_debug_args[item].items():
                    model_debug_args[item][k] = random.choice(v)

    config = {'model_name': model_name, 'dataset_name': 'fgmsa'}
    config.update(dataset_args)
    config.update(model_common_args)
    config.update(model_dataset_args)
    config.update(model_debug_args)
    config = edict(config)

    return config
