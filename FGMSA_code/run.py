import gc
import logging
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from easydict import EasyDict as edict

from .config import get_config, get_config_tune
from .data_loader import MMDataLoader
from .models import AMIO
from .trains import ATIO
from .utils import assign_gpu, count_parameters, setup_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"

logger = logging.getLogger('FGMSA')


def run_train(
        model_name: str,
        seeds=None,
        is_tune: bool = False,
        tune_times: int = 50,
        gpu_id=None,
        num_workers: int = 4,
        model_save_name: str = "",
        config_path: str="./FGMSA_code/config.json",
):
    if gpu_id is None:
        gpu_id = [0]
    if seeds is None:
        seeds = []
    seeds = seeds if seeds != [] else [1111]
    logger.info("======================================== Program Start ========================================")

    if is_tune:
        logger.info(f"Tuning with seed {seeds[0]}")
        initial_args = get_config_tune(model_name, config_path)
        initial_args[
            'model_save_path'] = Path(os.getcwd()) / "models_trained" / f"{initial_args['model_name']}.pth" if model_save_name == "" else Path(os.getcwd()) / "models_trained" / model_save_name
        initial_args['device'] = assign_gpu(gpu_id)
        initial_args['train_mode'] = 'regression'
        torch.cuda.set_device(initial_args['device'])

        res_save_dir = Path(os.getcwd()) / "results" / "tune"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        has_debuged = []
        csv_file = res_save_dir / f"{model_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
            for i in range(len(df)):
                has_debuged.append([df.loc[i, k] for k in initial_args['d_paras']])

        for i in range(tune_times):
            args = edict(**initial_args)
            random.seed(time.time())
            new_args = get_config_tune(model_name, config_path)
            args.update(new_args)

            args['cur_seed'] = i + 1
            logger.info(f"{'-' * 30} Tuning [{i + 1}/{tune_times}] {'-' * 30}")
            logger.info(f"Args: {args}")
            cur_param = [args[k] for k in args['d_paras']]
            if cur_param in has_debuged:
                logger.info(f"This set of parameters has been run. Skip.")
                time.sleep(1)
                continue
            setup_seed(seeds[0])
            print("Args:", args)
            result = _run(args, num_workers, is_tune)
            has_debuged.append(cur_param)

            if Path(csv_file).is_file():
                df2 = pd.read_csv(csv_file)
            else:
                df2 = pd.DataFrame(columns=[k for k in args.d_paras] + [k for k in result.keys()])
            res = [args[c] for c in args.d_paras]
            for col in result.keys():
                value = result[col]
                res.append(value)
            df2.loc[len(df2)] = res
            df2.to_csv(csv_file, index=None)
            logger.info(f"Results saved to {csv_file}.")

    else:
        args = get_config(model_name, config_path)
        args[
            'model_save_path'] = Path(os.getcwd()) / "models_trained" / f"{args['model_name']}.pth" if model_save_name == "" else Path(os.getcwd()) / "models_trained" / model_save_name
        args['device'] = assign_gpu(gpu_id)
        args['train_mode'] = 'regression'
        torch.cuda.set_device(args['device'])

        logger.info("Running with args:")
        logger.info(args)
        logger.info(f"Seeds: {seeds}")
        res_save_dir = Path(os.getcwd()) / "results" / "normal"
        res_save_dir.mkdir(parents=True, exist_ok=True)
        model_results = []
        for i, seed in enumerate(seeds):
            setup_seed(seed)
            args['cur_seed'] = i + 1
            logger.info(f"{'-' * 30} Running with seed {seed} [{i + 1}/{len(seeds)}] {'-' * 30}")

            result = _run(args, num_workers, is_tune)
            logger.info(f"Result for seed {seed}: {result}")
            model_results.append(result)
        criterions = list(model_results[0].keys())

        csv_file = res_save_dir / f"{model_name}.csv"
        if csv_file.is_file():
            df = pd.read_csv(csv_file)
        else:
            df = pd.DataFrame(columns=criterions)

        res = []
        for c in criterions:
            values = [r[c] for r in model_results]
            mean = round(np.max(values) * 100, 2)
            #std = round(np.std(values) * 100, 2)
            res.append(mean)
        df.loc[len(df)] = res
        df.to_csv(csv_file, index=None)
        logger.info(f"Results saved to {csv_file}.")


def _run(args, num_workers=4, is_tune=False, from_sena=False):
    dataloader = MMDataLoader(args, num_workers)
    model = AMIO(args).to(args['device'])
    logger.info(f'The model has {count_parameters(model)} trainable parameters')

    trainer = ATIO().getTrain(args)
    epoch_results = trainer.do_train(model, dataloader, return_epoch_results=from_sena)

    assert Path(args['model_save_path']).exists()
    model.load_state_dict(torch.load(args['model_save_path']))
    model.to(args['device'])
    if is_tune:
        results = trainer.do_test(model, dataloader['test'], mode="TEST")
        Path(args['model_save_path']).unlink(missing_ok=True)
    else:
        results = trainer.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    return results


def run_test(
        model_name: str,
        model_path: str,
        config_path: str,
        gpu_id=None,
        num_workers: int = 4,
):
    if gpu_id is None:
        gpu_id = [0]
    args = get_config(model_name, config_path)
    args['device'] = assign_gpu(gpu_id)
    args['train_mode'] = 'regression'

    dataloader = MMDataLoader(args, num_workers)
    model = AMIO(args)
    model.load_state_dict(torch.load(model_path))
    model.to(args['device'])

    trainer = ATIO().getTrain(args)
    results = trainer.do_test(model, dataloader['test'], mode="TEST")

    del model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    criterions = list(results.keys())
    csv_file = Path(os.getcwd()) / "results" / "test.csv"
    if csv_file.is_file():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)
    res = [model_name]
    for c in criterions:
        res.append(round(results[c] * 100, 2))
    df.loc[len(df)] = res
    df.to_csv(csv_file, index=None)
    logger.info(df)
    logger.info(f"Results saved to {csv_file}.")
