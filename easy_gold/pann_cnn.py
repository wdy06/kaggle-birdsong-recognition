import logging
import os
import random
import time
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from IPython.display import Audio
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold

import audioread
import hydra
import librosa
import librosa.display as display
import pann_utils
import soundfile as sf
import utils
from catalyst.dl import CheckpointCallback, SupervisedRunner

# from fastprogress import progress_bar


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/config_pann.yml")
def main(cfg):
    logger.info(cfg.pretty())
    logger.info(os.getcwd())
    logger.info(hydra.utils.get_original_cwd())
    utils.seed_everything()

    if cfg.debug:
        logger.info("running debug mode")
        EPOCH = 1
    else:
        EPOCH = cfg.epoch

    df = pd.read_csv(utils.DATA_DIR / cfg.train_csv)
    # remove row becase XC195038.mp3 cannot load
    df = df.drop(df[df.filename == "XC195038.mp3"].index)
    df = df.drop(
        df[(df.filename == "XC575512.mp3") & (df.ebird_code == "swathr")].index
    )
    df = df.drop(
        df[(df.filename == "XC433319.mp3") & (df.ebird_code == "aldfly")].index
    )
    df = df.drop(
        df[(df.filename == "XC471618.mp3") & (df.ebird_code == "redcro")].index
    )
    train_audio_dir = utils.DATA_DIR / cfg.train_audio_dir
    print(df.shape)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    df["fold"] = -1
    for fold_id, (train_index, val_index) in enumerate(skf.split(df, df["ebird_code"])):
        df.iloc[val_index, -1] = fold_id

    # # check the propotion
    fold_proportion = pd.pivot_table(
        df, index="ebird_code", columns="fold", values="xc_id", aggfunc=len
    )
    print(fold_proportion.shape)

    use_fold = 0
    if cfg.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    warnings.simplefilter("ignore")

    # loaders
    logging.info(f"fold: {use_fold}")
    loaders = {
        "train": data.DataLoader(
            # PANNsDataset(train_file_list, None),
            pann_utils.PANNsDataset(
                df=df.query("fold != @use_fold").reset_index(), datadir=train_audio_dir,
            ),
            shuffle=True,
            drop_last=True,
            **cfg.dataloader,
        ),
        "valid": data.DataLoader(
            # PANNsDataset(val_file_list, None),
            pann_utils.PANNsDataset(
                df=df.query("fold == @use_fold").reset_index(), datadir=train_audio_dir,
            ),
            shuffle=False,
            drop_last=False,
            **cfg.dataloader,
        ),
    }

    # model
    model_config = cfg.model
    model_config["classes_num"] = 527
    model = pann_utils.get_model(model_config)

    if cfg.multi and cfg.gpu:
        logger.info("Using pararell gpu")
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = pann_utils.PANNsLoss().to(device)
    callbacks = [
        pann_utils.F1Callback(input_key="targets", output_key="logits", prefix="f1"),
        pann_utils.mAPCallback(input_key="targets", output_key="logits", prefix="mAP"),
        CheckpointCallback(save_n_best=0),
    ]

    runner = SupervisedRunner(
        device=device, input_key="waveform", input_target_key="targets"
    )
    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=EPOCH,
        verbose=True,
        logdir=f"fold0",
        callbacks=callbacks,
        main_metric="epoch_f1",
        minimize_metric=False,
    )

    logging.info("train all...")
    loaders = {
        "train": data.DataLoader(
            # PANNsDataset(train_file_list, None),
            pann_utils.PANNsDataset(df=df.reset_index(), datadir=train_audio_dir,),
            shuffle=True,
            drop_last=True,
            **cfg.dataloader,
        ),
    }

    # model
    model_config = cfg.model
    model_config["classes_num"] = 527
    model = pann_utils.get_model(model_config)

    if cfg.multi and cfg.gpu:
        logger.info("Using pararell gpu")
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    criterion = pann_utils.PANNsLoss().to(device)
    callbacks = [
        pann_utils.F1Callback(input_key="targets", output_key="logits", prefix="f1"),
        pann_utils.mAPCallback(input_key="targets", output_key="logits", prefix="mAP"),
        CheckpointCallback(save_n_best=0),
    ]

    runner = SupervisedRunner(
        device=device, input_key="waveform", input_target_key="targets"
    )
    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=EPOCH,
        verbose=True,
        logdir=f"all",
        callbacks=callbacks,
        main_metric="epoch_f1",
        minimize_metric=False,
    )

    logger.info(os.getcwd())


if __name__ == "__main__":
    main()
